"""Sequences and structures for the entities in the pathway node space.

Three lookups, one per node type, for either organism:

  gene        CDS nucleotide sequence
  enzyme      amino-acid sequence  (by UniProt accession, and by EC + organism)
  metabolite  SMILES

    python build_sequences.py --organism ecoli --what all
    python build_sequences.py --organism human --what cds

Where each thing comes from, and why:

  gene CDS, human      Ensembl REST. The canonical transcript per gene, then
                       its CDS. Done through the BATCH endpoints (POST
                       /lookup/id, 300 ids; POST /sequence/id, 50 ids) rather
                       than one request per gene — 2,848 genes is 5,696 serial
                       calls the naive way and about 65 batched.
  gene CDS, E. coli    NCBI ASM584v2 cds_from_genomic.fna.gz, 1.4 MB, indexed
                       on [locus_tag=b####]. E. coli K-12 lives in Ensembl
                       Bacteria, which rest.ensembl.org does not serve, and one
                       genome file is more reliable than 1,516 REST calls.
  enzyme AA            UniProt. The GEMs already carry a UniProt accession per
                       gene (1,515/1,516 for iML1515, geneUniProtID for
                       Human-GEM), so the accession route is exact and is the
                       primary one. The EC + organism route is also built,
                       because that is the query you want when you have a
                       reaction and no gene: it is many-to-many by nature
                       (one EC returns every isozyme) and is written to its
                       own file rather than mixed into the per-gene set.
  metabolite SMILES    MetaNetX chem_prop.tsv. Both GEMs annotate metabolites
                       with an MNXM id (1,804/1,877 for iML1515), and one
                       274 MB download resolves every one of them; the
                       per-compound alternative is thousands of ChEBI or
                       PubChem calls.

Compartments are dropped for SMILES: iML1515's 1,877 metabolites are only
1,169 distinct chemical species, and atp_c and atp_p are the same molecule.
"""

import argparse
import csv
import gzip
import io
import json
import pickle
import re
import sys
import time
import urllib.request
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

ENSEMBL = "https://rest.ensembl.org"
UNIPROT = "https://rest.uniprot.org/uniprotkb"
MNX = "https://www.metanetx.org/cgi-bin/mnxget/mnxref"
METANETX = f"{MNX}/chem_prop.tsv"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey"
CHEBI_STRUCT = ("https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/structures.tsv.gz")
NCBI_ECOLI_CDS = ("https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
                  "GCF_000005845.2_ASM584v2/"
                  "GCF_000005845.2_ASM584v2_cds_from_genomic.fna.gz")

TAXID = {"human": 9606, "ecoli": 83333}      # 83333 = E. coli K-12

# UniProt accession format. Human-GEM's geneUniProtID column is not always an
# accession — ENSG00000135070 carries the gene SYMBOL "ISCA1" — and the
# /accessions endpoint rejects the WHOLE batch of 100 over one malformed id,
# so these have to be separated out before the request rather than after.
ACC_RE = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]"
                    r"|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "GraphPathModel/1.0"})


# ── transport ──────────────────────────────────────────────────────────────
#
# Ensembl returns transient 500s and 503s under load — not rarely, roughly one
# call in five during a full run — so every request retries with backoff and
# honours Retry-After on 429. Without this a build fails a third of the way in
# and the partial FASTA looks like missing annotation rather than a dropped
# connection.

def _req(method, url, tries=8, **kw):
    delay = 2.0
    for attempt in range(tries):
        try:
            r = SESSION.request(method, url, timeout=300, **kw)
        except requests.RequestException as e:
            if attempt == tries - 1:
                raise
            print(f"      {type(e).__name__}, retry {attempt+1}/{tries-1}", flush=True)
            time.sleep(delay); delay = min(delay * 2, 120)
            continue
        if r.ok:
            return r
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", delay))
            print(f"      rate limited, waiting {wait:.0f}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code in (500, 502, 503, 504) and attempt < tries - 1:
            print(f"      {r.status_code}, retry {attempt+1}/{tries-1}", flush=True)
            time.sleep(delay); delay = min(delay * 2, 120)
            continue
        r.raise_for_status()
    return r


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def write_fasta(path, records, width=60):
    """records: iterable of (id, description, sequence)."""
    with open(path, "w") as f:
        for rid, desc, seq in records:
            f.write(f">{rid}{(' ' + desc) if desc else ''}\n")
            for i in range(0, len(seq), width):
                f.write(seq[i:i + width] + "\n")
    return path


def read_fasta(handle):
    rid, buf = None, []
    for line in handle:
        line = line.rstrip("\n")
        if line.startswith(">"):
            if rid is not None:
                yield rid, "".join(buf)
            rid, buf = line[1:], []
        else:
            buf.append(line)
    if rid is not None:
        yield rid, "".join(buf)


# ── the single-id Ensembl API, kept for interactive use ────────────────────
#
# These three are the shape you reach for when checking one gene by hand. The
# bulk path below does NOT call them — it uses the POST endpoints, which are
# roughly 90x fewer requests for the same answer — but they are the readable
# reference for what that path is doing.

def get_transcripts(gene_id):
    r = SESSION.get(f"{ENSEMBL}/lookup/id/{gene_id}", params={"expand": 1},
                    headers={"Content-Type": "application/json"}, timeout=120)
    if not r.ok:
        print(f"Failed to get transcripts for {gene_id}")
        return []
    return [t["id"] for t in r.json().get("Transcript", [])]


def get_canonical_transcript(gene_id):
    r = SESSION.get(f"{ENSEMBL}/lookup/id/{gene_id}",
                    headers={"Content-Type": "application/json"}, timeout=120)
    if not r.ok:
        print(f"Failed to get canonical transcript for {gene_id}")
        return None
    return r.json().get("canonical_transcript")


def get_cds(transcript_id):
    r = SESSION.get(f"{ENSEMBL}/sequence/id/{transcript_id}", params={"type": "cds"},
                    headers={"Content-Type": "text/x-fasta"}, timeout=120)
    if not r.ok:
        print(f"Failed to get CDS for {transcript_id}")
        return None
    return r.text


# ── gene CDS ───────────────────────────────────────────────────────────────

def _ensembl_batch(url, ids, harvest, batch, cache_path, label):
    """
    POST `ids` to `url` in batches, `harvest(response_json, out)` filling the
    result dict, with two properties that matter over a 2,848-gene run:

    RESUMABLE. Results are checkpointed to `cache_path` after every batch, and
    a rerun starts from what is already there. Ensembl 503s often enough that
    a 40-minute build has a real chance of dying somewhere in the middle, and
    losing 30 minutes of successful calls to the last failed one is the
    difference between this being usable and not.

    ADAPTIVE. A batch that still fails after `_req` has exhausted its retries
    is split in half and each half retried, down to single ids. A whole batch
    is then never lost to one bad id, and a genuinely dead id costs one
    request instead of the whole chunk.
    """
    out = {}
    if cache_path.exists():
        out = json.loads(cache_path.read_text())
        ids = [i for i in ids if i not in out]
        print(f"    {label}: {len(out):,} already cached, {len(ids):,} to fetch")
    if not ids:
        return out

    def run(part, depth=0):
        try:
            r = _req("POST", url,
                     headers={"Content-Type": "application/json",
                              "Accept": "application/json"},
                     json={"ids": part})
            harvest(r.json(), out)
            return
        except Exception as e:
            if len(part) == 1:
                print(f"      dropping {part[0]}: {type(e).__name__}", flush=True)
                return
            mid = len(part) // 2
            print(f"      splitting {len(part)} -> {mid}+{len(part)-mid} "
                  f"after {type(e).__name__}", flush=True)
            run(part[:mid], depth + 1)
            run(part[mid:], depth + 1)

    done = 0
    for part in chunks(ids, batch):
        run(part)
        done += len(part)
        cache_path.write_text(json.dumps(out))
        print(f"    {label} {done:>5}/{len(ids)}  {len(out):,} held", flush=True)
    return out


def canonical_transcripts(gene_ids, cache, batch=100):
    """{gene_id: transcript_id}, version suffix stripped."""
    def harvest(js, out):
        for gid, rec in js.items():
            # a retired or non-human id comes back as null, not as an error
            if rec and rec.get("canonical_transcript"):
                out[gid] = rec["canonical_transcript"].split(".")[0]
    return _ensembl_batch(f"{ENSEMBL}/lookup/id", list(gene_ids), harvest,
                          batch, cache / "human_canonical.json", "lookup")


def cds_for_transcripts(tids, cache, batch=50):
    """{transcript_id: cds}. 50 is Ensembl's POST /sequence/id ceiling."""
    def harvest(js, out):
        for rec in js:
            if isinstance(rec, dict) and rec.get("seq"):
                out[rec["id"].split(".")[0]] = rec["seq"]
    return _ensembl_batch(f"{ENSEMBL}/sequence/id?type=cds", list(tids), harvest,
                          batch, cache / "human_cds.json", "cds")


def human_cds(gene_ids, out_dir, cache):
    print(f"  {len(gene_ids):,} Ensembl gene ids")
    t2g = canonical_transcripts(gene_ids, cache)
    seqs = cds_for_transcripts(sorted(set(t2g.values())), cache)
    recs = [(g, f"transcript={t} len={len(seqs[t])}", seqs[t])
            for g, t in sorted(t2g.items()) if t in seqs]
    p = write_fasta(out_dir / "human_gene_cds.fasta", recs)
    return p, {"genes_requested": len(gene_ids),
               "canonical_transcript_found": len(t2g),
               "cds_written": len(recs)}


_LOCUS = re.compile(r"\[locus_tag=([^\]]+)\]")


def ecoli_cds(gene_ids, out_dir, cache):
    src = cache / "GCF_000005845.2_ASM584v2_cds_from_genomic.fna.gz"
    if not src.exists():
        print(f"  downloading {NCBI_ECOLI_CDS.rsplit('/', 1)[-1]} ...", flush=True)
        req = urllib.request.Request(NCBI_ECOLI_CDS,
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            src.write_bytes(r.read())
    with gzip.open(src, "rt") as fh:
        by_locus = {}
        for header, seq in read_fasta(fh):
            m = _LOCUS.search(header)
            if m:
                by_locus[m.group(1)] = (header, seq)
    print(f"  genome file holds {len(by_locus):,} CDS with a locus_tag")
    want = set(gene_ids)
    recs = [(g, by_locus[g][0].split(" ", 1)[-1][:120], by_locus[g][1])
            for g in sorted(want) if g in by_locus]
    p = write_fasta(out_dir / "ecoli_gene_cds.fasta", recs)
    return p, {"genes_requested": len(want),
               "cds_in_genome_file": len(by_locus),
               "cds_written": len(recs)}


# ── enzyme amino-acid sequences ────────────────────────────────────────────

def uniprot_by_accession(acc_list, batch=100):
    """{accession: (header, sequence)}. 100 is the endpoint's stated ceiling."""
    out = {}
    for i, part in enumerate(chunks(acc_list, batch)):
        r = _req("GET", f"{UNIPROT}/accessions",
                 params={"accessions": ",".join(part), "format": "fasta"})
        for header, seq in read_fasta(io.StringIO(r.text)):
            acc = header.split("|")[1] if "|" in header else header.split()[0]
            out[acc] = (header, seq)
        print(f"    uniprot {min((i+1)*batch, len(acc_list)):>5}/{len(acc_list)}  "
              f"{len(out):,} sequences", flush=True)
    return out


def uniprot_by_symbol(symbols, taxid):
    """{symbol: accession} for entries whose id turned out to be a gene name."""
    out = {}
    for sym in sorted(symbols):
        r = _req("GET", f"{UNIPROT}/search",
                 params={"query": f"gene_exact:{sym} AND organism_id:{taxid} "
                                  f"AND reviewed:true",
                         "format": "list", "size": 1})
        hit = r.text.split("\n")[0].strip()
        if hit:
            out[sym] = hit
    return out


def uniprot_by_ec(ec_codes, taxid, reviewed_first=True):
    """
    {ec: [(accession, header, seq), ...]} — every protein in the organism
    annotated with that EC. Many-to-many on purpose: an EC names a chemical
    activity, not a gene, so a query returns all isozymes carrying it. Kept in
    its own file for that reason.
    """
    out = {}
    for i, ec in enumerate(ec_codes, 1):
        q = f"ec:{ec} AND organism_id:{taxid}"
        if reviewed_first:
            q += " AND reviewed:true"
        r = _req("GET", f"{UNIPROT}/search",
                 params={"query": q, "format": "fasta", "size": 500})
        hits = []
        for header, seq in read_fasta(io.StringIO(r.text)):
            acc = header.split("|")[1] if "|" in header else header.split()[0]
            hits.append((acc, header, seq))
        if hits:
            out[ec] = hits
        if i % 100 == 0 or i == len(ec_codes):
            print(f"    ec {i:>5}/{len(ec_codes)}  {len(out):,} with hits, "
                  f"{sum(len(v) for v in out.values()):,} proteins", flush=True)
    return out


# ── metabolite SMILES ──────────────────────────────────────────────────────

def _mnx_fetch(cache, name, refresh=False, note=""):
    src = cache / name
    if not src.exists() or refresh:
        print(f"  downloading {name}{note} ...", flush=True)
        req = urllib.request.Request(f"{MNX}/{name}",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3600) as r, open(src, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
    return src


def deprecated_map(cache, refresh=False):
    """
    {stale MNXM id: current id}.

    This is not an optional nicety. iML1515's metanetx.chemical annotations
    were assigned against an older MNXref: 646 of its 1,114 distinct species
    point at ids that no longer exist in chem_prop.tsv, so looking them up
    directly finds only 42% of the model. Following the deprecation chain
    first is what turns that into near-complete coverage.
    """
    src = _mnx_fetch(cache, "chem_depr.tsv", refresh, " (2 MB)")
    out = {}
    for line in open(src, encoding="utf8", errors="ignore"):
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0] and p[1]:
            out[p[0]] = p[1]
    # the table records one hop per version, so chains have to be followed
    resolved = {}
    for k in out:
        seen, cur = {k}, out[k]
        while cur in out and cur not in seen:
            seen.add(cur)
            cur = out[cur]
        resolved[k] = cur
    print(f"  chem_depr.tsv: {len(resolved):,} deprecated ids")
    return resolved


def bigg_to_mnx(cache, bigg_ids, refresh=False):
    """{BiGG metabolite id: MNX id} for anything the MNX route could not
    resolve — a second, independent way in, via chem_xref.tsv (678 MB)."""
    if not bigg_ids:
        return {}
    src = _mnx_fetch(cache, "chem_xref.tsv", refresh, " (678 MB, several minutes)")
    want = {f"bigg.metabolite:{b}" for b in bigg_ids}
    out = {}
    for line in open(src, encoding="utf8", errors="ignore"):
        if line.startswith("#"):
            continue
        i = line.find("\t")
        if i > 0 and line[:i] in want:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                out[p[0].split(":", 1)[1]] = p[1]
    print(f"  chem_xref.tsv: recovered {len(out):,} of {len(bigg_ids):,} "
          f"unresolved ids by BiGG id")
    return out


def pubchem_smiles(keys, batch=100, pause=0.25):
    """
    {InChIKey: SMILES} from PubChem.

    Needed because MetaNetX carries structure-less placeholder rows: a BiGG
    compound it could not reconcile still gets an MNXM id, with empty InChI
    and SMILES columns, and the deprecation chain leads straight to it.
    (R)-propane-1,2-diol is one — MNXM90191 -> MNXM1364597, a row whose only
    content is `biggM:12ppd__R`. So "resolved to an MNX id" and "has a
    structure" are different questions, and this answers the second.

    The response does NOT echo the query key, so InChIKey is requested as a
    property alongside SMILES; without it the results cannot be mapped back
    onto the inputs. PubChem allows 5 requests/second.
    """
    keys = sorted({k for k in keys if k})
    out = {}
    for i, part in enumerate(chunks(keys, batch)):
        r = _req("POST", f"{PUBCHEM}/property/InChIKey,SMILES/JSON",
                 data={"inchikey": ",".join(part)})
        try:
            props = r.json()["PropertyTable"]["Properties"]
        except (ValueError, KeyError):
            continue
        for rec in props:
            if rec.get("InChIKey") and rec.get("SMILES"):
                out[rec["InChIKey"]] = rec["SMILES"]
        print(f"    pubchem {min((i+1)*batch, len(keys)):>5}/{len(keys)}  "
              f"{len(out):,} structures", flush=True)
        time.sleep(pause)
    return out


def pubchem_by_cid(cids, batch=100, pause=0.25):
    """{CID: SMILES}. Human-GEM annotates metPubChemID but no InChIKey, so
    this is the route that covers it where MetaNetX has no structure."""
    cids = sorted({str(c).strip() for c in cids if str(c).strip().isdigit()})
    out = {}
    for i, part in enumerate(chunks(cids, batch)):
        r = _req("POST", "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                         "cid/property/SMILES/JSON",
                 data={"cid": ",".join(part)})
        try:
            props = r.json()["PropertyTable"]["Properties"]
        except (ValueError, KeyError):
            continue
        for rec in props:
            if rec.get("SMILES"):
                out[str(rec["CID"])] = rec["SMILES"]
        print(f"    pubchem-cid {min((i+1)*batch, len(cids)):>5}/{len(cids)}  "
              f"{len(out):,} structures", flush=True)
        time.sleep(pause)
    return out


def chebi_smiles(cache, wanted, refresh=False):
    """
    {ChEBI numeric id: SMILES}, from ChEBI's structures flat file.

    The fourth and last source. Human-GEM annotates ChEBI far more often than
    PubChem (649 of its unresolved metabolites against 177), so this is what
    covers the human side where MetaNetX has no structure.

    The file embeds a full molfile per row as a quoted multi-line field, so it
    has to be read with a real CSV reader rather than split on tabs, and the
    field size limit has to be raised. `default_structure` marks the preferred
    structure where a compound has several.
    """
    if not wanted:
        return {}
    src = cache / "chebi_structures.tsv.gz"
    if not src.exists() or refresh:
        print("  downloading ChEBI structures.tsv.gz (89 MB) ...", flush=True)
        req = urllib.request.Request(CHEBI_STRUCT,
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            src.write_bytes(r.read())
    csv.field_size_limit(1 << 30)
    out, n = {}, 0
    with gzip.open(src, "rt", errors="ignore", newline="") as fh:
        rd = csv.reader(fh, delimiter="\t", quotechar='"')
        ci = {c: i for i, c in enumerate(next(rd))}
        for row in rd:
            n += 1
            if len(row) <= ci["default_structure"]:
                continue
            cid, smi = row[ci["compound_id"]], row[ci["smiles"]]
            if smi and cid in wanted and (
                    cid not in out or row[ci["default_structure"]] in ("Y", "1")):
                out[cid] = smi
    print(f"  chebi structures: {n:,} rows scanned, {len(out):,} of the "
          f"{len(wanted):,} ChEBI ids wanted were found")
    return out


def metanetx_smiles(cache, wanted, refresh=False):
    """
    {MNXM id: row} for the ids in `wanted` only, plus the column header.

    chem_prop.tsv is 810 MB and ~1.3 M compounds, of which the two GEMs
    between them reference a few thousand. Holding all of it as split lists
    costs several GB, so this filters during the single streaming pass rather
    than after it. The 388-line comment block at the top carries the licence
    text of every contributing resource; the column names are the LAST of
    those lines, "#ID name reference formula charge mass InChI InChIKey SMILES".
    """
    src = _mnx_fetch(cache, "chem_prop.tsv", refresh, " (810 MB, several minutes)")
    table, header, n = {}, None, 0
    with open(src, "r", encoding="utf8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("#"):
                if line.startswith("#ID"):
                    header = line[1:].rstrip("\n").split("\t")
                continue
            n += 1
            i = line.find("\t")
            if i > 0 and line[:i] in wanted:
                table[line[:i]] = line.rstrip("\n").split("\t")
    print(f"  chem_prop.tsv: {n:,} compounds scanned, {len(table):,} of the "
          f"{len(wanted):,} ids wanted were found")
    print(f"  columns: {header}")
    return table, header


# ── entity lists from the two GEMs ─────────────────────────────────────────

def ecoli_entities():
    m = json.loads((ROOT / "data" / "ecoli" / "iML1515.json").read_text())
    genes = [g["id"] for g in m["genes"]]
    acc = {g["id"]: g.get("annotation", {}).get("uniprot", [None])[0]
           for g in m["genes"]}
    acc = {k: v for k, v in acc.items() if v}
    ecs = sorted({e for r in m["reactions"]
                  for e in r.get("annotation", {}).get("ec-code", [])})
    mets = {}
    for x in m["metabolites"]:
        base = x["id"].rsplit("_", 1)[0]          # atp_c and atp_p are one molecule
        an = x.get("annotation", {})
        mnx = (an.get("metanetx.chemical") or [None])[0]
        if base not in mets or (mnx and not mets[base]["mnx"]):
            mets[base] = {"name": x.get("name", ""), "mnx": mnx, "bigg": base,
                          "inchikey": (an.get("inchi_key") or [""])[0],
                          "cid": "", "chebi": (an.get("chebi") or [""])[0]}
    return genes, acc, ecs, mets


def human_entities():
    import numpy as np
    import scipy.io as sio
    cache = ROOT / "data" / "gem" / "gem_cache"
    meta = json.loads((ROOT / "data" / "gem" / "gem_ref" / "meta.json").read_text())
    genes = list(meta["genes"])

    acc = {}
    with open(cache / "genes.tsv") as fh:
        cols = [c.strip('"') for c in next(fh).rstrip("\n").split("\t")]
        gi, ui = cols.index("genes"), cols.index("geneUniProtID")
        for line in fh:
            p = [c.strip('"') for c in line.rstrip("\n").split("\t")]
            if len(p) > ui and p[ui]:
                acc[p[gi]] = p[ui].split(";")[0]

    g = sio.loadmat(cache / "Human-GEM.mat", struct_as_record=False,
                    squeeze_me=True)["humanGEM"]
    ecs = set()
    for v in np.asarray(g.eccodes, dtype=object).ravel():
        if isinstance(v, str):
            # Human-GEM packs multiple codes per reaction as "1.1.1.1;1.1.1.71"
            ecs.update(e.strip() for e in v.split(";") if e.strip())
    ecs = sorted(ecs)

    mets = {}
    with open(cache / "metabolites.tsv") as fh:
        cols = next(fh).rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(cols)}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            base = p[ci["metsNoComp"]]
            if base and base not in mets:
                def col(k):
                    i = ci[k]
                    return p[i] if i < len(p) else ""
                mets[base] = {"name": col("metBiGGID"), "mnx": col("metMetaNetXID"),
                              "bigg": col("metBiGGID"), "inchikey": "",
                              "cid": col("metPubChemID").split(";")[0],
                              "chebi": col("metChEBIID").split(";")[0]}
    return genes, acc, ecs, mets


# ── driver ─────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organism", choices=["human", "ecoli", "both"], default="both")
    ap.add_argument("--what", default="all",
                    help="comma-separated: cds, protein, ec, smiles, or all")
    ap.add_argument("--out-dir", default="seq_ref")
    ap.add_argument("--cache-dir", default="seq_cache")
    ap.add_argument("--max-ec", type=int, default=None,
                    help="cap the EC queries (one serial request each)")
    ap.add_argument("--no-chebi", action="store_true",
                    help="skip the ChEBI structures fallback")
    ap.add_argument("--no-pubchem", action="store_true",
                    help="skip the PubChem InChIKey fallback")
    ap.add_argument("--no-xref", action="store_true",
                    help="skip the 678 MB chem_xref.tsv fallback")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args(argv)

    what = ({"cds", "protein", "ec", "smiles"} if a.what == "all"
            else {w.strip() for w in a.what.split(",")})
    out = HERE / a.out_dir
    cache = HERE / a.cache_dir
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    orgs = ["human", "ecoli"] if a.organism == "both" else [a.organism]
    manifest_path = out / "manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() and not a.refresh else {})

    # entity lists for every organism first: the SMILES step streams an
    # 810 MB file, and it should do that once for the whole run, not per
    # organism
    ents = {o: (ecoli_entities() if o == "ecoli" else human_entities())
            for o in orgs}
    chem = chem_cols = None
    resolved = {}          # (org, base) -> current MNX id
    if "smiles" in what:
        print("\n-- MetaNetX --")
        depr = deprecated_map(cache, a.refresh)
        # two independent routes to a current MNX id: follow the model's own
        # annotation through the deprecation chain, and look the BiGG id up in
        # the cross-reference table. The first is preferred; the second covers
        # metabolites the model never annotated.
        by_bigg = {}
        if not a.no_xref:
            by_bigg = bigg_to_mnx(cache, {r["bigg"] for _, _, _, mets in ents.values()
                                          for r in mets.values() if r["bigg"]},
                                  a.refresh)
        for o, (_, _, _, mets) in ents.items():
            for base, r in mets.items():
                mnx, bigg = r["mnx"], r["bigg"]
                # Human-GEM packs several ids per metabolite as
                # "MNXM150165;MNXM27815"; passing the whole string as one id
                # resolves nothing
                cur = None
                for one in (mnx or "").split(";"):
                    one = one.strip()
                    if one:
                        cur = depr.get(one, one)
                        break
                resolved[(o, base)] = cur or by_bigg.get(bigg)
        wanted = {v for v in resolved.values() if v}
        chem, chem_cols = metanetx_smiles(cache, wanted, a.refresh)

    for org in orgs:
        print(f"\n{'=' * 74}\n{org}\n{'=' * 74}")
        genes, acc, ecs, mets = ents[org]
        print(f"  {len(genes):,} genes ({len(acc):,} with a UniProt accession), "
              f"{len(ecs):,} distinct EC codes, {len(mets):,} distinct metabolites")
        rec = manifest.setdefault(org, {})

        if "cds" in what:
            print("\n-- gene CDS --")
            p, s = (ecoli_cds(genes, out, cache) if org == "ecoli"
                    else human_cds(genes, out, cache))
            rec["cds"] = {**s, "file": p.name}
            print(f"  -> {p.name}: {s['cds_written']:,}/{s['genes_requested']:,} "
                  f"({s['cds_written']/max(len(genes),1):.0%})")

        if "protein" in what:
            print("\n-- enzyme amino-acid sequences, by accession --")
            odd = sorted({v for v in acc.values() if not ACC_RE.match(v)})
            if odd:
                print(f"  {len(odd)} id(s) are not UniProt accessions: "
                      f"{', '.join(odd[:6])}{' ...' if len(odd) > 6 else ''}")
                fixed = uniprot_by_symbol(odd, TAXID[org])
                print(f"  resolved {len(fixed)} of them as gene symbols: {fixed}")
                acc = {g: fixed.get(v, v) for g, v in acc.items()}
                acc = {g: v for g, v in acc.items() if ACC_RE.match(v)}
            seqs = uniprot_by_accession(sorted(set(acc.values())))
            recs = [(g, seqs[v][0].split(" ", 1)[-1][:120], seqs[v][1])
                    for g, v in sorted(acc.items()) if v in seqs]
            p = write_fasta(out / f"{org}_enzyme_aa.fasta", recs)
            rec["protein"] = {"genes_with_accession": len(acc),
                              "sequences_written": len(recs), "file": p.name}
            print(f"  -> {p.name}: {len(recs):,}/{len(genes):,} genes "
                  f"({len(recs)/max(len(genes),1):.0%})")

        if "ec" in what:
            use = ecs[:a.max_ec] if a.max_ec else ecs
            print(f"\n-- enzyme sequences by EC + organism ({len(use):,} EC codes, "
                  f"taxid {TAXID[org]}) --")
            hits = uniprot_by_ec(use, TAXID[org])
            flat, seen = [], set()
            for ec, lst in sorted(hits.items()):
                for a_, h, s_ in lst:
                    if (ec, a_) not in seen:
                        seen.add((ec, a_))
                        flat.append((f"{ec}|{a_}", h.split(" ", 1)[-1][:120], s_))
            p = write_fasta(out / f"{org}_ec_enzyme_aa.fasta", flat)
            (out / f"{org}_ec_index.json").write_text(json.dumps(
                {ec: [x[0] for x in lst] for ec, lst in hits.items()}, indent=1))
            rec["ec"] = {"ec_queried": len(use), "ec_with_hits": len(hits),
                         "proteins": len(flat), "file": p.name}
            print(f"  -> {p.name}: {len(hits):,}/{len(use):,} EC codes resolved, "
                  f"{len(flat):,} protein entries")

        if "smiles" in what:
            print("\n-- metabolite SMILES --")
            ci = {c: i for i, c in enumerate(chem_cols)}
            si, ki, fi = ci.get("SMILES"), ci.get("InChIKey"), ci.get("formula")

            def mnx_row(base):
                mnx = resolved.get((org, base))
                return mnx, (chem.get(mnx) if mnx else None)

            def field(row, i):
                return row[i] if row and i is not None and i < len(row) else ""

            # MetaNetX carries structure-less placeholder rows, so "resolved"
            # and "has a structure" are different questions. Collect the keys
            # still unanswered after the MNX pass and ask PubChem once.
            # three sources, tried in order: MetaNetX, then PubChem by
            # InChIKey, then PubChem by CID. iML1515 annotates InChIKeys and no
            # CIDs; Human-GEM annotates CIDs and no InChIKeys, so both routes
            # are needed to cover both organisms.
            unresolved = [b for b in mets if not field(mnx_row(b)[1], si)]
            by_key, by_cid, by_chebi = {}, {}, {}
            if unresolved:
                keys = {mets[b]["inchikey"] for b in unresolved if mets[b]["inchikey"]}
                cids = {mets[b]["cid"] for b in unresolved if mets[b]["cid"]}
                chs = {mets[b]["chebi"].replace("CHEBI:", "")
                       for b in unresolved if mets[b]["chebi"]}
                print(f"  {len(unresolved):,} without a MetaNetX structure: "
                      f"{len(keys):,} have an InChIKey, {len(cids):,} a PubChem CID, "
                      f"{len(chs):,} a ChEBI id")
                if not a.no_pubchem:
                    if keys:
                        by_key = pubchem_smiles(keys)
                    if cids:
                        by_cid = pubchem_by_cid(cids)
                if chs and not a.no_chebi:
                    by_chebi = chebi_smiles(cache, chs, a.refresh)

            def q(x):
                return '"' + str(x).replace('"', "'") + '"'

            rows = ["node_id,name,mnx_id,formula,smiles,inchikey,source"]
            hit = from_pubchem = 0
            for base, r in sorted(mets.items()):
                name, ikey, cid = r["name"], r["inchikey"], r["cid"]
                mnx, row = mnx_row(base)
                smi, src = field(row, si), "metanetx"
                key = field(row, ki) or ikey
                if not smi and ikey and ikey in by_key:
                    smi, src = by_key[ikey], "pubchem:inchikey"
                    from_pubchem += 1
                elif not smi and cid and cid in by_cid:
                    smi, src = by_cid[cid], "pubchem:cid"
                    from_pubchem += 1
                elif not smi and r["chebi"].replace("CHEBI:", "") in by_chebi:
                    smi, src = by_chebi[r["chebi"].replace("CHEBI:", "")], "chebi"
                    from_pubchem += 1
                if smi:
                    hit += 1
                else:
                    src = ""
                rows.append(",".join([q(base), q(name), q(mnx or ""),
                                      q(field(row, fi)), q(smi), q(key), q(src)]))
            p = out / f"{org}_metabolite_smiles.csv"
            p.write_text("\n".join(rows) + "\n")
            rec["smiles"] = {"metabolites": len(mets), "with_smiles": hit,
                             "from_pubchem": from_pubchem, "file": p.name}
            print(f"  -> {p.name}: {hit:,}/{len(mets):,} with SMILES "
                  f"({hit/max(len(mets),1):.0%}); "
                  f"{hit-from_pubchem:,} from MetaNetX, "
                  f"{from_pubchem:,} from PubChem/ChEBI")

    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"\nmanifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
