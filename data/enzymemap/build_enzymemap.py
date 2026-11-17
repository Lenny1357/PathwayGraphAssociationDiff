"""EnzymeMap -> reaction table, protein sequences, and a bridge to the GEMs.

EnzymeMap (Heid, Probst, Green & Madsen 2023) is BRENDA's enzymatic reactions
curated, atom-mapped and validated: 349,458 reaction records over 4,552 EC
codes and 7,468 organisms, each carrying atom-mapped and unmapped reaction
SMILES, a reaction rule, a quality score, and — for 30% of rows — UniProt
accessions for the protein that catalyses it.

    python build_enzymemap.py                 # everything
    python build_enzymemap.py --what parse,bridge

It matters to this project for one specific reason. The GEMs say a reaction
EXISTS; EnzymeMap says what it DOES, at the level of which bonds move and
which protein moves them. So it is the reference that can judge a generated
enzyme-reaction or metabolite-reaction edge on chemistry rather than on
whether the pair happens to appear in iML1515 — the gap the notebook's
`edge_precision` cannot cover because 98.5% of its reference is training data.

Three outputs, in `em_ref/`:

  reactions.csv.gz        the per-reaction table, without the bulky `rule` and
                          `mapped` columns (386 MB uncompressed with them)
  proteins.fasta          UniProt sequences for every accession EnzymeMap
                          cites, keyed by accession
  ec_index.csv            per EC code: reactions, organisms, proteins, and
                          whether iML1515 / Human-GEM use that EC
  species.csv             every distinct molecule, canonical SMILES and
                          InChIKey, and whether it appears in either GEM

The bridge is on InChIKey, not on SMILES strings. The same molecule is written
differently by different databases and a string comparison silently misses
most true matches; rdkit canonicalisation followed by InChIKey is what makes
"is this the same metabolite" answerable.
"""

import argparse
import ast
import csv
import gzip
import io
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "data" / "sequences"))

RAW = "https://raw.githubusercontent.com/hesther/enzymemap/main/data"
FILES = {
    "processed_reactions.csv.gz": "the curated reaction table (13 MB)",
    "raw_reactions.csv.gz": "before validation, for provenance (9 MB)",
    "compound_to_smiles.json": "BRENDA compound name -> SMILES (9 MB)",
}
# columns dropped from the derived table: `rule` and `mapped` are most of the
# 386 MB and neither is needed to ask which EC, protein or molecule is involved
KEEP = ["rxn_idx", "unmapped", "orig_rxn_text", "source", "steps", "quality",
        "natural", "organism", "protein_refs", "protein_db", "ec_num"]

csv.field_size_limit(1 << 30)


def fetch(cache, names, refresh=False):
    cache.mkdir(parents=True, exist_ok=True)
    for name in names:
        dst = cache / name
        if dst.exists() and not refresh:
            continue
        print(f"  downloading {name} — {FILES[name]} ...", flush=True)
        req = urllib.request.Request(f"{RAW}/{name}",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=900) as r:
            dst.write_bytes(r.read())
    return cache


def parse(cache, out):
    """Stream the table, write the trimmed copy, and return the summary."""
    src = cache / "processed_reactions.csv.gz"
    ecs, orgs, accs = {}, set(), set()
    ec_orgs, ec_accs = {}, {}
    n = 0
    with gzip.open(src, "rt", errors="ignore", newline="") as fh, \
            gzip.open(out / "reactions.csv.gz", "wt", newline="") as fo:
        rd = csv.DictReader(fh)
        wr = csv.DictWriter(fo, fieldnames=KEEP, extrasaction="ignore")
        wr.writeheader()
        for r in rd:
            n += 1
            wr.writerow(r)
            ec, org = r["ec_num"], r["organism"]
            ecs[ec] = ecs.get(ec, 0) + 1
            orgs.add(org)
            ec_orgs.setdefault(ec, set()).add(org)
            pr = r["protein_refs"]
            if pr and pr != "[]" and r["protein_db"] in ("uniprot", "swissprot"):
                try:
                    ids = ast.literal_eval(pr)
                except (ValueError, SyntaxError):
                    ids = []
                accs.update(ids)
                ec_accs.setdefault(ec, set()).update(ids)
            if n % 100_000 == 0:
                print(f"    {n:,} rows", flush=True)
    print(f"  {n:,} reactions, {len(ecs):,} EC codes, {len(orgs):,} organisms, "
          f"{len(accs):,} UniProt accessions")
    return {"rows": n, "ecs": ecs, "organisms": orgs, "accessions": accs,
            "ec_orgs": ec_orgs, "ec_accs": ec_accs}


def species(cache, limit=None):
    """
    {InChIKey: (canonical SMILES, n_reactions)} over every distinct molecule.

    Deduplicating the SMILES STRINGS first is what makes this affordable:
    349,458 reactions hold ~3.5 M participant strings but only ~30 k distinct
    ones, so rdkit is asked to parse each molecule once rather than 100 times.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    src = cache / "processed_reactions.csv.gz"
    counts = {}
    with gzip.open(src, "rt", errors="ignore", newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if limit and i >= limit:
                break
            for side in r["unmapped"].split(">>"):
                for smi in side.split("."):
                    smi = smi.strip()
                    if smi:
                        counts[smi] = counts.get(smi, 0) + 1
    print(f"  {sum(counts.values()):,} participants, {len(counts):,} distinct "
          f"SMILES strings")

    out, bad = {}, 0
    for j, (smi, c) in enumerate(counts.items(), 1):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            bad += 1
            continue
        try:
            key = Chem.MolToInchiKey(m)
        except Exception:
            bad += 1
            continue
        if not key:
            bad += 1
            continue
        if key in out:
            out[key] = (out[key][0], out[key][1] + c)
        else:
            out[key] = (Chem.MolToSmiles(m), c)
        if j % 5000 == 0:
            print(f"    canonicalised {j:,}/{len(counts):,}", flush=True)
    print(f"  {len(out):,} distinct molecules by InChIKey ({bad:,} unparseable)")
    return out


def gem_ecs():
    """{name: set of EC codes} for the two GEMs, for the EC overlap column."""
    import numpy as np
    import scipy.io as sio
    out = {}
    p = ROOT / "data" / "ecoli" / "iML1515.json"
    if p.exists():
        m = json.loads(p.read_text())
        out["iML1515"] = {e for r in m["reactions"]
                          for e in r.get("annotation", {}).get("ec-code", [])}
    p = ROOT / "data" / "gem" / "gem_cache" / "Human-GEM.mat"
    if p.exists():
        g = sio.loadmat(p, struct_as_record=False, squeeze_me=True)["humanGEM"]
        out["Human-GEM"] = {e.strip()
                            for v in np.asarray(g.eccodes, dtype=object).ravel()
                            if isinstance(v, str)
                            for e in v.split(";") if e.strip()}
    return out


def gem_inchikeys():
    """{name: {InChIKey: node_id}} from the SMILES tables data/sequences built."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = {}
    for name, fn in (("iML1515", "ecoli_metabolite_smiles.csv"),
                     ("Human-GEM", "human_metabolite_smiles.csv")):
        p = ROOT / "data" / "sequences" / "seq_ref" / fn
        if not p.exists():
            print(f"  {fn} missing — run data/sequences/build_sequences.py "
                  f"--what smiles; skipping the {name} molecule bridge")
            continue
        keys = {}
        for row in csv.DictReader(open(p)):
            # prefer the InChIKey already in the table; recompute from SMILES
            # where it is absent, so a metabolite with a structure is never
            # dropped from the bridge just because the key was not annotated
            key = (row.get("inchikey") or "").strip()
            if not key and row.get("smiles"):
                m = Chem.MolFromSmiles(row["smiles"])
                if m is not None:
                    try:
                        key = Chem.MolToInchiKey(m)
                    except Exception:
                        key = ""
            if key:
                keys.setdefault(key, row["node_id"])
        out[name] = keys
        print(f"  {name}: {len(keys):,} metabolites with an InChIKey")
    return out


def skeleton(key):
    """
    First block of an InChIKey — the molecular skeleton.

    Blocks two and three encode stereochemistry, isotopes and PROTONATION, and
    that last one is why a full-key comparison badly understates the overlap
    here: a GEM stores atp_c as ATP(4-) at physiological pH while BRENDA
    stores neutral ATP, so the same molecule gets different full keys. The
    skeleton block is the standard "same compound, ignoring charge and
    stereo" comparison. Both are reported, because the strict one is what you
    want if stereochemistry matters to the question being asked.
    """
    return key.split("-")[0] if key else ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", default="all",
                    help="comma-separated: parse, protein, bridge, or all")
    ap.add_argument("--out-dir", default="em_ref")
    ap.add_argument("--cache-dir", default="em_cache")
    ap.add_argument("--extras", action="store_true",
                    help="also download raw_reactions and compound_to_smiles")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows in the species scan, for a quick check")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args(argv)

    what = ({"parse", "protein", "bridge"} if a.what == "all"
            else {w.strip() for w in a.what.split(",")})
    out, cache = HERE / a.out_dir, HERE / a.cache_dir
    out.mkdir(parents=True, exist_ok=True)

    names = (list(FILES) if a.extras else ["processed_reactions.csv.gz"])
    fetch(cache, names, a.refresh)

    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}

    summary = None
    if "parse" in what:
        print("\n-- parse --")
        summary = parse(cache, out)
        manifest["reactions"] = {
            "rows": summary["rows"], "ec_codes": len(summary["ecs"]),
            "organisms": len(summary["organisms"]),
            "uniprot_accessions": len(summary["accessions"]),
            "file": "reactions.csv.gz"}
        (out / "_ec_accs.json").write_text(json.dumps(
            {k: sorted(v) for k, v in summary["ec_accs"].items()}))
        (out / "_ec_counts.json").write_text(json.dumps(summary["ecs"]))
        (out / "_ec_orgs.json").write_text(json.dumps(
            {k: len(v) for k, v in summary["ec_orgs"].items()}))

    if "protein" in what:
        print("\n-- protein sequences --")
        from build_sequences import uniprot_by_accession, write_fasta, ACC_RE
        accs = (sorted(summary["accessions"]) if summary else
                sorted({a_ for v in json.loads((out / "_ec_accs.json").read_text()).values()
                        for a_ in v}))
        good = [x for x in accs if ACC_RE.match(x)]
        print(f"  {len(accs):,} accessions cited, {len(good):,} well-formed")
        seqs = uniprot_by_accession(good)
        write_fasta(out / "proteins.fasta",
                    [(k, v[0].split(" ", 1)[-1][:120], v[1])
                     for k, v in sorted(seqs.items())])
        manifest["proteins"] = {"accessions_cited": len(accs),
                                "sequences": len(seqs), "file": "proteins.fasta"}
        print(f"  -> proteins.fasta: {len(seqs):,}/{len(good):,} "
              f"({len(seqs)/max(len(good),1):.0%})")

    if "bridge" in what:
        print("\n-- bridge to the GEMs --")
        gec = gem_ecs()
        counts = json.loads((out / "_ec_counts.json").read_text())
        norg = json.loads((out / "_ec_orgs.json").read_text())
        eacc = json.loads((out / "_ec_accs.json").read_text())
        names_ = sorted(gec)
        with open(out / "ec_index.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ec_num", "n_reactions", "n_organisms", "n_proteins"]
                       + [f"in_{n}" for n in names_])
            for ec in sorted(counts, key=lambda e: -counts[e]):
                w.writerow([ec, counts[ec], norg.get(ec, 0), len(eacc.get(ec, []))]
                           + [int(ec in gec[n]) for n in names_])
        for n in names_:
            hit = sum(1 for ec in counts if ec in gec[n])
            print(f"  EC codes: {hit:,}/{len(counts):,} of EnzymeMap's are used "
                  f"by {n}; {len(gec[n] - set(counts)):,} of {n}'s "
                  f"{len(gec[n]):,} are absent from EnzymeMap")
            manifest.setdefault("bridge", {})[f"ec_shared_{n}"] = hit

        print("\n  molecules")
        sp = species(cache, a.limit)
        gk = gem_inchikeys()
        gnames = sorted(gk)
        gs = {n: {skeleton(k): v for k, v in gk[n].items()} for n in gnames}
        with open(out / "species.csv", "w", newline="") as f:
            w = csv.writer(f)
            cols = ["inchikey", "smiles", "n_participations"]
            for n in gnames:
                cols += [f"{n}_node_exact", f"{n}_node_skeleton"]
            w.writerow(cols)
            for key, (smi, c) in sorted(sp.items(), key=lambda kv: -kv[1][1]):
                row = [key, smi, c]
                for n in gnames:
                    row += [gk[n].get(key, ""), gs[n].get(skeleton(key), "")]
                w.writerow(row)
        for n in gnames:
            ex = sum(1 for k in sp if k in gk[n])
            sk = sum(1 for k in sp if skeleton(k) in gs[n])
            gex = len({k for k in gk[n] if k in sp})
            gsk = len({skeleton(k) for k in gk[n] if skeleton(k) in
                       {skeleton(x) for x in sp}})
            print(f"    {n}: {sk:,} of {len(sp):,} EnzymeMap molecules match "
                  f"by skeleton ({ex:,} by exact key)")
            print(f"      -> {gsk:,}/{len(gk[n]):,} of {n}'s metabolites appear "
                  f"in EnzymeMap ({gex:,} exact) "
                  f"= {gsk/max(len(gk[n]),1):.0%}")
            manifest.setdefault("bridge", {})[f"molecules_shared_{n}"] = {
                "exact": ex, "skeleton": sk,
                "gem_covered_exact": gex, "gem_covered_skeleton": gsk,
                "gem_total": len(gk[n])}
        manifest.setdefault("bridge", {})["enzymemap_molecules"] = len(sp)

    mpath.write_text(json.dumps(manifest, indent=1))
    print(f"\nmanifest -> {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
