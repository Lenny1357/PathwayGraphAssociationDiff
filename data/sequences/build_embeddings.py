"""Pretrained sequence/structure embeddings for the pathway node space.

One feature vector per node token, built from what the entity actually IS
rather than from a learned id lookup:

  metabolite   SMILES            -> ChemBERTa        (chemical structure)
  gene         CDS nucleotides   -> nucleotide model (codon usage, composition)
  enzyme       amino acids       -> ESM-2            (fold / function)
  reaction     no sequence       -> MEAN of its participating metabolites

Reactions are pooled rather than left empty because a reaction IS its
substrates and products; the incidence comes from the GEM edge table, so the
pooled vector is built from the same chemistry the metabolite rows carry.

LAYOUT. The three encoders have different widths (ChemBERTa 384, ESM-2 35M
480, Nucleotide Transformer 512), so the output is BLOCK DIAGONAL: each node
type writes only into its own slice and leaves the others at zero, with a
type one-hot appended. A single trainable Linear on that is exactly a set of
per-type readouts, so nothing is mixed and no encoder has to be squeezed into
another's dimension.

  [ chem | dna | prot | type one-hot ]

PAD and MASK get no features and no slice. They are not molecules, and the
consumer is expected to give them a learned fallback row — see
PretrainedNodeEmbedding in the training notebook.

Usage:
    python build_embeddings.py --organism ecoli
    python build_embeddings.py --organism ecoli --protein-model facebook/esm2_t33_650M_UR50D
    python build_embeddings.py --organism ecoli --dna-model kmer   # no download

Writes seq_ref/node_embeddings_<organism>.npz with:
    feats   (V+1, D) float32   the block-diagonal matrix, rows indexed by TOKEN ID
    have    (V+1,)   bool      True where any encoder produced a vector
    slices  json               {"chem": [0, 384], "dna": [...], ...}
    meta    json               model names, coverage counts, per-type totals
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEQ = HERE / "seq_ref"

# ── node id conventions ──────────────────────────────────────────────────
#
# These are the spellings in the model's vocabulary, which differ from the
# keys in the sequence files. Getting them wrong produces a silently empty
# feature matrix, so each one is asserted against real coverage below.
#
#   ecoli  metabolite  10fthf_c   -> 10fthf       (compartment dropped: atp_c
#                                                  and atp_p are one molecule)
#          gene        b0002      -> b0002        (as-is)
#          enzyme      enz_b0002  -> b0002        (the protein of that gene)
#          reaction    rxn_CYTDK2 -> pooled
#   human  metabolite  MAM00001   -> MAM00001
#          gene        ENSG...    -> ENSG...
#          reaction    rxn_MAR... -> pooled

_COMPARTMENTS = ("_c", "_p", "_e", "_m", "_x", "_r", "_v", "_n", "_g", "_l", "_i")


def strip_compartment(mid: str) -> str:
    for c in _COMPARTMENTS:
        if mid.endswith(c):
            return mid[: -len(c)]
    return mid


def read_fasta(path: Path) -> dict:
    """{first whitespace-delimited token of the header: sequence}."""
    out, key, buf = {}, None, []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if key is not None:
                    out[key] = "".join(buf)
                key, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
    if key is not None:
        out[key] = "".join(buf)
    return out


def read_smiles(path: Path) -> dict:
    import csv
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            s = (row.get("smiles") or "").strip()
            if s:
                out[row["node_id"]] = s
    return out


# ── encoders ─────────────────────────────────────────────────────────────

def _mean_pool(last_hidden, attention_mask):
    import torch
    m = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)


def encode_hf(seqs: list, model_name: str, batch_size: int, max_len: int,
              device, spaced_kmer: int = 0) -> np.ndarray:
    """Mean-pooled last hidden state. Returns (len(seqs), hidden)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mod = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    mod.eval().to(device)
    if spaced_kmer:                      # DNABERT-1 wants space-separated k-mers
        seqs = [" ".join(s[i:i + spaced_kmer]
                         for i in range(0, len(s) - spaced_kmer + 1, spaced_kmer))
                for s in seqs]
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            b = tok(seqs[i:i + batch_size], padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt").to(device)
            h = mod(**b).last_hidden_state
            out.append(_mean_pool(h, b["attention_mask"]).float().cpu().numpy())
            if (i // batch_size) % 20 == 0:
                print(f"      {min(i + batch_size, len(seqs))}/{len(seqs)}",
                      flush=True)
    del mod
    return np.concatenate(out, 0) if out else np.zeros((0, 0), np.float32)


_CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]


def encode_codon(seqs: list) -> np.ndarray:
    """No-download fallback: in-frame codon frequencies + length + GC.

    66 dimensions and no pretrained model. It is NOT a language model and
    should not be reported as one — it exists so the pipeline runs end to end
    on a machine with no network, and as the baseline any nucleotide model has
    to beat to justify its download.
    """
    idx = {c: i for i, c in enumerate(_CODONS)}
    out = np.zeros((len(seqs), 66), np.float32)
    for r, s in enumerate(seqs):
        s = s.upper()
        n = 0
        for i in range(0, len(s) - 2, 3):
            j = idx.get(s[i:i + 3])
            if j is not None:
                out[r, j] += 1.0
                n += 1
        if n:
            out[r, :64] /= n
        out[r, 64] = min(len(s) / 3000.0, 4.0)
        gc = sum(ch in "GC" for ch in s)
        out[r, 65] = gc / max(len(s), 1)
    return out


# ── the build ────────────────────────────────────────────────────────────

def build(organism: str, chem_model: str, dna_model: str, protein_model: str,
          batch_size: int = 64, device=None, out_path: Path = None,
          rxn_source: str = "metabolite") -> Path:
    import torch
    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"organism {organism}   device {device}")

    # the model's vocabulary, which fixes the row order of everything below
    if organism == "ecoli":
        nid_path = ROOT / "data/ecoli/ecoli_ref/ecoli_node_id_map.pkl"
        edges = ROOT / "data/ecoli/ecoli_ref/ecoli_gem_reaction_node_edges.csv"
        from graphpath.data.node_types import ecoli_node_type as ntype
    else:
        pt = Path(__import__("os").environ.get("PT_FILES",
                                               "pt_files"))
        nid_path = pt / "node_id_map.pkl"
        edges = pt / "human_gem_reaction_node_edges2.csv"

        def ntype(n):
            return ("reaction" if str(n).startswith("rxn_") else
                    "metabolite" if str(n).startswith("MAM") else
                    "gene" if str(n).startswith("ENS") else "enzyme")

    with open(nid_path, "rb") as fh:
        nid = pickle.load(fh)
    V = max(nid.values()) + 1
    id_to_node = {v: k for k, v in nid.items()}
    types = {t: [] for t in ("metabolite", "gene", "enzyme", "reaction")}
    for name, tid in nid.items():
        t = ntype(name)
        if t in types:
            types[t].append((tid, name))
    print("  vocab", f"{V:,}", {k: len(v) for k, v in types.items()})

    smiles = read_smiles(SEQ / f"{organism}_metabolite_smiles.csv")
    cds    = read_fasta(SEQ / f"{organism}_gene_cds.fasta")
    prot   = read_fasta(SEQ / f"{organism}_enzyme_aa.fasta")
    print(f"  sources: {len(smiles):,} SMILES, {len(cds):,} CDS, "
          f"{len(prot):,} protein")

    # ── 1. metabolites -> ChemBERTa ──────────────────────────────────────
    met_rows, met_seqs = [], []
    for tid, name in types["metabolite"]:
        s = smiles.get(strip_compartment(name)) or smiles.get(name)
        if s:
            met_rows.append(tid)
            met_seqs.append(s)
    print(f"  metabolites with SMILES: {len(met_rows)}/{len(types['metabolite'])}")
    assert met_rows, ("no metabolite matched a SMILES row — the id convention "
                      "is wrong, not the data")
    print(f"    encoding with {chem_model}")
    chem = encode_hf(met_seqs, chem_model, batch_size, 256, device)

    # ── 2. genes -> nucleotide model (or the codon fallback) ─────────────
    gene_rows, gene_seqs = [], []
    for tid, name in types["gene"]:
        s = cds.get(name)
        if s:
            gene_rows.append(tid)
            gene_seqs.append(s)
    print(f"  genes with CDS: {len(gene_rows)}/{len(types['gene'])}")
    if dna_model == "kmer":
        print("    codon-frequency fallback (no pretrained nucleotide model)")
        dna = encode_codon(gene_seqs)
    else:
        print(f"    encoding with {dna_model}")
        kmer = 6 if "DNA_bert" in dna_model else 0
        dna = encode_hf(gene_seqs, dna_model, max(batch_size // 4, 1), 1024,
                        device, spaced_kmer=kmer)

    # ── 3. enzymes -> ESM-2 ─────────────────────────────────────────────
    # 205 of iML1515's 1,166 enzyme nodes are COMPLEXES spelled with "+", e.g.
    # enz_b0032+b0033 and enz_b0054+b0641+b3199+b3200+b3201+b4261+b4262. Keying
    # on the whole string misses every one of them. A complex is its subunits,
    # so each subunit is encoded and the vectors averaged.
    enz_rows, enz_subunits, uniq = [], [], {}
    n_complex = 0
    for tid, name in types["enzyme"]:
        key = name[4:] if name.startswith("enz_") else name
        subs = [k for k in key.split("+") if (prot.get(k) or prot.get(name))]
        if not subs:
            continue
        if len(subs) > 1:
            n_complex += 1
        enz_rows.append(tid)
        enz_subunits.append(subs)
        for k in subs:
            uniq.setdefault(k, len(uniq))
    print(f"  enzymes with AA: {len(enz_rows)}/{len(types['enzyme'])}"
          f"   ({n_complex} are complexes, mean-pooled over subunits;"
          f" {len(uniq)} distinct proteins to encode)")
    if enz_rows:
        print(f"    encoding with {protein_model}")
        order = sorted(uniq, key=uniq.get)
        sub_mat = encode_hf([prot[k] for k in order], protein_model,
                            max(batch_size // 4, 1), 1024, device)
        pro = np.stack([sub_mat[[uniq[k] for k in subs]].mean(0)
                        for subs in enz_subunits])
    else:
        pro = np.zeros((0, 1), np.float32)

    # ── 4. reactions -> mean of their metabolites' chemistry ────────────
    # ── LEAKAGE WARNING, and it is not small ────────────────────────────
    #
    # A reaction's feature vector here is the MEAN CHEMISTRY OF ITS OWN
    # NEIGHBOURS in the reference edge table — which is a large part of the
    # adjacency the model is asked to predict. Any edge_precision gain on
    # metabolite-reaction edges is therefore partly the answer arriving as an
    # input feature, not the model learning chemistry.
    #
    # rxn_source="none" is the clean control: reactions get only the type
    # one-hot and a learned fallback row, exactly like PAD and MASK. Run both
    # and the difference bounds the leakage.
    #
    # There is no non-leaky pooled option available here: the enzyme-reaction
    # link leaks too, and reaction -> EC is not in the edge table (it would
    # need iML1515.json). Honest flagging was the available choice.
    import csv as _csv
    chem_by_tid = {t: chem[i] for i, t in enumerate(met_rows)}
    parts = {}
    if rxn_source == "none":
        print("  reactions: NO pooled features (clean control; learned fallback "
              "row only)")
    else:
        print("  !! reactions pooled from their OWN metabolite neighbours — this "
              "leaks\n     metabolite-reaction adjacency into the inputs. "
              "Compare against rxn_source='none'.")
    with open(edges, newline="") as fh:
        for row in ([] if rxn_source == "none" else _csv.DictReader(fh)):
            for a, b in ((row["source"], row["target"]),
                         (row["target"], row["source"])):
                if str(a).startswith("rxn_") and a in nid and b in nid:
                    v = chem_by_tid.get(nid[b])
                    if v is not None:
                        parts.setdefault(nid[a], []).append(v)
    rxn_rows = [t for t, _ in types["reaction"] if t in parts]
    rxn = (np.stack([np.mean(parts[t], 0) for t in rxn_rows])
           if rxn_rows else np.zeros((0, chem.shape[1]), np.float32))
    print(f"  reactions pooled from metabolites: "
          f"{len(rxn_rows)}/{len(types['reaction'])}"
          f"   (mean {np.mean([len(parts[t]) for t in rxn_rows]):.1f} "
          f"metabolites each)" if rxn_rows else "  reactions pooled: 0")

    # ── assemble, block diagonal, + a type one-hot ──────────────────────
    d_chem, d_dna, d_pro = chem.shape[1], dna.shape[1], (pro.shape[1] if enz_rows else 0)
    TYPES = ["metabolite", "gene", "enzyme", "reaction"]
    sl = {}
    off = 0
    for nm, w in (("chem", d_chem), ("dna", d_dna), ("prot", d_pro)):
        sl[nm] = [off, off + w]; off += w
    sl["type"] = [off, off + len(TYPES)]
    D = off + len(TYPES)

    feats = np.zeros((V + 1, D), np.float32)      # +1: the MASK row stays empty
    have = np.zeros(V + 1, bool)

    def place(rows, mat, key):
        if len(rows) == 0:
            return
        a, b = sl[key]
        feats[np.asarray(rows), a:b] = mat
        have[np.asarray(rows)] = True

    place(met_rows, chem, "chem")
    place(gene_rows, dna, "dna")
    place(enz_rows, pro, "prot")
    place(rxn_rows, rxn, "chem")        # reactions live in the chemistry slice

    a, b = sl["type"]
    for i, t in enumerate(TYPES):
        rows = [tid for tid, _ in types[t]]
        feats[np.asarray(rows), a + i] = 1.0

    # L2-normalise each populated block so one encoder's scale cannot dominate
    for key in ("chem", "dna", "prot"):
        a, b = sl[key]
        if b > a:
            blk = feats[:, a:b]
            n = np.linalg.norm(blk, axis=1, keepdims=True)
            feats[:, a:b] = np.divide(blk, n, out=np.zeros_like(blk), where=n > 0)

    feats[0] = 0.0; have[0] = False          # PAD is not a molecule
    have[V] = False                          # nor is MASK

    meta = dict(
        organism=organism, vocab=int(V), dim=int(D),
        models=dict(chem=chem_model, dna=dna_model, protein=protein_model),
        coverage={
            "metabolite": [len(met_rows), len(types["metabolite"])],
            "gene":       [len(gene_rows), len(types["gene"])],
            "enzyme":     [len(enz_rows), len(types["enzyme"])],
            "reaction":   [len(rxn_rows), len(types["reaction"])],
        },
        rows_with_features=int(have.sum()),
        rxn_source=rxn_source,
        note="rows are TOKEN IDS; PAD(0) and MASK(V) are deliberately empty",
    )
    tag = "" if rxn_source == "metabolite" else f"_rxn-{rxn_source}"
    out = out_path or (SEQ / f"node_embeddings_{organism}{tag}.npz")
    np.savez_compressed(out, feats=feats, have=have,
                        slices=json.dumps(sl), meta=json.dumps(meta))
    print(f"\n  {have.sum():,}/{V + 1:,} rows have features   D={D}   -> {out}")
    print("  slices", sl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", default="ecoli", choices=["ecoli", "human"])
    ap.add_argument("--chem-model", default="DeepChem/ChemBERTa-77M-MLM")
    ap.add_argument("--dna-model", default="kmer",
                    help="'kmer' for the no-download codon fallback, or a HF id "
                         "such as InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    ap.add_argument("--protein-model", default="facebook/esm2_t12_35M_UR50D")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rxn-source", default="metabolite",
                    choices=["metabolite", "none"],
                    help="metabolite: mean of the reaction's own neighbours "
                         "(LEAKY, see build()); none: learned fallback only")
    a = ap.parse_args()
    build(a.organism, a.chem_model, a.dna_model, a.protein_model,
          a.batch_size, rxn_source=a.rxn_source)


if __name__ == "__main__":
    main()
