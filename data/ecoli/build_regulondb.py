#!/usr/bin/env python3
"""RegulonDB TF-gene network -> a second reference layer, same edge schema.

Human-GEM has no curated regulatory network to check generated links against;
E. coli does. Feeding this to a second KnownStructureBenchmark lets a link the
metabolic model calls "novel" be scored against curated regulation instead -
so a generated gene-gene edge can be right for a reason iML1515 cannot see.

Input: RegulonDB's TF-gene interaction table (from regulondb.ccg.unam.mx,
"network_tf_gene" / "TF-gene interactions"). Column layout has changed
between releases, so the parser sniffs rather than assumes: it looks for the
two name columns and an effect column (+ / - / +-) among whatever is there,
skipping the '#' comment header.

Gene symbols are mapped to b-numbers using node_name_map.csv from
build_reference.py, so the output lands in the same id space as the GEM graph.
Symbols that do not map (genes absent from iML1515) are dropped and counted.

Usage
    python build_regulondb.py --network network_tf_gene.txt \\
        --name-map ecoli_ref/node_name_map.csv --out-dir ecoli_ref
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys

import pandas as pd

EFFECT = {"+": "activates", "-": "represses", "+-": "dual", "?": "unknown"}
GENE_RE = re.compile(r"^b\d{4}$")


def read_network(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.rstrip("\n").split("\t")]
            if len(parts) < 2:
                parts = [p.strip() for p in re.split(r"\s{2,}|,", line.strip())]
            if len(parts) >= 2:
                rows.append(parts)
    if not rows:
        sys.exit(f"no data rows parsed from {path}")
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows)


def pick_columns(df: pd.DataFrame):
    """-> (tf_col, target_col, effect_col|None). Chosen by content, not header."""
    name_like = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{1,20}$")
    scores = {}
    for c in df.columns:
        col = df[c].astype(str)
        scores[c] = col.str.match(name_like).mean()
    ranked = sorted(scores, key=lambda c: -scores[c])
    cands = [c for c in ranked if scores[c] > 0.8]
    if len(cands) < 2:
        sys.exit("could not identify TF and target name columns; "
                 f"column name-likeness: {scores}")
    # the TF column repeats far more than the target column
    a, b = cands[0], cands[1]
    if df[a].nunique() > df[b].nunique():
        a, b = b, a
    eff = None
    for c in df.columns:
        vals = set(df[c].astype(str).str.strip().unique())
        if vals and vals <= set(EFFECT) | {""}:
            eff = c
            break
    return a, b, eff


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", required=True, help="RegulonDB TF-gene table")
    ap.add_argument("--name-map", required=True,
                    help="node_name_map.csv from build_reference.py")
    ap.add_argument("--out-dir", default="ecoli_ref")
    ap.add_argument("--out-name", default="ecoli_regulondb_edges.csv")
    ap.add_argument("--tf-col", type=int, default=None)
    ap.add_argument("--target-col", type=int, default=None)
    a = ap.parse_args()

    nm = pd.read_csv(a.name_map)
    # common_name -> b-number, restricted to gene nodes
    sym2b = {}
    for node, name in zip(nm["id"].astype(str), nm["common_name"].astype(str)):
        if GENE_RE.match(node) and name:
            sym2b.setdefault(name.lower(), node)
    print(f"{len(sym2b)} gene symbols available from the GEM")

    df = read_network(a.network)
    if a.tf_col is not None and a.target_col is not None:
        tf_c, tg_c, eff_c = a.tf_col, a.target_col, None
    else:
        tf_c, tg_c, eff_c = pick_columns(df)
    print(f"using columns: TF={tf_c}, target={tg_c}, effect={eff_c}")
    print("sample rows:")
    for _, r in df.head(3).iterrows():
        print("   ", list(r))

    edges, unmapped = [], collections.Counter()
    for _, r in df.iterrows():
        tf_sym, tg_sym = str(r[tf_c]).strip(), str(r[tg_c]).strip()
        tf = sym2b.get(tf_sym.lower())
        tg = sym2b.get(tg_sym.lower())
        if tf is None:
            unmapped[tf_sym] += 1
        if tg is None:
            unmapped[tg_sym] += 1
        if tf is None or tg is None or tf == tg:
            continue
        eff = EFFECT.get(str(r[eff_c]).strip(), "regulates") if eff_c is not None \
            else "regulates"
        edges.append((tf, tg, eff, "gene", "gene", "regulation", ""))

    seen, uniq = set(), []
    for e in edges:
        if (e[0], e[1], e[2]) not in seen:
            seen.add((e[0], e[1], e[2]))
            uniq.append(e)

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, a.out_name)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "target", "edge_type", "source_type",
                    "target_type", "subsystem", "reaction"])
        w.writerows(uniq)

    tfs = len({e[0] for e in uniq})
    print(f"\n{len(uniq)} regulatory edges over {tfs} regulators -> {out}")
    print(f"{len(unmapped)} symbols had no gene node in the GEM "
          f"(expected: iML1515 covers ~1515 of ~4400 genes)")
    print("most frequent unmapped:", ", ".join(
        f"{k}({v})" for k, v in unmapped.most_common(8)))
    print("""
Use it as a second reference in the benchmark cell:

    ref_reg = ReferenceNetwork(dataset.node_to_id, device=device,
                               num_nodes=dataset.vocab_size)
    ref_reg.load_edges(f"{OUT_DIR}/ecoli_regulondb_edges.csv")
    ref_reg.build_adjacency()
    known_reg = KnownStructureBenchmark(ref_reg, device=device)

An edge with edge_precision 0 against `ref` but > 0 against `ref_reg` is the
interesting population: chemistry says no, curated regulation says yes.""")


if __name__ == "__main__":
    main()
