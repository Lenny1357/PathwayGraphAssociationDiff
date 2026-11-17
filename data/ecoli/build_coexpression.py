#!/usr/bin/env python3
"""PRECISE-1K -> per-condition sparse gene co-expression, in node-id space.

The human pipeline's ExpressionLoader has a TISSUE axis. E. coli has no
tissues, so the axis becomes CONDITION STRATA drawn from PRECISE-1K's own
metadata - carbon source, strain, growth phase, or project. Everything
downstream (per_tissue_coexpr, best-tissue selection in the discovery cell,
tissue_heatmap) then works unchanged; only the labels change.

Inputs (from github.com/SBRG/precise1k, data/precise1k/):
    log_tpm_qc.csv    genes (b-numbers) x samples, log2 TPM
    metadata_qc.csv   samples x annotation columns

Output directory, read by expression_ecoli.EcoliExpressionLoader:
    meta.json          {tissues, space, vocab_size, genes, node_ids, params}
    coexp_000.npz      row/col/val of the sparse co-expression, per stratum
    expression.npy     (n_nodes, n_strata) mean expression per stratum

Why sparse top-k: a dense 4257^2 float32 per stratum is 72 MB, and the
notebook's propagator wants a sparse tensor anyway.

Usage
    # list the metadata columns, pick one, then run for real
    python build_coexpression.py --expr log_tpm_qc.csv --meta metadata_qc.csv --list-columns

    python build_coexpression.py \
        --expr log_tpm_qc.csv --meta metadata_qc.csv \
        --stratify-by project --node-id-map node_id_map.pkl \
        --out-dir ./ecoli_expr --top-k 32 --min-samples 12
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

GENE_RE = re.compile(r"^b\d{4}$")

# Metadata columns worth stratifying on, in preference order. PRECISE-1K's
# header has drifted between releases, so each is matched case-insensitively
# as a substring.
PREFERRED = ["carbon source", "project", "strain description", "growth phase",
             "condition", "supplement", "base media"]


def load_expression(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    # PRECISE-1K ships genes-as-rows. If it looks transposed, fix it.
    if not any(GENE_RE.match(str(i)) for i in df.index[:50]):
        if any(GENE_RE.match(str(c)) for c in df.columns[:50]):
            print("expression matrix looks transposed - transposing", file=sys.stderr)
            df = df.T
        else:
            print("warning: index does not look like b-numbers; gene ids will "
                  "not join to the GEM without a mapping", file=sys.stderr)
    df.index = df.index.astype(str)
    return df.loc[~df.index.duplicated()]


def pick_strata(meta: pd.DataFrame, column: str | None, samples):
    meta = meta.reindex(samples)
    if column is None:
        for want in PREFERRED:
            hit = [c for c in meta.columns if want in c.lower()]
            if hit:
                column = hit[0]
                print(f"stratifying by auto-picked column {column!r}", file=sys.stderr)
                break
    if column is None:
        print("no usable metadata column - falling back to one 'all' stratum",
              file=sys.stderr)
        return {"all": list(samples)}, "none"
    if column not in meta.columns:
        cand = [c for c in meta.columns if column.lower() in c.lower()]
        if not cand:
            sys.exit(f"column {column!r} not in metadata. Available:\n  " +
                     "\n  ".join(map(str, meta.columns)))
        column = cand[0]
    vals = meta[column].fillna("unknown").astype(str).str.strip()
    groups = {}
    for v, idx in vals.groupby(vals).groups.items():
        groups[str(v)] = list(idx)
    return groups, column


def topk_corr(X: np.ndarray, k: int, min_abs: float, absolute: bool):
    """X: genes x samples. -> (rows, cols, vals) of a symmetric top-k graph."""
    n_genes, n_samples = X.shape
    Z = X - X.mean(1, keepdims=True)
    sd = Z.std(1, ddof=1)
    live = sd > 1e-8
    Z = np.where(live[:, None], Z / np.where(live, sd, 1.0)[:, None], 0.0)
    C = (Z @ Z.T) / (n_samples - 1)
    np.clip(C, -1.0, 1.0, out=C)
    np.fill_diagonal(C, 0.0)

    rank_on = np.abs(C) if absolute else C
    k = min(k, n_genes - 1)
    part = np.argpartition(-rank_on, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n_genes), k)
    cols = part.ravel()
    vals = C[rows, cols]

    keep = np.abs(vals) >= min_abs
    rows, cols, vals = rows[keep], cols[keep], vals[keep]

    # symmetrise: keep the max magnitude of (i,j) and (j,i)
    lo = np.minimum(rows, cols)
    hi = np.maximum(rows, cols)
    key = lo.astype(np.int64) * n_genes + hi
    order = np.argsort(-np.abs(vals), kind="stable")
    _, first = np.unique(key[order], return_index=True)
    sel = order[first]
    lo, hi, v = lo[sel], hi[sel], vals[sel]
    return (np.concatenate([lo, hi]), np.concatenate([hi, lo]),
            np.concatenate([v, v]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expr", required=True, help="log_tpm_qc.csv")
    ap.add_argument("--meta", required=True, help="metadata_qc.csv")
    ap.add_argument("--out-dir", default="ecoli_expr")
    ap.add_argument("--stratify-by", default=None,
                    help="metadata column to use as the 'tissue' axis")
    ap.add_argument("--list-columns", action="store_true",
                    help="print the metadata columns and exit")
    ap.add_argument("--node-id-map", default=None,
                    help="node_id_map.pkl ({node_name: int_id}); emits matrices "
                         "in dataset node-id space so they can be multiplied "
                         "against ref.adj_sparse. Without it, gene space.")
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--min-samples", type=int, default=12)
    ap.add_argument("--min-abs", type=float, default=0.3)
    ap.add_argument("--signed", action="store_true",
                    help="rank neighbours by signed r instead of |r| "
                         "(default keeps strong anticorrelations too)")
    ap.add_argument("--max-strata", type=int, default=24,
                    help="keep the N largest strata; the tail is mostly "
                         "single-project noise")
    a = ap.parse_args()

    meta = pd.read_csv(a.meta, index_col=0)
    meta.index = meta.index.astype(str)
    if a.list_columns:
        print("metadata columns and their distinct-value counts:\n")
        for c in meta.columns:
            print(f"  {c:<40s} {meta[c].nunique():>5d} values")
        return

    expr = load_expression(a.expr)
    samples = [s for s in expr.columns.astype(str) if s in meta.index]
    if not samples:
        sys.exit("no overlap between expression columns and metadata index")
    expr.columns = expr.columns.astype(str)
    expr = expr[samples]
    print(f"expression: {expr.shape[0]} genes x {len(samples)} samples")

    # ---- restrict to genes that exist as nodes, and fix the output index ----
    node_id_map = None
    if a.node_id_map:
        with open(a.node_id_map, "rb") as fh:
            node_id_map = pickle.load(fh)
        if not isinstance(node_id_map, dict):
            sys.exit("node_id_map.pkl is not a dict")
        # accept {name: id} or {id: name}
        sample_key = next(iter(node_id_map))
        if isinstance(sample_key, int):
            node_id_map = {v: k for k, v in node_id_map.items()}
        genes = [g for g in expr.index if g in node_id_map]
        missing = len(expr.index) - len(genes)
        print(f"genes joined to graph nodes: {len(genes)} "
              f"({missing} expression genes absent from the graph)")
        index = np.array([node_id_map[g] for g in genes], dtype=np.int64)
        vocab = int(max(node_id_map.values())) + 1
        space = "node"
    else:
        genes = list(expr.index)
        index = np.arange(len(genes), dtype=np.int64)
        vocab = len(genes)
        space = "gene"
    if len(genes) < 2:
        sys.exit("fewer than two genes left after the join - check that the "
                 "graph's gene nodes really are b-numbers")
    X_all = expr.loc[genes].to_numpy(dtype=np.float32)

    # ---- strata --------------------------------------------------------
    groups, column = pick_strata(meta, a.stratify_by, samples)
    groups = {k: v for k, v in groups.items() if len(v) >= a.min_samples}
    if not groups:
        print(f"no stratum reaches --min-samples {a.min_samples}; using one "
              f"'all' stratum instead", file=sys.stderr)
        groups = {"all": samples}
    groups = dict(sorted(groups.items(), key=lambda kv: -len(kv[1]))[:a.max_strata])
    print(f"strata ({column}): " +
          ", ".join(f"{k}({len(v)})" for k, v in groups.items()))

    os.makedirs(a.out_dir, exist_ok=True)
    col_pos = {s: i for i, s in enumerate(expr.columns)}
    tissues, expression = [], []
    for i, (name, cols) in enumerate(groups.items()):
        X = X_all[:, [col_pos[c] for c in cols]]
        rows, cols_idx, vals = topk_corr(X, a.top_k, a.min_abs,
                                         absolute=not a.signed)
        np.savez_compressed(
            os.path.join(a.out_dir, f"coexp_{i:03d}.npz"),
            row=index[rows].astype(np.int32),
            col=index[cols_idx].astype(np.int32),
            val=vals.astype(np.float32),
        )
        tissues.append(name)
        expression.append(X.mean(1))
        print(f"  [{i:>3}] {name:<32s} {len(cols):>4d} samples  "
              f"{len(vals):>8d} edges  mean|r|={np.abs(vals).mean():.3f}")

    E = np.zeros((vocab, len(tissues)), dtype=np.float32)
    E[index] = np.stack(expression, axis=1)
    np.save(os.path.join(a.out_dir, "expression.npy"), E)

    with open(os.path.join(a.out_dir, "meta.json"), "w") as fh:
        json.dump({
            "tissues": tissues, "space": space, "vocab_size": vocab,
            "genes": genes, "node_ids": index.tolist(),
            "stratified_by": column,
            "params": {"top_k": a.top_k, "min_abs": a.min_abs,
                       "min_samples": a.min_samples, "signed": a.signed},
        }, fh, indent=1)

    print(f"\nwritten to {a.out_dir}/  ({len(tissues)} strata, space={space})")
    print("load it with expression_ecoli.EcoliExpressionLoader(path=...)")


if __name__ == "__main__":
    main()
