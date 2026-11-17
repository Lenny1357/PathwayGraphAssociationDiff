"""Drop-in replacement for graphpath.data.expression.ExpressionLoader, E. coli side.

Reads what build_coexpression.py writes and exposes the attributes the
notebook actually touches:

    .tissues                    list[str]           condition strata
    .co_expression[tissue]      sparse (V, V)       torch.sparse_coo_tensor
    .expression                 dense  (V, T)       mean log-TPM per stratum
    .n_tissues, .genes, .space, .vocab_size

Use it exactly where ExpressionLoader was used:

    from expression_ecoli import EcoliExpressionLoader
    expr_loader = EcoliExpressionLoader(path=EXPR_PATH, device=device)

If your CoExpressionBenchmark / PropagatedExpression reach for an attribute
that is not here, the error will name it - add it below rather than
reshaping the files.

Note on index space. `CoExpressionPropagator(ref.adj_sparse, expr_loader.
co_expression[tissue], vocab_size=dataset.vocab_size)` multiplies the
co-expression matrix against the metabolic adjacency, so both must be in the
same index space: dataset node ids. Build with --node-id-map to get that. If
your own ExpressionLoader instead keyed on gene symbols and re-indexed
internally, build without --node-id-map and set space='gene'.
"""

from __future__ import annotations

import json
import os

import numpy as np


class EcoliExpressionLoader:
    def __init__(self, path: str, device="cpu", dtype=None, lazy: bool = True):
        import torch

        self._torch = torch
        self.path = path
        self.device = device
        self.dtype = dtype or torch.float32

        with open(os.path.join(path, "meta.json")) as fh:
            meta = json.load(fh)
        self.tissues = list(meta["tissues"])
        self.space = meta.get("space", "node")
        self.vocab_size = int(meta["vocab_size"])
        self.genes = list(meta.get("genes", []))
        self.node_ids = np.asarray(meta.get("node_ids", []), dtype=np.int64)
        self.stratified_by = meta.get("stratified_by")
        self.params = meta.get("params", {})

        self.expression = torch.as_tensor(
            np.load(os.path.join(path, "expression.npy"))
        ).to(device=device, dtype=self.dtype)

        self.co_expression = _TissueDict(self, lazy=lazy)
        if not lazy:
            for t in self.tissues:
                _ = self.co_expression[t]

    # --- the bits benchmarks tend to ask for -------------------------------
    @property
    def n_tissues(self) -> int:
        return len(self.tissues)

    def __len__(self) -> int:
        return len(self.tissues)

    def __repr__(self) -> str:
        return (f"EcoliExpressionLoader({len(self.tissues)} strata by "
                f"{self.stratified_by!r}, space={self.space}, "
                f"V={self.vocab_size})")

    def tissue_index(self, tissue) -> int:
        if isinstance(tissue, int):
            return tissue
        return self.tissues.index(tissue)

    def get(self, tissue):
        return self.co_expression[tissue]

    def _load(self, i: int):
        torch = self._torch
        z = np.load(os.path.join(self.path, f"coexp_{i:03d}.npz"))
        idx = torch.from_numpy(
            np.stack([z["row"].astype(np.int64), z["col"].astype(np.int64)])
        )
        val = torch.from_numpy(z["val"]).to(self.dtype)
        V = self.vocab_size
        return torch.sparse_coo_tensor(idx, val, (V, V)).coalesce().to(self.device)

    # ── pair lookup and set statistics ─────────────────────────────────────
    #
    # The same contract as graphpath.data.expression.ExpressionLoader, so analyses
    # written against the human loader run here unchanged. Without these the
    # README's claim that this is a "drop-in" is only true for the attributes.

    STAT_COLUMNS = ("expr_count", "expr_mean", "coexpr_count", "coexpr_mean")

    def _sorted_keys(self, tissue):
        """Cached (key, value) arrays for one stratum, sorted for searchsorted.

        Looked up by sorted key rather than by broadcasting the query against
        every stored entry: the (n_pairs, nnz) match matrix the naive version
        builds is enormous for a few hundred genes against a 200k-entry table.
        """
        torch = self._torch
        if not hasattr(self, "_keycache"):
            self._keycache = {}
        if tissue not in self._keycache:
            sp = self.co_expression[tissue].coalesce()
            u, v = sp.indices()
            k = u.long() * self.vocab_size + v.long()
            order = torch.argsort(k)
            self._keycache[tissue] = (k[order].contiguous(),
                                      sp.values()[order].contiguous())
        return self._keycache[tissue]

    def _lookup(self, tissue, u, v):
        """Co-expression for pairs (u_i, v_i); 0 where absent."""
        torch = self._torch
        keys, vals = self._sorted_keys(tissue)
        V = self.vocab_size

        def probe(a, b):
            q = a.long() * V + b.long()
            pos = torch.searchsorted(keys, q).clamp(max=keys.numel() - 1)
            hit = keys[pos] == q
            return torch.where(hit, vals[pos], torch.zeros_like(vals[pos])), hit

        # the table may store only one orientation of a pair
        fwd, hit = probe(u, v)
        bwd, _ = probe(v, u)
        return torch.where(hit, fwd, bwd)

    def within_list_co_exp(self, node_ids):
        """
        (n_tissues, 4) — per stratum:
            expr_count    how many of these nodes are expressed at all
            expr_mean     mean expression over the nodes
            coexpr_count  how many of the pairs have a co-expression entry
            coexpr_mean   mean co-expression over all pairs (absent = 0)

        Column order is `STAT_COLUMNS`.
        """
        torch = self._torch
        node_ids = torch.as_tensor(node_ids, device=self.device).long()
        res = torch.zeros(len(self.tissues), 4, device=self.device)
        pairs = torch.combinations(node_ids, r=2)
        for i, t in enumerate(self.tissues):
            e = self.expression[:, i][node_ids]
            m = (self._lookup(t, pairs[:, 0], pairs[:, 1]) if pairs.numel()
                 else torch.zeros(0, device=self.device))
            res[i] = torch.tensor([
                float(e.count_nonzero()),
                float(e.mean()) if e.numel() else 0.0,
                float(m.count_nonzero()),
                float(m.mean()) if m.numel() else 0.0,
            ], device=self.device)
        return res

    def to(self, device):
        self.device = device
        self.expression = self.expression.to(device)
        self.co_expression.clear()
        if hasattr(self, "_keycache"):
            self._keycache.clear()
        return self


class _TissueDict(dict):
    """Lazy {tissue: sparse matrix}; behaves like a plain dict once touched."""

    def __init__(self, loader: EcoliExpressionLoader, lazy=True):
        super().__init__()
        self._loader = loader
        self._lazy = lazy

    def __missing__(self, key):
        i = self._loader.tissue_index(key)
        mat = self._loader._load(i)
        self[self._loader.tissues[i]] = mat
        return mat

    def __getitem__(self, key):
        if isinstance(key, int) and key not in self:
            key = self._loader.tissues[key]
        return super().__getitem__(key)

    def keys(self):
        return list(self._loader.tissues)

    def __iter__(self):
        return iter(self._loader.tissues)

    def __len__(self):
        return len(self._loader.tissues)

    def items(self):
        for t in self._loader.tissues:
            yield t, self[t]

    def values(self):
        for t in self._loader.tissues:
            yield self[t]


# ---------------------------------------------------------------------------
# numpy-only view, for checking the files without torch installed
# ---------------------------------------------------------------------------
def inspect(path: str):
    with open(os.path.join(path, "meta.json")) as fh:
        meta = json.load(fh)
    E = np.load(os.path.join(path, "expression.npy"))
    print(f"{len(meta['tissues'])} strata by {meta.get('stratified_by')!r}, "
          f"space={meta['space']}, V={meta['vocab_size']}, "
          f"expression {E.shape}")
    for i, t in enumerate(meta["tissues"]):
        z = np.load(os.path.join(path, f"coexp_{i:03d}.npz"))
        print(f"  [{i:>3}] {t:<32s} {len(z['val']):>8d} edges  "
              f"mean|r|={np.abs(z['val']).mean():.3f}")
    return meta


if __name__ == "__main__":
    import sys
    inspect(sys.argv[1] if len(sys.argv) > 1 else "ecoli_expr")
