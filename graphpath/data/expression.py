"""Tissue expression and co-expression tables.

Serves two consumers:

  * `graphpath.layers.conditioning` — the (n_tissues, V, V) sparse co-expression
    tensor a model conditions on (`stacked_coexpression`)
  * `benchmarks.CoExpressionBenchmark` — scoring whether the genes a model
    put in one pathway are actually co-expressed (`within_list_co_exp`)

Files expected under `path`:
    co_expression_sparse_<prefix>_<tissue>.pt   one sparse (V, V) per tissue
    tissue_expression_dict.pkl                  {tissue: (V,) expression}
"""

import glob
import os
import pickle
from typing import Optional, Sequence

import torch
from torch import Tensor

DEFAULT_TISSUES = ['lung', 'prostate', 'skin', 'small intestine', 'breast',
                   'bone marrow', 'testis', 'spleen', 'pancreas', 'ovary',
                   'adipose']

# columns of the within_list_co_exp result, one row per tissue
STAT_COLUMNS = ('expr_count', 'expr_mean', 'coexpr_count', 'coexpr_mean')


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device)


class ExpressionLoader:

    def __init__(
        self,
        path: str = 'pt_files',
        device=None,
        tissues: Optional[Sequence[str]] = None,
        file_prefix: str = 'co_expression_sparse_homo_sapiens_sra_',
        expression_pkl: str = 'tissue_expression_dict.pkl',
        rename: Optional[dict] = None,
        verbose: bool = False,
    ):
        """
        `path` is read directly — the working directory is left alone, unlike
        an os.chdir, which would silently break every relative path the
        caller holds.
        """
        self.path = path
        self.device = torch.device(
            device if device is not None
            else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tissues = list(tissues) if tissues is not None else list(DEFAULT_TISSUES)
        rename = {'adipose tissue': 'adipose'} if rename is None else rename

        # ── co-expression: one sparse (V, V) per tissue ────────────────────
        self.co_expression = {}
        files = sorted(glob.glob(os.path.join(path, '*.pt')))
        if verbose:
            print(f"found {len(files)} .pt files in {path}")
        for f in files:
            tissue = (os.path.basename(f)
                      .replace(file_prefix, '').replace('_', ' ').replace('.pt', ''))
            tissue = rename.get(tissue, tissue)
            sp = _torch_load(f, self.device).coalesce()
            # A zero-variance gene yields a NaN correlation, and real tables
            # carry them. NaN wins argmax while losing every comparison, so
            # it silently inflates any "which tissue scores highest" metric.
            n_nan = int(torch.isnan(sp.values()).sum())
            if n_nan:
                sp = torch.sparse_coo_tensor(sp.indices(),
                                             torch.nan_to_num(sp.values()),
                                             size=sp.shape, device=self.device).coalesce()
                if verbose:
                    print(f"  {tissue}: {n_nan} NaN co-expression values zeroed")
            self.co_expression[tissue] = sp
            if verbose:
                print(f"  {tissue}: {self.co_expression[tissue]._nnz()} nnz")

        # ── expression: {tissue: (V,)} ─────────────────────────────────────
        with open(os.path.join(path, expression_pkl), 'rb') as fh:
            self.expression = pickle.load(fh)
        for old, new in rename.items():
            if old in self.expression:
                self.expression[new] = self.expression.pop(old)
        self.expression = {t: torch.nan_to_num(v.to(self.device))
                           for t, v in self.expression.items()}

        missing = [t for t in self.tissues
                   if t not in self.expression or t not in self.co_expression]
        if missing:
            raise KeyError(f"no expression/co-expression data for {missing}; "
                           f"have expression for {sorted(self.expression)} and "
                           f"co-expression for {sorted(self.co_expression)}")

        # pre-stacked for fast batch indexing: (n_tissues, V)
        self.expression_stack = torch.stack([self.expression[t] for t in self.tissues])
        self.vocab_size = self.expression_stack.shape[1]

        # sorted (u*V + v) keys per tissue, for O(log nnz) pair lookup
        self._keys, self._vals = {}, {}
        for t in self.tissues:
            sp = self.co_expression[t].coalesce()
            u, v = sp.indices()
            k = u.long() * self.vocab_size + v.long()
            order = torch.argsort(k)
            self._keys[t] = k[order]
            self._vals[t] = sp.values()[order]

    # ── conditioning ───────────────────────────────────────────────────────

    def tissue_index(self, tissue: str) -> int:
        return self.tissues.index(tissue)

    def stacked_coexpression(self) -> Tensor:
        """
        Sparse (n_tissues, V, V) tensor — the form
        `graphpath.layers.conditioning.CoExpressionEncoder` expects as
        `co_expr["coexpr_sparse"]`.
        """
        idx, val = [], []
        for t_i, t in enumerate(self.tissues):
            sp = self.co_expression[t].coalesce()
            u, v = sp.indices()
            idx.append(torch.stack([torch.full_like(u, t_i), u, v]))
            val.append(sp.values())
        return torch.sparse_coo_tensor(
            torch.cat(idx, dim=1), torch.cat(val),
            size=(len(self.tissues), self.vocab_size, self.vocab_size),
            device=self.device,
        ).coalesce()

    def as_condition(self, tissue_ids: Tensor) -> dict:
        """Ready-made `co_expr` dict for compute_loss / sample."""
        return {"coexpr_sparse": self.stacked_coexpression(),
                "tissue_ids": tissue_ids.to(self.device).long()}

    # ── pair lookup ────────────────────────────────────────────────────────

    def _lookup(self, tissue: str, u: Tensor, v: Tensor) -> Tensor:
        """
        Co-expression values for pairs (u_i, v_i); 0 where absent.

        Looked up by sorted key rather than by broadcasting the query against
        every stored entry: the (num_pairs, nnz) match matrix the naive
        version builds is ~100 GB for a few hundred genes against a
        million-entry table.
        """
        keys, vals = self._keys[tissue], self._vals[tissue]
        V = self.vocab_size

        def probe(a, b):
            q = a.long() * V + b.long()
            pos = torch.searchsorted(keys, q).clamp(max=keys.numel() - 1)
            hit = keys[pos] == q
            return torch.where(hit, vals[pos], torch.zeros_like(vals[pos])), hit

        # the table may store only one orientation of a pair
        fwd, hit = probe(u, v)
        bwd, _   = probe(v, u)
        return torch.where(hit, fwd, bwd)

    # ── statistics used by the benchmark ───────────────────────────────────

    def within_list_co_exp(self, node_ids: Tensor) -> Tensor:
        """
        (n_tissues, 4) — per tissue:
            expr_count    how many of these nodes are expressed at all
            expr_mean     mean expression over the nodes
            coexpr_count  how many of the pairs have a co-expression entry
            coexpr_mean   mean co-expression over all pairs (absent = 0)

        Column order is `STAT_COLUMNS`.
        """
        node_ids = node_ids.to(self.device).long()
        res = torch.zeros(len(self.tissues), 4, device=self.device)

        pairs = torch.combinations(node_ids, r=2)
        for t_idx, t in enumerate(self.tissues):
            expr = self.expression[t][node_ids]
            if pairs.numel():
                matched = self._lookup(t, pairs[:, 0], pairs[:, 1])
            else:
                matched = torch.zeros(0, device=self.device)
            res[t_idx] = torch.tensor([
                float(expr.count_nonzero()),
                float(expr.mean()) if expr.numel() else 0.0,
                float(matched.count_nonzero()),
                float(matched.mean()) if matched.numel() else 0.0,
            ], device=self.device)
        return res

    # ── batch loading (used for model conditioning inputs) ─────────────────

    def load_expr_only(self, node_ids: Tensor, tissue: Optional[str] = None):
        """Expression per node, using the highest-expressing tissue per sample
        (or a fixed `tissue`)."""
        batch_size, num_nodes = node_ids.shape
        expr_all = self.expression_stack[:, node_ids].permute(1, 0, 2)
        max_tissue = expr_all.sum(dim=2).argmax(dim=1)
        if tissue is not None:
            max_tissue = torch.full((batch_size,), self.tissue_index(tissue),
                                    device=node_ids.device)
        b = torch.arange(batch_size, device=node_ids.device)[:, None].expand(-1, num_nodes)
        n = torch.arange(num_nodes, device=node_ids.device)[None].expand(batch_size, -1)
        expr = expr_all[b, max_tissue[:, None].expand(-1, num_nodes), n]
        return expr, [self.tissues[i] for i in max_tissue.tolist()]

    def load_expr(self, node_ids: Tensor, constraint: bool = False):
        """Expression plus a PyG-style co-expression batch, one graph per row."""
        batch_size, _ = node_ids.shape
        expr_batch, tissues_batch = self.load_expr_only(node_ids)

        feats, e_idx, e_attr, b_edge, b_node = [], [], [], [], []
        id_map, node_offset = {}, 0
        for b in range(batch_size):
            r = self.load_co_expr(self.co_expression[tissues_batch[b]],
                                  node_ids[b], offset=node_offset, constraint=constraint)
            e_idx.append(r['edge_index'])
            e_attr.append(r['edge_attr'])
            b_edge.append(torch.full(r['edge_attr'].shape, b))
            b_node.append(torch.full(r['node_feats'].shape, b))
            feats.append(r['node_feats'])
            id_map.update(r['sample_to_global'])
            node_offset += len(r['node_feats'])

        co_expr_batch = {
            'x': torch.cat(feats).view(-1, 1),
            'edge_index': torch.cat(e_idx, dim=1) if e_idx else
                          torch.zeros((2, 0), dtype=torch.long, device=node_ids.device),
            'edge_attr': torch.cat(e_attr) if e_attr else
                         torch.zeros((0,), dtype=torch.float, device=node_ids.device),
            'batch': torch.cat(b_edge),
            'batch_node': torch.cat(b_node),
            'map': id_map,
        }
        return expr_batch, co_expr_batch, tissues_batch

    def load_co_expr(self, co_expression_sparse: Tensor, node_ids: Tensor,
                     offset: int = 0, constraint: bool = False) -> dict:
        """
        Co-expression edges touching `node_ids`, reindexed to sample-local ids.
        `constraint=True` keeps only edges with BOTH endpoints in the set.
        """
        rows, cols = co_expression_sparse.indices()
        vals = co_expression_sparse.values()

        in_r, in_c = torch.isin(rows, node_ids), torch.isin(cols, node_ids)
        mask = (in_r & in_c) if constraint else (in_r | in_c)

        if not mask.any():
            return {'edge_index': torch.zeros((2, 0), dtype=torch.long, device=node_ids.device),
                    'edge_attr': torch.zeros((0,), dtype=vals.dtype, device=node_ids.device),
                    'node_feats': torch.tensor([], device=node_ids.device),
                    'global_to_sample': {}, 'sample_to_global': {}}

        edge_global = torch.stack([rows[mask], cols[mask]], dim=0)
        nodes = edge_global.unique()

        global_to_sample = {int(g): i + offset for i, g in enumerate(nodes.tolist())}
        sample_to_global = {i + offset: int(g) for i, g in enumerate(nodes.tolist())}

        # vectorised remap: position of each global id within `nodes`
        local = torch.searchsorted(nodes, edge_global.reshape(-1)).reshape(2, -1) + offset

        return {'edge_index': local,
                'edge_attr': vals[mask],
                'global_to_sample': global_to_sample,
                'sample_to_global': sample_to_global,
                'node_feats': nodes.long()}
