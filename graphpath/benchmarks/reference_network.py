"""The curated network the model is supposed to be reproducing.

Loads the ground-truth edge table (e.g. Human-GEM), maps it into node-id
space, and answers the two questions the benchmarks need:

  * is this node / this edge part of the reference network at all?
  * which subsystems does a set of nodes belong to?

Used by `KnownStructureBenchmark`, and by `known_edge_index` to highlight
recovered edges in the sample plots (`create_graph_objs(subgraphs=...)`).
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor


class ReferenceNetwork:

    def __init__(self, node_id_map: dict, device='cuda',
                 num_nodes: Optional[int] = None, undirected: bool = True):
        """
        node_id_map : {node name -> id}, i.e. dataset.node_to_id
        num_nodes   : vocabulary size. Defaults to one past the largest id in
            the edge table, but pass the real vocabulary size when the model
            can emit ids beyond it — otherwise membership tests index out of
            bounds on exactly those novel nodes.
        undirected  : mirror every edge. Generated adjacency is symmetric, so
            comparing it against a directed reference would score every
            edge stored in the "wrong" direction as a miss.
        """
        self.node_id_map = node_id_map
        self.device = device
        self.undirected = undirected
        self.num_nodes = num_nodes
        self.edges = None
        self.adj_sparse = None
        self._edge_keys = None          # sorted u*num_nodes + v, for membership tests
        self.n_dropped_edges = 0

    # ── loading ────────────────────────────────────────────────────────────

    def load_edges(self, csv_path: str, source='source', target='target'):
        """Load and clean the reference edge table."""
        edges = pd.read_csv(csv_path)
        missing = [c for c in (source, target) if c not in edges.columns]
        if missing:
            raise KeyError(
                f"{csv_path} has no column(s) {missing}; found "
                f"{list(edges.columns)}. Pass the right names, e.g. "
                f"load_edges(path, source='from', target='to')."
            )
        edges = edges[edges[source].notna() & edges[target].notna()].copy()
        edges[source] = edges[source].astype(str).str.strip()
        edges[target] = edges[target].astype(str).str.strip()
        self.edges = edges
        self._src_col, self._tgt_col = source, target
        return edges

    def build_pathway_associations(self, subsystem='SUBSYSTEM') -> dict:
        """{node id -> [subsystems it participates in]}."""
        if self.edges is None:
            raise ValueError("Edges not loaded. Call load_edges() first.")
        df = self.edges.copy()
        if subsystem not in df.columns:
            raise KeyError(
                f"no '{subsystem}' column in the edge table; found "
                f"{list(df.columns)}. Pass the right name to "
                f"build_pathway_associations(subsystem=...), or skip it — "
                f"node_to_subsystems is optional."
            )
        df[subsystem] = df[subsystem].astype(str).str.strip()

        nodes = pd.concat([
            df[[self._src_col, subsystem]].rename(columns={self._src_col: 'node'}),
            df[[self._tgt_col, subsystem]].rename(columns={self._tgt_col: 'node'}),
        ])
        nodes['node'] = nodes['node'].map(self.node_id_map)
        nodes = nodes.dropna(subset=['node'])          # names outside the vocabulary
        nodes['node'] = nodes['node'].astype(int)

        return (nodes.groupby('node')[subsystem]
                     .unique().apply(list).to_dict())

    def build_adjacency(self) -> Tensor:
        """Sparse adjacency of the reference network, in node-id space."""
        if self.edges is None:
            raise ValueError("Edges not loaded. Call load_edges() first.")

        src = self.edges[self._src_col].map(self.node_id_map)
        dst = self.edges[self._tgt_col].map(self.node_id_map)

        keep = src.notna() & dst.notna()
        self.n_dropped_edges = int((~keep).sum())      # endpoints outside the vocabulary
        src, dst = src[keep].astype(int), dst[keep].astype(int)

        idx = torch.tensor(np.stack([src.values, dst.values]), dtype=torch.long)
        if self.undirected:
            idx = torch.cat([idx, idx.flip(0)], dim=1)

        max_id = int(idx.max()) + 1 if idx.numel() else 0
        if self.num_nodes is None:
            self.num_nodes = max_id
        elif max_id > self.num_nodes:
            raise ValueError(f"edge table references id {max_id - 1} but "
                             f"num_nodes={self.num_nodes}")

        self.adj_sparse = torch.sparse_coo_tensor(
            idx, torch.ones(idx.shape[1], dtype=torch.float32),
            size=(self.num_nodes, self.num_nodes), device=self.device,
        ).coalesce()
        # coalesce SUMS duplicates; membership is boolean, so clamp
        self.adj_sparse = torch.sparse_coo_tensor(
            self.adj_sparse.indices(), self.adj_sparse.values().clamp(max=1.0),
            size=self.adj_sparse.shape, device=self.device,
        ).coalesce()

        u, v = self.adj_sparse.indices()
        self._edge_keys = torch.sort(u * self.num_nodes + v).values
        return self.adj_sparse

    # ── queries ────────────────────────────────────────────────────────────

    @property
    def known_node_ids(self) -> Tensor:
        """Node ids appearing anywhere in the reference network."""
        self._require_adj()
        return torch.unique(self.adj_sparse.indices())

    def has_node(self, node_ids: Tensor) -> Tensor:
        """Boolean mask, same shape as `node_ids`."""
        self._require_adj()
        known = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        known[self.known_node_ids] = True
        ids = node_ids.to(self.device)
        in_range = (ids >= 0) & (ids < self.num_nodes)
        out = torch.zeros_like(ids, dtype=torch.bool)
        out[in_range] = known[ids[in_range]]
        return out

    def has_edge(self, u: Tensor, v: Tensor) -> Tensor:
        """Boolean mask over pairs (u_i, v_i), any broadcastable shape."""
        self._require_adj()
        u, v = u.to(self.device), v.to(self.device)
        valid = (u >= 0) & (u < self.num_nodes) & (v >= 0) & (v < self.num_nodes)
        keys = torch.where(valid, u * self.num_nodes + v,
                           torch.full_like(u, -1))
        return torch.isin(keys, self._edge_keys) & valid

    def reference_adjacency_for(self, node_toks: Tensor) -> Tensor:
        """
        (N, N) bool — which pairs of SLOTS are joined in the reference
        network, given one graph's node tokens. Directly comparable to the
        model's (N, N) adjacency, duplicated tokens included.
        """
        ids = node_toks.to(self.device)
        N = ids.numel()
        u = ids[:, None].expand(N, N)
        v = ids[None, :].expand(N, N)
        return self.has_edge(u, v)

    def known_edge_index(self, node_toks: Tensor) -> Tensor:
        """
        (2, E) SLOT-index pairs whose entities are joined in the reference
        network — the format `create_graph_objs(subgraphs=...)` expects, so
        recovered edges can be drawn highlighted.
        """
        ref = self.reference_adjacency_for(node_toks)
        iu = torch.triu_indices(ref.shape[0], ref.shape[0], offset=1, device=ref.device)
        keep = ref[iu[0], iu[1]]
        return iu[:, keep]

    def extract_subgraph(self, node_ids) -> Tensor:
        """Sparse adjacency induced on `node_ids` (de-duplicated, sorted)."""
        self._require_adj()
        node_ids = (torch.as_tensor(node_ids, device=self.device)
                    if not isinstance(node_ids, Tensor) else node_ids.to(self.device))
        node_ids = node_ids[(node_ids >= 0) & (node_ids < self.num_nodes)].unique()

        rows, cols = self.adj_sparse.indices()
        vals = self.adj_sparse.values()

        node_set = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        node_set[node_ids] = True
        mask = node_set[rows] & node_set[cols]

        # unique() returns sorted ids, which is what searchsorted needs
        rows_new = torch.searchsorted(node_ids, rows[mask])
        cols_new = torch.searchsorted(node_ids, cols[mask])

        n = node_ids.numel()
        return torch.sparse_coo_tensor(
            torch.stack([rows_new, cols_new]), vals[mask],
            size=(n, n), device=self.device,
        ).coalesce()

    def _require_adj(self):
        if self.adj_sparse is None:
            raise ValueError("Adjacency not built. Call build_adjacency() first.")
