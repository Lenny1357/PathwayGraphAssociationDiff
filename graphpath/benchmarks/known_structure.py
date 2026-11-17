"""How much of a generated pathway is real, and how much is invented.

Every generated node and edge is checked against the curated reference
network (`ReferenceNetwork`). Two things are being measured at once, and
they pull in opposite directions:

  * fidelity  — are the entities and links real biology, or hallucinated?
  * novelty   — is the model recombining known parts into something new, or
                just copying a pathway it saw in training?

A model scoring 100% on fidelity with 0% novel edges has memorised the
reference; one scoring 0% fidelity has invented everything. Report both.
"""

from typing import Optional

import torch
from torch import Tensor

from .base import Benchmark, BenchmarkResult
from .reference_network import ReferenceNetwork


class KnownStructureBenchmark(Benchmark):

    name = "known vs novel structure"

    def __init__(self, reference: ReferenceNetwork,
                 node_to_subsystems: Optional[dict] = None,
                 pad_id: int = 0, device=None):
        """
        reference : a built ReferenceNetwork (load_edges + build_adjacency).
        node_to_subsystems : optional {node id -> [subsystem]} from
            reference.build_pathway_associations(); enables the subsystem
            coherence stats — a real pathway concentrates in a few
            subsystems, a random node set sprays across many.
        """
        self.ref = reference
        self.node_to_subsystems = node_to_subsystems
        self.pad_id = pad_id
        self.device = device or reference.device

    # ── per-graph ──────────────────────────────────────────────────────────

    def _score_graph(self, node_toks: Tensor, adj: Optional[Tensor]) -> dict:
        ids = node_toks.to(self.device)
        real = ids != self.pad_id
        ids_real = ids[real]
        n_nodes = int(real.sum())
        if n_nodes == 0:
            return {}

        known_node = self.ref.has_node(ids_real)
        out = {
            "node_known_frac": float(known_node.float().mean()),
            "unique_node_frac": float(ids_real.unique().numel() / n_nodes),
            "n_nodes": n_nodes,
        }

        if self.node_to_subsystems is not None:
            subs = []
            for i in ids_real.tolist():
                subs.extend(self.node_to_subsystems.get(i, []))
            if subs:
                counts = {}
                for s in subs:
                    counts[s] = counts.get(s, 0) + 1
                top = max(counts.values())
                out["n_subsystems"] = len(counts)
                # share of node-subsystem memberships in the largest subsystem
                out["subsystem_purity"] = top / sum(counts.values())

        if adj is not None:
            adj = adj.to(self.device).bool()
            ref = self.ref.reference_adjacency_for(ids)
            # upper triangle over real slots only; the adjacency is symmetric
            N = ids.numel()
            iu = torch.triu_indices(N, N, offset=1, device=self.device)
            valid = real[iu[0]] & real[iu[1]]
            gen_e = adj[iu[0], iu[1]] & valid
            ref_e = ref[iu[0], iu[1]] & valid

            hit = int((gen_e & ref_e).sum())
            n_gen, n_ref = int(gen_e.sum()), int(ref_e.sum())
            out.update({
                "n_edges_generated": n_gen,
                "n_edges_reference": n_ref,
                "edge_precision": hit / n_gen if n_gen else float('nan'),
                "edge_recall":    hit / n_ref if n_ref else float('nan'),
                "novel_edge_frac": 1 - hit / n_gen if n_gen else float('nan'),
            })
            p, r = out["edge_precision"], out["edge_recall"]
            out["edge_f1"] = (2 * p * r / (p + r)
                              if n_gen and n_ref and (p + r) > 0 else float('nan'))
        return out

    # ── the benchmark ──────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, node_toks: Tensor, adjacency: Optional[Tensor] = None,
                 **kwargs) -> BenchmarkResult:
        node_toks = node_toks if node_toks.dim() == 2 else node_toks.unsqueeze(0)
        per_graph = []
        for i, row in enumerate(node_toks):
            adj = adjacency[i] if adjacency is not None else None
            s = self._score_graph(row, adj)
            if s:
                per_graph.append(s)

        notes = []
        if self.ref.n_dropped_edges:
            notes.append(f"{self.ref.n_dropped_edges} reference edge(s) dropped: "
                         f"endpoint not in the node id map")
        if adjacency is None:
            notes.append("no adjacency passed — node metrics only, "
                         "edge fidelity not measured")
        if not per_graph:
            notes.append("no graph had any non-padding node")
            return BenchmarkResult(self.name, None, {}, len(node_toks), 0, notes)

        keys = [k for k in per_graph[0] if k != "n_nodes"]
        summary = {}
        for k in keys:
            vals = torch.tensor([g[k] for g in per_graph if k in g], dtype=torch.float)
            vals = vals[~vals.isnan()]
            if vals.numel():
                summary[k] = float(vals.mean())

        values = torch.tensor([[g.get(k, float('nan')) for k in keys]
                               for g in per_graph], dtype=torch.float)
        by_metric = {k: values[:, i] for i, k in enumerate(keys)}
        return BenchmarkResult(self.name, values, summary,
                               n_graphs=len(node_toks), n_scored=len(per_graph),
                               notes=notes, per_graph=by_metric)

    # ── for plotting ───────────────────────────────────────────────────────

    def known_edge_indices(self, node_toks: Tensor):
        """
        One (2, E) slot-index tensor per graph, ready to pass as
        `create_graph_objs(..., subgraphs=...)` so reference edges are drawn
        highlighted in the sample figures.
        """
        node_toks = node_toks if node_toks.dim() == 2 else node_toks.unsqueeze(0)
        return [self.ref.known_edge_index(row) for row in node_toks]
