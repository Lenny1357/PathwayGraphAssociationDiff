"""How far apart are the endpoints of an edge the model invented?

`KnownStructureBenchmark` asks whether a generated edge exists in the curated
network. This asks the follow-up question about the ones that do NOT, and about
the ones that do but were never in the training graphs:

    for every generated edge absent from TRAINING, how many hops apart are its
    endpoints in the REFERENCE network, and what kind of edge is it?

The hop count is the whole point, because it separates three very different
things that "novel edge" lumps together:

    1 hop     the edge is IN the reference but was never in a training chunk.
              The model put back something real that it had not been shown —
              the strongest evidence of generalisation this suite produces.
    2 hops    the endpoints share a reference neighbour. For a metabolic graph
              that usually means "two metabolites of the same reaction" or
              "a gene and a metabolite of the reaction it catalyses" — a
              plausible shortcut rather than a discovery.
    3+ hops   a genuine long-range claim, and mostly a mistake.
    unreachable   different components entirely.

Read the distribution against the `random` arm, never on its own: a network
this dense has a small diameter, so a large share of RANDOM pairs also sit 2-3
hops apart. Only the excess is informative, which is why `evaluate` scores a
type-matched random control alongside every batch.
"""

from collections import Counter, defaultdict
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import torch
from torch import Tensor

from .base import Benchmark, BenchmarkResult
from graphpath.data.node_types import default_node_type


def _canon(a: int, b: int):
    return (a, b) if a <= b else (b, a)


class NovelEdgeBenchmark(Benchmark):
    """
    train_dataset : the graphs the model actually saw. An edge counts as novel
        only if it appears in none of them.
    reference : a built `ReferenceNetwork` (load_edges + build_adjacency).
    max_hops : BFS cut-off. Anything further is reported as ">max" rather than
        measured, since the exact number stops meaning much past 4 or 5.
    """

    name = "novel edges"

    def __init__(self, train_dataset, reference, device="cpu",
                 node_type_fn: Callable[[str], str] = default_node_type,
                 max_hops: int = 6, pad_id: int = 0, n_random: int = 4000,
                 seed: int = 0):
        if reference.adj_sparse is None:
            raise ValueError("reference has no adjacency — call "
                             "reference.load_edges(...) then build_adjacency()")
        self.ds, self.ref, self.device = train_dataset, reference, device
        self.max_hops, self.pad_id = max_hops, pad_id
        self.n_random, self.seed = n_random, seed
        self.id_to_node = train_dataset.id_to_node
        self.type_of = {t: node_type_fn(str(n)) for t, n in self.id_to_node.items()}

        # every edge the model was shown, as canonical token pairs
        self.train_edges = set()
        for i in range(len(train_dataset)):
            adj, nodes, mask = train_dataset[i][:3]
            nodes = torch.as_tensor(nodes).long()
            live = torch.as_tensor(mask).bool().nonzero().ravel()
            a = (torch.as_tensor(adj) > 0)[live][:, live].triu(1).nonzero()
            for u, v in a.tolist():
                x, y = int(nodes[live[u]]), int(nodes[live[v]])
                if x != y and x != pad_id and y != pad_id:
                    self.train_edges.add(_canon(x, y))

        # CSR view of the reference, for BFS
        idx = reference.adj_sparse.coalesce().indices().cpu().numpy()
        n = reference.num_nodes
        self.n_nodes = n
        self.csr = sp.csr_matrix(
            (np.ones(idx.shape[1], dtype=np.int8), (idx[0], idx[1])), shape=(n, n))
        self.csr = ((self.csr + self.csr.T) > 0).astype(np.int8)
        self.ref_pairs = {_canon(int(u), int(v)) for u, v in idx.T}

    # ── hop distances ──────────────────────────────────────────────────────

    def _hops(self, pairs, chunk: int = 48) -> np.ndarray:
        """
        Shortest-path length for each (u, v), capped at max_hops.
        inf means further than the cap or in another component.

        Grouped by source so one BFS answers every query that starts there —
        a generated batch repeats endpoints heavily, so this is typically an
        order of magnitude fewer traversals than one per pair.
        """
        if not pairs:
            return np.zeros(0)
        by_src = defaultdict(list)
        for k, (u, v) in enumerate(pairs):
            by_src[u].append((v, k))
        out = np.full(len(pairs), np.inf)
        srcs = sorted(by_src)
        for i in range(0, len(srcs), chunk):
            block = srcs[i:i + chunk]
            # dijkstra rather than shortest_path: only dijkstra takes `limit`,
            # and stopping at max_hops is what keeps this cheap on a network
            # with a small diameter
            D = dijkstra(self.csr, directed=False, indices=block,
                         unweighted=True, limit=self.max_hops)
            for r, s in enumerate(block):
                for v, k in by_src[s]:
                    out[k] = D[r, v]
        return out

    def _type_pair(self, a: int, b: int):
        return tuple(sorted((self.type_of.get(a, "?"), self.type_of.get(b, "?"))))

    def _bucket(self, d) -> str:
        if not np.isfinite(d):
            return "unreachable"
        d = int(d)
        return {0: "same node", 1: "1 (in reference)", 2: "2 (shares a neighbour)"}.get(
            d, f"{d}" if d <= self.max_hops else f">{self.max_hops}")

    # ── the benchmark ──────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, node_toks: Tensor, adjacency: Optional[Tensor] = None,
                 **kw) -> BenchmarkResult:
        if adjacency is None:
            raise ValueError("this benchmark needs an adjacency")
        B = node_toks.shape[0]
        novel, seen_n, total_n, per_graph = [], 0, 0, []
        for b in range(B):
            nodes = node_toks[b].cpu().long()
            keep = (nodes != self.pad_id).nonzero().ravel()
            a = (adjacency[b].cpu() > 0)[keep][:, keep].triu(1).nonzero()
            g_new = 0
            for u, v in a.tolist():
                x, y = int(nodes[keep[u]]), int(nodes[keep[v]])
                if x == y:
                    continue
                total_n += 1
                p = _canon(x, y)
                if p in self.train_edges:
                    seen_n += 1
                else:
                    novel.append(p); g_new += 1
            per_graph.append(g_new)

        # a type-matched random control: same type-pair mix, random endpoints
        rng = np.random.default_rng(self.seed)
        by_type = defaultdict(list)
        for t, _ in self.id_to_node.items():
            by_type[self.type_of.get(t, "?")].append(t)
        want = Counter(self._type_pair(*p) for p in novel)
        ctrl = []
        n_ctrl = min(self.n_random, max(len(novel), 1) * 4)
        for tp, cnt in want.items():
            share = int(round(n_ctrl * cnt / max(len(novel), 1)))
            pool_a, pool_b = by_type.get(tp[0], []), by_type.get(tp[1], [])
            if not pool_a or not pool_b:
                continue
            for _ in range(share):
                x, y = int(rng.choice(pool_a)), int(rng.choice(pool_b))
                if x != y:
                    ctrl.append(_canon(x, y))

        d_new, d_ctrl = self._hops(novel), self._hops(ctrl)
        b_new = Counter(self._bucket(d) for d in d_new)
        b_ctrl = Counter(self._bucket(d) for d in d_ctrl)

        n_new = max(len(novel), 1)
        in_ref = sum(1 for p in novel if p in self.ref_pairs)
        summary = {
            "edges_total": float(total_n),
            "novel_rate": len(novel) / max(total_n, 1),
            "novel_per_graph": float(np.mean(per_graph)) if per_graph else 0.0,
            "novel_in_reference": in_ref / n_new,
            "hops_le2": float(np.mean(d_new <= 2)) if len(d_new) else 0.0,
            "hops_median": float(np.median(d_new[np.isfinite(d_new)]))
                            if np.isfinite(d_new).any() else float("inf"),
            "unreachable": float(np.mean(~np.isfinite(d_new))) if len(d_new) else 0.0,
            "control_hops_le2": float(np.mean(d_ctrl <= 2)) if len(d_ctrl) else float("nan"),
            "control_unreachable": float(np.mean(~np.isfinite(d_ctrl)))
                                    if len(d_ctrl) else float("nan"),
        }
        summary["excess_le2_over_control"] = (summary["hops_le2"]
                                              - summary["control_hops_le2"])
        notes = []
        if not novel:
            notes.append("no novel edges — every generated edge was in training")
        return BenchmarkResult(
            name=self.name, values=None, summary=summary,
            n_graphs=B, n_scored=B,
            per_graph={"novel_edges": torch.tensor(per_graph, dtype=torch.float)},
            distributions={
                "hops": d_new, "hops_control": d_ctrl,
                "hop_buckets": dict(b_new), "hop_buckets_control": dict(b_ctrl),
                "type_pairs": {f"{a}-{b}": c for (a, b), c in want.items()},
                "type_pair_hops": self._by_type(novel, d_new),
            },
            notes=notes)

    def _by_type(self, pairs, dists) -> dict:
        acc = defaultdict(list)
        for p, d in zip(pairs, dists):
            acc[self._type_pair(*p)].append(d)
        return {f"{a}-{b}": {"n": len(v),
                             "hops_le2": float(np.mean(np.array(v) <= 2)),
                             "median": float(np.median([x for x in v if np.isfinite(x)]))
                                        if any(np.isfinite(v)) else float("inf")}
                for (a, b), v in sorted(acc.items(), key=lambda kv: -len(kv[1]))}


def novel_edge_table(result: BenchmarkResult) -> str:
    """Hop histogram, generated against its type-matched control."""
    d = result.distributions
    keys = ["same node", "1 (in reference)", "2 (shares a neighbour)",
            "3", "4", "5", "6", f">{6}", "unreachable"]
    new, ctrl = d.get("hop_buckets", {}), d.get("hop_buckets_control", {})
    n1, n2 = max(sum(new.values()), 1), max(sum(ctrl.values()), 1)
    rows = [f"{'hops apart in the reference':<28}{'novel':>10}{'control':>10}{'ratio':>8}"]
    for k in keys:
        if not new.get(k) and not ctrl.get(k):
            continue
        a, b = new.get(k, 0) / n1, ctrl.get(k, 0) / n2
        rows.append(f"  {k:<26}{a:>10.1%}{b:>10.1%}"
                    f"{(a / b if b else float('inf')):>8.1f}")
    rows.append("")
    rows.append(f"{'by node-type pair':<28}{'n':>8}{'<=2 hops':>10}{'median':>8}")
    for tp, v in list(d.get("type_pair_hops", {}).items())[:8]:
        rows.append(f"  {tp:<26}{v['n']:>8,}{v['hops_le2']:>10.1%}{v['median']:>8.1f}")
    return "\n".join(rows)
