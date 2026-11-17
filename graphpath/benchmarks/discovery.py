"""Novel generated links that indirect evidence nonetheless supports.

A generated edge absent from the curated network is either a hallucination
or a hypothesis, and `KnownStructureBenchmark` cannot tell them apart — it
only counts the absence. This looks for corroboration from a source the
curated network does not contain: co-expression.

Co-expression is measured between GENES, but the metabolic network lets it
propagate. A metabolite sits next to a reaction, which sits next to the gene
encoding its enzyme; so if metabolite M reaches gene A within a couple of
hops, and the other endpoint reaches gene B, and A and B are strongly
co-expressed, then the M-to-endpoint link has support that never appears in
any gene-level table:

    metabolite ─hops─ gene A  ~~co-expressed~~  gene B ─hops─ reaction
    └───────────────── the generated edge ─────────────────┘

Support alone proves nothing — hub metabolites reach half the network in two
hops. Score novel edges against randomly drawn non-edges (`control_scores`)
and read the ranking, not the absolute number.
"""

from typing import Optional

import numpy as np
import torch
from torch import Tensor


class DiscoveryFinder:

    def __init__(self, reference, expr_loader, dataset, node_type_fn=None,
                 max_hops: int = 2, max_anchors: int = 40, pad_id: int = 0,
                 name_fn=None, exclude=None):
        """
        max_hops   : how far from an endpoint to look for a gene. 1 finds
            almost nothing, 3+ reaches most of the network through hubs and
            the score stops discriminating; 2 is the useful setting.
        max_anchors: cap on genes per endpoint, nearest hops first — keeps
            hub nodes from dominating by sheer reach.
        name_fn    : token id -> display label. Defaults to the raw entity
            id; pass `lambda i: get_common_name(dataset.id_to_node[i])` for
            readable names.
        exclude    : token ids (e.g. graphpath.data.cofactors.cofactor_ids)
            that may be neither an endpoint of a scored link nor a step of a
            support route, and are left out of the random control too.
        """
        from graphpath.data.node_types import default_node_type
        import scipy.sparse as ssp

        self.ref = reference
        self.expr = expr_loader
        self.dataset = dataset
        self.type_fn = node_type_fn or default_node_type
        self.max_hops = max_hops
        self.max_anchors = max_anchors
        self.pad_id = pad_id
        self.name_fn = name_fn or (lambda i: dataset.id_to_node.get(int(i), f"<{i}>"))
        self.exclude = frozenset(int(i) for i in (exclude or ()))

        sp = reference.adj_sparse.coalesce()
        idx = sp.indices().cpu().numpy()
        V = reference.num_nodes
        self.A = ssp.coo_matrix((np.ones(idx.shape[1]), (idx[0], idx[1])),
                                shape=(V, V)).tocsr()
        self.V = V
        self.is_gene = np.zeros(V, dtype=bool)
        for i, name in dataset.id_to_node.items():
            if 0 <= int(i) < V and self.type_fn(name) == "gene":
                self.is_gene[int(i)] = True

    def _id(self, i: int) -> Tensor:
        return torch.tensor([int(i)], device=self.ref.device)

    # ── reachability ───────────────────────────────────────────────────────

    def gene_anchors(self, node_id: int) -> dict:
        """
        {gene id: (hops, path)} for genes within max_hops of `node_id`.

        The path is kept, not just the distance — the evidence is the route,
        and a chain the reader cannot trace is not evidence.
        """
        start = int(node_id)
        prev = {start: None}
        frontier = [start]
        out = {}
        if self.is_gene[start]:
            out[start] = 0
        for h in range(1, self.max_hops + 1):
            nxt = []
            for n in frontier:
                for m in self.A.indices[self.A.indptr[n]:self.A.indptr[n + 1]]:
                    m = int(m)
                    if m in prev or m in self.exclude:
                        continue
                    prev[m] = n
                    nxt.append(m)
                    if self.is_gene[m] and m not in out:
                        out[m] = h
            frontier = nxt
            if len(out) >= self.max_anchors:
                break

        def path_to(g):
            p, cur = [], g
            while cur is not None:
                p.append(cur)
                cur = prev[cur]
            return p[::-1]

        ordered = sorted(out.items(), key=lambda kv: (kv[1], kv[0]))[:self.max_anchors]
        return {g: (h, path_to(g)) for g, h in ordered}

    # ── scoring ────────────────────────────────────────────────────────────

    def score_pair(self, u: int, v: int, tissue: str):
        """
        Strongest co-expression reachable between the two endpoints, as a
        dict with the anchor genes, hop counts and the routes taken.
        Support is 0.0 when nothing connects them.
        """
        empty = dict(support=0.0, gene_a=None, gene_b=None, hops_a=None,
                     hops_b=None, path_a=None, path_b=None)
        ga, gb = self.gene_anchors(u), self.gene_anchors(v)
        if not ga or not gb:
            return empty
        a = torch.tensor(list(ga), device=self.expr.device)
        b = torch.tensor(list(gb), device=self.expr.device)
        A = a[:, None].expand(-1, b.numel()).reshape(-1)
        B = b[None, :].expand(a.numel(), -1).reshape(-1)
        keep = A != B
        if not bool(keep.any()):
            return empty
        w = self.expr._lookup(tissue, A[keep], B[keep])
        best = int(w.argmax())
        if float(w[best]) == 0.0:
            return empty
        ga_id, gb_id = int(A[keep][best]), int(B[keep][best])
        return dict(support=float(w[best]), gene_a=ga_id, gene_b=gb_id,
                    hops_a=ga[ga_id][0], hops_b=gb[gb_id][0],
                    path_a=ga[ga_id][1], path_b=gb[gb_id][1])

    # ── search ─────────────────────────────────────────────────────────────

    def find(self, node_toks: Tensor, adjacency: Tensor, tissue: str,
             top_k: Optional[int] = None, max_pairs: int = 4000) -> list:
        """
        EVERY novel generated edge, scored and ranked by reachable
        co-expression support — including the ones scoring zero, so the
        supported fraction can be compared against chance on equal terms.
        Pass top_k to truncate.

        "Novel" means absent from the curated network. Training graphs are
        drawn from that network, so this also excludes anything the model
        could have copied.
        """
        seen, out = set(), []
        node_toks = node_toks if node_toks.dim() == 2 else node_toks.unsqueeze(0)
        for b in range(node_toks.shape[0]):
            ids = node_toks[b].cpu()
            adj = adjacency[b].bool()
            N = ids.shape[0]
            # build the index pairs on the adjacency's device: `sel` inherits
            # it, and a CUDA mask cannot index a CPU tensor
            iu = torch.triu_indices(N, N, offset=1, device=adj.device)
            sel = adj[iu[0], iu[1]]
            for i, j in zip(iu[0][sel].tolist(), iu[1][sel].tolist()):
                u, v = int(ids[i]), int(ids[j])
                if u == self.pad_id or v == self.pad_id or u == v:
                    continue
                if u in self.exclude or v in self.exclude:
                    continue
                key = (min(u, v), max(u, v))
                if key in seen or len(seen) >= max_pairs:
                    continue
                seen.add(key)
                if bool(self.ref.has_edge(self._id(u), self._id(v))[0]):
                    continue                      # already known, not a discovery
                out.append(dict(u=u, v=v, graph=b, slot_u=i, slot_v=j,
                                **self.score_pair(u, v, tissue)))
        out.sort(key=lambda d: -d["support"])
        return out[:top_k] if top_k else out

    def control_scores(self, n: int, tissue: str, seed: int = 0) -> np.ndarray:
        """
        The same score for random NON-edges — what an arbitrary pair scores.
        A discovery only means something above this.
        """
        g = torch.Generator().manual_seed(seed)
        pool = torch.tensor([i for i in range(self.V)
                             if i != self.pad_id and i not in self.exclude
                             and self.A.indptr[i + 1] > self.A.indptr[i]])
        out = []
        while len(out) < n:
            p = pool[torch.randint(0, len(pool), (2,), generator=g)]
            u, v = int(p[0]), int(p[1])
            if u == v or bool(self.ref.has_edge(self._id(u), self._id(v))[0]):
                continue
            out.append(self.score_pair(u, v, tissue)['support'])
        return np.array(out)

    def describe(self, d: dict) -> str:
        n = self.name_fn
        if not d["support"]:
            return f"{n(d['u'])} — {n(d['v'])}   no reachable support"
        return (f"{n(d['u'])} — {n(d['v'])}   support {d['support']:.3f}  "
                f"via {n(d['gene_a'])} (+{d['hops_a']}h) ~ "
                f"{n(d['gene_b'])} (+{d['hops_b']}h)")
