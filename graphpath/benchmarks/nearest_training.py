"""How close is each generated pathway to its nearest training example?

This is the memorisation test. A generative model that scores well on
fidelity metrics may be doing so by reproducing pathways it was trained on,
and no fidelity metric can tell the difference — a copied pathway is, by
construction, perfectly faithful.

For every generated graph we find the training graph sharing the most nodes
and report how close that match is. Three numbers, each answering something
different:

    jaccard      |A n B| / |A u B| — overall set similarity
    containment  |A n B| / |A|     — how much of the GENERATED graph is
                                     contained in one training example.
                                     High containment with low Jaccard means
                                     the model reproduced a fragment.
    edge jaccard  the same for the edges of that best-matching pair — the
                                     decisive one: identical nodes wired
                                     differently is recombination, identical
                                     nodes wired identically is a copy.

Run it on the training set itself (with `exclude_identical=True`) for the
reference: that is how similar real pathways are to *each other*, and a
model whose graphs sit at that level is as novel as a held-out pathway.
"""

from typing import Optional

import torch
from torch import Tensor

from .base import Benchmark, BenchmarkResult


class NearestTrainingBenchmark(Benchmark):

    name = "nearest training pathway"

    def __init__(self, dataset, pad_id: int = 0, device=None,
                 max_train: Optional[int] = None,
                 duplicate_threshold: float = 0.9,
                 exclude_identical: bool = False):
        """
        duplicate_threshold : node Jaccard at or above which a generated
            graph is counted as a near-duplicate of a training example.
        exclude_identical : skip candidates with Jaccard == 1. Set this when
            scoring the training set against itself, or every graph matches
            itself and the answer is trivially 1.0.
        """
        self.dataset = dataset
        self.pad_id = pad_id
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.duplicate_threshold = duplicate_threshold
        self.exclude_identical = exclude_identical

        n = len(dataset) if max_train is None else min(max_train, len(dataset))
        self.n_train = n
        self.vocab_size = int(getattr(dataset, 'vocab_size', 0)) or self._infer_vocab(n)

        # (n_train, V) membership, plus each training graph's adjacency kept
        # as a node-id pair set for the edge comparison
        self.member = torch.zeros(n, self.vocab_size, dtype=torch.float32,
                                  device=self.device)
        self.train_edges = []
        for i in range(n):
            adj, node_feats, mask = dataset[i][:3]
            node_feats = torch.as_tensor(node_feats).long()
            mask = torch.as_tensor(mask).bool()
            ids = node_feats[mask]
            ids = ids[(ids != pad_id) & (ids < self.vocab_size)]
            # the dataset yields CPU tensors; `member` may live on the GPU,
            # and a CPU index cannot address a CUDA buffer
            self.member[i, ids.to(self.device)] = 1.0
            self.train_edges.append(self._edge_key_set(
                torch.as_tensor(adj), node_feats, mask))
        self.train_sizes = self.member.sum(1)                       # (n_train,)

    def _infer_vocab(self, n):
        return int(max(torch.as_tensor(self.dataset[i][1]).max() for i in range(n))) + 1

    def _edge_key_set(self, adj: Tensor, node_feats: Tensor, mask: Optional[Tensor] = None):
        """Undirected edges as a set of node-ID keys, so two graphs are
        comparable regardless of which slot an entity landed in."""
        adj = adj.bool()
        N = adj.shape[0]
        if mask is not None:
            mask = torch.as_tensor(mask).bool().to(adj.device)
            adj = adj & mask[:, None] & mask[None, :]
        iu = torch.triu_indices(N, N, offset=1, device=adj.device)
        node_feats = node_feats.to(adj.device)
        sel = adj[iu[0], iu[1]]
        u = node_feats[iu[0][sel]].long()
        v = node_feats[iu[1][sel]].long()
        keep = (u != self.pad_id) & (v != self.pad_id) & (u != v)
        u, v = u[keep], v[keep]
        lo, hi = torch.minimum(u, v), torch.maximum(u, v)
        return set((lo * self.vocab_size + hi).cpu().tolist())

    # ── the benchmark ──────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, node_toks: Tensor, adjacency: Optional[Tensor] = None,
                 **kwargs) -> BenchmarkResult:
        node_toks = node_toks if node_toks.dim() == 2 else node_toks.unsqueeze(0)
        B = node_toks.shape[0]

        # (V, B) query membership -> (n_train, B) intersection sizes
        q = torch.zeros(self.vocab_size, B, device=self.device)
        gen_sizes = torch.zeros(B, device=self.device)
        for b, row in enumerate(node_toks):
            ids = row.to(self.device).long()
            ids = ids[(ids != self.pad_id) & (ids < self.vocab_size)].unique()
            q[ids, b] = 1.0
            gen_sizes[b] = float(ids.numel())

        inter = self.member @ q                                     # (n_train, B)
        union = self.train_sizes[:, None] + gen_sizes[None, :] - inter
        jac = inter / union.clamp(min=1e-9)
        if self.exclude_identical:
            jac = jac.masked_fill(jac >= 1.0 - 1e-9, -1.0)

        best_j, best_i = jac.max(dim=0)                             # (B,)
        contain = (inter.gather(0, best_i[None, :]).squeeze(0)
                   / gen_sizes.clamp(min=1e-9))

        scored = (gen_sizes > 0) & (best_j >= 0)
        notes = []
        if not bool(scored.any()):
            notes.append("no graph had any non-padding node")
            return BenchmarkResult(self.name, None, {}, B, 0, notes)

        per_graph = {
            "node_jaccard":  best_j[scored].cpu(),
            "containment":   contain[scored].cpu(),
            "match_index":   best_i[scored].cpu(),
            "n_nodes":       gen_sizes[scored].cpu(),
        }

        # edge agreement with that same best match
        if adjacency is not None:
            e_j = []
            idx = torch.arange(B)[scored.cpu()]
            for b in idx.tolist():
                gen_e = self._edge_key_set(adjacency[b].cpu(), node_toks[b].cpu())
                ref_e = self.train_edges[int(best_i[b])]
                if not gen_e and not ref_e:
                    e_j.append(float('nan'))
                else:
                    inter_e = len(gen_e & ref_e)
                    e_j.append(inter_e / max(len(gen_e | ref_e), 1))
            per_graph["edge_jaccard"] = torch.tensor(e_j)
        else:
            notes.append("no adjacency passed — node overlap only")

        summary = {
            "nn_node_jaccard_mean":  float(per_graph["node_jaccard"].mean()),
            "nn_node_jaccard_max":   float(per_graph["node_jaccard"].max()),
            "nn_containment_mean":   float(per_graph["containment"].mean()),
            "near_duplicate_rate":   float((per_graph["node_jaccard"]
                                            >= self.duplicate_threshold).float().mean()),
            "distinct_matches":      int(per_graph["match_index"].unique().numel()),
        }
        if "edge_jaccard" in per_graph:
            ej = per_graph["edge_jaccard"]
            ej = ej[~ej.isnan()]
            if ej.numel():
                summary["nn_edge_jaccard_mean"] = float(ej.mean())
                summary["nn_edge_jaccard_max"]  = float(ej.max())

        n_scored = int(scored.sum())
        if summary["distinct_matches"] < n_scored / 2:
            notes.append(f"{n_scored} graphs matched only "
                         f"{summary['distinct_matches']} distinct training pathways — "
                         f"generation may be collapsing onto a few examples")
        return BenchmarkResult(self.name, None, summary, B, n_scored, notes,
                               per_graph=per_graph)

    # ── inspection ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def nearest(self, node_toks: Tensor, k: int = 5):
        """
        Top-k training matches per generated graph, as
        (indices (B, k), jaccard (B, k)) — for eyeballing what it matched.
        """
        node_toks = node_toks if node_toks.dim() == 2 else node_toks.unsqueeze(0)
        B = node_toks.shape[0]
        q = torch.zeros(self.vocab_size, B, device=self.device)
        sizes = torch.zeros(B, device=self.device)
        for b, row in enumerate(node_toks):
            ids = row.to(self.device).long()
            ids = ids[(ids != self.pad_id) & (ids < self.vocab_size)].unique()
            q[ids, b] = 1.0
            sizes[b] = float(ids.numel())
        inter = self.member @ q
        jac = inter / (self.train_sizes[:, None] + sizes[None, :] - inter).clamp(min=1e-9)
        if self.exclude_identical:
            jac = jac.masked_fill(jac >= 1.0 - 1e-9, -1.0)
        top = jac.topk(min(k, jac.shape[0]), dim=0)
        return top.indices.T.cpu(), top.values.T.cpu()
