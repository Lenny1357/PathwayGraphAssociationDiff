"""Hold out biological edges from training so they can be predicted back.

The usual 90/10 split in this project holds out whole GRAPHS, and it leaks:
chunks are cut from one connected network, so a held-out chunk sits at ~0.85
Jaccard to a training chunk and the model has seen almost all of it.

This splits on EDGES instead. A held-out edge — say a specific enzyme-reaction
association — is removed from every graph it appears in, so the model never
sees it anywhere. Its endpoints stay in the vocabulary and keep their other
edges, which is what makes the task link prediction rather than cold start.

Two properties that have to hold, and both are checked in test_offline.py:

  GLOBAL     an edge is identified by its canonical (token, token) pair, not
             by a position in one graph. Removing it from graph 7 but leaving
             it in graph 12 would leak it outright.
  TYPED      edges are held out per node-type pair, so you can ask separately
             whether the model knows enzyme->reaction associations and
             whether it knows which metabolites share a reaction.
"""

from collections import Counter, defaultdict
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


def canon(a: int, b: int):
    return (a, b) if a <= b else (b, a)


class EdgeSplit:
    """
    edge_types : iterable of unordered node-type pairs to hold out, e.g.
                 [("enzyme", "reaction"), ("gene", "enzyme"),
                  ("metabolite", "reaction")]
                 Pass None for "every type pair".
    frac       : share of the eligible edges to hold out.
    min_degree : never hold out an edge if it would leave either endpoint with
                 fewer than this many remaining edges ANYWHERE in the corpus.
                 Holding out a node's only edge makes it unpredictable rather
                 than held out — there is nothing left to predict from.
    """

    def __init__(self, dataset, node_type_fn: Callable[[str], str],
                 edge_types: Optional[Sequence] = None, frac: float = 0.10,
                 seed: int = 0, min_degree: int = 1, pad_id: int = 0):
        self.dataset, self.pad_id = dataset, pad_id
        self.type_of_token = {}
        for tok, name in dataset.id_to_node.items():
            self.type_of_token[tok] = node_type_fn(name)

        want = None if edge_types is None else {frozenset(t) for t in edge_types}

        # ── collect every edge in the corpus, once, canonically ───────────
        occur = Counter()          # canonical pair -> how many graphs contain it
        degree = Counter()         # token -> distinct partners
        partners = defaultdict(set)
        pairs_of_type = defaultdict(list)
        for i in range(len(dataset)):
            adj, nodes, mask = dataset[i][:3]
            nodes = torch.as_tensor(nodes).long()
            mask = torch.as_tensor(mask).bool()
            adj = torch.as_tensor(adj)
            live = mask.nonzero().ravel()
            sub = (adj[live][:, live] > 0).triu(1).nonzero()
            for u, v in sub.tolist():
                a, b = int(nodes[live[u]]), int(nodes[live[v]])
                if a == self.pad_id or b == self.pad_id or a == b:
                    continue
                p = canon(a, b)
                occur[p] += 1
                partners[a].add(b); partners[b].add(a)
                tp = frozenset((self.type_of_token.get(a, "?"),
                                self.type_of_token.get(b, "?")))
                pairs_of_type[tp].append(p)
        for t, ps in partners.items():
            degree[t] = len(ps)

        self.n_edges_total = len(occur)
        self.type_counts = {tuple(sorted(k)): len(set(v))
                            for k, v in pairs_of_type.items()}

        # ── choose which to hold out ──────────────────────────────────────
        eligible = []
        for tp, ps in pairs_of_type.items():
            if want is not None and tp not in want:
                continue
            eligible.extend(sorted(set(ps)))
        eligible = sorted(set(eligible))
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(eligible), generator=g).tolist()

        held, skipped, deg = [], 0, dict(degree)
        target = int(round(frac * len(eligible)))
        for k in order:
            if len(held) >= target:
                break
            a, b = eligible[k]
            if deg.get(a, 0) - 1 < min_degree or deg.get(b, 0) - 1 < min_degree:
                skipped += 1
                continue
            held.append((a, b)); deg[a] -= 1; deg[b] -= 1

        self.held_out = set(held)
        self.n_eligible = len(eligible)
        self.n_skipped_min_degree = skipped
        self.held_by_type = Counter(
            tuple(sorted((self.type_of_token.get(a, "?"),
                          self.type_of_token.get(b, "?"))))
            for a, b in self.held_out)

    # ── use ────────────────────────────────────────────────────────────────

    def mask_graph(self, adj: Tensor, nodes: Tensor) -> Tensor:
        """Zero every held-out edge in one graph. Returns a copy."""
        adj = torch.as_tensor(adj).clone()
        nodes = torch.as_tensor(nodes).long()
        N = adj.shape[0]
        idx = (adj > 0).triu(1).nonzero()
        for u, v in idx.tolist():
            if canon(int(nodes[u]), int(nodes[v])) in self.held_out:
                adj[u, v] = adj[v, u] = 0
        return adj

    def positives_in(self, nodes: Tensor) -> list:
        """(slot, slot) positions in this graph whose edge was held out."""
        nodes = torch.as_tensor(nodes).long()
        live = (nodes != self.pad_id).nonzero().ravel().tolist()
        out = []
        for i_, u in enumerate(live):
            for v in live[i_ + 1:]:
                if canon(int(nodes[u]), int(nodes[v])) in self.held_out:
                    out.append((u, v))
        return out

    def summary(self) -> str:
        L = [f"EdgeSplit: {len(self.held_out):,} of {self.n_eligible:,} eligible "
             f"edges held out ({len(self.held_out)/max(self.n_eligible,1):.1%}); "
             f"{self.n_edges_total:,} distinct edges in the corpus"]
        if self.n_skipped_min_degree:
            L.append(f"  {self.n_skipped_min_degree:,} skipped: would have left an "
                     f"endpoint below min_degree")
        L.append("  held out by type pair:")
        for t, n in self.held_by_type.most_common():
            avail = self.type_counts.get(t, 0)
            L.append(f"    {t[0]:<11}-{t[1]:<11} {n:>6,} of {avail:,}")
        return "\n".join(L)


class MaskedEdgeDataset(Dataset):
    """The training view: the same graphs with the held-out edges removed."""

    def __init__(self, dataset, split: EdgeSplit):
        self.ds, self.split = dataset, split
        self.id_to_node = dataset.id_to_node
        self.node_to_id = getattr(dataset, "node_to_id", None)
        self.vocab_size = dataset.vocab_size
        self._cache = {}

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        if i not in self._cache:
            item = list(self.ds[i])
            item[0] = self.split.mask_graph(item[0], item[1])
            self._cache[i] = tuple(item)
        return self._cache[i]


# ── evaluation ──────────────────────────────────────────────────────────────

def _auc(pos: Tensor, neg: Tensor) -> float:
    """Rank-based AUC. Ties count a half, which matters at t=0 where a model
    can emit identical logits for every position."""
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    allv = torch.cat([pos, neg])
    r = allv.argsort().argsort().to(torch.float64) + 1        # average-free ranks
    # handle ties by averaging ranks within equal values
    uniq, inv = torch.unique(allv, return_inverse=True)
    mean_r = torch.zeros_like(uniq, dtype=torch.float64).scatter_reduce(
        0, inv, r, reduce="mean", include_self=False)
    r = mean_r[inv]
    n1, n0 = pos.numel(), neg.numel()
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def link_prediction(model, dataset, split: EdgeSplit, t: int = 5,
                    n_graphs: int = 64, n_neg_per_pos: int = 20,
                    device=None, seed: int = 0) -> dict:
    """
    Can the model put a held-out edge back?

    Each graph goes in exactly as the model saw it in training — held-out
    edges removed — lightly corrupted at timestep `t`, and P(edge) is read at
    two kinds of position:

      positives   the held-out edges, which really belong
      negatives   true non-edges with the SAME node-type pair

    Matching the negatives by type pair is not optional. Without it the
    learned type-pair bias separates the two sets on its own and the AUC
    measures nothing but "does it know a gene does not bind a gene".

    AUC > 0.5 means the model generalises to associations it never saw. It is
    a genuine held-out number, unlike the graph-level split, where a held-out
    chunk sits at ~0.85 Jaccard to a training one.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:            # a parameterless scorer is legitimate
            device = torch.device("cpu")
    was = model.training
    model.eval()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n_graphs].tolist()

    scores = defaultdict(lambda: ([], []))       # type pair -> (pos, neg)
    n_pos_total = n_graph_used = 0
    for i in idx:
        adj, nodes, mask = dataset[i][:3]
        nodes = torch.as_tensor(nodes).long()
        pos = split.positives_in(nodes)
        if not pos:
            continue
        n_graph_used += 1
        adj_in = split.mask_graph(adj, nodes).long().to(device)
        nodes_d = nodes.to(device)
        tt = torch.full((1,), t, dtype=torch.long, device=device)

        node_in = nodes_d[None]
        edge_in = (adj_in > 0).long()[None]
        if t > 0 and hasattr(model, "schedule") and hasattr(model.schedule, "q_sample"):
            try:
                edge_in = model.schedule.q_sample(edge_in, tt, model.edge_vocab)
            except TypeError:
                pass
        _, edge_logits = model.forward(node_in, edge_in, tt, tt)
        p_edge = (edge_logits[0, :, :, 1] - edge_logits[0, :, :, 0]).cpu()

        live = (nodes != split.pad_id).nonzero().ravel().tolist()
        by_type = defaultdict(list)
        for a_ in range(len(live)):
            for b_ in range(a_ + 1, len(live)):
                u, v = live[a_], live[b_]
                tp = tuple(sorted((split.type_of_token.get(int(nodes[u]), "?"),
                                   split.type_of_token.get(int(nodes[v]), "?"))))
                by_type[tp].append((u, v))
        held = set(pos)
        real = (torch.as_tensor(adj) > 0)
        for u, v in pos:
            tp = tuple(sorted((split.type_of_token.get(int(nodes[u]), "?"),
                               split.type_of_token.get(int(nodes[v]), "?"))))
            cand = [(a_, b_) for (a_, b_) in by_type[tp]
                    if (a_, b_) not in held and not bool(real[a_, b_])]
            if not cand:
                continue
            k = torch.randperm(len(cand), generator=g)[:n_neg_per_pos].tolist()
            scores[tp][0].append(float(p_edge[u, v]))
            scores[tp][1].extend(float(p_edge[cand[j][0], cand[j][1]]) for j in k)
            n_pos_total += 1

    if was:
        model.train()

    out, allp, alln = {}, [], []
    for tp, (p, n) in scores.items():
        P, N = torch.tensor(p), torch.tensor(n)
        allp.append(P); alln.append(N)
        out[f"{tp[0]}-{tp[1]}"] = {"auc": _auc(P, N), "n_pos": len(p), "n_neg": len(n)}
    if allp:
        out["overall"] = {"auc": _auc(torch.cat(allp), torch.cat(alln)),
                          "n_pos": n_pos_total,
                          "n_graphs_with_a_held_out_edge": n_graph_used}
    else:
        out["overall"] = {"auc": float("nan"), "n_pos": 0,
                          "n_graphs_with_a_held_out_edge": n_graph_used}
    return out


def link_table(res: dict) -> str:
    rows = [f"{'type pair':<26}{'AUC':>8}{'n pos':>8}{'n neg':>8}"]
    for k, v in sorted(res.items(), key=lambda kv: kv[0] == "overall"):
        rows.append(f"{k:<26}{v['auc']:>8.3f}{v['n_pos']:>8,}"
                    f"{v.get('n_neg', 0):>8,}")
    rows.append("  0.5 = chance; the negatives are type-matched, so anything")
    rows.append("  above chance is real generalisation to unseen associations.")
    return "\n".join(rows)
