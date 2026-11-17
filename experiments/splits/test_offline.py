"""Checks for the edge split. No dataset, no model, no network."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root
from experiments.splits.edge_split import (EdgeSplit, MaskedEdgeDataset, canon, _auc,
                        link_prediction, link_table)

OK = FAIL = 0
def check(label, got, want, tol=1e-9):
    global OK, FAIL
    good = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    print(f"  {'ok  ' if good else 'FAIL'}  {label:<56} got {got!r}")
    OK, FAIL = OK + good, FAIL + (not good)


# ── a toy corpus with real node types ───────────────────────────────────────
TYPES = ["gene", "enzyme", "reaction", "metabolite"]
V, MAXN = 60, 12
names = {}
for t in range(1, V):                      # 0 is PAD
    names[t] = f"{TYPES[t % 4]}_{t}"
type_fn = lambda n: n.split("_")[0]

class ToyDS:
    """Graphs built from a fixed global edge set, so an edge recurs across
    graphs — which is exactly what a global hold-out has to cope with."""
    def __init__(self, n_graphs=40, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.id_to_node = names
        self.vocab_size = V
        # a fixed bipartite-ish truth: enzymes->reactions, genes->enzymes,
        # metabolites->reactions
        self.truth = set()
        toks = list(names)
        for a in toks:
            for b in toks:
                if a >= b: continue
                ta, tb = type_fn(names[a]), type_fn(names[b])
                p = frozenset((ta, tb))
                if p in ({"enzyme","reaction"}, {"gene","enzyme"},
                         {"metabolite","reaction"}) and \
                   torch.rand(1, generator=g).item() < 0.25:
                    self.truth.add((a, b))
        self.graphs = []
        for _ in range(n_graphs):
            k = int(torch.randint(6, MAXN + 1, (1,), generator=g))
            ids = torch.randperm(V - 1, generator=g)[:k] + 1
            a = torch.zeros(MAXN, MAXN)
            for i in range(k):
                for j in range(i + 1, k):
                    if canon(int(ids[i]), int(ids[j])) in self.truth:
                        a[i, j] = a[j, i] = 1
            nodes = torch.zeros(MAXN, dtype=torch.long); nodes[:k] = ids
            m = torch.zeros(MAXN); m[:k] = 1
            self.graphs.append((a, nodes, m))
    def __len__(self): return len(self.graphs)
    def __getitem__(self, i): return (*self.graphs[i], 0, 0)

ds = ToyDS()
print("EdgeSplit")
sp = EdgeSplit(ds, type_fn, edge_types=[("enzyme", "reaction")], frac=0.3, seed=0)
check("only the requested type pair is held out",
      set(sp.held_by_type) == {("enzyme", "reaction")}, True)
check("held out something", len(sp.held_out) > 0, True)

# GLOBAL: a held-out edge must be gone from EVERY graph
mds = MaskedEdgeDataset(ds, sp)
leaks = 0
for i in range(len(ds)):
    adj, nodes, _, *_ = mds[i]
    live = (nodes != 0).nonzero().ravel()
    for a_ in range(len(live)):
        for b_ in range(a_ + 1, len(live)):
            u, v = int(live[a_]), int(live[b_])
            if canon(int(nodes[u]), int(nodes[v])) in sp.held_out and adj[u, v] > 0:
                leaks += 1
check("no held-out edge survives in any graph", leaks, 0)

# and the rest is untouched
kept_ok = True
for i in range(len(ds)):
    a0 = torch.as_tensor(ds[i][0]); a1 = torch.as_tensor(mds[i][0])
    nodes = torch.as_tensor(ds[i][1])
    diff = (a0 != a1).nonzero()
    for u, v in diff.tolist():
        if canon(int(nodes[u]), int(nodes[v])) not in sp.held_out:
            kept_ok = False
check("nothing else was removed", kept_ok, True)

n_removed = sum(int((torch.as_tensor(ds[i][0]) > 0).sum() -
                    (torch.as_tensor(mds[i][0]) > 0).sum()) for i in range(len(ds)))
print(f"        {len(sp.held_out)} distinct edges held out -> {n_removed//2} edge "
      f"instances removed across {len(ds)} graphs")

sp_all = EdgeSplit(ds, type_fn, edge_types=None, frac=0.2, seed=0)
check("edge_types=None covers several pairs", len(sp_all.held_by_type) > 1, True)

sp_deg = EdgeSplit(ds, type_fn, edge_types=None, frac=1.0, seed=0, min_degree=1)
check("min_degree guard refuses some edges at frac=1.0",
      sp_deg.n_skipped_min_degree > 0, True)

print("\nAUC harness")
check("perfect separation", _auc(torch.tensor([3., 4.]), torch.tensor([1., 2.])), 1.0)
check("reversed", _auc(torch.tensor([1., 2.]), torch.tensor([3., 4.])), 0.0)
check("all tied is chance", _auc(torch.ones(5), torch.ones(7)), 0.5)
check("interleaved", _auc(torch.tensor([2., 4.]), torch.tensor([1., 3.])), 0.75)

print("\nlink_prediction end to end")
class FakeModel(nn.Module):
    """edge logits from a rule, so the AUC is known in advance."""
    def __init__(self, ds, mode): super().__init__(); self.ds, self.mode = ds, mode
    edge_vocab = 2
    def forward(self, node, edge, t_n, t_e):
        B, N = node.shape
        el = torch.zeros(B, N, N, 2)
        if self.mode == "oracle":
            for b in range(B):
                for u in range(N):
                    for v in range(N):
                        if canon(int(node[b, u]), int(node[b, v])) in self.ds.truth:
                            el[b, u, v, 1] = 5.0
        elif self.mode == "random":
            el[..., 1] = torch.rand(B, N, N, generator=torch.Generator().manual_seed(1))
        return torch.zeros(B, N, 2), el

for mode, lo, hi in (("oracle", 0.999, 1.001), ("random", 0.35, 0.65)):
    r = link_prediction(FakeModel(ds, mode), ds, sp, t=0, n_graphs=40, seed=0)
    a = r["overall"]["auc"]
    check(f"{mode} model AUC in [{lo}, {hi}]", lo <= a <= hi, True)
    print(f"        {mode}: AUC {a:.3f} over {r['overall']['n_pos']} positives")


# ── version split, against the cached release files ────────────────────────
print("\nVersionSplit")
CACHE = Path(__file__).resolve().parents[2] / "data" / "gem" / "gem_cache"
have = (CACHE / "Human-GEM-v1.13.0.mat").exists() and (CACHE / "Human-GEM-main.mat").exists()
if not have:
    print("  ..    release .mat files not cached — run splits.ipynb cell 3 first")
else:
    from version_split import Release, VersionSplit
    old = Release.from_mat(CACHE / "Human-GEM-v1.13.0.mat", "v1.13.0")
    new = Release.from_mat(CACHE / "Human-GEM-main.mat", "main")
    check("old release parsed", len(old.rxns) > 12000, True)
    vs = VersionSplit(old, new)
    check("gained and lost are disjoint", len(vs.gained & vs.lost), 0)
    check("provenance buckets partition gained",
          vs.gained_from_new_reactions | vs.gained_from_edits == vs.gained, True)
    check("provenance buckets do not overlap",
          len(vs.gained_from_new_reactions & vs.gained_from_edits), 0)
    # the invariant that catches a direction bug: reversing the releases must
    # swap gained and lost exactly
    rev = VersionSplit(new, old)
    check("reversing the releases swaps gained and lost",
          rev.gained == vs.lost and rev.lost == vs.gained, True)
    check("only shared entities appear in the target",
          all(a in vs.shared_entities and b in vs.shared_entities
              for a, b in list(vs.gained)[:500]), True)

    class DS:
        def __init__(self, ents):
            self.id_to_node = {i + 1: e for i, e in enumerate(sorted(ents))}
            self.vocab_size = len(self.id_to_node) + 1
        def __len__(self): return 0
    ents = {e for pr in list(vs.gained)[:3000] for e in pr}
    ds2 = DS(ents)
    tf = lambda n: "gene" if str(n).startswith("ENSG") else "metabolite"
    es = vs.to_edge_split(ds2, tf, which="new_chemistry")
    check("to_edge_split yields canonical ordered pairs",
          all(a <= b for a, b in es.held_out), True)
    check("pairs outside the vocabulary are counted, not dropped silently",
          es.n_not_in_vocab + len(es.held_out) >= len(vs.gained_from_new_reactions), True)
    print(f"        {vs.old.tag} -> {vs.new.tag}: {len(vs.gained):,} gained "
          f"({len(vs.gained_from_new_reactions):,} new chemistry, "
          f"{len(vs.gained_from_edits):,} re-annotation), {len(vs.lost):,} lost")

print(f"\n{OK} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
