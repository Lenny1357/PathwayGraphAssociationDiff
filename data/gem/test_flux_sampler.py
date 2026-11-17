"""Checks for FluxSampler. Needs gem_ref/ built; no network, no model."""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from graphpath.fba import StoichModel
from flux_sampler import FluxSampler

OK = FAIL = 0
def check(label, got, want, tol=1e-9):
    global OK, FAIL
    good = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    print(f"  {'ok  ' if good else 'FAIL'}  {label:<58} got {got!r}")
    OK, FAIL = OK + good, FAIL + (not good)

if not (HERE / "gem_ref" / "gem.npz").exists():
    print("gem_ref/ not built — run build_gem.py first"); sys.exit(0)

st = StoichModel(HERE / "gem_ref")
MAXN = 64
s = FluxSampler(st, max_nodes=MAXN, gene_frac=0.20)
rng = np.random.default_rng(0)
graphs = s.sample(n=60, seed=0, verbose=False)
print(f"sampled {len(graphs)} graphs at max_nodes={MAXN}\n")

print("completeness — the property the whole sampler exists for")
worst = 1.0
for g in graphs:
    ms = set(g["mets"])
    for j in g["rxns"]:
        need = set(int(m) for m in s.rxn_mets[j])
        worst = min(worst, len(need & ms) / max(len(need), 1))
check("every reaction has ALL of its metabolites present", worst, 1.0)

print("\nbudget")
sizes = [len(g["rxns"]) + len(g["mets"]) + len(g["genes"]) for g in graphs]
check("no graph exceeds max_nodes", max(sizes) <= MAXN, True)
check("chemistry alone respects the reserved gene budget",
      max(len(g["rxns"]) + len(g["mets"]) for g in graphs) <= int(round(MAXN * 0.8)), True)
check("genes actually appear", float(np.mean([len(g["genes"]) for g in graphs])) > 5, True)
check("min_reactions honoured", min(len(g["rxns"]) for g in graphs) >= 3, True)

print("\nflux and mass")
sc = [s.score(g) for g in graphs]
check("every kept graph has a nonzero flux space",
      min(x["null_dim"] for x in sc) >= 1, True)
check("mass balance is near-perfect",
      float(np.mean([x["mass_balanced_frac"] for x in sc])) > 0.95, True)
print(f"        mass-balanced {np.mean([x['mass_balanced_frac'] for x in sc]):.3f}   "
      f"null_dim {np.mean([x['null_dim'] for x in sc]):.1f}   "
      f"residual {np.mean([x['flux_residual'] for x in sc]):.3f}")

print("\nhubs: kept as nodes, not traversed through")
deg = np.diff(s.Scsr.indptr)
hub = np.flatnonzero(deg > s.expand_max_degree)
present = sum(1 for g in graphs if set(g["mets"]) & set(hub.tolist()))
check("hub metabolites DO appear as nodes", present > 0, True)
check("but are excluded from expansion",
      bool((~s.expandable[hub]).all()), True)
print(f"        {len(hub)} metabolites above degree {s.expand_max_degree}; "
      f"{present} of {len(graphs)} graphs contain at least one")

print("\nto_dataset")
ds, vocab = s.to_dataset(graphs)
inv = {v: k for k, v in vocab.items()}
g0 = ds[0]
A = g0["adj_matrix"]
check("PathwayGraphDataset keys present",
      set(g0) >= {"node_ids", "adj_matrix", "num_nodes", "original_group_id",
                  "center_node"}, True)
check("adjacency is square and matches num_nodes",
      A.shape == (g0["num_nodes"], g0["num_nodes"]), True)
check("adjacency is symmetric", bool((A == A.T).all()), True)
check("no self-loops", float(np.diagonal(A).sum()), 0.0)
check("reaction ids carry the project's rxn_ prefix",
      all(str(inv[t]).startswith(("rxn_", "MAM", "ENSG")) for g in ds
          for t in g["node_ids"]), True)

# no metabolite-metabolite or gene-gene edges: everything routes via a reaction
def kind(t):
    n = str(inv[t])
    return "r" if n.startswith("rxn_") else "g" if n.startswith("ENSG") else "m"
bad = 0
for g in ds[:20]:
    ks = [kind(t) for t in g["node_ids"]]
    a = np.asarray(g["adj_matrix"])
    for i, j in zip(*np.nonzero(np.triu(a, 1))):
        if "r" not in (ks[i], ks[j]):
            bad += 1
check("every edge touches a reaction node", bad, 0)
check("token ids are dense and 0-based",
      sorted(set(vocab.values())) == list(range(len(vocab))), True)

print("\ndeterminism")
a1 = s.sample(n=20, seed=7, verbose=False)
a2 = s.sample(n=20, seed=7, verbose=False)
check("same seed gives the same graphs",
      [g["rxns"] for g in a1] == [g["rxns"] for g in a2], True)

print("\nmax_nodes=128")
s2 = FluxSampler(st, max_nodes=128, gene_frac=0.20)
g2 = s2.sample(n=20, seed=0, verbose=False)
sz2 = [len(g["rxns"]) + len(g["mets"]) + len(g["genes"]) for g in g2]
check("respects the larger budget", max(sz2) <= 128, True)
check("and uses it", float(np.mean(sz2)) > 90, True)

print(f"\n{OK} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
