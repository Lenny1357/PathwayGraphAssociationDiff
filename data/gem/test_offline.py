"""Checks for the torch FBA layer on networks whose answers can be worked out
by hand, plus a few against Human-GEM itself. No solver, no cobra."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphpath.fba import (StoichModel, null_dim, flux_residual,
                       dead_end_metabolites, mass_imbalance)

D = torch.float64
OK = FAIL = 0


def check(label, got, want, tol=1e-6):
    global OK, FAIL
    good = (abs(got - want) <= tol) if isinstance(want, (int, float)) else (got == want)
    print(f"  {'ok  ' if good else 'FAIL'}  {label:<58} got {got!r}")
    OK, FAIL = OK + good, FAIL + (not good)


print("null_dim and flux_residual on hand-worked networks")

# A -> B -> C, both irreversible. Internal metabolite B; A and C are the
# severed boundary, so only B constrains. S over {B} = [1, -1].
S = torch.tensor([[1., -1.]], dtype=D)
check("chain A->B->C: null_dim", int(null_dim(S)[0]), 1)
check("chain A->B->C: residual", float(flux_residual(S, torch.zeros(2, dtype=torch.bool))), 0.0)

# A -> B, nothing consumes B. S over {B} = [1]. Rank 1 of 1 column.
S = torch.tensor([[1.]], dtype=D)
check("dangling A->B: null_dim", int(null_dim(S)[0]), 0)

# Cycle A -> B -> A. Null vector (1,1), both fluxes forward.
S = torch.tensor([[-1., 1.], [1., -1.]], dtype=D)
check("cycle A<->B: null_dim", int(null_dim(S)[0]), 1)
check("cycle A<->B: residual", float(flux_residual(S, torch.zeros(2, dtype=torch.bool))), 0.0)

# Two parallel irreversible A -> B. Null space is spanned by (1,-1), which
# needs one reaction to run backwards, so a nonneg steady state does NOT
# exist even though null_dim is 1 — the documented gap between the two.
S = torch.tensor([[-1., -1.], [1., 1.]], dtype=D)
check("parallel A->B (both irrev): null_dim", int(null_dim(S)[0]), 1)
r_irrev = float(flux_residual(S, torch.zeros(2, dtype=torch.bool), iters=400))
check("parallel A->B (both irrev): residual > 0", r_irrev > 1e-3, True)
check("parallel A->B (both irrev): residual == 2", r_irrev, 2.0, tol=1e-3)
r_rev = float(flux_residual(S, torch.tensor([False, True]), iters=400))
check("parallel A->B (one reversible): residual", r_rev, 0.0, tol=1e-8)

print("\nbatching and gradients")
Sb = torch.stack([torch.tensor([[1., -1.]], dtype=D),
                  torch.tensor([[1.,  1.]], dtype=D)])
nd = null_dim(Sb)
check("batched null_dim", nd.tolist(), [1, 1])
res = flux_residual(Sb, torch.zeros(2, dtype=torch.bool), iters=400)
check("batched residual: balanced one is 0", float(res[0]), 0.0, tol=1e-8)
check("batched residual: unbalanced one > 0", float(res[1]) > 1e-3, True)

# the property guidance depends on: d(residual)/d(soft adjacency)
# at an UNBALANCED point — a residual already at its 0 minimum has zero
# gradient by definition, which would test nothing
A = torch.tensor([[1., 1.]], dtype=D, requires_grad=True)
S0 = torch.tensor([[1., 1.]], dtype=D)
loss = flux_residual(S0 * A, torch.zeros(2, dtype=torch.bool), iters=50)
loss.backward()
check("gradient reaches a soft adjacency", A.grad is not None and bool((A.grad != 0).any()), True)

print("\ndead ends and mass balance")
S = torch.tensor([[1., -1.], [1., 0.]], dtype=D)     # met 1 only ever produced
lb = torch.zeros(2, dtype=D); ub = torch.full((2,), 1000., dtype=D)
de = dead_end_metabolites(S, lb, ub)[0]
check("dead-end detection", de.tolist(), [False, True])
# r1 is the ONLY reaction touching met 1, so r1 is the one that has to be
# reversible for met 1 to be consumable — making r2 reversible changes nothing
lb2 = torch.tensor([-1000., 0.], dtype=D)
check("reversibility rescues a dead end",
      dead_end_metabolites(S, lb2, ub)[0].tolist(), [False, False])

# H2O + CO2 -> H2CO3 balances; dropping the O does not
E = torch.tensor([[0., 1., 1.], [2., 0., 2.], [1., 2., 3.]], dtype=D)  # C,H,O
S = torch.tensor([[-1.], [-1.], [1.]], dtype=D)
check("balanced reaction", float(mass_imbalance(E, S)[0]), 0.0)
E_bad = E.clone(); E_bad[2, 2] = 2.
check("unbalanced reaction detected", float(mass_imbalance(E_bad, S)[0]) > 0.5, True)

print("\nagainst Human-GEM itself")
ref = Path(__file__).resolve().parent / "gem_ref"
if not (ref / "gem.npz").exists():
    print("  (skipped — run build_gem.py first)")
else:
    m = StoichModel(ref)
    check("dimensions", (m.n_mets, m.n_rxns), (8461, 12931))
    check("curated reactions mostly balance",
          float(m.rxn_balanced.sum()) / float(m.rxn_checkable.sum()) > 0.97, True)
    ix = m.index(["rxn_MAR03905", "MAM01965c", "ENSG00000130203", "enz_X"])
    check("id bridge: reaction", ix["rxn_idx"].tolist(), [0])
    check("id bridge: metabolite found", len(ix["met_idx"]) == 1, True)
    check("id bridge: gene found", len(ix["gene_idx"]) == 1, True)
    check("id bridge: enzyme unmatched", ix["unmatched"], ["enz_X"])

    # a real subsystem should be far more balanceable than a random reaction set
    gly = [i for i, s in enumerate(m.subsystems) if s.startswith("Glycolysis")][:25]
    gi = torch.tensor(gly, device=m.device)
    mets = (m.dense.index_select(1, gi) != 0).any(1).nonzero().ravel()
    Sg = m.submatrix(mets, gi)
    lbg, ubg = m.bounds(gi)
    rg = float(flux_residual(Sg, (lbg < 0), iters=300))
    g = torch.Generator().manual_seed(0)
    ri = torch.randperm(m.n_rxns, generator=g)[:25].to(m.device)
    mets_r = (m.dense.index_select(1, ri) != 0).any(1).nonzero().ravel()
    lbr, ubr = m.bounds(ri)
    rr = float(flux_residual(m.submatrix(mets_r, ri), (lbr < 0), iters=300))
    print(f"        glycolysis({len(gly)} rxns, {len(mets)} mets) residual {rg:.4g}  "
          f"vs random({len(ri)} rxns, {len(mets_r)} mets) {rr:.4g}")
    check("real subsystem balances better than a random reaction set", rg < rr, True)


# ── the validity layer, on graphs cut out of Human-GEM itself ───────────────
if (ref / "gem.npz").exists():
    print("\nFluxValidityBenchmark and guidance_residual")
    from graphpath.benchmarks.flux import FluxValidityBenchmark, guidance_residual

    class FakeDS:
        """Only the id_to_node map the benchmark actually needs."""
        def __init__(self, names):
            self.id_to_node = {i + 1: n for i, n in enumerate(names)}   # 0 = PAD
            self.node_to_id = {n: i for i, n in self.id_to_node.items()}

    gly = [i for i, s in enumerate(m.subsystems) if s.startswith("Glycolysis")][:14]
    cols = torch.as_tensor(gly, device=m.device)
    mets = (m.dense.index_select(1, cols) != 0).any(1).nonzero().ravel()
    names = [f"rxn_{m.rxns[c]}" for c in cols] + [m.mets[i] for i in mets]
    N = 64
    names = names[:N]
    ds = FakeDS(names)
    n_rxn_slots = min(len(cols), len(names))

    def build(wire, seed=0):
        g = torch.Generator().manual_seed(seed)
        toks = torch.zeros(N, dtype=torch.long)
        for k, nm in enumerate(names):
            toks[k] = ds.node_to_id[nm]
        adj = torch.zeros(N, N, dtype=torch.long)
        for a in range(n_rxn_slots):
            for b in range(n_rxn_slots, len(names)):
                real = float(m.dense[mets[b - n_rxn_slots], cols[a]]) != 0
                on = real if wire == "true" else (torch.rand(1, generator=g).item() < 0.15)
                if on:
                    adj[a, b] = adj[b, a] = 1
        return toks, adj

    t_true, a_true = build("true")
    t_rand, a_rand = build("random")
    bench = FluxValidityBenchmark(m, ds, mode="asserted", iters=200)
    r_true = bench.evaluate(t_true[None], a_true[None])
    r_rand = bench.evaluate(t_rand[None], a_rand[None])
    for lbl, r in (("correctly wired", r_true), ("randomly wired ", r_rand)):
        print(f"        {lbl}: precision {r.summary['participation_precision']:.3f}  "
              f"residual {r.summary['flux_residual']:.4g}  "
              f"unusable mets {r.summary['unusable_frac']:.2f} "
              f"(dead {r.summary['dead_end_frac']:.2f} / isolated "
              f"{r.summary['isolated_frac']:.2f})  "
              f"mass-balanced {r.summary['mass_balanced_frac']:.2f}  "
              f"edges {r.summary['edges_drawn']:.0f}")
    check("true wiring: participation precision is 1",
          r_true.summary["participation_precision"], 1.0, tol=1e-9)
    check("random wiring: precision far lower",
          r_rand.summary["participation_precision"] < 0.5, True)
    # compare on `unusable`, not `dead_end_frac`: the random graph draws few
    # edges, so most of its metabolites are ISOLATED and a dead-end count that
    # only looks at wired-in metabolites flatters it
    check("true wiring: fewer unusable metabolites",
          r_true.summary["unusable_frac"] < r_rand.summary["unusable_frac"], True)
    check("true wiring: more mass-balanced reactions",
          r_true.summary["mass_balanced_frac"] >= r_rand.summary["mass_balanced_frac"], True)
    check("benchmark scored the graph", r_true.n_scored, 1)

    tables = m.token_tables(ds.id_to_node)
    ep = torch.full((2, N, N), 0.5, dtype=torch.float64, requires_grad=True)
    g = guidance_residual(m, tables, torch.stack([t_true, t_rand]), ep, iters=20)
    g.sum().backward()
    check("guidance_residual: one score per graph", tuple(g.shape), (2,))
    check("guidance gradient nonzero at the edge probabilities",
          bool((ep.grad != 0).any()), True)
    check("guidance gradient finite", bool(torch.isfinite(ep.grad).all()), True)
    print(f"        guidance residual {[round(x,4) for x in g.detach().tolist()]}, "
          f"grad |max| {float(ep.grad.abs().max()):.3g}")


    # device agreement: StoichModel lives on the model's device, but
    # evaluate_dataset hands over CPU tensors from the dataset
    print("\ndevice handling")
    if torch.cuda.is_available():
        st_gpu = StoichModel(ref, device="cuda")
        b_gpu = FluxValidityBenchmark(st_gpu, ds, mode="asserted", iters=50)
        r = b_gpu.evaluate(t_true[None].cpu(), a_true[None].cpu())
        check("CPU inputs against a CUDA StoichModel", r.n_scored, 1)
        tb = st_gpu.token_tables(ds.id_to_node)
        ep2 = torch.full((1, N, N), 0.5, dtype=torch.float64, requires_grad=True)
        gg = guidance_residual(st_gpu, tb, t_true[None].cpu(), ep2, iters=10)
        gg.sum().backward()
        check("guidance with CPU inputs against a CUDA StoichModel",
              ep2.grad is not None and bool(torch.isfinite(ep2.grad).all()), True)
    else:
        print("  ..    no CUDA here — the cross-device path is UNTESTED on this box")

print(f"\n{OK} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
