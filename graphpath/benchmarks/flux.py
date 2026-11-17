"""Stoichiometric validity of generated pathways, and a guidance signal.

Two entry points, for the two things the user asked for:

  FluxValidityBenchmark   scores a batch of generated graphs the same way
                          every other graphpath benchmark does, so it can be read
                          against `evaluate_dataset` on real pathways.

  guidance_residual       a differentiable badness score over a SOFT
                          adjacency, batched, for steering the reverse
                          diffusion loop.

Read the benchmark against the training reference, never against zero. A real
pathway chunk is cut out of a bigger network, so its boundary metabolites have
no source or sink inside the chunk and it scores badly in absolute terms too.
The question is only ever whether generated graphs score like real ones.
"""

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from graphpath.benchmarks.base import Benchmark, BenchmarkResult
from graphpath.fba import (StoichModel, dead_end_metabolites, flux_residual,
                           mass_imbalance, null_dim)


class FluxValidityBenchmark(Benchmark):
    """
    mode="asserted"   judge the edges the model DREW: the stoichiometry is the
                      reference coefficient wherever the generated adjacency
                      connects a metabolite to a reaction, and zero elsewhere.
                      This is the mode that scores the model's structure.
    mode="reference"  ignore the drawn edges and use the reference
                      stoichiometry among whatever entities were selected.
                      Scores the node SET alone, which isolates "did it pick a
                      coherent set of reactions" from "did it wire them up".
    """

    name = "stoichiometric validity"

    def __init__(self, stoich: StoichModel, dataset, mode: str = "asserted",
                 iters: int = 200, pad_id: int = 0):
        assert mode in ("asserted", "reference")
        self.st, self.mode, self.iters, self.pad_id = stoich, mode, iters, pad_id
        self.id_to_node = dataset.id_to_node

    def _one(self, toks: Tensor, adj: Optional[Tensor]) -> Optional[dict]:
        st = self.st
        # Everything downstream indexes GEM tensors with slot indices derived
        # from `toks`, so the two have to agree on a device. They usually do
        # not: StoichModel is built on the model's device, while
        # evaluate_dataset hands over CPU tensors straight from the dataset.
        toks = toks.to(st.device)
        if adj is not None:
            adj = adj.to(st.device)
        keep = toks != self.pad_id
        slots = keep.nonzero().ravel()
        names = [self.id_to_node.get(int(t), "") for t in toks[slots]]
        ix = st.index(names)
        rs, ms = slots[ix["rxn_slots"]], slots[ix["met_slots"]]
        nr, nm = len(rs), len(ms)
        if nr == 0 or nm == 0:
            return None

        S_ref = st.lookup(ix["met_idx"][:, None], ix["rxn_idx"][None, :])   # (nm, nr)
        drawn = None
        if adj is not None:
            drawn = (adj[ms][:, rs] > 0).to(S_ref.dtype)
        S = S_ref * drawn if (self.mode == "asserted" and drawn is not None) else S_ref

        lb, ub = st.bounds(ix["rxn_idx"])
        real = S_ref != 0
        nd = int(null_dim(S)[0])
        dead = dead_end_metabolites(S, lb, ub)[0]
        isolated = ~(S != 0).any(1)
        out = {
            "n_rxns": float(nr), "n_mets": float(nm),
            "mapped_frac": float((len(rs) + len(ms) + len(ix["gene_idx"]))
                                 / max(len(names), 1)),
            "flux_residual": float(flux_residual(S, (lb < 0), iters=self.iters)),
            "null_dim": float(nd),
            "blocked": float(nd == 0),
            # dead-end counts only metabolites that are actually wired in;
            # an isolated one is excluded by that definition and is arguably
            # worse, so both are reported and `unusable` is their union
            "dead_end_frac": float(dead.to(torch.float64).mean()),
            "isolated_frac": float(isolated.to(torch.float64).mean()),
            "unusable_frac": float((dead | isolated).to(torch.float64).mean()),
            "edges_drawn": float((drawn > 0).sum()) if drawn is not None else float("nan"),
        }
        if st.E.shape[0] > 0:
            E = st.E.index_select(1, ix["met_idx"])
            imb = mass_imbalance(E, S)
            multi = (S != 0).sum(0) >= 2
            out["mass_imbalance"] = float(imb[multi].mean()) if bool(multi.any()) else 0.0
            out["mass_balanced_frac"] = (float((imb[multi] < 1e-6).to(torch.float64).mean())
                                         if bool(multi.any()) else 0.0)
        if drawn is not None:
            # of the metabolite-reaction edges the model drew, how many are
            # participations that really exist; and of the real ones among the
            # chosen entities, how many it found
            d = drawn.bool()
            out["participation_precision"] = (float((d & real).sum() / d.sum())
                                              if bool(d.any()) else 0.0)
            out["participation_recall"] = (float((d & real).sum() / real.sum())
                                           if bool(real.any()) else 0.0)
        return out

    @torch.no_grad()
    def evaluate(self, node_toks: Tensor, adjacency: Optional[Tensor] = None,
                 **kw) -> "BenchmarkResult":
        rows = []
        for b in range(node_toks.shape[0]):
            r = self._one(node_toks[b],
                          adjacency[b] if adjacency is not None else None)
            if r is not None:
                rows.append(r)
        keys = sorted({k for r in rows for k in r})
        per = {k: torch.tensor([r.get(k, float("nan")) for r in rows])
               for k in keys}
        summary = {k: float(v.nanmean()) for k, v in per.items()}
        notes = []
        n_skip = node_toks.shape[0] - len(rows)
        if n_skip:
            notes.append(f"{n_skip} graph(s) skipped for having no reaction or "
                         f"no metabolite node")
        return BenchmarkResult(
            name=f"{self.name} [{self.mode}]", values=None, summary=summary,
            n_graphs=int(node_toks.shape[0]), n_scored=len(rows),
            per_graph=per, notes=notes)


# ── guidance ────────────────────────────────────────────────────────────────

def guidance_residual(stoich: StoichModel, tables: dict, node_toks: Tensor,
                      edge_prob: Tensor, iters: int = 30) -> Tensor:
    """
    (B,) differentiable steady-state residual over a SOFT adjacency.

    node_toks : (B, N) long  — the current decoded tokens
    edge_prob : (B, N, N)    — P(edge present), e.g. edge_logits.softmax(-1)[...,1]
    tables    : stoich.token_tables(dataset.id_to_node), built once

    Every slot is treated as both a candidate metabolite row and a candidate
    reaction column; the (kind) tables zero out the combinations that make no
    sense, and `edge_prob` scales what survives. So the score is a smooth
    function of the model's edge beliefs and autograd reaches the edge logits.

    To use it as classifier guidance, subtract its gradient from the edge
    logits before the posterior is formed:

        p = edge_logits.softmax(-1)[..., 1].requires_grad_()
        g = torch.autograd.grad(guidance_residual(...).sum(), p)[0]
        edge_logits[..., 1] -= scale * g

    Keep `scale` small and check it against the benchmark: pushing the residual
    down is not the same as making the graph biologically better, and a large
    scale will happily produce a graph that balances because it is empty.
    """
    # same device contract as the benchmark. For guidance this should be a
    # no-op — build StoichModel on the model's device, or every reverse step
    # pays a host copy.
    node_toks = node_toks.to(stoich.device)
    edge_prob = edge_prob.to(stoich.device)          # differentiable; the grad
                                                     # flows back to the caller
    B, N = node_toks.shape
    kind, gem = tables["kind"][node_toks], tables["gem"][node_toks]     # (B,N)
    is_m, is_r = kind == 2, kind == 1
    safe = gem.clamp(min=0)

    S = stoich.lookup(safe[:, :, None], safe[:, None, :])               # (B,N,N)
    S = S * (is_m[:, :, None] & is_r[:, None, :]).to(S.dtype)
    S = S * edge_prob.to(S.dtype)

    rxn_gem = torch.where(is_r, safe, torch.zeros_like(safe))
    rev = (stoich.lb[rxn_gem] < 0) & is_r
    return flux_residual(S, rev, iters=iters, col_mask=is_r)
