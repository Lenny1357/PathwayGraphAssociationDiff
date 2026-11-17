"""Sample subgraphs of Human-GEM that can actually carry flux.

The samplers elsewhere in this project grow by NODE: BFS outward until the
budget is spent. That cuts reactions in half — a reaction ends up in the graph
with three of its five metabolites — and a half-reaction is not chemistry. Mass
balance is undefined for it, and the steady-state residual measures the
truncation rather than the biology.

This one grows by REACTION. Adding a reaction adds ALL of its metabolites, so
every reaction in the result is complete and the subgraph is a real, if small,
metabolic network: E @ S is meaningful, and `flux_residual` measures whether
the thing can hold a steady state rather than how badly it was cut.

    from flux_sampler import FluxSampler
    s = FluxSampler(stoich, max_nodes=64)
    graphs = s.sample(n=2000, seed=0)          # PathwayGraphDataset format

Three things drive the growth, and all three are what make the output usable:

  COMPLETENESS   a reaction is admitted only if all of its metabolites fit in
                 the remaining budget. Never a partial reaction.
  FLUX CLOSURE   candidates are scored by how many currently DANGLING
                 metabolites they resolve. A metabolite that is only produced
                 inside the subgraph, and never consumed, cannot sit at steady
                 state; a reaction that consumes it closes the loop. This is
                 what turns a connected blob into something with a nonzero
                 flux space.
  BUDGET         adding a reaction that drags in six new metabolites costs six
                 nodes. Candidates are scored per node spent, so the sampler
                 prefers reactions that reuse what is already there.

Currency metabolites are INCLUDED by default and that is deliberate: water and
protons are real participants and dropping them makes every reaction fail mass
balance. They cost only one node each here, unlike a node-growing sampler where
they are hubs the traversal runs through. `--drop-currency` removes them if you
want the main-chain topology instead, at the price of mass balance.
"""

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp


class FluxSampler:
    """
    stoich : a built `StoichModel`.
    max_nodes : node budget per graph, counting reactions + metabolites
        (+ genes when include_genes). 64 and 128 are the sizes the project uses.
    include_genes : add the reactions' gene nodes, so the graphs carry all the
        node types the model expects. Genes are added LAST, from the leftover
        budget, because they cannot affect flux or mass balance.
    drop_currency : exclude the top-N metabolites by degree. Off by default —
        see the module docstring.
    """

    def __init__(self, stoich, max_nodes: int = 64, include_genes: bool = True,
                 drop_currency: int = 0, min_reactions: int = 3,
                 require_balanced: bool = True, expand_max_degree: int = 30,
                 gene_frac: float = 0.20):
        self.st = stoich
        self.max_nodes = max_nodes
        self.include_genes = include_genes
        # Reserve part of the budget for gene nodes. Without this, reactions
        # and metabolites spend all of it and genes almost never appear —
        # 184 across 400 graphs in the first version — which leaves the model
        # a node type it never sees. Genes cannot affect flux or mass balance,
        # so the reservation costs only chemistry-per-graph, not correctness.
        self.gene_frac = gene_frac if include_genes else 0.0
        self.min_reactions = min_reactions
        self.require_balanced = require_balanced

        S = sp.csc_matrix(stoich.S.to_dense().cpu().numpy()) \
            if not sp.issparse(stoich.S) else sp.csc_matrix(stoich.S)
        idx = stoich.S.coalesce().indices().cpu().numpy()
        val = stoich.S.coalesce().values().cpu().numpy()
        self.n_mets, self.n_rxns = stoich.n_mets, stoich.n_rxns
        self.Scsc = sp.csc_matrix((val, (idx[0], idx[1])),
                                  shape=(self.n_mets, self.n_rxns))
        self.Scsr = self.Scsc.tocsr()

        self.lb = stoich.lb.cpu().numpy()
        self.ub = stoich.ub.cpu().numpy()
        self.balanced = stoich.rxn_balanced.cpu().numpy()
        self.checkable = stoich.rxn_checkable.cpu().numpy()

        met_deg = np.diff(self.Scsr.indptr)
        # Hubs are kept as NODES — mass balance needs water and protons — but
        # they are not traversed THROUGH. 2% of metabolites (degree > 30) carry
        # 49% of all incidences, so expanding from them enumerates thousands of
        # candidates per step and lands the graph anywhere in the network. This
        # is the same node/traversal distinction the other samplers make, and
        # it is what makes growth fast enough to be usable.
        self.expandable = met_deg <= expand_max_degree
        self.expand_max_degree = expand_max_degree
        self.currency = set(np.argsort(-met_deg)[:drop_currency].tolist()) \
            if drop_currency else set()

        # reactions with no participants left after any exclusion are unusable
        self.rxn_mets = [
            np.array([m for m in self.Scsc.indices[self.Scsc.indptr[j]:
                                                   self.Scsc.indptr[j + 1]]
                      if m not in self.currency])
            for j in range(self.n_rxns)]
        self.usable = np.array([len(m) >= 2 and len(m) <= max_nodes // 2
                                for m in self.rxn_mets])
        if require_balanced:
            self.usable &= (self.balanced | ~self.checkable)

        self.rxn_genes = None
        if include_genes and getattr(stoich, "rxn_gene", None) is not None:
            rg = sp.csr_matrix(
                (np.ones(stoich.rxn_gene._nnz()),
                 (stoich.rxn_gene.indices()[0].cpu().numpy(),
                  stoich.rxn_gene.indices()[1].cpu().numpy())),
                shape=(self.n_rxns, len(stoich.genes)))
            self.rxn_genes = [rg.indices[rg.indptr[j]:rg.indptr[j + 1]]
                              for j in range(self.n_rxns)]

    # ── growth ─────────────────────────────────────────────────────────────

    def _sides(self, j: int):
        """(produced, consumed) metabolites of reaction j, honouring direction."""
        fwd, bwd = self.ub[j] > 0, self.lb[j] < 0
        lo, hi = self.Scsc.indptr[j], self.Scsc.indptr[j + 1]
        prod, cons = set(), set()
        for m, v in zip(self.Scsc.indices[lo:hi], self.Scsc.data[lo:hi]):
            m = int(m)
            if m in self.currency:
                continue
            if (v > 0 and fwd) or (v < 0 and bwd):
                prod.add(m)
            if (v < 0 and fwd) or (v > 0 and bwd):
                cons.add(m)
        return prod, cons

    def _dangling(self, mets: set, rxns: list) -> set:
        """
        Metabolites the current subgraph can only make, or only use.

        A metabolite is fine if some included reaction can produce it AND some
        included reaction can consume it, where a reversible reaction counts
        both ways. Anything else cannot sit at steady state inside the graph.

        `grow` maintains the two sets incrementally; this recomputes them and
        is kept for callers that have a graph but no growth history.
        """
        prod, cons = set(), set()
        for j in rxns:
            p, c = self._sides(j)
            prod |= p; cons |= c
        return {m for m in mets if not (m in prod and m in cons)}

    def _candidates(self, mets, rxns: set) -> set:
        """
        Reactions touching the current metabolite set, not already in it.

        Only EXPANDABLE metabolites are used to enumerate — see `expandable`.
        Without that, one water molecule in the set puts 2,722 reactions in the
        candidate list on every step.
        """
        out = set()
        for m in mets:
            if not self.expandable[m]:
                continue
            lo, hi = self.Scsr.indptr[m], self.Scsr.indptr[m + 1]
            for j in self.Scsr.indices[lo:hi]:
                j = int(j)
                if j not in rxns and self.usable[j]:
                    out.add(j)
        return out

    def grow(self, seed: int, rng) -> Optional[dict]:
        seed = int(seed)
        rxns, mets = [seed], set(int(m) for m in self.rxn_mets[seed])
        if not mets:
            return None
        budget = self.max_nodes
        chem_budget = int(round(budget * (1.0 - self.gene_frac)))
        prod, cons = self._sides(seed)      # maintained incrementally below
        rxn_set = {seed}

        while len(rxns) + len(mets) < chem_budget:
            dang = {m for m in mets if not (m in prod and m in cons)}
            cands = self._candidates(mets, rxn_set)
            if not cands:
                break
            best, best_score = None, -np.inf
            for j in cands:
                new = [int(m) for m in self.rxn_mets[j] if m not in mets]
                cost = 1 + len(new)                      # the reaction + its new mets
                if len(rxns) + len(mets) + cost > chem_budget:
                    continue
                resolved = sum(1 for m in self.rxn_mets[j] if int(m) in dang)
                # per node spent: closing dangling ends is the objective,
                # reusing existing metabolites is how the budget lasts
                score = (2.0 * resolved + 0.5 * (len(self.rxn_mets[j]) - len(new))) / cost
                score += 0.15 * rng.random()             # break ties stochastically
                if score > best_score:
                    best, best_score = j, score
            if best is None:
                break
            rxns.append(best); rxn_set.add(best)
            mets.update(int(m) for m in self.rxn_mets[best])
            p, c = self._sides(best)
            prod |= p; cons |= c

        if len(rxns) < self.min_reactions:
            return None
        g = {"rxns": rxns, "mets": sorted(mets), "genes": []}

        # genes last, out of whatever budget is left: they cannot affect flux
        # or mass balance, so they never compete with a reaction for space
        if self.rxn_genes is not None:
            room = budget - len(rxns) - len(mets)
            pool = sorted({int(x) for j in rxns for x in self.rxn_genes[j]})
            if room > 0 and pool:
                if len(pool) > room:
                    pool = [pool[k] for k in rng.permutation(len(pool))[:room]]
                g["genes"] = sorted(pool)
        return g

    # ── scoring, so a caller can filter ────────────────────────────────────

    def score(self, g: dict) -> dict:
        import torch
        from torch_fba import flux_residual, dead_end_metabolites, null_dim
        st = self.st
        mi = torch.as_tensor(g["mets"], dtype=torch.long, device=st.device)
        ri = torch.as_tensor(g["rxns"], dtype=torch.long, device=st.device)
        Ssub = st.submatrix(mi, ri)
        lb, ub = st.bounds(ri)
        E = st.E.index_select(1, mi)
        imb = (E @ Ssub).abs().amax(-2)
        multi = (Ssub != 0).sum(0) >= 2
        return {
            "n_rxns": len(g["rxns"]), "n_mets": len(g["mets"]),
            "flux_residual": float(flux_residual(Ssub, (lb < 0), iters=200)),
            "null_dim": int(null_dim(Ssub)[0]),
            "dead_end_frac": float(dead_end_metabolites(Ssub, lb, ub)[0]
                                   .to(torch.float64).mean()),
            "mass_balanced_frac": (float((imb[multi] < 1e-6).to(torch.float64).mean())
                                   if bool(multi.any()) else 1.0),
        }

    # ── the dataset ────────────────────────────────────────────────────────

    def sample(self, n: int = 1000, seed: int = 0, max_tries: Optional[int] = None,
               min_null_dim: int = 1, verbose: bool = True) -> list:
        """
        Grow `n` subgraphs and keep the ones that can carry flux.

        `min_null_dim=1` rejects graphs whose steady-state flux space is empty
        — no assignment of rates holds every internal metabolite balanced, so
        the graph is not a pathway in any useful sense. Set 0 to keep them.
        """
        rng = np.random.default_rng(seed)
        pool = np.flatnonzero(self.usable)
        tries = max_tries or max(6 * n, 200)
        out, seen, rejected = [], set(), defaultdict(int)
        order = rng.permutation(pool)
        for t in range(tries):
            if len(out) >= n:
                break
            g = self.grow(int(order[t % len(order)]), rng)
            if g is None:
                rejected["too small"] += 1
                continue
            key = frozenset(g["rxns"])
            if key in seen:
                rejected["duplicate"] += 1
                continue
            sc = self.score(g)
            if sc["null_dim"] < min_null_dim:
                rejected["blocked (null_dim 0)"] += 1
                continue
            seen.add(key)
            g["score"] = sc
            out.append(g)
        if verbose:
            print(f"kept {len(out):,} of {t + 1:,} attempts; rejected: "
                  + ", ".join(f"{k} {v:,}" for k, v in rejected.items()))
        return out

    # ── export ─────────────────────────────────────────────────────────────

    def to_dataset(self, graphs: Sequence[dict], name_of=None) -> tuple:
        """
        (graph list, node_id_map) in the format `PathwayGraphDataset` consumes.

        Node ids follow this project's conventions so `default_node_type` works
        unchanged: reactions get a `rxn_` prefix, metabolites and genes keep
        their native Human-GEM ids (MAM..., ENSG...).

        Edges are metabolite-reaction participations and gene-reaction
        assignments. There is no metabolite-metabolite edge: two metabolites of
        one reaction are connected THROUGH it, which is what the reaction node
        is for, and adding the clique would triple the edge count without
        adding information.
        """
        st = self.st
        vocab, out = {}, []

        def tok(nm):
            if nm not in vocab:
                vocab[nm] = len(vocab)
            return vocab[nm]

        for g in graphs:
            names = ([f"rxn_{st.rxns[j]}" for j in g["rxns"]]
                     + [st.mets[m] for m in g["mets"]]
                     + [st.genes[x] for x in g.get("genes", [])])
            ids = [tok(n) for n in names]
            nr, nm = len(g["rxns"]), len(g["mets"])
            pos = {("r", j): k for k, j in enumerate(g["rxns"])}
            pos.update({("m", m): nr + k for k, m in enumerate(g["mets"])})
            pos.update({("g", x): nr + nm + k for k, x in enumerate(g.get("genes", []))})
            n = len(names)
            A = np.zeros((n, n), dtype=np.float32)
            for j in g["rxns"]:
                a = pos[("r", j)]
                for m in self.rxn_mets[j]:
                    b = pos.get(("m", int(m)))
                    if b is not None:
                        A[a, b] = A[b, a] = 1.0
                if self.rxn_genes is not None:
                    for x in self.rxn_genes[j]:
                        b = pos.get(("g", int(x)))
                        if b is not None:
                            A[a, b] = A[b, a] = 1.0
            out.append({"node_ids": ids, "adj_matrix": A, "num_nodes": n,
                        "original_group_id": st.subsystems[g["rxns"][0]],
                        "center_node": f"rxn_{st.rxns[g['rxns'][0]]}",
                        "score": g.get("score", {})})
        return out, vocab
