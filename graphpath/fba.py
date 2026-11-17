"""Steady-state flux checks on Human-GEM, batched and differentiable, in torch.

What this is FOR, and what it is not
------------------------------------
It is not a replacement for COBRApy. COBRApy calls a real LP solver (GLPK,
Gurobi) through optlang and returns a certified optimum in exact simplex
arithmetic. Nothing here does that, and for "what is the maximum biomass flux
of Human-GEM" you should use COBRApy and believe it over this.

This exists for the two things a solver cannot do in a diffusion loop:

  BATCHING       COBRApy solves one model per call, on the CPU, through a
                 Python object graph. Scoring a batch of 32 generated graphs
                 at every one of 200 reverse steps is 6,400 solver calls per
                 sample. Here a batch of subnetworks is one dense (B, m, n)
                 tensor and one batched SVD.

  GRADIENTS      Guidance needs d(score)/d(edge probability), so the score has
                 to be a differentiable function of a SOFT adjacency. An LP
                 solver returns a number, and the vertex it came from is a
                 discontinuous function of the input. `flux_residual` is a
                 smooth function of the stoichiometry it is handed, so
                 autograd flows back to the model's edge logits.

The tradeoff is honest: `flux_residual` is a projected-gradient minimum of a
Rayleigh quotient, not an LP optimum. It answers "how close can this
subnetwork get to a steady state" rather than "is it feasible, yes or no".
`null_dim` IS exact, but it ignores irreversibility. Use both, and calibrate
against real pathway chunks rather than against an absolute threshold —
that is what `validity.py` does.

A note on scale. The guidance case works on subnetworks of <= 64 nodes, so
the extracted matrices are perhaps 40x25 and the batched dense treatment here
is the right one. Do not point this at the full 8,461 x 12,931 model.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import json
import numpy as np
import torch
from torch import Tensor


# ── the model ───────────────────────────────────────────────────────────────

class StoichModel:
    """Human-GEM's stoichiometry as tensors, plus the id bridge."""

    def __init__(self, ref_dir, device="cpu", dtype=torch.float64):
        ref = Path(ref_dir)
        z = np.load(ref / "gem.npz", allow_pickle=False)
        meta = json.loads((ref / "meta.json").read_text())

        self.device, self.dtype = torch.device(device), dtype
        self.n_mets, self.n_rxns = (int(x) for x in z["S_shape"])

        self.S = torch.sparse_coo_tensor(
            np.stack([z["S_row"], z["S_col"]]), z["S_val"],
            (self.n_mets, self.n_rxns), dtype=dtype, device=self.device).coalesce()
        for k in ("lb", "ub", "c"):
            setattr(self, k, torch.as_tensor(z[k], dtype=dtype, device=self.device))
        self.E = torch.as_tensor(z["E"], dtype=dtype, device=self.device)
        self.met_charges = torch.as_tensor(z["met_charges"], dtype=dtype,
                                           device=self.device)
        self.rxn_balanced = torch.as_tensor(z["rxn_balanced"], device=self.device)
        # (n_rxns, n_genes) incidence — already in the npz, exposed so samplers
        # can attach gene nodes without re-reading the file
        self.rxn_gene = torch.sparse_coo_tensor(
            np.stack([z["rgm_row"], z["rgm_col"]]),
            np.ones(len(z["rgm_row"]), dtype=np.float32),
            (len(meta["rxns"]), len(meta["genes"]))).coalesce()
        self.rxn_checkable = torch.as_tensor(z["rxn_checkable"], device=self.device)

        self.rxns, self.mets, self.genes = meta["rxns"], meta["mets"], meta["genes"]
        self.elements = meta["elements"]
        self.subsystems = meta["subsystems"]
        self.rxn_of = {r: i for i, r in enumerate(self.rxns)}
        self.met_of = {m: i for i, m in enumerate(self.mets)}
        self.gene_of = {g: i for i, g in enumerate(self.genes)}
        self._dense = None

    def __repr__(self):
        return (f"StoichModel({self.n_mets:,} mets x {self.n_rxns:,} rxns, "
                f"{self.S._nnz():,} nnz, {self.device})")

    @property
    def dense(self) -> Tensor:
        """
        (n_mets, n_rxns) dense S — 875 MB at float64, and it lands on
        self.device, so on a GPU it is 875 MB of VRAM. Nothing in the library
        needs it; `lookup` and `submatrix` both go through the sparse index.
        Materialise it only for interactive slicing where that cost is fine.
        """
        if self._dense is None:
            self._dense = self.S.to_dense()
        return self._dense

    # ── bridge from this project's node ids ────────────────────────────────
    #
    # graphpath node ids carry a type prefix: reactions are "rxn_MAR03905",
    # metabolites are bare "MAM01965c", genes are bare "ENSG00000130203".
    # Anything else (an "enz_..." node) has no counterpart in the GEM.

    @staticmethod
    def strip(name: str) -> str:
        s = str(name)
        return s[4:] if s.startswith("rxn_") else s

    def index(self, names: Sequence[str]) -> dict:
        """
        Sort a list of node names into GEM row/column indices.

        Returns slot-aligned arrays so a generated graph's adjacency can be
        cut down to the same slots: `rxn_slots[k]` is the position in `names`
        of the reaction whose GEM column is `rxn_idx[k]`.
        """
        rxn_idx, rxn_slots, met_idx, met_slots, gene_idx, unmatched = [], [], [], [], [], []
        for k, nm in enumerate(names):
            s = self.strip(nm)
            if s in self.rxn_of:
                rxn_idx.append(self.rxn_of[s]); rxn_slots.append(k)
            elif s in self.met_of:
                met_idx.append(self.met_of[s]); met_slots.append(k)
            elif s in self.gene_of:
                gene_idx.append(self.gene_of[s])
            else:
                unmatched.append(nm)
        L = lambda v: torch.as_tensor(v, dtype=torch.long, device=self.device)
        return {"rxn_idx": L(rxn_idx), "rxn_slots": L(rxn_slots),
                "met_idx": L(met_idx), "met_slots": L(met_slots),
                "gene_idx": L(gene_idx), "unmatched": unmatched}

    def submatrix(self, met_idx: Tensor, rxn_idx: Tensor) -> Tensor:
        """
        (len(met_idx), len(rxn_idx)) slice of S, dense in the OUTPUT only —
        routed through the sparse lookup so the full 875 MB matrix is never
        built.
        """
        if len(met_idx) == 0 or len(rxn_idx) == 0:
            return torch.zeros(len(met_idx), len(rxn_idx), dtype=self.dtype,
                               device=self.device)
        return self.lookup(met_idx[:, None], rxn_idx[None, :])

    def bounds(self, rxn_idx: Tensor) -> Tuple[Tensor, Tensor]:
        return self.lb.index_select(0, rxn_idx), self.ub.index_select(0, rxn_idx)

    # ── elementwise lookup, for the guidance path ──────────────────────────

    def _build_lookup(self):
        """One sorted key per nonzero, so S[m, r] is a searchsorted away.

        The dense matrix is 8,461 x 12,931 — 875 MB at float64 — and the
        guidance path only ever needs the ~4,096 (metabolite, reaction) pairs
        a 64-slot graph can express. Same trick as CoexprLookup in
        graphpath/layers/pairwise_conditioning.py.
        """
        idx = self.S.indices()
        key = idx[0] * self.n_rxns + idx[1]
        order = torch.argsort(key)
        self._key = key[order].contiguous()
        self._val = self.S.values()[order].contiguous()

    def lookup(self, met_idx: Tensor, rxn_idx: Tensor) -> Tensor:
        """
        S[met_idx, rxn_idx] elementwise, broadcasting; 0 where absent.
        Shapes of the two must broadcast against each other.
        """
        if not hasattr(self, "_key"):
            self._build_lookup()
        met_idx, rxn_idx = torch.broadcast_tensors(met_idx, rxn_idx)
        q = (met_idx * self.n_rxns + rxn_idx).reshape(-1)
        pos = torch.searchsorted(self._key, q).clamp(max=self._key.numel() - 1)
        hit = self._key[pos] == q
        return torch.where(hit, self._val[pos], torch.zeros((), dtype=self.dtype,
                                                            device=self.device)
                           ).reshape(met_idx.shape)

    def token_tables(self, id_to_node: dict) -> dict:
        """
        Precompute token -> (kind, GEM index) so the guidance path is pure
        indexing rather than a per-graph Python loop over names.

        kind: 0 nothing, 1 reaction, 2 metabolite, 3 gene.
        """
        V = max(id_to_node) + 1
        kind = torch.zeros(V, dtype=torch.long, device=self.device)
        gem = torch.full((V,), -1, dtype=torch.long, device=self.device)
        for tok, nm in id_to_node.items():
            s = self.strip(nm)
            if s in self.rxn_of:
                kind[tok], gem[tok] = 1, self.rxn_of[s]
            elif s in self.met_of:
                kind[tok], gem[tok] = 2, self.met_of[s]
            elif s in self.gene_of:
                kind[tok], gem[tok] = 3, self.gene_of[s]
        return {"kind": kind, "gem": gem}


# ── exact: the dimension of the steady-state flux space ─────────────────────

def null_dim(S: Tensor, tol: float = 1e-9) -> Tensor:
    """
    n_reactions - rank(S), batched over a leading dimension.

    S v = 0 has a nonzero solution iff this is > 0, so 0 means the subnetwork
    is completely blocked: no assignment of fluxes, however large, holds every
    internal metabolite at steady state. Exact (batched SVD), no iteration.

    It IGNORES irreversibility, so it is a necessary condition for a usable
    flux distribution, not a sufficient one. Cheap, and it never lies in the
    direction that matters: null_dim == 0 is a hard fail.
    """
    if S.ndim == 2:
        S = S[None]
    B, m, n = S.shape
    if n == 0:
        return torch.zeros(B, dtype=torch.long, device=S.device)
    s = torch.linalg.svdvals(S)                                  # (B, min(m,n))
    cutoff = s[:, :1] * max(m, n) * tol
    rank = (s > cutoff.clamp(min=torch.finfo(S.dtype).tiny)).sum(-1)
    return n - rank


# ── differentiable: how close to a steady state can it get ──────────────────

def flux_residual(S: Tensor, reversible: Optional[Tensor] = None,
                  iters: int = 200, lr: float = 1.0, tol: float = 1e-12,
                  col_mask: Optional[Tensor] = None,
                  return_flux: bool = False):
    """
    min over v of  ||S v||^2 / ||v||^2,  subject to v_j >= 0 where the reaction
    is irreversible. Batched over a leading dimension, differentiable in S.

    Why the Rayleigh quotient rather than a plain least-squares. ||S v||^2 is
    minimised at v = 0, which is a steady state and tells you nothing. Dividing
    by ||v||^2 makes the objective scale-free, so the trivial solution is
    excluded without an extra constraint, and the flux MAGNITUDE bounds drop
    out entirely — only the SIGN constraints survive, which is exactly the part
    of lb/ub that carries biology.

    0 means an exact steady state exists in the allowed orthant. Larger means
    mass cannot balance no matter how the fluxes are set. Read it against real
    pathway chunks, not against an absolute number: a chunk cut out of a bigger
    network has severed boundary metabolites and scores badly too.

    Initialised from the smallest right singular vector, which is the exact
    unconstrained optimum, so with no irreversible reactions it converges
    immediately and the iteration only does work when signs actually bind.

    `col_mask` (B, n) marks the columns that are really reactions. It matters
    whenever the batch is padded to a common width: an all-zero column is a
    free variable that satisfies S v = 0 at no cost, so the minimiser parks
    all of v there and every padded graph scores a perfect 0. Masked columns
    are held at v_j = 0 and dropped from the normalisation.

    `lr` is a MULTIPLE of the stable step, not a raw step. The objective's
    curvature is 2*sigma_max(S)^2, which varies by orders of magnitude between
    subnetworks, so a fixed step that is stable for one is divergent for the
    next. Anything below 2.0 is stable; 1.0 is the usual choice.
    """
    squeeze = S.ndim == 2
    if squeeze:
        S = S[None]
    B, m, n = S.shape
    if n == 0:
        z = torch.zeros(B, dtype=S.dtype, device=S.device)
        return (z, torch.zeros(B, 0, dtype=S.dtype, device=S.device)) if return_flux else z

    if col_mask is not None:
        cm0 = col_mask.to(S.device).bool()
        if cm0.ndim == 1:
            cm0 = cm0[None].expand(B, n)
        S = S * cm0[:, None, :].to(S.dtype)

    # The flux vector is found WITHOUT gradient and then detached. Two
    # reasons, and the second is the load-bearing one:
    #   * torch.linalg.svd's backward divides by (sigma_i^2 - sigma_j^2), which
    #     is NaN whenever two singular values coincide — and a stoichiometric
    #     matrix has a whole block of zeros, so that is the normal case, not an
    #     edge case. Differentiating through the search produces NaN gradients.
    #   * it is not needed. v* minimises R, so dR/dv = 0 there and the total
    #     derivative of R with respect to S equals the partial one. Detaching v
    #     gives the exact gradient at the optimum, not an approximation.
    with torch.no_grad():
        v = _solve(S, reversible, iters, lr, tol, col_mask)
    res = ((S @ v[:, :, None]).squeeze(-1) ** 2).sum(-1)
    if squeeze:
        res, v = res[0], v[0]
    return (res, v) if return_flux else res


def _solve(S: Tensor, reversible, iters: int, lr: float, tol: float,
           col_mask=None) -> Tensor:
    """
    Projected-gradient search for the flux direction minimising ||S v|| / ||v||.
    Runs under no_grad; the caller re-evaluates the residual with S attached.
    """
    B, m, n = S.shape
    # full_matrices=True is required, not a preference: with fewer metabolites
    # than reactions — the normal case for a subnetwork — full_matrices=False
    # returns only min(m, n) right singular vectors, so Vh[-1] would be the
    # LARGEST direction rather than the null one and the search would start at
    # the worst possible point.
    sv = torch.linalg.svd(S, full_matrices=True)
    v = sv.Vh[:, -1, :].clone()
    # curvature of ||Sv||^2 on the unit sphere is 2*sigma_max^2; stepping at
    # lr/sigma_max^2 keeps every batch element inside its own stability limit
    step = (lr / (sv.S[:, 0] ** 2).clamp(min=torch.finfo(S.dtype).tiny))[:, None]

    cm = None
    if col_mask is not None:
        cm = col_mask.to(S.device).bool()
        if cm.ndim == 1:
            cm = cm[None].expand(B, n)

    rev = None
    if reversible is not None:
        rev = reversible.to(S.device).bool()
        if rev.ndim == 1:
            rev = rev[None].expand(B, n)
        # the sign of a null direction is arbitrary; keep whichever orientation
        # puts more of it in the allowed orthant before projecting
        pos = torch.where(rev, v, v.clamp(min=0)).norm(dim=-1)
        neg = torch.where(rev, -v, (-v).clamp(min=0)).norm(dim=-1)
        v = torch.where((neg > pos)[:, None], -v, v)
        v = torch.where(rev, v, v.clamp(min=0))

    if cm is not None:
        v = v * cm.to(v.dtype)
        # the null direction can live entirely on the masked-out columns,
        # leaving nothing behind; fall back to a flat vector on the live ones
        dead = v.norm(dim=-1) < 1e-12
        if bool(dead.any()):
            v = torch.where(dead[:, None], cm.to(v.dtype), v)
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-30)

    if rev is None:
        return v                          # unconstrained: the SVD is exact

    StS = S.transpose(1, 2) @ S                                   # (B, n, n)
    for _ in range(iters):
        num = (v[:, None, :] @ StS @ v[:, :, None]).squeeze(-1)    # ||Sv||^2
        grad = 2 * ((StS @ v[:, :, None]).squeeze(-1) - num * v)   # dR/dv, ||v||=1
        # stop before rounding noise starts being amplified
        if float(grad.abs().amax()) < tol:
            break
        v = v - step * grad
        v = torch.where(rev, v, v.clamp(min=0))
        if cm is not None:
            v = v * cm.to(v.dtype)
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    return v


# ── exact, no LP: structural checks ─────────────────────────────────────────

def dead_end_metabolites(S: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
    """
    (B, m) bool — metabolites that can only be produced or only consumed, so
    they can never sit at steady state inside this subnetwork.

    Accounts for reversibility: a coefficient's achievable sign depends on
    whether its reaction may run backwards.
    """
    if S.ndim == 2:
        S = S[None]
    fwd = (ub > 0).to(S.dtype)
    bwd = (lb < 0).to(S.dtype)
    while fwd.ndim < S.ndim - 1:
        fwd, bwd = fwd[None], bwd[None]
    prod = ((S > 0) * fwd.unsqueeze(-2) + (S < 0) * bwd.unsqueeze(-2)).sum(-1) > 0
    cons = ((S < 0) * fwd.unsqueeze(-2) + (S > 0) * bwd.unsqueeze(-2)).sum(-1) > 0
    present = (S != 0).any(-1)
    return present & ~(prod & cons)


def mass_imbalance(E: Tensor, S: Tensor) -> Tensor:
    """
    (..., n_rxns) largest net atom count over all elements, per reaction.

    E @ S is atoms created or destroyed. Zero is balanced. Independent of any
    flux, so it is the cheapest real check there is and the one that catches a
    model inventing chemistry outright.
    """
    return (E @ S).abs().amax(-2)
