"""
GraphDiffBERT v8.1 — Marginal Transition Matrix Diffusion (Nodes + Edges)
=========================================================================

Theory
──────
Replaces the uniform transition matrix of v7 with a MARGINAL transition
matrix (Vignac et al., 2023 "DiGress"; the general D3PM family of
Austin et al., 2021).

Forward process  q(x_t | x_{t-1})
  At each step a token stays the same with probability α_t and jumps to
  a token drawn from the DATA MARGINAL m with probability (1-α_t).

  In matrix form:  Q_t = α_t I  +  (1-α_t) 1 mᵀ
  where m ∈ Δ^{K-1} is the empirical class distribution of the dataset.
  (v7 was the special case m = 1/K · 1.)

  Because mᵀ1 = 1, the rank-one term is idempotent under composition and
  Q̄_t = Π Q_s collapses in exactly the same way as the uniform kernel:

    q(x_t = k | x_0 = j) = ᾱ_t · 1[k=j] + (1 - ᾱ_t) · m_k

  So at t→T every token approaches m — not the uniform distribution.

Why this matters
  For a sparse graph the edge marginal is roughly m = [0.99, 0.01].
  Under uniform noise q(x_T) puts ~50% of mass on "edge", so the prior
  sample is a near-complete graph and every intermediate state x_t is
  orders of magnitude denser than anything in the data. The denoiser
  then spends its capacity learning to delete edges rather than learning
  structure, and it is evaluated far off the data manifold.
  Under marginal noise every x_t has approximately the right density,
  and the reverse trajectory stays where the data lives.

  This is the categorical analogue of standardising your data before
  Gaussian diffusion: make the terminal distribution match the data's
  aggregate statistics instead of an arbitrary one.

Reverse process  p_θ(x_{t-1} | x_t)
  The model predicts x_0 logits from (x_t, t); the posterior is then
  computed in closed form. Writing j = x_t (observed) and noting that
  q(x_t=j | x_{t-1}=k) = α_t·1[k=j] + (1-α_t)·m_j depends on k only
  through the indicator:

    A_k   = (1-α_t)·m_j + α_t·1[k=j]                  (likelihood)
    C_kc  = ᾱ_{t-1}·1[k=c] + (1-ᾱ_{t-1})·m_k          (prior from x_0=c)
    Z_c   = Σ_k A_k C_kc = ᾱ_{t-1}·A_c + (1-ᾱ_{t-1})·m_j

  and the EXACT marginalisation over the predicted x_0 is

    p(x_{t-1}=k | x_t) = Σ_c p_θ(x_0=c) · A_k·C_kc / Z_c
                       = A_k · [ ᾱ_{t-1}·p_θ(k)/Z_k
                                 + (1-ᾱ_{t-1})·m_k·Σ_c p_θ(c)/Z_c ]

  — still O(K), no matrix inverse, and it sums to exactly 1.

  NOTE: v7 used the cheaper approximation A_k·(ᾱ_{t-1}p_θ(k) + (1-ᾱ_{t-1})/K),
  which drops the per-c normaliser Z_c. Measured against brute force, that
  approximation sits at mean total-variation ≈ 0.010 from the exact
  posterior with occasional excursions to TV ≈ 0.85 (worst case) — and,
  contrary to what you might expect, this is roughly skew-INDEPENDENT: it
  is about as wrong for m = [0.5, 0.5] as for m = [0.998, 0.002]. The exact
  form costs two extra elementwise ops, so it is the default here;
  `exact_posterior=False` restores v7's behaviour for ablation.

Verified
────────
  The Q̄ composition and the O(K) posterior in this file are checked against
  an explicit brute-force construction of the K×K transition matrices in
  tests/test_marginal_process.py (K ∈ {2,5,9,10}, uniform and skewed
  marginals, all t): max abs error < 1e-10, posterior sums to 1 within 1e-9.
"""

from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphpath.config import MarginalConfig
from graphpath.processes.alphas import _cosine_alphas_cumprod, _linear_alphas_cumprod


def _smooth(m: Tensor, floor: float) -> Tensor:
    """
    Floor every class at `floor` and renormalise.

    This is not optional. A class with m_k = 0 is unreachable by the noise
    process, and the posterior's Z_c denominator becomes 0 for that class,
    producing NaNs on the first reverse step.
    """
    m = m.double().clamp(min=floor)
    return (m / m.sum()).float()


@torch.no_grad()
def estimate_marginals(
    loader:       Iterable,
    node_vocab:   int,
    edge_vocab:   int,
    floor:        float = 1e-4,
    max_batches:  Optional[int] = None,
    unpack=None,
) -> Tuple[Tensor, Tensor]:
    """
    Empirical class marginals over the dataset.

    loader   : yields batches; by default unpacked as
               (adjcs, node_feats, mask, *rest) to match the v7 loader.
    unpack   : optional fn(batch) -> (adjcs, node_feats, pad_mask)
               for a differently-shaped loader.

    Counts respect padding, and edges are counted on valid OFF-DIAGONAL
    pairs only — counting the diagonal would bias m_edge toward class 0
    by exactly N/(N²) and counting padding would bias it much harder.

    Returns (m_node, m_edge), both smoothed and normalised.
    """
    node_counts = torch.zeros(node_vocab, dtype=torch.double)
    edge_counts = torch.zeros(edge_vocab, dtype=torch.double)

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if unpack is not None:
            adjcs, node_feats, pad_mask = unpack(batch)
        else:
            adjcs, node_feats, pad_mask = batch[0], batch[1], batch[2]

        adjcs      = adjcs.long().cpu()
        node_feats = node_feats.long().cpu()
        pad_mask   = pad_mask.bool().cpu()
        N          = node_feats.shape[1]

        node_counts += torch.bincount(
            node_feats[pad_mask].reshape(-1), minlength=node_vocab
        ).double()

        valid = pad_mask[:, :, None] & pad_mask[:, None, :]
        valid = valid & ~torch.eye(N, dtype=torch.bool)[None]
        edge_counts += torch.bincount(
            adjcs[valid].reshape(-1), minlength=edge_vocab
        ).double()

    m_node = _smooth(node_counts / node_counts.sum().clamp(min=1), floor)
    m_edge = _smooth(edge_counts / edge_counts.sum().clamp(min=1), floor)
    return m_node, m_edge


class MarginalTransitionSchedule(nn.Module):
    """
    Marginal transition matrix diffusion for a K-class discrete variable.

    Forward marginal (closed form):
        q(x_t | x_0) = Cat( ᾱ_t · e_{x_0}  +  (1-ᾱ_t) · m )

    Reverse posterior (closed form, exact, given predicted x_0):
        see module docstring — O(K), no matrix ops.

    The schedule itself is marginal-agnostic: the marginal vector `m` is
    passed per call, so one schedule serves both the node and edge streams.
    """

    def __init__(self, cfg: MarginalConfig):
        super().__init__()
        self.cfg = cfg
        fn = _cosine_alphas_cumprod if cfg.schedule == "cosine" \
             else _linear_alphas_cumprod
        ac = fn(cfg.T)                                     # (T,)
        self.register_buffer("ac", ac)                     # ᾱ_t
        ac_prev = torch.cat([torch.ones(1), ac[:-1]])      # ᾱ_{t-1}
        self.register_buffer("ac_prev", ac_prev)
        alpha_t = ac / ac_prev.clamp(min=1e-8)             # α_t  (T,)
        self.register_buffer("alpha_t", alpha_t)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def sample_prior(m: Tensor, shape, device) -> Tensor:
        """Draw i.i.d. samples from Cat(m) — the t=T prior, and the noise
        source for the forward process."""
        cdf = m.double().cumsum(0).contiguous()
        u   = torch.rand(int(np.prod(shape)), device=device, dtype=torch.double)
        idx = torch.searchsorted(cdf.to(device), u)
        return idx.clamp_(max=m.numel() - 1).reshape(shape).long()

    # ── forward diffusion ─────────────────────────────────────────────────────

    def q_sample(self, x0: Tensor, t: Tensor, m: Tensor) -> Tensor:
        """
        Sample x_t ~ q(x_t | x_0).

        x0 : (B, ...)  long  — clean tokens in {0, ..., K-1}
        t  : (B,)      long  — timestep indices
        m  : (K,)      float — class marginal for this stream

        With probability ᾱ_t keep x0, else draw from Cat(m).
        """
        ac = self.ac[t]
        for _ in range(x0.dim() - 1):
            ac = ac.unsqueeze(-1)                          # (B, 1, ..., 1)
        ac = ac.expand_as(x0.float())

        keep  = torch.bernoulli(ac).bool()
        noise = self.sample_prior(m, tuple(x0.shape), x0.device)
        return torch.where(keep, x0, noise)

    # ── posterior for reverse step ────────────────────────────────────────────

    def posterior_logits(
        self,
        x0_logits: Tensor,    # (*, K) — model's x_0 prediction
        x_t:       Tensor,    # (*,)   long — current noisy token
        t_idx:     int,       # scalar timestep
        m:         Tensor,    # (K,)   class marginal
        exact:     bool = True,
    ) -> Tensor:              # (*, K) — log probs of x_{t-1}
        """
        log q(x_{t-1} = k | x_t, x_0_pred) for all k.

        Exact form (default):
            A_k  = (1-α_t)·m_j + α_t·1[k=j]                 j = x_t
            Z_c  = ᾱ_{t-1}·A_c + (1-ᾱ_{t-1})·m_j
            post = A ⊙ [ ᾱ_{t-1}·p₀/Z + (1-ᾱ_{t-1})·m·Σ_c p₀_c/Z_c ]

        Approximate form (exact=False) reproduces v7:
            post = A ⊙ [ ᾱ_{t-1}·p₀ + (1-ᾱ_{t-1})·m ]

        Computed in float64 — Z_c can be ~1e-7 when m_j is small, and the
        division is the one genuinely delicate step in the reverse process.
        """
        device = x0_logits.device
        m      = m.to(device).double()
        at     = self.alpha_t[t_idx].to(device).double()
        acp    = self.ac_prev[t_idx].to(device).double()

        p0   = F.softmax(x0_logits.double(), dim=-1)          # (*,K)
        m_xt = m[x_t].unsqueeze(-1)                           # (*,1)  = m_j

        # A_k = (1-α_t)·m_j + α_t·1[k=j]
        K = m.numel()
        A = ((1.0 - at) * m_xt).expand_as(p0) \
            + at * F.one_hot(x_t, K).to(p0.dtype)             # (*,K)

        if exact:
            Z    = acp * A + (1.0 - acp) * m_xt               # (*,K), index c
            S    = (p0 / Z.clamp(min=1e-300)).sum(-1, keepdim=True)
            post = A * (acp * p0 / Z.clamp(min=1e-300) + (1.0 - acp) * m * S)
        else:
            post = A * (acp * p0 + (1.0 - acp) * m)           # v7 behaviour

        log_post = torch.log(post.clamp(min=1e-300))
        log_post = log_post - torch.logsumexp(log_post, dim=-1, keepdim=True)
        return log_post.to(x0_logits.dtype)



def _mirror_upper(x: Tensor) -> Tensor:
    """
    Take the strict upper triangle of x and mirror it into the lower one.

    This is the unbiased way to make a sampled adjacency symmetric. v7 used
    `OR` over both triangles, which maps an edge probability p to 2p - p²
    — nearly doubling the density for the small p that a marginal-noise
    model is specifically trying to preserve.
    """
    N = x.shape[-1]
    upper = torch.triu(
        torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
    )
    return torch.where(upper[None], x, x.transpose(-1, -2))

