"""Uniform transition-matrix diffusion schedule."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphpath.config import Config
from graphpath.processes.alphas import _cosine_alphas_cumprod, _linear_alphas_cumprod


class UniformTransitionSchedule(nn.Module):
    """
    Uniform transition matrix diffusion for a K-class discrete variable.

    Forward marginal (closed form):
        q(x_t | x_0) = Cat( ᾱ_t · e_{x_0}  +  (1-ᾱ_t)/K · 1_K )

    i.e. the noisy token equals x_0 with prob ᾱ_t, else drawn uniformly.

    Reverse posterior (closed form, given predicted x_0):
        q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) · q(x_{t-1} | x_0)

    All quantities are O(K) — no matrix operations needed.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        fn = _cosine_alphas_cumprod if cfg.schedule == "cosine" \
             else _linear_alphas_cumprod
        ac = fn(cfg.T)                                     # (T,)
        self.register_buffer("ac", ac)                     # ᾱ_t
        # Precompute one-step α_t = ᾱ_t / ᾱ_{t-1}  (for reverse posterior)
        ac_prev = torch.cat([torch.ones(1), ac[:-1]])      # ᾱ_{t-1}
        self.register_buffer("ac_prev", ac_prev)
        alpha_t = ac / ac_prev.clamp(min=1e-8)             # α_t  (T,)
        self.register_buffer("alpha_t", alpha_t)

    # ── forward diffusion ─────────────────────────────────────────────────────

    def q_sample(self, x0: Tensor, t: Tensor, K: int,
                 marginal: Optional[Tensor] = None) -> Tensor:
        """
        Sample x_t ~ q(x_t | x_0) for a K-class variable.

        x0 : (B, ...)  long  — clean tokens in {0, ..., K-1}
        t  : (B,)      long  — timestep indices
        K  : int             — vocabulary size for this stream
        marginal : (K,) or None — the distribution a corrupted token is drawn
            from. None means uniform, i.e. the plain D3PM uniform kernel:
                Q_t = (1-b_t) I + b_t (1/K) 11^T
            Passing the DATA marginal m instead gives
                Q_t = (1-b_t) I + b_t 1 m^T
            which is the other kernel D3PM defines, and it matters whenever
            the classes are imbalanced. A binary adjacency at 5.5% density has
            a uniform stationary distribution of 50% density, so the reverse
            chain would have to transport an order of magnitude; with the data
            marginal it starts where the data already is.

        With probability a_bar_t keep x0, else draw from `marginal`.
        """
        ac = self.ac[t]
        for _ in range(x0.dim() - 1):
            ac = ac.unsqueeze(-1)                          # (B, 1, ..., 1)
        ac = ac.expand_as(x0.float())

        keep = torch.bernoulli(ac).bool()
        corrupted = self.prior_sample(x0.shape, K, marginal, x0.device)
        return torch.where(keep, x0, corrupted)

    @staticmethod
    def prior_sample(shape, K: int, marginal: Optional[Tensor] = None,
                     device=None) -> Tensor:
        """
        Draw from the stationary distribution — what x_T looks like, and what
        `sample()` must start from for the reverse chain to be consistent
        with the forward one.
        """
        if marginal is None:
            return torch.randint(0, K, shape, device=device)
        m = marginal.to(device).clamp(min=0)
        idx = torch.multinomial(m / m.sum(), int(torch.tensor(shape).prod()),
                                replacement=True)
        return idx.view(shape)

    # ── posterior for reverse step ────────────────────────────────────────────

    def posterior_logits(
        self,
        x0_logits: Tensor,    # (*, K) — model's x_0 prediction
        x_t:       Tensor,    # (*,)   long — current noisy token
        t_idx:     int,       # scalar timestep
        K:         int,
        marginal:  Optional[Tensor] = None,   # (K,) — None means uniform
    ) -> Tensor:              # (*, K) — log probs of x_{t-1}
        """
        Compute  log q(x_{t-1} = k | x_t, x_0_pred)  for all k.

        q(x_{t-1}=k | x_t=j, x_0=c) ∝
            q(x_t=j | x_{t-1}=k) · q(x_{t-1}=k | x_0=c)

        For the uniform kernel:
            q(x_t=j | x_{t-1}=k) = α_t·1[j=k] + (1-α_t)/K
            q(x_{t-1}=k | x_0=c) = ᾱ_{t-1}·1[k=c] + (1-ᾱ_{t-1})/K

        We marginalise over c using the predicted p(x_0=c | x_t):
            q(x_{t-1}=k | x_t) = Σ_c p_θ(x_0=c) · q(x_{t-1}=k | x_t, x_0=c)

        All computed in log-space for stability.
        """
        device = x0_logits.device
        at     = self.alpha_t[t_idx].to(device)           # scalar α_t
        ac_t1  = self.ac_prev[t_idx].to(device)           # scalar ᾱ_{t-1}

        if marginal is not None:
            # Same derivation with 1/K replaced by m:
            #   q(x_t=j | x_{t-1}=k) = a_t 1[j=k] + (1-a_t) m_j
            #   q(x_{t-1}=k | x_0=c) = ab_{t-1} 1[k=c] + (1-ab_{t-1}) m_k
            # The first factor is evaluated at the OBSERVED j, so its
            # off-diagonal value now depends on x_t rather than being the
            # constant (1-a_t)/K.
            m = (marginal.to(device) / marginal.sum()).clamp(min=1e-30)
            m_j = m[x_t].unsqueeze(-1)                     # (*,1)
            log_qt = torch.log((1.0 - at) * m_j + 1e-30).expand_as(x0_logits).clone()
            on = torch.log(at + (1.0 - at) * m_j + 1e-30)
            log_qt.scatter_(-1, x_t.unsqueeze(-1), on)
            p0 = F.log_softmax(x0_logits, dim=-1).exp()
            log_qtm1 = torch.log(ac_t1 * p0 + (1.0 - ac_t1) * m + 1e-30)
            joint = log_qt + log_qtm1
            return joint - torch.logsumexp(joint, dim=-1, keepdim=True)

        # log q(x_t=j | x_{t-1}=k):  shape (*,K) — index over k
        # = log( α_t·1[j=k] + (1-α_t)/K )
        # For k==j (on-diagonal): α_t + (1-α_t)/K
        # For k!=j (off-diagonal): (1-α_t)/K
        off_diag = math.log((1.0 - at.item()) / K + 1e-30)
        on_diag  = math.log(at.item() + (1.0 - at.item()) / K + 1e-30)

        # Build log q(x_t | x_{t-1}=k) for each k, evaluated at observed x_t
        # Shape: (*,K)
        log_qt_given_xtm1 = torch.full(
            x0_logits.shape, off_diag, device=device
        )
        # x_t is the observed value; positions where k == x_t get on_diag
        xt_idx = x_t.unsqueeze(-1)                        # (*,1)
        log_qt_given_xtm1.scatter_(-1, xt_idx, on_diag)

        # log q(x_{t-1}=k | x_0=c):  for each (k, c) pair
        # = ᾱ_{t-1}·1[k=c] + (1-ᾱ_{t-1})/K
        # We need Σ_c p_θ(c) · q(x_{t-1}=k | x_0=c)
        # = ᾱ_{t-1} · p_θ(x_0=k) + (1-ᾱ_{t-1})/K · Σ_c p_θ(c)
        # = ᾱ_{t-1} · p_θ(x_0=k) + (1-ᾱ_{t-1})/K          [probs sum to 1]
        log_p0     = F.log_softmax(x0_logits, dim=-1)     # (*,K)
        p0         = log_p0.exp()
        # Σ_c p_θ(c) · q(x_{t-1}=k | x_0=c)  in log space
        log_qtm1   = torch.log(
            ac_t1 * p0 + (1.0 - ac_t1) / K + 1e-30
        )                                                  # (*,K)

        # Joint in log-space, then normalise
        log_joint  = log_qt_given_xtm1 + log_qtm1         # (*,K)
        return log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True)
