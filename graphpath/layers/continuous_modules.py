"""The "separate towers" backend for continuous diffusion.

`ContinuousGraphDiffusion` defaults to the interleaved `ContinuousGraphTower`
(node and edge state exchange information at every layer). These two modules
are the original two-tower alternative: a node denoiser that never sees the
edge state, followed by an adjacency head that reads only its output. Kept
because they are cheaper (no O(N²·d) edge state) and directly comparable.

Select them with `ContinuousGraphDiffusion(..., backend="separate")`.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphpath.config import Config
from graphpath.layers.common import FiLMCondition


class NodeDenoiser(nn.Module):
    """
    Predicts x̂_0 (clean embedding) from noisy_embeds at time t.
    Input: [noisy_emb ‖ prev_emb ‖ cond_emb]  →  project  →  FiLM(t)
           →  n_layers of TransformerEncoderLayer  →  Linear → bottleneck_dim.
    """

    def __init__(self, time_emb, cfg: Config):
        super().__init__()
        d   = cfg.d_model
        bn  = cfg.bottleneck_dim

        # 3× because we concat [noisy, prev, cond]
        self.input_proj = nn.Linear(3 * bn, d)
        self.input_norm = nn.LayerNorm(d)

        self.time_emb   = time_emb
        self.time_proj  = nn.Linear(d, d)
        self.film       = FiLMCondition(d)

        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model     = d,
                nhead       = cfg.n_heads,
                dim_feedforward = cfg.d_ff,
                dropout     = cfg.dropout,
                batch_first = True,
                norm_first  = True,
            )
            for _ in range(cfg.n_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, bn),
            nn.LayerNorm(bn),
        )

    def forward(
        self,
        noisy_embeds: Tensor,     # (B, N, bottleneck_dim)
        adj_t:        Tensor,     # (B, N, N)  noisy adjacency (unused here)
        t:            Tensor,     # (B,) int
        prev_embeds:  Tensor,     # (B, N, bottleneck_dim)  self-conditioning
        cond_embeds:  Tensor,     # (B, N, bottleneck_dim)  external cond (zeros ok)
    ) -> Tensor:                  # (B, N, bottleneck_dim)  x̂_0 prediction
        h = noisy_embeds

        # ── concat self-conditioning + external conditioning ──────────────
        h = torch.cat([h, prev_embeds, cond_embeds], dim=-1)    # (B,N,3·bn)
        h = self.input_norm(self.input_proj(h))                  # (B,N,d)

        # ── FiLM time conditioning ────────────────────────────────────────
        t_emb = self.time_proj(self.time_emb(t))                 # (B,d)
        # broadcast over sequence: (B,1,d)
        h = self.film(h, t_emb[:, None, :])

        # ── transformer blocks ────────────────────────────────────────────
        for block in self.transformer_blocks:
            h = block(h)

        return self.output_proj(h)                               # (B,N,bn)


class AdjacencyHead(nn.Module):
    """
    Takes node hidden states h (B,N,in_dim) + noisy adj prev_adj (B,N,N)
    and predicts clean edge logits (B,N,N).

    Pipeline:
      1. Project h to hidden_dim, add time embedding.
      2. n_layers of multi-head self-attention (node updater).
      3. Pair features [h_i ‖ h_j ‖ prev_adj[i,j]] → MLP → scalar logit.
      4. Symmetrise, suppress diagonal.
    """

    def __init__(self, time_emb, cfg: Config):
        super().__init__()
        d   = cfg.adj_hidden_dim
        H   = cfg.adj_n_heads
        d_k = d // H
        assert d % H == 0

        self.H, self.d_k = H, d_k

        in_dim = cfg.bottleneck_dim
        self.input_proj = (
            nn.Linear(in_dim, d) if in_dim != d else nn.Identity()
        )

        self.time_emb  = time_emb
        self.time_proj = nn.Linear(d, d)

        self.layers = nn.ModuleList()
        self.norms  = nn.ModuleList()
        for _ in range(cfg.adj_n_layers):
            self.layers.append(nn.ModuleDict({
                "W_q": nn.Linear(d, d, bias=False),
                "W_k": nn.Linear(d, d, bias=False),
                "W_v": nn.Linear(d, d, bias=False),
                "out": nn.Linear(d, d),
            }))
            self.norms.append(nn.LayerNorm(d))

        # Edge MLP: [h_i ‖ h_j ‖ scalar] → 1
        edge_in = 2 * d + (1 if cfg.use_prev_adj else 0)
        eh      = cfg.adj_edge_hidden
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, eh), nn.SiLU(),
            nn.Linear(eh, eh),      nn.SiLU(),
            nn.Linear(eh, 1),
        )
        self.use_prev_adj = cfg.use_prev_adj

    @staticmethod
    def _pair_features(h: Tensor) -> Tensor:
        """h (B,N,d) → (B,N,N,2d)"""
        hi = h.unsqueeze(2).expand(-1, -1, h.size(1), -1)
        hj = h.unsqueeze(1).expand(-1, h.size(1), -1, -1)
        return torch.cat([hi, hj], dim=-1)

    def forward(self, h: Tensor, t: Tensor, prev_adj: Optional[Tensor] = None) -> Tensor:
        """
        h        : (B, N, in_dim)
        t        : (B,) int timestep
        prev_adj : (B, N, N) float — noisy adjacency (self-conditioning)
        Returns  : adj_logits (B, N, N)
        """
        B, N, _ = h.shape
        H, d_k  = self.H, self.d_k

        h = self.input_proj(h)
        h = h + self.time_proj(self.time_emb(t))[:, None, :]

        diag = torch.eye(N, device=h.device, dtype=torch.bool)

        for layer, norm in zip(self.layers, self.norms):
            Q = layer["W_q"](h).view(B, N, H, d_k).transpose(1, 2)  # (B,H,N,dk)
            K = layer["W_k"](h).view(B, N, H, d_k).transpose(1, 2)
            V = layer["W_v"](h).view(B, N, H, d_k).transpose(1, 2)

            scores = (Q @ K.transpose(-1, -2)) / math.sqrt(d_k)     # (B,H,N,N)
            scores = scores.masked_fill(diag[None, None], -1e4)
            attn   = F.softmax(scores, dim=-1)

            out = (attn @ V).transpose(1, 2).reshape(B, N, H * d_k)
            h   = norm(h + layer["out"](out))

        # ── edge prediction ───────────────────────────────────────────────
        pair = self._pair_features(h)                          # (B,N,N,2d)
        if self.use_prev_adj and prev_adj is not None:
            pair = torch.cat([pair, prev_adj.unsqueeze(-1)], dim=-1)

        logits = self.edge_mlp(pair).squeeze(-1)               # (B,N,N)
        logits = 0.5 * (logits + logits.transpose(-1, -2))     # symmetrise
        logits = logits.masked_fill(diag[None], -1e4)          # no self-loops
        return logits
