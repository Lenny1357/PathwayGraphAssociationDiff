"""GraphTransformer block — Dwivedi & Bresson, "A Generalization of
Transformer Networks to Graphs" (https://arxiv.org/abs/2012.09699).

An alternative to `GraphBlock`, selected with `Config.backbone`. Edge
features gate attention, and the pre-softmax scores — which already fuse the
node pair with the edge — are recycled as the new edge representation:

    what_ij^k = ( (q_i^k . k_j^k) / sqrt(d_k) ) * eta_ij^k      per head k
    w_ij^k    = softmax_j( what_ij^k )
    h_i'      = O_h( ||_k  sum_j w_ij^k V^k h_j )
    e_ij'     = O_e( ||_k  what_ij^k )

then residual + norm + FFN on both streams.

Adaptations, and why
--------------------
* The paper sparsifies attention to graph neighbours. Here the adjacency is
  the thing being generated, not a fixed input, so attention is dense over
  all N slots and gated by the (noisy) edge embedding — the model has to
  learn to discount edges it believes absent.
* LayerNorm rather than BatchNorm: batch statistics are unreliable with
  padded graphs of varying size.
* No Laplacian positional encoding — it presumes a fixed known graph. Node
  identity and the shared front-end's time embedding take its place.
* This block is a drop-in for `GraphBlock`, so it inherits the tower's
  embeddings, time conditioning, type embedding, identity features and
  co-expression conditioner unchanged. That is deliberate: swapping only the
  layer means a backbone comparison measures the attention mechanism rather
  than a different front-end.
* The paper's per-layer timestep FiLM is available via
  `Config.gt_time_film`, but defaults OFF — `GraphBlock` has no per-layer
  time conditioning, and giving it to one backbone only would confound the
  comparison.
* `edge_known` is ignored: the edge token embedding already distinguishes
  MASK from a real class in the discrete variants, and it is constant under
  continuous diffusion.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphpath.config import Config


class GraphTransformerBlock(nn.Module):
    """One Dwivedi-Bresson block, with `GraphBlock`'s interface."""

    def __init__(self, cfg: Config):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        assert d % H == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_k = d, H, d // H

        self.Q = nn.Linear(d, d)
        self.K = nn.Linear(d, d)
        self.V = nn.Linear(d, d)
        self.E = nn.Linear(d, d)          # edge -> per-head gate
        self.O_h = nn.Linear(d, d)
        self.O_e = nn.Linear(H, d)        # concat(heads) of raw scores -> edge repr

        self.norm1_h, self.norm1_e = nn.LayerNorm(d), nn.LayerNorm(d)
        self.norm2_h, self.norm2_e = nn.LayerNorm(d), nn.LayerNorm(d)

        m = cfg.gt_ffn_mult
        self.ffn_h = nn.Sequential(nn.Linear(d, m * d), nn.GELU(), nn.Linear(m * d, d))
        self.ffn_e = nn.Sequential(nn.Linear(d, m * d), nn.GELU(), nn.Linear(m * d, d))
        self.drop = nn.Dropout(cfg.dropout)

        # per-layer timestep FiLM — off by default, see the module docstring
        self.use_time_film = cfg.gt_time_film
        if self.use_time_film:
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
            nn.init.zeros_(self.time_mlp[-1].weight)
            nn.init.zeros_(self.time_mlp[-1].bias)

    def reset_zeros(self):
        """Called by the tower's post_init_zeros, after its generic sweep."""
        if self.use_time_film:
            nn.init.zeros_(self.time_mlp[-1].weight)
            nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(self, h: Tensor, e: Tensor, edge_known: Tensor,
                diag_mask: Tensor, t_emb: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        B, N, d = h.shape
        H, dk = self.n_heads, self.d_k

        if self.use_time_film and t_emb is not None:
            gamma, beta = self.time_mlp(t_emb).chunk(2, dim=-1)
            h = h * (1.0 + gamma).unsqueeze(1) + beta.unsqueeze(1)

        q = self.Q(h).view(B, N, H, dk)
        k = self.K(h).view(B, N, H, dk)
        v = self.V(h).view(B, N, H, dk)
        eta = self.E(e).view(B, N, N, H, dk)

        # (q_i . k_j / sqrt(dk)) * eta_ij, summed over the head dimension
        qk = q.unsqueeze(2) * k.unsqueeze(1) / math.sqrt(dk)      # (B,N,N,H,dk)
        raw = (qk * eta).sum(-1)                                  # (B,N,N,H)

        # the diagonal is suppressed for parity with GraphBlock
        scores = raw.masked_fill(diag_mask[None, :, :, None], -1e4)
        attn = F.softmax(scores, dim=2)
        if getattr(self, "store_attn", False):
            self.last_attn = attn.detach().permute(0, 3, 1, 2)   # (B,H,N,N)
            self.last_content = raw.detach().permute(0, 3, 1, 2)
            self.last_edge_bias = None      # edges gate the score, not added to it
        w = self.drop(attn)

        attn = torch.einsum("bijh,bjhd->bihd", w, v).reshape(B, N, d)
        h = self.norm1_h(h + self.drop(self.O_h(attn)))
        e = self.norm1_e(e + self.drop(self.O_e(raw)))   # scores recycled as edges

        h = self.norm2_h(h + self.drop(self.ffn_h(h)))
        e = self.norm2_e(e + self.drop(self.ffn_e(e)))
        return h, e


BLOCKS = {"tower": None, "graph_transformer": GraphTransformerBlock}


def make_block(cfg: Config) -> nn.Module:
    """Block factory for `_BaseGraphTower`, keyed by `Config.backbone`."""
    from graphpath.layers.graph_block import GraphBlock
    if cfg.backbone == "tower":
        return GraphBlock(cfg)
    if cfg.backbone == "graph_transformer":
        return GraphTransformerBlock(cfg)
    raise ValueError(f"unknown backbone {cfg.backbone!r}; "
                     f"expected one of {sorted(BLOCKS)}")
