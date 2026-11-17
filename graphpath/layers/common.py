"""Shared building blocks: time embedding, FFN, and a reusable MHA block."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d

    def forward(self, t: Tensor) -> Tensor:
        half  = self.d // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


def _make_ff(d: int, d_ff: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(d_ff, d), nn.Dropout(dropout),
    )


def _mha_block(
    h:        Tensor,           # (*, S, d)
    layer:    nn.ModuleDict,
    H:        int,
    d_k:      int,
    mask:     Optional[Tensor], # (S, S) bool — positions to suppress
) -> Tensor:                    # (*, S, d)
    orig_shape = h.shape
    S, d = orig_shape[-2], orig_shape[-1]
    Beff = h.numel() // (S * d)
    h_flat = h.reshape(Beff, S, d)

    Q = layer["W_q"](h_flat).view(Beff, S, H, d_k).transpose(1, 2)
    K = layer["W_k"](h_flat).view(Beff, S, H, d_k).transpose(1, 2)
    V = layer["W_v"](h_flat).view(Beff, S, H, d_k).transpose(1, 2)

    scores = (Q @ K.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask[None, None], -1e4)

    out = (F.softmax(scores, dim=-1) @ V).transpose(1, 2).reshape(Beff, S, d)
    out = layer["out"](out)
    return out.reshape(orig_shape)


class FiLMCondition(nn.Module):
    """
    Feature-wise linear modulation: h <- h * (1 + scale) + shift, with
    (scale, shift) predicted from a conditioning vector.

    `normalize=True` LayerNorms h first (the usual formulation, used for
    time conditioning inside the continuous denoiser). `normalize=False`
    modulates h in place, which is what the co-expression front-end wants:
    there h is a sum of token / time / type / identity embeddings whose
    relative scales carry information worth preserving.

    Zero-initialised, so it starts as the identity (or a plain LayerNorm)
    and the model learns how much modulation to apply — the same rationale
    as the zero-inits used for the per-layer edge attention bias.

    NOTE: `Modules.Cont_Helpers.FiLMCondition` was not supplied with the
    continuous-diffusion code, so this is a reimplementation.
    """

    def __init__(self, d_cond: int, d_model: Optional[int] = None,
                 normalize: bool = True):
        super().__init__()
        d_model = d_cond if d_model is None else d_model
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False) if normalize else None
        self.to_scale_shift = nn.Linear(d_cond, 2 * d_model)
        self.zero_init()

    def zero_init(self):
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        """h: (B, S, d_model); cond: (B, d_cond) or (B, 1, d_cond)."""
        if cond.dim() == h.dim() - 1:
            cond = cond.unsqueeze(-2)
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        if self.norm is not None:
            h = self.norm(h)
        return h * (1 + scale) + shift
