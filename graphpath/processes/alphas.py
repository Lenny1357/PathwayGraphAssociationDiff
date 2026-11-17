"""ᾱ (alphas_cumprod) tables, shared by the two discrete schedules.

`AbsorbingMaskSchedule` and `UniformTransitionSchedule` read ᾱ_t as the
probability of keeping the clean token; they differ only in what happens
otherwise (absorb to MASK vs resample uniformly), so the table itself is
common to both. The Gaussian schedule builds its own — there ᾱ is a
variance-preserving mixing coefficient, not a probability.
"""

import math

import torch
from torch import Tensor


def _cosine_alphas_cumprod(T: int) -> Tensor:
    s = 0.008
    steps = torch.arange(T + 1, dtype=torch.float64)
    f  = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(0, 0.999)
    return torch.cumprod(1.0 - betas, 0).float()


def _linear_alphas_cumprod(T: int, beta_min=1e-4, beta_max=0.02) -> Tensor:
    betas = torch.linspace(beta_min, beta_max, T)
    return torch.cumprod(1.0 - betas, 0)
