"""Gaussian (continuous) diffusion schedule with a deterministic DDIM step."""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class ContinuousCosineSchedule(nn.Module):
    """
    Cosine ᾱ schedule for Gaussian diffusion:  x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε

    Distinct from `AbsorbingMaskSchedule` / `UniformTransitionSchedule`,
    which carry a *categorical* ᾱ (probability of keeping the clean token).
    Here ᾱ is the usual variance-preserving mixing coefficient.
    """

    def __init__(self, diffusion_steps: int, offset: float = 0.01):
        super().__init__()

        self.diffusion_steps = diffusion_steps

        steps = torch.linspace(0, 1, diffusion_steps + 1)
        alpha_bar = torch.cos(((steps + offset) / (1 + offset)) * torch.pi / 2) ** 2

        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1 - alpha_bar))

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(0, self.diffusion_steps, (batch_size,), device=device)

    def _extract(self, arr: Tensor, t: Tensor, x_shape) -> Tensor:
        out = arr[t]  # (B,)
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    def q_sample(self, x0: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = self._extract(self.sqrt_alpha_bar, t, x0.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alpha_bar, t, x0.shape)

        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt

    def get_coefficients(self, t: Tensor, x_shape):
        return (
            self._extract(self.sqrt_alpha_bar, t, x_shape),
            self._extract(self.sqrt_one_minus_alpha_bar, t, x_shape),
        )

    def ddim(self, x_t: Tensor, x0_hat: Tensor, t: Tensor, t_next: Tensor) -> Tensor:
        """
        One deterministic DDIM step from t → t_next.
        x_t, x0_hat : same shape; t, t_next : (B,) or scalar tensors.
        """
        extra = x_t.dim() - 1

        def _get(buf, idx):
            return buf[idx].view((-1,) + (1,) * extra)

        sa_t  = _get(self.sqrt_alpha_bar,           t)
        so_t  = _get(self.sqrt_one_minus_alpha_bar, t)
        sa_t1 = _get(self.sqrt_alpha_bar,           t_next)
        so_t1 = _get(self.sqrt_one_minus_alpha_bar, t_next)

        eps   = (x_t - sa_t * x0_hat) / so_t.clamp(min=1e-8)
        return sa_t1 * x0_hat + so_t1 * eps
