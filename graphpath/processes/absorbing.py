"""Absorbing-mask discrete diffusion noise schedule."""

import torch
import torch.nn as nn
from torch import Tensor

from graphpath.processes.alphas import _cosine_alphas_cumprod, _linear_alphas_cumprod
from graphpath.config import Config


class AbsorbingMaskSchedule(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        fn = _cosine_alphas_cumprod if cfg.schedule == "cosine" \
             else _linear_alphas_cumprod
        self.register_buffer("ac", fn(cfg.T))

    # ── FIX 1 + FIX 8: low-t² timestep sampler with a uniform-t mix-in ────
    def sample_t(self, B: int, T: int, device: torch.device,
                 exp: float = 2.0, uniform_mix_p: float = 0.0) -> Tensor:
        """
        Sample B timesteps biased toward small t (exp=1 → uniform;
        exp=2 → quadratic, ~50% of samples have t < T/4), but with
        probability `uniform_mix_p` draw that sample's t fully uniformly
        instead. This keeps enough heavily-masked / high-t examples in
        training to match the regime sampling actually starts from.
        """
        u = torch.rand(B, device=device)
        biased = (u ** exp * T).long()
        if uniform_mix_p > 0:
            uniform_t  = torch.randint(0, T, (B,), device=device)
            use_uniform = torch.rand(B, device=device) < uniform_mix_p
            t = torch.where(use_uniform, uniform_t, biased)
        else:
            t = biased
        return t.clamp(0, T - 1)

    def q_sample(self, x0: Tensor, t: Tensor, mask_id: int) -> Tensor:
        alpha = self.ac[t]
        for _ in range(x0.dim() - 1):
            alpha = alpha.unsqueeze(-1)
        keep = torch.bernoulli(alpha.expand_as(x0.float())).bool()
        return torch.where(keep, x0, x0.new_full(x0.shape, mask_id))
