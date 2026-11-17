"""Node class balancing (FIX 7).

Edge loss balances positive/negative classes; node loss did not, which let a
few high-frequency node types (e.g. near-duplicate lipid species) dominate
the CE gradient and get over-sampled at generation time.
"""

import torch
from torch import Tensor


def inverse_freq_weights(counts, power: float, expected_n: int, device) -> Tensor:
    """
    power=0   -> uniform (no-op)
    power=0.5 -> sqrt-inverse-frequency (gentle, recommended default)
    power=1.0 -> full inverse-frequency (aggressive; can overweight
                 rare/near-noise node types — watch validation loss)

    Weights are normalized so the mean weight is ≈ 1.
    """
    counts = torch.as_tensor(counts, dtype=torch.float, device=device).clamp(min=1)
    assert counts.numel() == expected_n, \
        f"expected {expected_n} counts, got {counts.numel()}"
    if power <= 0:
        return torch.ones_like(counts)
    inv = counts.pow(-power)
    return inv * (inv.numel() / inv.sum())
