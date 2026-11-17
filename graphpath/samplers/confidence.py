"""Confidence-guided decoding helpers for absorbing-mask sampling (FIX 9 / 10).

Used only by GraphDiffBERT.sample — the other two variants have no notion of
a "still masked" position to reveal or send back.
"""

from typing import Optional

import torch
from torch import Tensor


class _FractionalBudget:
    """
    Turns a per-step fractional budget into an integer count without losing
    the remainder.

    Rounding `frac * n_eligible` independently each step silently drops every
    budget below 0.5: with N=64 nodes the reveal fraction p_t stays under
    0.008 until t≈139, so `round` yields 0 and the first ~60 steps of a
    200-step trajectory reveal nothing at all. Carrying the fractional part
    forward instead means the same expected number of positions is revealed,
    but the trajectory starts moving immediately.
    """

    def __init__(self, B: int, device):
        self.carry = torch.zeros(B, device=device)

    def take(self, frac: float, n_eligible: Tensor) -> Tensor:
        """(B,) integer budget for this step, given (B,) eligible counts."""
        want = self.carry + frac * n_eligible.float()
        k = want.floor()
        self.carry = want - k                       # always in [0, 1)
        return k.long()


def _confidence_topk_mask(
    score:       Tensor,            # (B, K) higher = more likely to be SELECTED
    eligible:    Tensor,            # (B, K) bool — only True positions are candidates
    frac:        Optional[float] = None,  # fraction of eligible entries to select
    gumbel_temp: float = 0.0,
    k:           Optional[Tensor] = None, # (B,) explicit budget; overrides `frac`
    force:       Optional[Tensor] = None, # (B, K) bool — always select these
) -> Tensor:
    """
    Per-row top-k selection with a variable budget per row (since the number
    of eligible/masked positions can differ across the batch), implemented
    without a python loop over B via an argsort + rank trick.

    Used both to pick which masked positions to reveal (pass +confidence)
    and which revealed positions to reconsider/remask (pass -confidence, so
    the LEAST confident revealed positions are the ones selected).

    Pass `k` (from a _FractionalBudget) instead of `frac` to avoid the
    per-step rounding loss. Pass `force` to select positions regardless of
    the budget — used to let confident predictions run ahead of the schedule.
    """
    B, K = score.shape
    if gumbel_temp > 0:
        u = torch.rand_like(score).clamp(min=1e-9, max=1 - 1e-9)
        gumbel = -torch.log(-torch.log(u))
        score = score + gumbel_temp * gumbel

    score = score.masked_fill(~eligible, float("-inf"))
    n_eligible = eligible.sum(dim=1)
    if k is None:
        k = (frac * n_eligible.float()).round().long()
    k = k.clamp(min=0)

    if int(n_eligible.max().item()) == 0:
        return torch.zeros_like(eligible)

    sorted_idx = torch.argsort(score, dim=1, descending=True)
    rank = torch.empty_like(sorted_idx)
    arangeK = torch.arange(K, device=score.device).unsqueeze(0).expand(B, K)
    rank.scatter_(1, sorted_idx, arangeK)

    selected = (rank < k.unsqueeze(1)) & eligible
    if force is not None:
        selected = selected | (force & eligible)
    return selected
