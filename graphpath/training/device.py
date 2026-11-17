"""Make the graphpath stack runnable on Apple MPS.

One PyTorch bug blocks it, and it is worth stating precisely because the
failure mode is a hard abort with no Python traceback:

    nn.Linear applied to an EXPANDED (stride-0) 4-D tensor aborts the process
    on MPS in torch 2.8.0:

        /...MPSNDArray.mm:893: failed assertion
        `[MPSNDArray initWithBufferImpl:...] buffer is not large enough.'

Bisected to exactly that: `Linear(h.unsqueeze(1).expand(-1,N,-1,-1))` aborts,
`Linear(...expand(...).contiguous())` does not. In graphpath the single site
is `self.msg_val(hj)` in GraphBlock (blocks.py), the GAT message projection;
every other 4-D Linear goes through a torch.cat first, which materialises a
contiguous tensor.

The patch below is deliberately general rather than a rewrite of GraphBlock:
it makes any stride-0 input contiguous before Linear, on MPS only. That way a
site I have not found still works, and CUDA/CPU behaviour is untouched.

    from mps_compat import patch_linear_for_mps, pick_device
    patch_linear_for_mps()
    device = pick_device()

Worth upstreaming separately: `msg_val(hj)` projects the SAME (B,N,d) tensor
N times, since hj[b,i,j,:] = h[b,j,:] and Linear acts only on the last axis.
Computing `msg_val(h)` once and expanding afterwards is identical and N times
cheaper — a real speedup, not just an MPS workaround. Left alone here so this
folder does not change shared code.
"""

import torch
import torch.nn as nn

_PATCHED = False


def patch_linear_for_mps() -> bool:
    """Idempotent. Returns True if the patch is now active."""
    global _PATCHED
    if _PATCHED or not torch.backends.mps.is_available():
        return _PATCHED
    _orig = nn.Linear.forward

    def _safe(self, x):
        # stride 0 on any axis == this tensor came from expand/broadcast
        if x.device.type == "mps" and 0 in x.stride():
            x = x.contiguous()
        return _orig(self, x)

    nn.Linear.forward = _safe
    _PATCHED = True
    return True


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        patch_linear_for_mps()
        return torch.device("mps")
    return torch.device("cpu")
