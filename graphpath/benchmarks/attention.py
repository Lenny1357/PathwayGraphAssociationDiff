"""What the model attends to, and what that says about how it decides.

Three questions, each with a figure:

  1. WHERE does attention go? Per-layer, per-head maps over the node slots.
  2. Does it respect entity TYPES? Attention aggregated by type pair, which
     is where a learned gene -> enzyme -> reaction -> metabolite grammar
     would show up — the model is never told that grammar directly, only the
     type embedding and a bias table.
  3. Does it follow the EDGES it currently believes in, and how much of that
     comes from the edge stream rather than from node content? `GraphBlock`
     adds a learned edge bias to the attention score, so the two terms can
     be separated exactly rather than inferred.

Capture is opt-in and off by default: `capture_attention(model)` sets a flag
on each block, which stores the softmax output and its two terms.
"""

from contextlib import contextmanager
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor


@contextmanager
def capture_attention(model):
    """Record attention inside `model`'s blocks for the duration of the block."""
    blocks = [b for b in getattr(model.tower, "blocks", [])]
    for b in blocks:
        b.store_attn = True
    try:
        yield blocks
    finally:
        for b in blocks:
            b.store_attn = False
            for attr in ("last_attn", "last_content", "last_edge_bias", "last_gate"):
                if hasattr(b, attr):
                    delattr(b, attr)


@torch.no_grad()
def collect(model, node_toks: Tensor, edge_toks: Tensor, t: int,
            device=None, **forward_kwargs) -> dict:
    """
    Run one forward pass at timestep `t` and return everything captured:

        attn        (L, B, H, N, N)  post-softmax attention
        content     (L, B, H, N, N)  the Q.K term alone
        edge_bias   (L, B, H, N, N)  the edge stream's additive contribution
                                     (None for backbones without one)
        gate        (L, B, N, N)     GAT-style message gate, where present
    """
    device = device or next(model.parameters()).device
    node_toks = node_toks.to(device)
    edge_toks = edge_toks.to(device)
    tt = torch.full((node_toks.shape[0],), int(t), dtype=torch.long, device=device)

    was_training = model.training
    model.eval()
    with capture_attention(model) as blocks:
        model.forward(node_toks, edge_toks, tt, tt, **forward_kwargs)
        out = {"attn": torch.stack([b.last_attn for b in blocks]).cpu()}
        if all(getattr(b, "last_content", None) is not None for b in blocks):
            out["content"] = torch.stack([b.last_content for b in blocks]).cpu()
        if all(getattr(b, "last_edge_bias", None) is not None for b in blocks):
            out["edge_bias"] = torch.stack([b.last_edge_bias for b in blocks]).cpu()
        if all(hasattr(b, "last_gate") for b in blocks):
            out["gate"] = torch.stack([b.last_gate for b in blocks]).cpu()
    if was_training:
        model.train()
    return out


def type_flow(attn: Tensor, type_ids: Tensor, n_types: int) -> np.ndarray:
    """
    (L, n_types, n_types): attention mass from a type to a type, per layer,
    normalised so each source-type row sums to 1. This is the abstraction the
    model would have to build to respect a type grammar.
    """
    L = attn.shape[0]
    out = np.zeros((L, n_types, n_types))
    ti = type_ids.cpu()
    for l in range(L):
        a = attn[l].mean(1)                      # average heads -> (B, N, N)
        for b in range(a.shape[0]):
            for s in range(n_types):
                src = (ti[b] == s)
                if not src.any():
                    continue
                row = a[b][src]                  # (n_src, N)
                for d in range(n_types):
                    dst = (ti[b] == d)
                    if dst.any():
                        out[l, s, d] += float(row[:, dst].sum())
    row_sum = out.sum(axis=2, keepdims=True)
    return out / np.clip(row_sum, 1e-9, None)


def edge_alignment(cap: dict, edge_toks: Tensor, edge_mask_id: Optional[int] = None
                   ) -> dict:
    """
    Does attention follow the edges the model currently believes in?

    Returns, per layer: mean attention on believed-present pairs vs absent
    ones (their ratio is the readable number), and how much of the attention
    score the edge stream contributed relative to node content.
    """
    attn = cap["attn"]
    L, B, H, N, _ = attn.shape
    e = edge_toks.cpu()
    present = (e == 1)
    absent = (e == 0)
    off = ~torch.eye(N, dtype=torch.bool)[None]
    present, absent = present & off, absent & off

    rows = []
    for l in range(L):
        a = attn[l].mean(1)                      # (B, N, N)
        p = float(a[present].mean()) if present.any() else float("nan")
        q = float(a[absent].mean()) if absent.any() else float("nan")
        row = {"layer": l, "attn_present": p, "attn_absent": q,
               "ratio": p / q if q else float("nan")}
        if "edge_bias" in cap and "content" in cap:
            eb = cap["edge_bias"][l].abs().mean().item()
            ct = cap["content"][l].abs().mean().item()
            row["edge_share"] = eb / (eb + ct) if (eb + ct) else float("nan")
        rows.append(row)
    return {"per_layer": rows,
            "masked_frac": float((e == edge_mask_id).float().mean())
                           if edge_mask_id is not None else float("nan")}


def attention_entropy(attn: Tensor) -> np.ndarray:
    """(L, H) mean entropy in nats. log(N) = uniform, 0 = one slot."""
    a = attn.clamp(min=1e-12)
    ent = -(a * a.log()).sum(-1)                 # (L, B, H, N)
    return ent.mean(dim=(1, 3)).numpy()
