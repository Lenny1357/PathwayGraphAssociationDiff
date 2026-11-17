"""Controls to read a benchmark score against.

A score means nothing on its own. Three references bracket it:

    training data   what real pathways score — the target
    shuffled        the model's own node vocabulary, regrouped at random.
                    Isolates whether the model's *grouping* carries signal,
                    or whether any set drawn from its token distribution
                    would score the same.
    random          uniform tokens at matched density — the absolute floor.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


def random_like(node_toks: Tensor, adjacency: Optional[Tensor], vocab_size: int,
                pad_id: int = 0, seed: int = 0) -> Tuple[Tensor, Optional[Tensor]]:
    """Uniform node tokens, edges at the same density as `adjacency`."""
    g = torch.Generator().manual_seed(seed)
    B, N = node_toks.shape
    lo = pad_id + 1 if pad_id == 0 else 0
    nodes = torch.randint(lo, vocab_size, (B, N), generator=g)

    adj = None
    if adjacency is not None:
        off = ~torch.eye(N, dtype=torch.bool)
        density = adjacency.cpu().bool()[:, off].float().mean()
        a = (torch.rand(B, N, N, generator=g) < density).long()
        adj = (a | a.transpose(1, 2))
        adj.diagonal(dim1=1, dim2=2).zero_()
        adj = adj.to(adjacency.device)
    return nodes.to(node_toks.device), adj


def shuffled_like(node_toks: Tensor, adjacency: Optional[Tensor],
                  seed: int = 0) -> Tuple[Tensor, Optional[Tensor]]:
    """
    The same multiset of tokens, redealt across graphs. Topology is kept as
    generated, so any drop against the real score is attributable to *which
    entities were grouped together*, not to the token distribution or to the
    graph's shape.
    """
    g = torch.Generator().manual_seed(seed)
    B, N = node_toks.shape
    flat = node_toks.reshape(-1)
    perm = torch.randperm(flat.numel(), generator=g).to(flat.device)
    return flat[perm].reshape(B, N), adjacency


def compare(results: dict, keys=None) -> str:
    """Render several BenchmarkResults side by side."""
    names = list(results)
    if keys is None:
        keys, seen = [], set()
        for r in results.values():
            for k in r.summary:
                if k not in seen and isinstance(r.summary[k], (int, float)):
                    seen.add(k); keys.append(k)
    w = max(len(k) for k in keys) + 2
    head = f"{'metric':<{w}}" + "".join(f"{n:>18}" for n in names)
    lines = [head, "-" * len(head)]
    for k in keys:
        row = f"{k:<{w}}"
        for n in names:
            v = results[n].summary.get(k)
            row += f"{v:>18.4f}" if isinstance(v, float) else f"{str(v):>18}"
        lines.append(row)
    lines.append("-" * len(head))
    lines.append(f"{'graphs scored':<{w}}" +
                 "".join(f"{results[n].n_scored:>13}/{results[n].n_graphs:<4}" for n in names))
    return "\n".join(lines)
