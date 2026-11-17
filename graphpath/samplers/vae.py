"""One-shot VAE sampler: draw a latent from the prior, decode once.

`model` is a `GraphVAE`.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


@torch.no_grad()
def sample_vae(model, N: int, B: int = 1, temperature: float = 1.0,
           device: Optional[torch.device] = None,
           z: Optional[Tensor] = None,
           greedy: bool = False,
           type_adjacency_mask: Optional[Tensor] = None,
           co_expr: Optional[dict] = None) -> Tuple[Tensor, Tensor]:
    """
    Draw z from the prior and decode once. `z` overrides the draw, for
    interpolation or reconstruction studies.
    """
    if device is None:
        device = next(model.parameters()).device
    if z is None:
        z = torch.randn(B, model.latent_dim, device=device)

    teacher = getattr(model.cfg, "vae_teacher_types", False)
    node_logits, edge_logits = model.decode(z, N, co_expr=co_expr, raw_edges=teacher)
    full_type_mask = model._build_full_type_mask(type_adjacency_mask, device)

    tau = max(temperature, 1e-6)
    node_toks = (node_logits.argmax(-1) if greedy else
                 torch.distributions.Categorical(
                     logits=node_logits / tau).sample())
    if teacher:
        # edges see the tokens actually drawn, as they saw the true ones in training
        edge_logits = model.edge_head(edge_logits, node_toks)

    # the hard constraint reads the FINAL node types, so it is applied
    # after the tokens are decided
    edge_logits = model._apply_type_mask(edge_logits, node_toks, full_type_mask)
    edge_toks = (edge_logits.argmax(-1) if greedy else
                 torch.distributions.Categorical(
                     logits=edge_logits / tau).sample())

    adj = (edge_toks == 1).long()
    if getattr(model.cfg, "vae_symmetric_edges", False):
        upper = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), 1)
        adj = adj * upper.long()                       # one draw per pair, mirrored
        adj = adj | adj.transpose(-1, -2)
        pad = node_toks == 0                           # no edges on PAD slots
        adj = adj.masked_fill(pad[:, :, None] | pad[:, None, :], 0)
    else:
        adj = adj | adj.transpose(-1, -2)              # old behaviour: two draws OR-ed
    diag = torch.eye(N, device=device, dtype=torch.bool)
    return node_toks, adj.masked_fill(diag[None], 0)
