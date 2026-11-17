"""Gaussian sampler: deterministic DDIM over node embeddings and a
real-valued adjacency, self-conditioned, decoded to tokens at the end.

`model` is a `ContinuousGraphDiffusion`.
"""

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


@torch.no_grad()
def sample_continuous(
    model,
    N:                   int,
    B:                   int   = 1,
    temperature:         float = 1.0,
    edge_thresh:         float = 0.5,
    device:              Optional[torch.device] = None,
    type_adjacency_mask: Optional[Tensor] = None,   # (n_types, n_types) bool
    co_expr:             Optional[dict]   = None,   # co-expression conditioning
    guidance_scale:      float = 1.0,               # CFG over co_expr; 1.0 = off
    callback:            Optional[Callable] = None,  # per-step trajectory hook
) -> Tuple[Tensor, Tensor]:
    """
    Generate B graphs with N nodes using DDIM reverse diffusion.
    Returns (node_tokens (B,N), adj_binary (B,N,N)).

    FIX 11 — `type_adjacency_mask` clamps the edge logit of disallowed
    type pairs at every step, using the types estimated from the current
    node state, and once more against the FINAL decoded tokens before
    binarisation, so the returned adjacency is violation-free.
    """
    if device is None:
        device = next(model.parameters()).device

    T  = model.cfg.T
    bn = model.cfg.bottleneck_dim

    # FIX 5: one random identity draw per node, held fixed for the whole
    # trajectory (tower backend only — the separate backend has no slot
    # for it).
    node_id_feat = None
    if model.cfg.use_random_node_features and model.backend == "tower":
        node_id_feat = torch.randn(B, N, model.cfg.rand_feat_dim, device=device)

    full_type_mask = model._build_full_type_mask(type_adjacency_mask, device)

    # Pooled conditioning vector, computed once before the DDIM loop.
    cond_pooled, cond_pooled_null = None, None
    if model.tower is not None and model.tower.use_coexpr and co_expr is not None:
        cond_pooled = model.tower.encode_condition(co_expr)
        if guidance_scale > 1.0:
            cond_pooled_null = model.tower.null_condition(co_expr)

    embeds_t = torch.randn(B, N, bn, device=device)
    adj_t    = torch.randn(B, N, N, device=device)
    adj_t    = 0.5 * (adj_t + adj_t.transpose(-1, -2))

    prev_embeds = torch.zeros_like(embeds_t)
    prev_adj    = torch.zeros_like(adj_t)

    for t_val in range(T - 1, -1, -1):
        t      = torch.full((B,), t_val, dtype=torch.long, device=device)
        t_next = torch.full((B,), max(t_val - 1, 0), dtype=torch.long, device=device)

        type_ids = model._estimate_type_ids(embeds_t)
        embeds_hat, adj_logits = model.forward(
            embeds_t, adj_t, t, prev_embeds, prev_adj,
            node_id_feat=node_id_feat, type_ids=type_ids,
            cond_pooled=cond_pooled,
        )

        # Classifier-free guidance on the conditioning signal, applied to
        # the x̂_0 predictions of both streams.
        if guidance_scale > 1.0 and cond_pooled_null is not None:
            embeds_hat_u, adj_logits_u = model.forward(
                embeds_t, adj_t, t, prev_embeds, prev_adj,
                node_id_feat=node_id_feat, type_ids=type_ids,
                cond_pooled=cond_pooled_null,
            )
            embeds_hat = embeds_hat_u + guidance_scale * (embeds_hat - embeds_hat_u)
            adj_logits = adj_logits_u + guidance_scale * (adj_logits - adj_logits_u)
        adj_logits = model._apply_type_mask_binary(adj_logits, type_ids, full_type_mask)
        prev_adj = torch.sigmoid(adj_logits)

        if t_val > 0:
            embeds_t = model.schedule.ddim(embeds_t, embeds_hat, t, t_next)
            adj_pred = prev_adj
            adj_t    = model.schedule.ddim(adj_t, adj_pred, t, t_next)
            adj_t    = 0.5 * (adj_t + adj_t.transpose(-1, -2))
        else:
            embeds_t = embeds_hat
            adj_t    = prev_adj

        if callback is not None:
            # report the CURRENT x̂_0 estimate in the same token/edge form
            # the discrete variants emit, so trajectories are comparable
            callback(t_val,
                     model.embedding.decode(embeds_hat).argmax(-1),
                     (prev_adj > 0.5).long())

        prev_embeds = embeds_hat

    # Decode node tokens
    logits = model.embedding.decode(embeds_t)                 # (B,N,vocab)
    # Apply temperature scaling before argmax
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-6)
    node_toks = logits.argmax(dim=-1)                        # (B,N)

    # Binarise adjacency
    adj_binary = (adj_t > edge_thresh).long()
    adj_binary = adj_binary | adj_binary.transpose(-1, -2)
    diag       = torch.eye(N, device=device, dtype=torch.bool)
    adj_binary = adj_binary.masked_fill(diag[None], 0)

    # FIX 11: enforce the hard constraint against the FINAL node types.
    if full_type_mask is not None:
        allowed = model._type_allowed(model.token_to_type[node_toks], full_type_mask)
        adj_binary = adj_binary * allowed.long()

    return node_toks, adj_binary
