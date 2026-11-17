"""Uniform-transition sampler: ancestral sampling through the analytic
reverse posterior. Every token is revisable at every step.

`model` is a `GraphDiffBERTUniform`.
"""

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor


@torch.no_grad()
def sample_uniform(
    model,
    N:                   int,
    B:                   int   = 1,
    temperature:         float = 1.0,
    device:              Optional[torch.device] = None,
    type_adjacency_mask: Optional[Tensor] = None,   # (n_types, n_types) bool
    co_expr:             Optional[dict]   = None,   # co-expression conditioning
    guidance_scale:      float = 1.0,               # CFG over co_expr; 1.0 = off
    callback:            Optional[Callable] = None,  # per-step trajectory hook
) -> Tuple[Tensor, Tensor]:
    """
    Ancestral sampling via the analytic reverse posterior.

    At each step t → t-1:
      1. Run model to get x_0 logits.
      2. Compute posterior log q(x_{t-1} | x_t, x_0_pred) analytically.
      3. Sample x_{t-1} ~ Cat(posterior).

    No "force-decode" step needed — the posterior at t=1 gives x_0
    directly (since ᾱ_0 = 1 means q(x_0 | x_1, x_0) = δ(x_0)).

    `callback(t, node_state, edge_state)` is invoked after every reverse
    step, which is how the denoising trajectory is captured for
    visualisation. States are detached clones.

    FIX 11 — `type_adjacency_mask` is an (n_types, n_types) bool tensor
    where True means "an edge between these two types is allowed".
    Disallowed pairs get their "edge present" logit hard-clamped every
    step. Unlike the absorbing model, node tokens are never masked here,
    so the constraint is enforced on every step and the final adjacency
    is guaranteed violation-free with respect to the FINAL node types.
    """
    if device is None:
        device = next(model.parameters()).device

    T = model.cfg.T

    # FIX 5: one random identity draw per node, held FIXED for the whole
    # trajectory — a node's identity tag shouldn't drift step to step,
    # only its decoded content should change.
    node_id_feat = None
    if model.cfg.use_random_node_features:
        node_id_feat = torch.randn(B, N, model.cfg.rand_feat_dim, device=device)

    full_type_mask = model._build_full_type_mask(type_adjacency_mask, device)

    # Pooled conditioning vector, computed once before the reverse loop.
    cond_pooled, cond_pooled_null = None, None
    if model.tower.use_coexpr and co_expr is not None:
        cond_pooled = model.tower.encode_condition(co_expr)
        if guidance_scale > 1.0:
            cond_pooled_null = model.tower.null_condition(co_expr)

    # Initialise from prior: Uniform over all classes
    node_t = torch.randint(0, model.node_vocab, (B, N), device=device)
    # x_T must come from the kernel's stationary distribution, or the
    # reverse chain starts somewhere the forward process never reaches
    edge_t = model.schedule.prior_sample((B, N, N), model.edge_vocab,
                                        model._edge_marg, device)

    # Enforce no model-loops from the start
    diag = torch.eye(N, device=device, dtype=torch.bool)
    edge_t = edge_t.masked_fill(diag[None], 0)

    for t_val in range(T - 1, -1, -1):
        t = torch.full((B,), t_val, dtype=torch.long, device=device)

        node_logits, edge_logits = model.forward(
            node_t, edge_t, t, t, node_id_feat=node_id_feat,
            cond_pooled=cond_pooled,
        )

        # Classifier-free guidance on the conditioning signal: blend the
        # conditional x_0 logits away from the unconditional ones before
        # the posterior is formed.
        if guidance_scale > 1.0 and cond_pooled_null is not None:
            node_logits_u, edge_logits_u = model.forward(
                node_t, edge_t, t, t, node_id_feat=node_id_feat,
                cond_pooled=cond_pooled_null,
            )
            node_logits = node_logits_u + guidance_scale * (node_logits - node_logits_u)
            edge_logits = edge_logits_u + guidance_scale * (edge_logits - edge_logits_u)

        # ── Node reverse step ─────────────────────────────────────────
        # Scale logits by temperature before computing posterior
        node_logits_temp = node_logits / max(temperature, 1e-6)  # (B,N,Kn)

        if t_val > 0:
            # Compute posterior for each node position independently
            # node_t: (B, N),  node_logits_temp: (B, N, Kn)
            post_log = model.schedule.posterior_logits(
                node_logits_temp,   # (B, N, Kn)
                node_t,             # (B, N)
                t_val,
                model.node_vocab,
            )                                              # (B, N, Kn)
            node_t = torch.distributions.Categorical(
                logits=post_log
            ).sample()                                     # (B, N)
        else:
            # t=0: just take argmax of x_0 prediction
            node_t = node_logits_temp.argmax(-1)           # (B, N)

        # ── Edge reverse step ─────────────────────────────────────────
        # FIX 11: mask with the JUST-UPDATED node tokens, so the edges of
        # x_{t-1} are constrained by the node types of x_{t-1} (and, on
        # the final step, by the definitive node types).
        edge_logits = model._apply_type_mask(edge_logits, node_t, full_type_mask)
        edge_logits_temp = edge_logits / max(temperature, 1e-6)  # (B,N,N,Ke)

        if t_val > 0:
            # Flatten (B,N,N) → (B*N*N) for posterior, then reshape
            BNN = B * N * N
            post_log_e = model.schedule.posterior_logits(
                edge_logits_temp.reshape(BNN, model.edge_vocab),
                edge_t.reshape(BNN),
                t_val,
                model.edge_vocab,
                marginal=model._edge_marg,
            ).reshape(B, N, N, model.edge_vocab)            # (B,N,N,Ke)

            edge_t = torch.distributions.Categorical(
                logits=post_log_e
            ).sample()                                     # (B, N, N)
        else:
            edge_t = edge_logits_temp.argmax(-1)           # (B, N, N)

        # Enforce symmetry and no model-loops at every step
        edge_t = edge_t.masked_fill(diag[None], 0)
        if getattr(model.cfg, "edge_symmetrise", "or") == "triu":
            # one draw per undirected edge; keeps the sampled density at
            # the training density
            edge_t = torch.triu(edge_t, 1)
            edge_t = edge_t + edge_t.transpose(1, 2)
        else:
            # OR of two independent draws — inflates density every step.
            # See Config.edge_symmetrise.
            edge_t = ((edge_t == 1) | (edge_t.transpose(1, 2) == 1)).long()

        if callback is not None:
            callback(t_val, node_t.clone(), edge_t.clone())

    # Final adjacency
    adj_binary = edge_t
    adj_binary = adj_binary.masked_fill(diag[None], 0)
    return node_t, adj_binary
