"""Marginal-transition sampler: ancestral sampling through the exact reverse
posterior, prior at t=T drawn from Cat(m), unbiased triangle mirroring, and
an optional clamped context for inpainting.

`model` is a `GraphDiffBERTMarginal` or `GraphDiffBERTJoint`.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from graphpath.processes.marginal import _mirror_upper


@torch.no_grad()
def sample_marginal(
    model,
    N:            int,
    B:            int   = 1,
    temperature:  float = 1.0,
    device:       Optional[torch.device] = None,
    node_clamp:   Optional[Tuple[Tensor, Tensor]] = None,
    edge_clamp:   Optional[Tuple[Tensor, Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Ancestral sampling via the analytic reverse posterior.

    At each step t → t-1:
      1. Run model to get x_0 logits.
      2. Compute posterior log q(x_{t-1} | x_t, x_0_pred) analytically.
      3. Sample x_{t-1} ~ Cat(posterior).

    Prior at t=T is Cat(m) — not Uniform{K} — so the trajectory starts
    at roughly the data's density instead of at a near-complete graph.

    node_clamp / edge_clamp : optional (mask, values) pairs. Where mask
    is True the tokens are held at `values` for the whole trajectory.
    This is the inpainting / gap-filling hook: clamp the observed part
    of the graph, let the model denoise the rest.
    """
    if device is None:
        device = next(model.parameters()).device

    T    = model.cfg.T
    diag = torch.eye(N, device=device, dtype=torch.bool)

    def _apply_clamps(nt, et):
        if node_clamp is not None:
            nm, nv = node_clamp
            nt = torch.where(nm.to(device), nv.to(device), nt)
        if edge_clamp is not None:
            em, ev = edge_clamp
            et = torch.where(em.to(device), ev.to(device), et)
        return nt, et

    # Initialise from prior: Cat(m) for each stream
    _shift = model._edge_logit_shift()
    node_t = model.schedule.sample_prior(model.m_node, (B, N), device)
    edge_t = model.schedule.sample_prior(model.m_edge, (B, N, N), device)
    edge_t = _mirror_upper(edge_t).masked_fill(diag[None], 0)
    node_t, edge_t = _apply_clamps(node_t, edge_t)

    for t_val in range(T - 1, -1, -1):
        t = torch.full((B,), t_val, dtype=torch.long, device=device)

        node_logits, edge_logits = model.forward(node_t, edge_t, t, t)

        # ── Node reverse step ─────────────────────────────────────────
        node_logits_temp = node_logits / max(temperature, 1e-6)

        if t_val > 0:
            post_log = model.schedule.posterior_logits(
                node_logits_temp, node_t, t_val,
                model.m_node, exact=model.cfg.exact_posterior,
            )                                              # (B, N, Kn)
            node_t = torch.distributions.Categorical(
                logits=post_log
            ).sample()
        else:
            node_t = node_logits_temp.argmax(-1)

        # ── Edge reverse step ─────────────────────────────────────────
        # remove the class-weight bias BEFORE the posterior sees it, so
        # the reverse step is driven by p(edge|context) and not by the
        # rebalanced q the weighted loss actually optimises
        if _shift is not None:
            edge_logits = edge_logits - _shift
        edge_logits_temp = edge_logits / max(temperature, 1e-6)

        if t_val > 0:
            BNN = B * N * N
            post_log_e = model.schedule.posterior_logits(
                edge_logits_temp.reshape(BNN, model.edge_vocab),
                edge_t.reshape(BNN), t_val,
                model.m_edge, exact=model.cfg.exact_posterior,
            ).reshape(B, N, N, model.edge_vocab)

            edge_t = torch.distributions.Categorical(
                logits=post_log_e
            ).sample()
        else:
            edge_t = edge_logits_temp.argmax(-1)

        # Unbiased symmetrisation + no model-loops at every step
        if model.cfg.symmetric_edges:
            edge_t = _mirror_upper(edge_t)
        edge_t = edge_t.masked_fill(diag[None], 0)
        node_t, edge_t = _apply_clamps(node_t, edge_t)
        edge_t = edge_t.masked_fill(diag[None], 0)

    return node_t, edge_t
