"""Absorbing-mask sampler: confidence-guided unmasking with self-correcting
remasking, classifier-free guidance and known-node conditioning (FIX 9-11).

`model` is a `GraphDiffBERT`; the sampler only reads its config, schedule,
tower and typed-connectivity helpers, and calls `model.forward`.
"""

from typing import Callable, Optional, Tuple

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from graphpath.samplers.confidence import _FractionalBudget, _confidence_topk_mask


@torch.no_grad()
def sample_masked(
    model,
    N:                       int,
    B:                       int   = 1,
    temperature:             float = 1.0,
    known_nodes:             Optional[Tensor] = None,   # (N,) or (B, N); -1 = unknown
    guidance_scale:          float = 1.0,                # CFG strength over known_nodes AND co_expr; 1.0 = off
    device:                  Optional[torch.device] = None,
    use_confidence_unmask:   Optional[bool]  = None,
    remask_frac:             Optional[float] = None,
    remask_stop_frac:        Optional[float] = None,
    confidence_temperature:  Optional[float] = None,
    carry_reveal_budget:     Optional[bool]  = None,
    confidence_reveal_thresh:      Optional[float] = None,
    edge_confidence_reveal_thresh: Optional[float] = None,
    type_adjacency_mask:     Optional[Tensor] = None,   # (n_types, n_types) bool; see FIX 11
    co_expr:                 Optional[dict]   = None,   # co-expression conditioning
    callback:                Optional[Callable] = None, # per-step trajectory hook
    stats:                   Optional[dict]     = None, # filled in with remask/reveal counters
    free_pad_cap:            Optional[float] = None,    # max share of free slots that may become PAD
    pad_scale:               Optional[float] = None,    # multiply P(PAD) on free slots by this (<1 damps it)
    pad_schedule:            str = "constant",          # constant | time | until_first (see below)
    pad_relax:               float = 0.5,               # time: damping eases off below this share of T
    pad_scale_after:         float = 1.0,               # until_first: scale once a graph holds a PAD
) -> Tuple[Tensor, Tensor]:
    """
    Generate B graphs with N nodes.

    FIX 9 — confidence-guided unmasking:
      Instead of an independent Bernoulli(p) coin-flip per masked
      position, each step reveals (approximately) the p-fraction of
      currently-masked positions the model is MOST confident about
      (with a little Gumbel noise mixed in so it isn't a brittle
      deterministic argmax-forever). This stops low-information guesses
      made at high t from permanently anchoring the rest of the
      trajectory.

    FIX 10 — model-correcting remasking:
      Absorbing diffusion is normally monotonic: once a token is
      revealed it is never revisited, so an early mistake is permanent.
      Each step (until the final `remask_stop_frac` of the trajectory)
      we additionally take a small fraction of already-REVEALED,
      non-fixed positions where the model is currently LEAST confident
      in its own earlier guess, and send them back to MASK so they get a
      fresh chance to be redecoded with more context around them.
      Remasking intensity is tapered down and then hard-disabled for the
      last `remask_stop_frac` of steps so the trajectory still converges
      to a fully decoded graph rather than oscillating forever.

    FIX 11 — hard type-adjacency constraint (optional):
      If your type-connectivity rule is a hard/deterministic constraint
      (e.g. "genes only ever connect to enzymes") rather than a soft
      statistical tendency, pass `type_adjacency_mask`, an (n_types,
      n_types) bool tensor where True means "an edge between these two
      types is allowed". Disallowed type pairs get their "edge present"
      logit hard-clamped to -inf every step, guaranteeing zero
      violations in the output regardless of what the network learned.
      Requires cfg.n_node_types > 0 and set_node_types() to have been
      called. Only enforced once both endpoints have a known (decoded)
      type — masked nodes are left unconstrained until they're revealed.
      Assumes the binary edge_vocab=2 convention (class 1 = edge present).

    `callback(t, node_state, edge_state)` is invoked after every reverse
    step (and once more on the fully decoded result, with t = -1), which
    is how the denoising trajectory is captured for visualisation. States
    are detached clones, so the callback may keep them; edge states carry
    the MASK id wherever an edge is still undecided.

    Reveal and remask budgets are carried as fractions across steps
    (`carry_reveal_budget`), so a per-step quota below 1 accumulates
    rather than rounding to zero. Setting `confidence_reveal_thresh`
    additionally lets positions the model is already sure about be
    revealed ahead of the schedule; see Config for the trade-off.

    Pass a dict as `stats` to have it filled with remasking diagnostics:
    per-step remask/reveal counts, cumulative per-slot counts, and how
    often a remasked node came back with a DIFFERENT token. Note that
    *how many* positions are remasked per step is fixed by the schedule
    (`remask_frac * t/T` of the revealed ones); what the model chooses is
    *which* ones, so `revisions_changed` is the informative number.
    """
    if device is None:
        device = next(model.parameters()).device

    use_confidence_unmask  = model.cfg.use_confidence_unmask  if use_confidence_unmask  is None else use_confidence_unmask
    remask_frac             = model.cfg.remask_frac            if remask_frac            is None else remask_frac
    remask_stop_frac        = model.cfg.remask_stop_frac       if remask_stop_frac       is None else remask_stop_frac
    confidence_temperature  = model.cfg.confidence_temperature if confidence_temperature is None else confidence_temperature
    carry_reveal_budget     = model.cfg.carry_reveal_budget    if carry_reveal_budget    is None else carry_reveal_budget
    if confidence_reveal_thresh is None:
        confidence_reveal_thresh = model.cfg.confidence_reveal_thresh
    if edge_confidence_reveal_thresh is None:
        edge_confidence_reveal_thresh = model.cfg.edge_confidence_reveal_thresh

    T  = model.cfg.T
    ac = model.schedule.ac

    # FIX 5: one random identity draw per node, held FIXED for the whole
    # trajectory (and reused for both the conditional and CFG-blind pass
    # within a step) — a node's identity tag shouldn't drift step to
    # step, only its decoded content should change.
    node_id_feat = None
    if model.cfg.use_random_node_features:
        node_id_feat = torch.randn(B, N, model.cfg.rand_feat_dim, device=device)

    # Pooled conditioning vector, computed ONCE before the T-step loop
    # (and its nulled counterpart for the CFG-blind pass) — it does not
    # depend on the current node/edge state, so there is nothing to
    # recompute per step.
    cond_pooled, cond_pooled_null = None, None
    if model.tower.use_coexpr and co_expr is not None:
        cond_pooled = model.tower.encode_condition(co_expr)
        if guidance_scale > 1.0:
            cond_pooled_null = model.tower.null_condition(co_expr)

    # Fractional budgets: carry the sub-1 remainder of every step's
    # reveal/remask quota forward instead of rounding it away.
    budget_reveal_n = _FractionalBudget(B, device)
    budget_reveal_e = _FractionalBudget(B, device)
    budget_remask_n = _FractionalBudget(B, device)
    budget_remask_e = _FractionalBudget(B, device)

    # Remasking diagnostics (FIX 10). `pending` marks slots sent back to
    # MASK whose replacement value we have not seen yet; `pre_val` holds
    # the token that was discarded, so we can tell a genuine revision
    # from a re-decode that landed on the same token.
    if stats is not None:
        node_remask_count = torch.zeros(B, N, dtype=torch.long, device=device)
        node_reveal_count = torch.zeros(B, N, dtype=torch.long, device=device)
        edge_remask_total = 0
        pending  = torch.zeros(B, N, dtype=torch.bool, device=device)
        pre_val  = torch.zeros(B, N, dtype=torch.long, device=device)
        n_changed = n_same = 0
        per_step = []

    # ── known-node setup ──────────────────────────────────────────────
    if known_nodes is not None:
        known_nodes = known_nodes.to(device=device, dtype=torch.long)
        if known_nodes.dim() == 1:
            assert known_nodes.shape[0] == N
            known_nodes = known_nodes.unsqueeze(0).expand(B, -1).clone()
        else:
            assert known_nodes.shape == (B, N)

        known_mask = known_nodes != -1
        node_t = torch.where(
            known_mask, known_nodes, torch.full_like(known_nodes, model.node_mask)
        )
    else:
        known_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        node_t = torch.full((B, N), model.node_mask, dtype=torch.long, device=device)

    edge_t = torch.full((B, N, N), model.edge_mask, dtype=torch.long, device=device)
    diag     = torch.eye(N, device=device, dtype=torch.bool)
    off_diag = ~diag
    edge_t = edge_t.masked_fill(diag[None], 0)

    # FIX 11: precompute the (n_types+1, n_types+1) allow-matrix; pairs
    # touching the reserved "unknown type" slot stay unconstrained, i.e.
    # we never constrain an edge whose endpoint is still masked.
    full_type_mask = model._build_full_type_mask(type_adjacency_mask, device)

    for t_val in range(T - 1, -1, -1):
        t = torch.full((B,), t_val, dtype=torch.long, device=device)

        # Conditional forward pass
        node_logits, edge_logits = model.forward(node_t, edge_t, t, t,
                                                 node_id_feat=node_id_feat,
                                                 co_expr=co_expr,
                                                 cond_pooled=cond_pooled)

        # CFG — second blind pass when guidance_scale > 1
        if guidance_scale > 1.0:
            node_t_blind = node_t.masked_fill(~known_mask, model.node_mask)
            node_logits_u, edge_logits_u = model.forward(
                node_t_blind, edge_t, t, t, node_id_feat=node_id_feat,
                cond_pooled=cond_pooled_null,
            )   # no co_expr: the blind pass drops the whole condition
            node_logits = node_logits_u + guidance_scale * (
                node_logits - node_logits_u
            )
            edge_logits = edge_logits_u + guidance_scale * (
                edge_logits - edge_logits_u
            )

        # FIX 11: hard-clamp disallowed type-pair edges BEFORE they
        # influence confidence-based unmasking/remasking or sampling.
        edge_logits = model._apply_type_mask(edge_logits, node_t, full_type_mask)

        # PAD on free slots. PAD is one token holding ~12% of human slots, so
        # under confidence-ranked reveals it tops the ranking everywhere, is
        # revealed first, and every PAD revealed makes the next more likely
        # (pad/human/masked in compare_oct2: 2.4 real nodes per graph). With
        # `free_pad_cap` set, PAD stays emittable but (a) slots are RANKED by
        # their best non-PAD token, so PAD no longer jumps the queue, and (b) a
        # graph may turn at most that share of its free slots into PAD; once
        # there, PAD is banned on its remaining free slots. 0 bans PAD outright.
        free_n = ~known_mask
        # `pad_scale` damps PAD instead of capping it: P(PAD) on every free slot
        # is multiplied by the scale before renormalising, so the early PAD
        # draws that tip a graph into "small graph" mode become rare, while a
        # graph can still shrink as far as the model takes it.
        #   constant     the same damping at every step
        #   time         full damping while t/T >= pad_relax, then easing log-linearly to 1 at t=0
        #   until_first  damping only while the graph holds no PAD; once one is placed the scale
        #                becomes pad_scale_after, so "is this graph small" is one damped decision
        #                and the model's own cascade then sets how small (data sizes are
        #                all-or-nothing: 64 nodes, or ~20)
        if pad_scale is not None and pad_scale != 1.0:
            ls = math.log(max(pad_scale, 1e-12))
            if pad_schedule == "time":
                frac = t_val / T
                ls_b = torch.full((B,), ls if frac >= pad_relax else ls * frac / max(pad_relax, 1e-9),
                                  device=device)
            elif pad_schedule == "until_first":
                has_pad = ((node_t == 0) & free_n).any(1)
                ls_b = torch.where(has_pad, torch.full((B,), math.log(max(pad_scale_after, 1e-12)), device=device),
                                   torch.full((B,), ls, device=device))
            else:
                ls_b = torch.full((B,), ls, device=device)
            node_logits = node_logits.clone()
            node_logits[..., 0] = node_logits[..., 0] + free_n.float() * ls_b[:, None]
        if free_pad_cap is not None:
            pad_cap = torch.floor(free_pad_cap * free_n.sum(1).float()).long()            # (B,)
            pad_now = ((node_t == 0) & free_n).sum(1)
            node_logits = node_logits.clone()
            node_logits[..., 0] = node_logits[..., 0].masked_fill(
                free_n & (pad_now >= pad_cap)[:, None], -1e4)

        ac_t  = ac[t_val]
        ac_t1 = ac[t_val - 1] if t_val > 0 else torch.ones(1, device=device)
        p = ((ac_t1 - ac_t) / (1.0 - ac_t + 1e-8)).clamp(0, 1).item()

        node_probs = F.softmax(node_logits / max(temperature, 1e-6), dim=-1)
        edge_probs = F.softmax(edge_logits / max(temperature, 1e-6), dim=-1)
        # Selection is scored in LOG-probability space. Gumbel noise added
        # to raw probabilities is scale-mismatched: competing candidates
        # can differ by ~1e-3 in probability while the noise has std
        # ~1.28*temp, which swamps the confidence signal entirely and
        # makes "confidence-guided" unmasking equivalent to random choice.
        # Log-probs span many units, so the temperature is meaningful.
        node_logp = F.log_softmax(node_logits / max(temperature, 1e-6), dim=-1)
        edge_logp = F.log_softmax(edge_logits / max(temperature, 1e-6), dim=-1)

        # ── node unmasking ────────────────────────────────────────────
        eligible_n = (node_t == model.node_mask) & ~known_mask
        if use_confidence_unmask:
            node_score = (node_logp[..., 1:] if free_pad_cap is not None else node_logp
                          ).max(dim=-1).values                 # log p, for ranking
            k_n = (budget_reveal_n.take(p, eligible_n.sum(1)) if carry_reveal_budget
                   else None)
            # the threshold stays in probability space — easier to reason
            # about ("commit anything the model is 80% sure of")
            force_n = (node_probs.max(dim=-1).values >= confidence_reveal_thresh) \
                if confidence_reveal_thresh is not None else None
            do_n = _confidence_topk_mask(node_score, eligible_n, p,
                                         confidence_temperature, k=k_n, force=force_n)
        else:
            do_n = (torch.bernoulli(torch.full((B, N), p, device=device)).bool()
                    & eligible_n)
        node_pred = torch.distributions.Categorical(probs=node_probs).sample()
        if free_pad_cap is not None:
            # several slots can draw PAD in one step: keep a random subset up to
            # the cap and send the rest back to <MASK> for a later step
            new_pad = do_n & free_n & (node_pred == 0)
            room = (pad_cap - pad_now).clamp(min=0)
            order = torch.rand(B, N, device=device).masked_fill(~new_pad, 2.0).argsort(1).argsort(1)
            do_n = do_n & ~(new_pad & (order >= room[:, None]))
        node_t = torch.where(do_n, node_pred, node_t)
        if known_nodes is not None:
            node_t = torch.where(known_mask, known_nodes, node_t)

        if stats is not None:
            node_reveal_count += do_n.long()
            # a previously-remasked slot has now been re-decoded: did the
            # model actually change its mind, or restate the same token?
            just     = pending & do_n
            changed  = just & (node_t != pre_val)
            n_changed += int(changed.sum())
            n_same    += int((just & ~changed).sum())
            pending   = pending & ~do_n
            step_remask_n = 0

        # ── edge unmasking ────────────────────────────────────────────
        eligible_e = (edge_t == model.edge_mask) & off_diag[None]
        if use_confidence_unmask:
            edge_score    = edge_logp.max(dim=-1).values.reshape(B, N * N)
            eligible_e_fl = eligible_e.reshape(B, N * N)
            k_e = (budget_reveal_e.take(p, eligible_e_fl.sum(1)) if carry_reveal_budget
                   else None)
            force_e = (edge_probs.max(dim=-1).values.reshape(B, N * N)
                       >= edge_confidence_reveal_thresh) \
                if edge_confidence_reveal_thresh is not None else None
            do_e_flat = _confidence_topk_mask(edge_score, eligible_e_fl, p,
                                              confidence_temperature, k=k_e, force=force_e)
            do_e = do_e_flat.reshape(B, N, N)
        else:
            do_e = (torch.bernoulli(torch.full((B, N, N), p, device=device)).bool()
                    & eligible_e)
        edge_pred = torch.distributions.Categorical(probs=edge_probs).sample()
        edge_t = torch.where(do_e, edge_pred, edge_t)

        # Symmetrise newly-revealed edges
        sym_val  = edge_t.transpose(1, 2)
        copy_sym = (sym_val != model.edge_mask) & (edge_t == model.edge_mask) & off_diag[None]
        edge_t   = torch.where(copy_sym, sym_val, edge_t)

        # ── FIX 10: model-correcting remasking ───────────────────────────
        if remask_frac > 0 and t_val > remask_stop_frac * T:
            # taper intensity toward 0 as t -> 0, on top of the hard
            # cutoff above, so the trajectory still fully converges
            frac_t = remask_frac * (t_val / T)

            revealed_n = (node_t != model.node_mask) & ~known_mask & ~do_n
            if revealed_n.any():
                # log p of the token currently sitting there; the least
                # likely under the model's own current belief go back
                node_self_conf = node_logp.gather(
                    -1, node_t.clamp(max=model.node_vocab - 1).unsqueeze(-1)
                ).squeeze(-1)
                k_rn = (budget_remask_n.take(frac_t, revealed_n.sum(1))
                        if carry_reveal_budget else None)
                remask_n = _confidence_topk_mask(-node_self_conf, revealed_n, frac_t,
                                                  confidence_temperature, k=k_rn)
                if stats is not None:
                    # capture the discarded token BEFORE it is overwritten
                    pre_val = torch.where(remask_n, node_t, pre_val)
                    pending = pending | remask_n
                    node_remask_count += remask_n.long()
                    step_remask_n = int(remask_n.sum())
                node_t = torch.where(remask_n, torch.full_like(node_t, model.node_mask), node_t)
                if known_nodes is not None:
                    node_t = torch.where(known_mask, known_nodes, node_t)

            revealed_e = (edge_t != model.edge_mask) & off_diag[None] & ~do_e
            if revealed_e.any():
                edge_self_conf = edge_logp.gather(
                    -1, edge_t.clamp(max=model.edge_vocab - 1).unsqueeze(-1)
                ).squeeze(-1).reshape(B, N * N)
                revealed_e_fl = revealed_e.reshape(B, N * N)
                k_re = (budget_remask_e.take(frac_t, revealed_e_fl.sum(1))
                        if carry_reveal_budget else None)
                remask_e_flat = _confidence_topk_mask(
                    -edge_self_conf, revealed_e_fl, frac_t, confidence_temperature, k=k_re
                )
                remask_e = remask_e_flat.reshape(B, N, N)
                remask_e = remask_e | remask_e.transpose(1, 2)   # keep pairs symmetric
                if stats is not None:
                    edge_remask_total += int(remask_e.sum())
                edge_t = torch.where(remask_e, torch.full_like(edge_t, model.edge_mask), edge_t)

        if stats is not None:
            per_step.append({"t": t_val,
                             "revealed": int(do_n.sum()),
                             "remasked": step_remask_n})

        if callback is not None:
            callback(t_val, node_t.clone(), edge_t.clone())

    # Force-decode any remaining masks
    t0 = torch.zeros(B, dtype=torch.long, device=device)
    nl, el = model.forward(node_t, edge_t, t0, t0, node_id_feat=node_id_feat,
                          co_expr=co_expr, cond_pooled=cond_pooled)
    node_t = torch.where(node_t == model.node_mask, nl.argmax(-1), node_t)
    # FIX 11: node_t is now fully decoded, so re-apply the hard type mask
    # (using the FINAL node types) before force-decoding remaining edges.
    el = model._apply_type_mask(el, node_t, full_type_mask)
    if known_nodes is not None:
        node_t = torch.where(known_mask, known_nodes, node_t)
    still  = (edge_t == model.edge_mask) & off_diag[None]
    edge_t = torch.where(still, el.argmax(-1), edge_t)

    adj_binary = (edge_t == 1).long()
    adj_binary = adj_binary | adj_binary.transpose(-1, -2)
    adj_binary = adj_binary.masked_fill(diag[None], 0)
    if stats is not None:
        stats.update({
            "per_step":           per_step,
            "node_remask_count":  node_remask_count.cpu(),   # (B, N)
            "node_reveal_count":  node_reveal_count.cpu(),   # (B, N)
            "node_remask_total":  int(node_remask_count.sum()),
            "node_reveal_total":  int(node_reveal_count.sum()),
            "edge_remask_total":  edge_remask_total,
            "revisions_changed":  n_changed,   # remasked, came back different
            "revisions_same":     n_same,      # remasked, came back identical
            "still_pending":      int(pending.sum()),  # remasked, force-decoded at the end
        })

    if callback is not None:
        callback(-1, node_t.clone(), edge_t.clone())
    return node_t, adj_binary
