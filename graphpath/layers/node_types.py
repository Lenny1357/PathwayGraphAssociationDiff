"""Typed connectivity (FIX 11), shared by both diffusion variants.

Nothing in the base architecture is told that node identities fall into a
small number of TYPES with hard rules about which types may be adjacent.
The model has to rediscover that abstraction purely from co-occurrence
statistics over specific entity tokens, which is much harder to generalize
than an explicit categorical feature — this is usually why loss/graph-shape
can look fine while type-adjacency is still wrong.

This mixin provides the three pieces:
  * `token_to_type`  — node token id -> type id lookup buffer
  * `type_pair_bias` — a tiny learned (n_types+1)^2 x edge_vocab additive
                       prior on the edge logits
  * `type_adjacency_mask` support — a hard -inf clamp on the "edge present"
                       logit for disallowed type pairs, used at sampling time
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class NodeTypeMixin:

    # ── construction ────────────────────────────────────────────────────────

    def _setup_node_types(self, cfg, vocab_size: int, unknown_is_mask: bool):
        """
        `token_to_type` maps each node token id (0..node_vocab-1) to a type
        id in [0, n_types). When `unknown_is_mask` (absorbing diffusion) the
        table has one extra row for the MASK id, which always resolves to the
        reserved "unknown type" slot (= n_types) and is never overwritten by
        set_node_types().

        The learned type_pair_bias is *tiny* — since it's shared across every
        specific entity pair of a given type combination, it can learn "these
        two types are simply incompatible" from far less data than
        distributing that same rule across the whole node embedding space
        would require, and it generalizes immediately to entity pairs never
        co-observed in training. Zero-init so it starts as a no-op.
        """
        self.n_types = cfg.n_node_types
        if self.n_types > 0:
            n_rows = vocab_size if unknown_is_mask else self.node_vocab
            self.register_buffer(
                "token_to_type",
                torch.full((n_rows,), self.n_types, dtype=torch.long),
            )
            if cfg.use_type_bias:
                self.type_pair_bias = nn.Parameter(
                    torch.zeros(self.n_types + 1, self.n_types + 1, self.edge_vocab)
                )
            else:
                self.type_pair_bias = None
        else:
            self.token_to_type  = None
            self.type_pair_bias = None

    def set_node_types(self, token_to_type):
        """
        Register the (deterministic) node-token -> node-type lookup used for
        typed connectivity. Requires cfg.n_node_types > 0 (rebuild the model
        with n_node_types set if it currently isn't).

            # e.g. gene=0, enzyme=1, reaction=2, metabolite=3
            token_to_type = torch.tensor([...])   # length == model.node_vocab
            model.set_node_types(token_to_type)

        `token_to_type` should have length >= model.node_vocab, one type id
        in [0, n_node_types) per real node token; extra entries are ignored.
        The MASK token's type always stays at the reserved "unknown" slot.
        """
        if self.n_types == 0:
            raise RuntimeError(
                "Config.n_node_types must be > 0 to use node types; "
                "rebuild the model with n_node_types set."
            )
        token_to_type = torch.as_tensor(
            token_to_type, dtype=torch.long, device=self.token_to_type.device
        )
        assert token_to_type.numel() >= self.node_vocab, \
            f"expected at least {self.node_vocab} entries, got {token_to_type.numel()}"
        self.token_to_type[: self.node_vocab] = token_to_type[: self.node_vocab]

    # ── use ─────────────────────────────────────────────────────────────────

    def _type_ids(self, node_toks: Tensor) -> Optional[Tensor]:
        """(B, N) type ids, or None when typing is disabled."""
        return self.token_to_type[node_toks] if self.n_types > 0 else None

    def _add_type_pair_bias(self, edge_logits: Tensor,
                            type_ids: Optional[Tensor]) -> Tensor:
        """
        Add the learned per-type-pair bias directly onto the edge logits.
        A fast-to-learn additive prior layered on top of the pairwise
        hidden-state term computed by the graph tower. Under absorbing
        diffusion it only carries information once both endpoints have a
        known (non-mask) type, so early in sampling it's naturally inert and
        sharpens as nodes get decoded.
        """
        if self.type_pair_bias is None or type_ids is None:
            return edge_logits
        N  = type_ids.shape[1]
        ti = type_ids.unsqueeze(2).expand(-1, -1, N)   # (B, N, N)
        tj = type_ids.unsqueeze(1).expand(-1, N, -1)   # (B, N, N)
        return edge_logits + self.type_pair_bias[ti, tj]  # (B, N, N, edge_vocab)

    # ── hard adjacency constraint (sampling) ────────────────────────────────

    def _build_full_type_mask(self, type_adjacency_mask: Optional[Tensor],
                              device) -> Optional[Tensor]:
        """
        Expand a user-supplied (n_types, n_types) allow-matrix to
        (n_types+1, n_types+1), padding any pair that touches the reserved
        "unknown type" slot with True — i.e. we never constrain an edge whose
        endpoint is still masked/undecoded.
        """
        if type_adjacency_mask is None:
            return None
        assert self.n_types > 0, "type_adjacency_mask requires cfg.n_node_types > 0"
        assert self.edge_vocab == 2, \
            "hard type masking assumes binary edge_vocab=2 (class 1 = edge present)"
        full = torch.ones(self.n_types + 1, self.n_types + 1,
                          dtype=torch.bool, device=device)
        full[:self.n_types, :self.n_types] = type_adjacency_mask.to(
            device=device, dtype=torch.bool
        )
        return full

    def _type_allowed(self, type_ids: Tensor,
                      full_type_mask: Tensor) -> Tensor:
        """(B, N, N) bool — is an edge between these two endpoints allowed?"""
        N  = type_ids.shape[1]
        ti = type_ids.unsqueeze(2).expand(-1, -1, N)
        tj = type_ids.unsqueeze(1).expand(-1, N, -1)
        return full_type_mask[ti, tj]

    def _apply_type_mask(self, edge_logits: Tensor, node_toks: Tensor,
                         full_type_mask: Optional[Tensor]) -> Tensor:
        """Hard-clamp the 'edge present' logit for disallowed type pairs."""
        if full_type_mask is None:
            return edge_logits
        allowed = self._type_allowed(self.token_to_type[node_toks], full_type_mask)
        out = edge_logits.clone()
        out[..., 1] = out[..., 1].masked_fill(~allowed, -1e4)
        return out

    # ── single-logit (Bernoulli) edge parameterisation ──────────────────────
    # The continuous variant predicts ONE logit per pair rather than a
    # 2-class distribution. A binary logit is exactly logit(present) -
    # logit(absent), so the corresponding type prior is the difference of the
    # two channels of the same shared table — no second parameter needed.

    def _add_type_pair_bias_binary(self, logits: Tensor,
                                   type_ids: Optional[Tensor]) -> Tensor:
        if self.type_pair_bias is None or type_ids is None:
            return logits
        assert self.edge_vocab == 2, \
            "binary edge logits require edge_vocab=2"
        N    = type_ids.shape[1]
        ti   = type_ids.unsqueeze(2).expand(-1, -1, N)
        tj   = type_ids.unsqueeze(1).expand(-1, N, -1)
        pair = self.type_pair_bias[ti, tj]                  # (B, N, N, 2)
        return logits + (pair[..., 1] - pair[..., 0])

    def _apply_type_mask_binary(self, logits: Tensor, type_ids: Optional[Tensor],
                                full_type_mask: Optional[Tensor]) -> Tensor:
        if full_type_mask is None or type_ids is None:
            return logits
        allowed = self._type_allowed(type_ids, full_type_mask)
        return logits.masked_fill(~allowed, -1e4)
