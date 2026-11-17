"""GraphDiffBERT — absorbing-mask discrete diffusion over nodes and adjacency."""

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphpath.layers.graph_block import GraphTower
from graphpath.config import Config
from graphpath.layers.node_types import NodeTypeMixin
from graphpath.layers.pairwise_conditioning import CoexprLookup, PairwiseEdgeBias
from graphpath.processes.absorbing import AbsorbingMaskSchedule
from graphpath.samplers.masked import sample_masked
from graphpath.training.class_weights import inverse_freq_weights


class GraphDiffBERT(NodeTypeMixin, nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg        = cfg
        self.vocab_size = vocab_size
        self.node_vocab = vocab_size - 1
        self.node_mask  = self.node_vocab
        cfg.node_vocab  = self.node_vocab
        self.edge_vocab = cfg.edge_vocab
        self.edge_mask  = cfg.edge_vocab

        self.schedule = AbsorbingMaskSchedule(cfg)
        self.tower    = GraphTower(cfg)   # FIX 12: interleaved node/edge stack

        d = cfg.d_model
        self.node_norm = nn.LayerNorm(d)
        self.node_head = nn.Linear(d, self.node_vocab)
        self.edge_norm = nn.LayerNorm(d)
        self.edge_head = nn.Linear(d, self.edge_vocab)

        # FIX 7: per-class node loss weights. Defaults to all-ones (no-op);
        # call set_node_class_weights(counts) once you have dataset stats.
        self.register_buffer("node_class_weight", torch.ones(self.node_vocab))

        # FIX 11: typed connectivity — see graphpath/layers/node_types.py.
        self._setup_node_types(cfg, vocab_size, unknown_is_mask=True)

        # Relational co-expression conditioning. Both stay inert until
        # set_coexpression() supplies the tables.
        self.coexpr_lookup = None
        self.pairwise_bias = (PairwiseEdgeBias(self.edge_vocab, cfg.pairwise_hidden)
                              if cfg.use_pairwise_bias else None)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

        # Zero-init the per-layer edge attention bias (so training starts at
        # the no-edge-bias baseline) and the conditioning gate/FiLM (so it
        # starts unconditional). Both live on the tower.
        self.tower.post_init_zeros()
        if self.pairwise_bias is not None:
            self.pairwise_bias.reset_zeros()

    # ── FIX 7: node class weighting setup ──────────────────────────────────
    def set_coexpression(self, coexpr_sparse, top_k: Optional[int] = None,
                         verbose: bool = True):
        """
        Register the (n_tissues, V, V) co-expression tensor the relational
        conditioning reads — `expr_loader.stacked_coexpression()`, or the
        propagated network from `propagate_all_tissues(...)` so metabolites
        and reactions carry a value too.

        Required by cfg.use_pairwise_bias and cfg.use_coexpr_attn; both are
        no-ops without it.
        """
        if not (self.cfg.use_pairwise_bias or self.cfg.use_coexpr_attn):
            raise RuntimeError(
                "set_coexpression() needs cfg.use_pairwise_bias or "
                "cfg.use_coexpr_attn; rebuild the model with one of them set.")
        self.coexpr_lookup = CoexprLookup(
            coexpr_sparse, self.vocab_size,
            top_k=top_k or self.cfg.coexpr_attn_k, verbose=verbose
        ).to(next(self.parameters()).device)

    def _coexpr_context(self, node_toks: Tensor, co_expr: Optional[dict],
                        cond_drop: Optional[Tensor] = None):
        """
        (neighbour embeddings, weights, pair values) for this batch.

        `cond_drop` marks samples whose condition is dropped for CFG. It has
        to reach here, not just the pooled vector: at sampling time the blind
        pass gets no co-expression at all, so if training only ever dropped
        the pooled half, the blind pass would be a state the model has never
        seen and guidance would extrapolate from nothing.
        """
        if self.coexpr_lookup is None or co_expr is None:
            return None, None, None, None, None
        tissue_ids = co_expr["tissue_ids"].to(node_toks.device).long()
        if tissue_ids.numel() == 1 and node_toks.shape[0] > 1:
            # one tissue for the whole batch — the pooled conditioner already
            # broadcasts this way, so accept it here too
            tissue_ids = tissue_ids.reshape(1).expand(node_toks.shape[0])
        if tissue_ids.shape[0] != node_toks.shape[0]:
            raise ValueError(
                f"co_expr['tissue_ids'] has {tissue_ids.shape[0]} entries but "
                f"the batch has {node_toks.shape[0]} graphs; pass one tissue "
                f"per graph (when sampling, one per requested sample).")
        nbr_emb = nbr_w = pair_vals = nbr_slots = None
        if self.cfg.use_coexpr_attn:
            idx, wts = self.coexpr_lookup.neighbours(tissue_ids, node_toks)
            if cond_drop is not None and cond_drop.any():
                # dropping the condition drops the partners with it
                wts = wts * (~cond_drop).to(wts.dtype)[:, None, None]
            # compact to the slots that actually have partners; the embedding
            # lookup and every layer's projections then run on those only
            k = wts.shape[-1]
            nbr_slots = (wts > 0).any(-1).reshape(-1).nonzero(as_tuple=True)[0]
            if nbr_slots.numel():
                nbr_w = wts.reshape(-1, k).index_select(0, nbr_slots)
                nbr_emb = self.tower.tok_emb(
                    idx.reshape(-1, k).index_select(0, nbr_slots))
            else:
                nbr_slots = None
        if self.cfg.use_pairwise_bias:
            N = node_toks.shape[1]
            u = node_toks[:, :, None].expand(-1, -1, N)
            v = node_toks[:, None, :].expand(-1, N, -1)
            pair_vals = self.coexpr_lookup.pair(tissue_ids, u, v)
        keep = None if cond_drop is None else (~cond_drop).to(node_toks.device)
        return nbr_emb, nbr_w, pair_vals, nbr_slots, keep

    def set_node_class_weights(self, counts, power: Optional[float] = None):
        """
        Call once you have training-set node-token frequencies, e.g.:

            counts = torch.bincount(all_node_feats[pad_mask].flatten(),
                                     minlength=model.node_vocab)
            model.set_node_class_weights(counts)

        power=0   -> uniform (no-op)
        power=0.5 -> sqrt-inverse-frequency (gentle, recommended default)
        power=1.0 -> full inverse-frequency (aggressive; can overweight
                     rare/near-noise node types — watch validation loss)
        """
        power = self.cfg.node_class_weight_power if power is None else power
        w = inverse_freq_weights(counts, power, self.node_vocab,
                                 self.node_class_weight.device)
        self.node_class_weight.copy_(w)

    def forward(
        self,
        node_toks:    Tensor,                  # (B, N)     long
        edge_toks:    Tensor,                  # (B, N, N)  long
        t_node:       Tensor,                  # (B,)       long
        t_edge:       Tensor,                  # (B,)       long
        node_id_feat: Optional[Tensor] = None, # see FIX 5
        co_expr:      Optional[dict]   = None, # co-expression conditioning
        cond_drop:    Optional[Tensor] = None, # (B,) bool — null the condition
        cond_pooled:  Optional[Tensor] = None, # precomputed via tower.encode_condition()
    ) -> Tuple[Tensor, Tensor]:
        # FIX 11: look up each node's type (node_mask id -> "unknown" slot).
        type_ids = self._type_ids(node_toks)
        nbr_emb, nbr_w, pair_vals, nbr_slots, pair_keep = self._coexpr_context(
            node_toks, co_expr, cond_drop)

        h, e        = self.tower(node_toks, edge_toks, t_node, t_edge,
                                  node_id_feat=node_id_feat, type_ids=type_ids,
                                  co_expr=co_expr, cond_drop=cond_drop,
                                  cond_pooled=cond_pooled,
                                  nbr_emb=nbr_emb, nbr_w=nbr_w,
                                  nbr_slots=nbr_slots)
        node_logits = self.node_head(self.node_norm(h))
        edge_logits = self.edge_head(self.edge_norm(e))

        # FIX 11: learned per-type-pair prior on the edge logits.
        edge_logits = self._add_type_pair_bias(edge_logits, type_ids)

        # co-expression between the two endpoints, straight onto the edge
        # logits — the one place a pairwise fact can act directly
        if self.pairwise_bias is not None and pair_vals is not None:
            edge_logits = self.pairwise_bias(edge_logits, pair_vals,
                                             keep=pair_keep)

        return node_logits, edge_logits

    # ── loss ─────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        node_feats:       Tensor,    # (B, N)    long
        adjcs:            Tensor,    # (B, N, N) long (0/1)
        pad_mask:         Tensor,    # (B, N)    bool
        loss_edge_weight: float = 1.0,
        co_expr:          Optional[dict] = None,
    ) -> dict:
        B, N   = node_feats.shape
        device = node_feats.device

        # FIX 1 + FIX 8: low-t² sampling with a uniform-t mix-in so the
        # heavily-masked regime sampling starts from isn't under-trained.
        t_node = self.schedule.sample_t(B, self.cfg.T, device,
                                         self.cfg.t_sample_exp, self.cfg.uniform_t_mix_p)
        t_edge = self.schedule.sample_t(B, self.cfg.T, device,
                                         self.cfg.t_sample_exp, self.cfg.uniform_t_mix_p)

        # ── Forward diffusion ─────────────────────────────────────────────
        # With supervise_pad the padding slots are masked and supervised like
        # any other: PAD becomes a token the model can learn to EMIT, which is
        # how it says "this graph is shorter than N". Without it, padding is
        # held clean and dropped from the loss, so nothing ever teaches the
        # model to stop — and at sampling, where there is no pad_mask, it can
        # only fill every slot.
        sup_pad = self.cfg.supervise_pad
        node_t = self.schedule.q_sample(node_feats, t_node, self.node_mask)
        if not sup_pad:
            node_t = torch.where(pad_mask, node_t, node_feats)

        valid_pair = pad_mask[:, :, None] & pad_mask[:, None, :]
        edge_t     = self.schedule.q_sample(adjcs, t_edge, self.edge_mask)
        if not sup_pad:
            edge_t = torch.where(valid_pair, edge_t, adjcs)

        # CFG-style conditioning dropout — a fraction of training samples
        # see the null condition, so the model also learns the
        # unconditional distribution that guidance needs at sampling time.
        cond_drop = None
        if co_expr is not None and self.training and (
                self.tower.use_coexpr or self.cfg.use_pairwise_bias
                or self.cfg.use_coexpr_attn):
            cond_drop = (self.tower.conditioner.train_drop_mask(B, device)
                         if self.tower.use_coexpr else
                         torch.rand(B, device=device) < self.cfg.cond_dropout_p)

        # ── Predict ───────────────────────────────────────────────────────
        node_logits, edge_logits = self.forward(node_t, edge_t, t_node, t_edge,
                                                co_expr=co_expr, cond_drop=cond_drop)

        # ── Node CE — FIX 7: inverse-frequency class balancing ─────────────
        node_masked = ((node_t == self.node_mask) if sup_pad
                       else ((node_t == self.node_mask) & pad_mask))
        loss_n = torch.tensor(0.0, device=device)
        if node_masked.any():
            per = F.cross_entropy(
                node_logits[node_masked],
                node_feats[node_masked],
                weight=self.node_class_weight if self.cfg.use_node_class_weights else None,
                reduction="none",
            )
            if sup_pad and self.cfg.pad_loss_weight != 1.0:
                w = torch.where(pad_mask[node_masked], torch.ones_like(per),
                                torch.full_like(per, self.cfg.pad_loss_weight))
                loss_n = (per * w).sum() / w.sum().clamp(min=1e-6)
            else:
                loss_n = per.mean()

        # ── Edge CE — balanced, full matrix (pre-existing) ─────────────────
        diag       = torch.eye(N, device=device, dtype=torch.bool)
        # supervise_pad: every off-diagonal pair counts, so pairs touching a
        # PAD slot are supervised as "no edge" rather than left unconstrained
        edge_masked = ((edge_t == self.edge_mask) & ~diag[None]) if sup_pad else (
            (edge_t == self.edge_mask)
            & valid_pair
            & ~diag[None]
        )

        loss_e = torch.tensor(0.0, device=device)
        if edge_masked.any():
            flat_logits = edge_logits[edge_masked]
            flat_tgts   = adjcs[edge_masked]

            loss_e = F.cross_entropy(flat_logits, flat_tgts, reduction="mean")

        loss = loss_n + loss_edge_weight * loss_e
        return {
            "loss":      loss,
            "loss_node": loss_n.item() if isinstance(loss_n, Tensor) else loss_n,
            "loss_edge": loss_e.item() if isinstance(loss_e, Tensor) else loss_e,
        }

    # ── sampling ─────────────────────────────────────────────────────────────

    def sample(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """Reverse-time generation. Implemented in `graphpath.samplers.masked`."""
        return sample_masked(self, *args, **kwargs)
