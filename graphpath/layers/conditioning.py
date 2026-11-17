"""Co-expression conditioning, shared by all three diffusion variants.

Pathway generation is conditioned on a tissue's gene co-expression graph: a
sparse ``(n_tissues, V, V)`` tensor of pairwise co-expression strengths over
the node vocabulary. The signal is reduced to a single pooled vector per
sample and injected ONCE at the tower's input (additive gate + FiLM), never
per-layer — it still reaches every layer, but by propagating through
self-attention and message passing like any other input feature (type
embedding, identity features), so GraphBlock stays untouched.

Because the injection point is the shared `_BaseGraphTower` front-end, the
same conditioner works for token-based and continuous node state alike.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from graphpath.config import Config
from graphpath.layers.common import FiLMCondition


class CoExpressionEncoder(nn.Module):
    """
    Streamlined co-expression encoder, traded for training speed against a
    cross-attention + pairwise-lookup design. Three deliberate simplifications:

      1. Only touches the tissues actually present in a batch. The sparse
         (n_tissues, V, V) graph is partitioned by tissue ONCE (cached — see
         _ensure_split) rather than being re-scanned in full on every
         forward call; a batch typically touches a handful of tissues out of
         possibly dozens, so this alone cuts most of the wasted work.
      2. ONE round of degree-weighted mean pooling over each tissue's top-k
         hub genes — no iterative multi-round message passing.
      3. Produces a SINGLE pooled (B, d_cond) vector per sample, not a
         V-sized memory bank — so there is nothing for a per-layer
         cross-attention to attend over, and none is used.

    Trade-off, stated plainly: this discards per-gene granularity (no memory
    bank, no exact pairwise gene-pair lookup) in exchange for a coarse
    per-tissue summary.

    Gene representations are supplied by the caller rather than owned here,
    so the pooled summary lives in whatever space the model already uses for
    entities: the node token embedding table for the discrete variants, the
    UnifiedEmbeddingModule table for the continuous one.
    """

    def __init__(self, cfg: Config, gene_dim: int):
        super().__init__()
        d = cfg.cond_d_model
        self.d = d
        self.n_tissues = cfg.n_tissues
        self.V = cfg.node_vocab
        self.top_k_hubs = cfg.top_k_hubs

        self.node_proj = nn.Linear(gene_dim, d)
        self.null_embed = nn.Parameter(torch.empty(1, d))
        nn.init.trunc_normal_(self.null_embed, std=0.02)

        self._cache_id = None
        self._per_tissue_edges = None   # list[(src, tgt, val)] per tissue, built lazily once

    def _ensure_split(self, coexpr_sparse: Tensor):
        """Partition the sparse graph by tissue once; reused every call
        thereafter as long as the same tensor object is passed in (true for
        a whole training run, since the co-expression graph is built once)."""
        if self._cache_id == id(coexpr_sparse) and self._per_tissue_edges is not None:
            return
        sp = coexpr_sparse.coalesce()
        idx = sp.indices()
        val = sp.values().float().abs()
        t_e, src_e, tgt_e = idx[0], idx[1], idx[2]

        order = torch.argsort(t_e)
        t_sorted = t_e[order]
        boundaries = torch.searchsorted(
            t_sorted, torch.arange(self.n_tissues + 1, device=t_e.device)
        )
        per_tissue = []
        for t in range(self.n_tissues):
            s, e = boundaries[t].item(), boundaries[t + 1].item()
            sel = order[s:e]
            per_tissue.append((src_e[sel], tgt_e[sel], val[sel]))
        self._per_tissue_edges = per_tissue
        self._cache_id = id(coexpr_sparse)

    def _pool_one_tissue(self, t: int, base: Tensor, device: torch.device) -> Tensor:
        src, tgt, val = self._per_tissue_edges[t]
        if src.numel() == 0:
            return self.null_embed[0]

        deg = torch.zeros(self.V, device=device)
        deg.index_add_(0, src, val)
        deg.index_add_(0, tgt, val)

        if self.top_k_hubs is not None:
            k = min(self.top_k_hubs, int((deg > 0).sum().item()))
            if k == 0:
                return self.null_embed[0]
            topk_val, topk_idx = deg.topk(k)
            w = topk_val / topk_val.sum().clamp(min=1e-6)
            return (base[topk_idx] * w.unsqueeze(-1)).sum(dim=0)
        else:
            total = deg.sum().clamp(min=1e-6)
            if total <= 1e-6:
                return self.null_embed[0]
            w = deg / total
            return (base * w.unsqueeze(-1)).sum(dim=0)

    def forward(
        self,
        gene_embeds:   Tensor,             # (V, gene_dim)
        coexpr_sparse: Optional[Tensor],   # sparse_coo (n_tissues, V, V), or None
        tissue_ids:    Tensor,             # (B,) long
        drop_mask:     Optional[Tensor] = None,
    ) -> Tensor:                           # (B, d_cond)
        device = tissue_ids.device
        B = tissue_ids.shape[0]
        d = self.d

        if coexpr_sparse is None:
            return self.null_embed.expand(B, d)

        self._ensure_split(coexpr_sparse)
        base = self.node_proj(gene_embeds)   # (V, d_cond)

        unique_t = torch.unique(tissue_ids).tolist()
        tissue_vecs = {t: self._pool_one_tissue(t, base, device) for t in unique_t}
        pooled = torch.stack([tissue_vecs[t] for t in tissue_ids.tolist()], dim=0)

        if drop_mask is not None and drop_mask.any():
            pooled = torch.where(drop_mask[:, None], self.null_embed.expand(B, d), pooled)

        return pooled


class GraphConditioner(nn.Module):
    """
    Encoder + injection, kept together so a tower only has to call
    `encode(...)` once and `apply(h, pooled)` per forward.

    Injection is additive (a learned gate) plus FiLM, both zero-initialised
    so an untrained model starts exactly at the unconditional baseline and
    learns how far to move away from it.
    """

    def __init__(self, cfg: Config, gene_dim: int, d_model: int):
        super().__init__()
        self.encoder = CoExpressionEncoder(cfg, gene_dim)
        self.gate = nn.Linear(cfg.cond_d_model, d_model)
        self.film = FiLMCondition(cfg.cond_d_model, d_model, normalize=False)
        self.cond_dropout_p = cfg.cond_dropout_p
        self.zero_init()

    def zero_init(self):
        """Re-applied after a model's generic weight-init sweep."""
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.film.zero_init()

    def encode(self, co_expr: dict, gene_embeds: Tensor,
               drop_mask: Optional[Tensor] = None) -> Tensor:
        """
        `co_expr` is a dict with:
            "coexpr_sparse" : sparse_coo (n_tissues, V, V), or None
            "tissue_ids"    : (B,) long
        Returns the pooled (B, d_cond) conditioning vector. Split out from
        the forward pass so samplers can compute it ONCE before their T-step
        loop instead of once per step.
        """
        return self.encoder(
            gene_embeds   = gene_embeds,
            coexpr_sparse = co_expr.get("coexpr_sparse"),
            tissue_ids    = co_expr["tissue_ids"],
            drop_mask     = drop_mask,
        )

    def null_pooled(self, co_expr: dict, gene_embeds: Tensor) -> Tensor:
        """The fully-unconditional counterpart, for the CFG-blind pass."""
        B = co_expr["tissue_ids"].shape[0]
        drop = torch.ones(B, dtype=torch.bool, device=co_expr["tissue_ids"].device)
        return self.encode(co_expr, gene_embeds, drop_mask=drop)

    def train_drop_mask(self, B: int, device) -> Tensor:
        """
        CFG-style conditioning dropout: a fraction of training samples see
        the null vector instead of their tissue's, so the model also learns
        the unconditional distribution that guidance needs at sampling time.
        """
        return torch.rand(B, device=device) < self.cond_dropout_p

    def apply(self, h: Tensor, pooled: Tensor) -> Tensor:
        h = h + self.gate(pooled)[:, None, :]
        return self.film(h, pooled)
