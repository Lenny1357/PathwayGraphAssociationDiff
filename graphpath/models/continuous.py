"""ContinuousGraphDiffusion — joint Gaussian diffusion over node embeddings
and adjacency, decoded back to tokens.

Contrast with the two discrete variants:
  GraphDiffBERT         absorbing-mask on node TOKENS; edges are 2-class
  GraphDiffBERTUniform  uniform transition on node TOKENS; edges are 2-class
  this                  Gaussian noise on node EMBEDDINGS and on a
                        real-valued adjacency; tokens are recovered by
                        decoding x̂₀ at the end. Self-conditioning on both
                        streams (p = cfg.self_cond_p during training),
                        deterministic DDIM reverse steps.

All three expose the same interface — compute_loss(node_feats, adjcs,
pad_mask, loss_edge_weight) and sample(N, B, ...) — and are driven by the
same train_step()/train() in graphpath/training/loop.py.
"""

import random
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphpath.layers.graph_block import ContinuousGraphTower
from graphpath.config import Config
from graphpath.layers.continuous_modules import AdjacencyHead, NodeDenoiser
from graphpath.processes.gaussian import ContinuousCosineSchedule
from graphpath.layers.embedding import UnifiedEmbeddingModule
from graphpath.layers.common import FiLMCondition, SinusoidalTimeEmbedding
from graphpath.layers.node_types import NodeTypeMixin
from graphpath.samplers.continuous import sample_continuous
from graphpath.training.class_weights import inverse_freq_weights


class ContinuousGraphDiffusion(NodeTypeMixin, nn.Module):

    def __init__(
        self,
        cfg:        Config,
        vocab_size: int,
        backend:    str = "tower",          # "tower" | "separate"
        schedule:   Optional[nn.Module] = None,
        embedding:  Optional[nn.Module] = None,
        node_model: Optional[nn.Module] = None,   # backend="separate" only
        adj_model:  Optional[nn.Module] = None,   # backend="separate" only
    ):
        """
        vocab_size = dataset.vocab_size + 1   (API compat with the discrete
        models — the +1 MASK slot is unused here, so the embedding table has
        vocab_size - 1 rows).

        backend="tower"    — the interleaved GraphBlock stack shared with the
                             discrete models, fed continuous state instead of
                             token embeddings. Node and edge streams inform
                             each other at every layer, and node types are
                             available to the tower.
        backend="separate" — NodeDenoiser (edge-blind) followed by
                             AdjacencyHead. Cheaper, no edge state.

        Every component is injectable; anything left as None is built from cfg.
        """
        super().__init__()
        if backend not in ("tower", "separate"):
            raise ValueError(f"backend must be 'tower' or 'separate', got {backend!r}")

        self.cfg        = cfg
        self.vocab_size = vocab_size
        self.node_vocab = vocab_size - 1
        self.edge_vocab = cfg.edge_vocab      # only used to shape the type prior
        self.backend    = backend
        cfg.node_vocab  = self.node_vocab

        self.schedule = schedule if schedule is not None else \
            ContinuousCosineSchedule(cfg.T)

        self.embedding = embedding if embedding is not None else \
            UnifiedEmbeddingModule(
                vocab_size     = self.node_vocab,
                embed_dim      = cfg.embed_dim,
                bottleneck_dim = cfg.bottleneck_dim,
            )

        bn = cfg.bottleneck_dim
        if cfg.use_coexpr and backend != "tower":
            raise ValueError(
                "co-expression conditioning is injected at the graph tower's "
                "input; use backend='tower' (the default) or set use_coexpr=False"
            )

        if backend == "tower":
            # node state in  = [noisy ‖ prev (self-cond) ‖ cond]
            # edge state in  = [adj_t, prev_adj]
            self.tower = ContinuousGraphTower(cfg, node_in_dim=3 * bn, edge_in_dim=2,
                                              gene_embed_table=self.embedding.embed)
            d = cfg.d_model
            self.node_out = nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, bn), nn.LayerNorm(bn),
            )
            self.edge_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            self.denoiser = self.adj_head = None
        else:
            time_emb      = SinusoidalTimeEmbedding(cfg.d_model)
            self.denoiser = node_model if node_model is not None else NodeDenoiser(time_emb, cfg)
            self.adj_head = adj_model  if adj_model  is not None else AdjacencyHead(time_emb, cfg)
            self.tower    = None

        # FIX 7 — optional per-class weights for the token CE term.
        self.register_buffer("node_class_weight", torch.ones(self.node_vocab))

        # FIX 11 — typed connectivity. There is no token to read a type from
        # mid-trajectory here, so types are ESTIMATED by decoding the current
        # noisy state; see _estimate_type_ids.
        self._setup_node_types(cfg, vocab_size, unknown_is_mask=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

        # Zero-init the per-layer edge attention bias and the conditioning
        # gate/FiLM (tower backend only).
        if self.tower is not None:
            self.tower.post_init_zeros()

        # FiLM is zero-init by construction; re-zero it after the sweep above
        # so it still starts as a plain LayerNorm.
        for m in self.modules():
            if isinstance(m, FiLMCondition):
                m.zero_init()

    def set_node_class_weights(self, counts, power: Optional[float] = None):
        """See GraphDiffBERT.set_node_class_weights — same contract."""
        power = self.cfg.node_class_weight_power if power is None else power
        w = inverse_freq_weights(counts, power, self.node_vocab,
                                 self.node_class_weight.device)
        self.node_class_weight.copy_(w)

    # ── node types in embedding space ───────────────────────────────────────

    @torch.no_grad()
    def _estimate_type_ids(self, embeds: Tensor) -> Optional[Tensor]:
        """
        The discrete variants read a node's type straight off its token. Here
        the node state is a continuous vector, so we decode it and take the
        type of the most likely token — falling back to the reserved "unknown
        type" slot wherever the decoder is less confident than
        cfg.type_conf_thresh.

        That confidence gate is what makes this behave like the absorbing
        model's MASK -> "unknown" mapping: at high noise the decode is
        uncertain, so the type signal switches itself off instead of feeding
        the tower random types. Crucially this reads the NOISY state, never
        the clean tokens, so training and sampling see the same distribution
        and no label information leaks into the CE target.
        """
        if self.n_types == 0:
            return None
        probs, idx = self.embedding.decode(embeds).exp().max(dim=-1)
        types = self.token_to_type[idx]
        return torch.where(probs >= self.cfg.type_conf_thresh, types,
                           torch.full_like(types, self.n_types))

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        noisy_embeds: Tensor,           # (B, N, bottleneck_dim)
        adj_t:        Tensor,           # (B, N, N) float
        t:            Tensor,           # (B,) int
        prev_embeds:  Optional[Tensor] = None,
        prev_adj:     Optional[Tensor] = None,
        node_id_feat: Optional[Tensor] = None,
        type_ids:     Optional[Tensor] = None,
        co_expr:      Optional[dict]   = None,   # co-expression conditioning
        cond_drop:    Optional[Tensor] = None,
        cond_pooled:  Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns
        ───────
        embeds_hat  : (B, N, bottleneck_dim)  predicted x̂_0 in embed space
        adj_logits  : (B, N, N)               edge logits (pre-sigmoid)
        """
        B, N, _ = noisy_embeds.shape

        if prev_embeds is None:
            prev_embeds = torch.zeros_like(noisy_embeds)
        if prev_adj is None:
            prev_adj = torch.zeros_like(adj_t)

        cond_embeds = torch.zeros_like(noisy_embeds)  # external conditioning placeholder

        if type_ids is None:
            type_ids = self._estimate_type_ids(noisy_embeds)

        if self.backend == "tower":
            node_state = torch.cat([noisy_embeds, prev_embeds, cond_embeds], dim=-1)
            edge_state = torch.stack([adj_t, prev_adj], dim=-1)     # (B,N,N,2)
            h, e = self.tower(node_state, edge_state, t, t,
                              node_id_feat=node_id_feat, type_ids=type_ids,
                              co_expr=co_expr, cond_drop=cond_drop,
                              cond_pooled=cond_pooled)
            embeds_hat = self.node_out(h)                           # (B,N,bn)
            adj_logits = self.edge_out(e).squeeze(-1)               # (B,N,N)
        else:
            embeds_hat = self.denoiser(
                noisy_embeds = noisy_embeds,
                adj_t        = adj_t,
                t            = t,
                prev_embeds  = prev_embeds,
                cond_embeds  = cond_embeds,
            )
            adj_logits = self.adj_head(embeds_hat, t, prev_adj=prev_adj)

        # FIX 11: learned per-type-pair prior, as a single binary logit.
        adj_logits = self._add_type_pair_bias_binary(adj_logits, type_ids)

        # Symmetrise and suppress self-loops (the separate backend already
        # does both internally; doing it here keeps the two paths identical).
        diag = torch.eye(N, device=adj_logits.device, dtype=torch.bool)
        adj_logits = 0.5 * (adj_logits + adj_logits.transpose(-1, -2))
        adj_logits = adj_logits.masked_fill(diag[None], -1e4)
        return embeds_hat, adj_logits

    # ── loss ─────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        node_feats:       Tensor,       # (B, N)    long  — clean node ids
        adjcs:            Tensor,       # (B, N, N) long  — clean binary adj
        pad_mask:         Tensor,       # (B, N)    bool   True=valid
        loss_edge_weight: float = 1.0,
        co_expr:          Optional[dict] = None,
    ) -> dict:
        """
        Three terms, all restricted to valid (non-padding) positions:
          • MSE between x̂₀ and the clean embedding
          • CE after decoding x̂₀ back to tokens
          • BCE on the adjacency logits, pos_weight'd because graphs are sparse
        """
        B, N   = node_feats.shape
        device = node_feats.device

        t = torch.randint(0, self.cfg.T, (B,), device=device)

        # ── embed clean nodes ─────────────────────────────────────────────
        embeds_0 = self.embedding.encode(node_feats)          # (B,N,bn)

        # ── forward diffusion ─────────────────────────────────────────────
        embeds_t = self.schedule.q_sample(embeds_0, t)
        adj_f    = adjcs.float()
        adj_t    = self.schedule.q_sample(adj_f, t)

        # ── self-conditioning ─────────────────────────────────────────────
        # CFG-style conditioning dropout (see GraphConditioner). Drawn once
        # and reused for the self-conditioning pass so both see the same
        # condition.
        cond_drop = None
        if self.tower is not None and self.tower.use_coexpr and co_expr is not None \
                and self.training:
            cond_drop = self.tower.conditioner.train_drop_mask(B, device)

        prev_embeds = torch.zeros_like(embeds_t)
        prev_adj    = torch.zeros_like(adj_t)
        if random.random() < self.cfg.self_cond_p:
            with torch.no_grad():
                prev_embeds, prev_adj_logits = self.forward(
                    embeds_t, adj_t, t, co_expr=co_expr, cond_drop=cond_drop)
                prev_adj = torch.sigmoid(prev_adj_logits)

        # ── predict x̂_0 ──────────────────────────────────────────────────
        embeds_hat, adj_logits = self.forward(
            embeds_t, adj_t, t, prev_embeds, prev_adj,
            co_expr=co_expr, cond_drop=cond_drop,
        )

        # ── node MSE loss (valid positions only) ──────────────────────────
        node_w   = pad_mask.float().unsqueeze(-1)                 # (B,N,1)
        n_valid  = pad_mask.float().sum().clamp(min=1)
        loss_mse = ((embeds_hat - embeds_0).pow(2).mean(-1, keepdim=True)
                    * node_w).sum() / n_valid

        # ── node CE after decoding back to tokens (valid positions only) ──
        logits_hat = self.embedding.decode(embeds_hat)            # (B,N,V) log-probs
        node_ce = F.cross_entropy(
            logits_hat.reshape(B * N, self.node_vocab),
            node_feats.reshape(B * N),
            weight=self.node_class_weight if self.cfg.use_node_class_weights else None,
            reduction="none",
        ).reshape(B, N)
        cross_entropy = (node_ce * pad_mask.float()).sum() / n_valid

        # ── adjacency BCE loss (valid off-diagonal pairs) ─────────────────
        diag    = torch.eye(N, device=device, dtype=torch.bool)
        vp      = (pad_mask[:, :, None] & pad_mask[:, None, :]) & ~diag[None]
        edge_ce = F.binary_cross_entropy_with_logits(
            adj_logits,
            adj_f,
            pos_weight=torch.tensor(self.cfg.adj_pos_weight, device=device),
            reduction="none",
        )
        m_valid  = vp.float().sum().clamp(min=1)
        loss_adj = (edge_ce * vp.float()).sum() / m_valid

        loss = (self.cfg.loss_mse_weight * loss_mse
                + self.cfg.loss_ce_weight * cross_entropy
                + loss_edge_weight * self.cfg.loss_adj_weight * loss_adj)

        return {
            "loss":          loss,
            "loss_node_mse": loss_mse.item(),
            "loss_node_ce":  cross_entropy.item(),
            "loss_edge":     loss_adj.item(),
        }

    # ── sampling ─────────────────────────────────────────────────────────────

    def sample(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """Reverse-time generation. Implemented in `graphpath.samplers.continuous`."""
        return sample_continuous(self, *args, **kwargs)
