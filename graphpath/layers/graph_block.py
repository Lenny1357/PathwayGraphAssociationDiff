"""Interleaved node/edge graph block and the tower that stacks them (FIX 12).

Previously: NodeTransformer ran cfg.n_layers of pure self-attention, with
edges touching nodes exactly ONCE beforehand (gat_pre) and via a shared
edge_bias_proj computed from the static edge-token embedding; then the
fully separate AdjacencyTransformer ran its own cfg.n_layers using only
the FINAL h_node. Nodes and edges were two disjoint towers joined at a
single seam.

Now: one stack of cfg.n_layers GraphBlocks. Each block does, in order:
  1. node self-attention, biased by the CURRENT edge representation `e`
  2. node <- edge message passing (GAT-style gate over the CURRENT `e`)
  3. node FFN
  4. edge <- node pair update (fresh h_i/h_j injected) + row/col attention
  5. edge FFN
So information now flows node->edge and edge->node at EVERY layer, each
direction conditioned on the other's most up-to-date state, rather than
once each at the two ends of the pipeline.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphpath.config import Config
from graphpath.layers.conditioning import GraphConditioner
from graphpath.layers.pairwise_conditioning import NeighbourhoodCrossAttention
from graphpath.layers.graph_transformer import make_block
from graphpath.layers.common import SinusoidalTimeEmbedding, _make_ff, _mha_block


class GraphBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.Hn, self.He = cfg.n_heads, cfg.edge_n_heads
        assert d % self.Hn == 0 and d % self.He == 0
        self.d_kn, self.d_ke = d // self.Hn, d // self.He

        # -- 1. node self-attention (edge-biased) --
        self.node_attn = nn.ModuleDict({
            "W_q": nn.Linear(d, d, bias=False), "W_k": nn.Linear(d, d, bias=False),
            "W_v": nn.Linear(d, d, bias=False), "out": nn.Linear(d, d),
        })
        self.node_attn_norm = nn.LayerNorm(d)
        self.edge_bias_proj = nn.Linear(d, self.Hn, bias=False)  # e -> per-head attn bias, THIS layer

        # -- 2. node <- edge message passing (GAT-style, reads current e) --
        self.msg_gate = nn.Sequential(
            nn.Linear(3 * d + 1, d), nn.SiLU(), nn.Linear(d, d), nn.Sigmoid(),
        )
        self.msg_val  = nn.Linear(d, d, bias=False)
        self.msg_out  = nn.Linear(d, d)
        self.msg_norm = nn.LayerNorm(d)

        # -- 3. node FFN --
        self.node_ff      = _make_ff(d, cfg.d_ff, cfg.dropout)
        self.node_ff_norm = nn.LayerNorm(d)

        # -- 4. edge <- node pair update, then row/col attention --
        self.pair_proj      = nn.Linear(2 * d, d)
        self.pair_proj_norm = nn.LayerNorm(d)
        self.row_attn = nn.ModuleDict({
            "W_q": nn.Linear(d, d, bias=False), "W_k": nn.Linear(d, d, bias=False),
            "W_v": nn.Linear(d, d, bias=False), "out": nn.Linear(d, d),
        })
        self.row_norm = nn.LayerNorm(d)
        self.col_attn = nn.ModuleDict({
            "W_q": nn.Linear(d, d, bias=False), "W_k": nn.Linear(d, d, bias=False),
            "W_v": nn.Linear(d, d, bias=False), "out": nn.Linear(d, d),
        })
        self.col_norm = nn.LayerNorm(d)

        # -- 5. edge FFN --
        self.edge_ff      = _make_ff(d, cfg.d_ff, cfg.dropout)
        self.edge_ff_norm = nn.LayerNorm(d)

        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        h:          Tensor,  # (B, N, d)
        e:          Tensor,  # (B, N, N, d)
        edge_known: Tensor,  # (B, N, N, 1) float — 1 where edge_toks != MASK
        diag_mask:  Tensor,  # (N, N) bool
    ) -> Tuple[Tensor, Tensor]:
        B, N, d = h.shape
        Hn, d_kn = self.Hn, self.d_kn

        # 1. node self-attention biased by the CURRENT edge state
        edge_bias = self.edge_bias_proj(e).permute(0, 3, 1, 2)   # (B, Hn, N, N)
        Q = self.node_attn["W_q"](h).view(B, N, Hn, d_kn).transpose(1, 2)
        K = self.node_attn["W_k"](h).view(B, N, Hn, d_kn).transpose(1, 2)
        V = self.node_attn["W_v"](h).view(B, N, Hn, d_kn).transpose(1, 2)
        content = (Q @ K.transpose(-1, -2)) / math.sqrt(d_kn)
        scores = (content + edge_bias).masked_fill(diag_mask[None, None], -1e4)
        attn = F.softmax(scores, dim=-1)
        if getattr(self, "store_attn", False):
            # kept for interpretability; `content` and `edge_bias` are stored
            # separately so the edge stream's contribution to the attention
            # can be measured rather than inferred
            self.last_attn = attn.detach()
            self.last_content = content.detach()
            self.last_edge_bias = edge_bias.detach()
        out = (attn @ V).transpose(1, 2).reshape(B, N, Hn * d_kn)
        h = self.node_attn_norm(h + self.node_attn["out"](out))

        # 2. node <- edge message passing (GAT-style gate reads the fresh `e`)
        hi = h.unsqueeze(2).expand(-1, -1, N, -1)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)
        gate = self.msg_gate(torch.cat([hi, hj, e, edge_known], dim=-1))
        gate = gate.masked_fill(diag_mask[None, :, :, None], 0.0)
        if getattr(self, "store_attn", False):
            self.last_gate = gate.detach().mean(-1)      # (B, N, N)
        msg  = (gate * self.msg_val(hj)).sum(dim=2)
        h = self.msg_norm(h + self.msg_out(self.drop(msg)))

        # 3. node FFN
        h = self.node_ff_norm(h + self.node_ff(h))

        # 4. edge <- fresh node pair info, then row + column attention
        hi2 = h.unsqueeze(2).expand(-1, -1, N, -1)
        hj2 = h.unsqueeze(1).expand(-1, N, -1, -1)
        e = self.pair_proj_norm(e + self.pair_proj(torch.cat([hi2, hj2], dim=-1)))

        e_row = e.reshape(B * N, N, d)
        out_r = _mha_block(e_row, self.row_attn, self.He, self.d_ke, diag_mask)
        e = self.row_norm(e + out_r.reshape(B, N, N, d))

        e_col = e.transpose(1, 2).reshape(B * N, N, d)
        out_c = _mha_block(e_col, self.col_attn, self.He, self.d_ke, diag_mask)
        e = self.col_norm(e + out_c.reshape(B, N, N, d).transpose(1, 2))

        # 5. edge FFN
        e = self.edge_ff_norm(e + self.edge_ff(e))

        return h, e


class _BaseGraphTower(nn.Module):
    """
    Everything in the tower that does not depend on how the node/edge state
    is represented: time conditioning, random node-identity features, node
    type embedding, the GraphBlock stack, and the final symmetrisation.

    Subclasses supply the input front-end by overriding `_embed_nodes`,
    `_embed_edges` and `_edge_known`. `GraphTower` embeds discrete tokens;
    `ContinuousGraphTower` projects continuous state. The block stack itself
    is identical either way — GraphBlock only ever sees (B,N,d) node states
    and (B,N,N,d) edge states.
    """

    def __init__(self, cfg: Config, gene_dim: int):
        super().__init__()
        d = cfg.d_model
        self.time_emb       = SinusoidalTimeEmbedding(d)
        self.time_proj_node = nn.Linear(d, d)
        self.time_proj_edge = nn.Linear(d, d)

        # FIX 5: random node-identity feature projection
        self.use_rand_feat = cfg.use_random_node_features
        self.rand_feat_dim = cfg.rand_feat_dim
        self.rand_feat_scope = getattr(cfg, "rand_feat_scope", "all")
        if self.rand_feat_scope not in ("all", "placeholders"):
            raise ValueError(f"rand_feat_scope must be 'all' or 'placeholders', "
                             f"got {self.rand_feat_scope!r}")
        self.pad_id = getattr(cfg, "pad_id", 0)
        if self.use_rand_feat:
            self.node_id_proj = nn.Linear(self.rand_feat_dim, d)

        # FIX 11: explicit node-type embedding (+1 row = "type unknown")
        self.n_types = cfg.n_node_types
        if self.n_types > 0:
            self.type_emb = nn.Embedding(self.n_types + 1, d)

        # Co-expression conditioning, injected once here rather than
        # per-layer; identical for every front-end, so all three variants
        # share one implementation. `gene_dim` is the width of whatever
        # per-entity table the model pools over.
        self.node_vocab = cfg.node_vocab
        self.use_coexpr = cfg.use_coexpr
        if self.use_coexpr:
            self.conditioner = GraphConditioner(cfg, gene_dim=gene_dim, d_model=d)

        # nodes attending over their co-expressed partners, applied between
        # blocks so the retrieved context can influence every layer rather
        # than having to survive all of them from the input
        self.use_coexpr_attn = cfg.use_coexpr_attn
        if self.use_coexpr_attn:
            which = {"all": range(cfg.n_layers), "first": [0],
                     "last": [cfg.n_layers - 1]}[cfg.coexpr_attn_layers]
            self.coexpr_attn_at = set(which)
            self.coexpr_attn = nn.ModuleList(
                NeighbourhoodCrossAttention(d, cfg.coexpr_attn_heads, cfg.dropout)
                if i in self.coexpr_attn_at else nn.Identity()
                for i in range(cfg.n_layers))

        self.blocks = nn.ModuleList(make_block(cfg) for _ in range(cfg.n_layers))
        self.drop = nn.Dropout(cfg.dropout)

    def post_init_zeros(self):
        """
        Re-apply the zero-inits that a model's generic trunc_normal sweep
        would otherwise clobber: the per-layer edge attention bias (so
        training starts at the no-edge-bias baseline) and the conditioning
        gate/FiLM (so it starts at the unconditional baseline).
        """
        for block in self.blocks:
            # GraphBlock has a per-layer edge attention bias to zero; other
            # backbones expose their own reset instead
            if hasattr(block, "edge_bias_proj"):
                nn.init.zeros_(block.edge_bias_proj.weight)
            elif hasattr(block, "reset_zeros"):
                block.reset_zeros()
        if self.use_coexpr:
            self.conditioner.zero_init()
        if getattr(self, "use_coexpr_attn", False):
            for m in self.coexpr_attn:
                if hasattr(m, "reset_zeros"):
                    m.reset_zeros()

    def gene_embeddings(self) -> Tensor:
        """Per-entity table the co-expression encoder pools over."""
        raise NotImplementedError

    def encode_condition(self, co_expr: dict, drop_mask: Optional[Tensor] = None) -> Tensor:
        """Pooled (B, d_cond) conditioning vector; call once per sampling run."""
        return self.conditioner.encode(co_expr, self.gene_embeddings(), drop_mask)

    def null_condition(self, co_expr: dict) -> Tensor:
        return self.conditioner.null_pooled(co_expr, self.gene_embeddings())

    # ── front-end hooks ─────────────────────────────────────────────────────

    def _embed_nodes(self, node_in: Tensor) -> Tensor:      # -> (B, N, d)
        raise NotImplementedError

    def _placeholder_mask(self, node_in: Tensor) -> Optional[Tensor]:
        """
        (B, N) bool — True where a slot holds PAD or MASK, i.e. carries no
        identity of its own. None means "cannot tell", and the caller then
        falls back to tagging every slot.
        """
        return None

    def _embed_edges(self, edge_in: Tensor) -> Tensor:      # -> (B, N, N, d)
        raise NotImplementedError

    def _edge_known(self, edge_in: Tensor) -> Tensor:       # -> (B, N, N, 1)
        raise NotImplementedError

    # ── shared body ─────────────────────────────────────────────────────────

    def forward(
        self,
        node_in:      Tensor,                  # tokens (B,N) or state (B,N,*)
        edge_in:      Tensor,                  # tokens (B,N,N) or state (B,N,N,*)
        t_node:       Tensor,                  # (B,) long
        t_edge:       Tensor,                  # (B,) long
        node_id_feat: Optional[Tensor] = None,
        type_ids:     Optional[Tensor] = None,
        co_expr:      Optional[dict]   = None,
        cond_drop:    Optional[Tensor] = None,
        cond_pooled:  Optional[Tensor] = None,
        nbr_emb:      Optional[Tensor] = None,   # (M, k, d) partner embeddings
        nbr_w:        Optional[Tensor] = None,   # (M, k) co-expression strengths
        nbr_slots:    Optional[Tensor] = None,   # (M,) flat slot index into B*N
    ) -> Tuple[Tensor, Tensor]:
        h = self._embed_nodes(node_in)
        B, N, _ = h.shape

        h = h + self.time_proj_node(self.time_emb(t_node))[:, None, :]
        if self.n_types > 0 and type_ids is not None:
            h = h + self.type_emb(type_ids)
        if self.use_rand_feat:
            if node_id_feat is None:
                node_id_feat = torch.randn(B, N, self.rand_feat_dim, device=h.device, dtype=h.dtype)
            rf = self.node_id_proj(node_id_feat)
            if self.rand_feat_scope == "placeholders":
                # Only tag the slots that are actually indistinguishable. A
                # slot holding a real token already has an identity; adding
                # noise to it only competes with the signal the node head has
                # to read back out.
                ph = self._placeholder_mask(node_in)
                if ph is not None:
                    rf = rf * ph.unsqueeze(-1).to(rf.dtype)
            h = h + rf

        # Co-expression conditioning: additive gate + FiLM, applied to the
        # assembled input state before the block stack runs.
        if self.use_coexpr:
            if cond_pooled is None and co_expr is not None:
                cond_pooled = self.encode_condition(co_expr, drop_mask=cond_drop)
            if cond_pooled is not None:
                h = self.conditioner.apply(h, cond_pooled)

        h = self.drop(h)

        e = self._embed_edges(edge_in)
        e = e + self.time_proj_edge(self.time_emb(t_edge))[:, None, None, :]
        e = self.drop(e)

        edge_known = self._edge_known(edge_in)                              # (B,N,N,1)
        diag_mask  = torch.eye(N, device=h.device, dtype=torch.bool)

        t_emb = self.time_proj_node(self.time_emb(t_node))
        for i, block in enumerate(self.blocks):
            if (self.use_coexpr_attn and nbr_emb is not None
                    and i in self.coexpr_attn_at):
                h = self.coexpr_attn[i](h, nbr_emb, nbr_w, nbr_slots)
            h, e = (block(h, e, edge_known, diag_mask, t_emb)
                    if getattr(block, "use_time_film", False)
                    else block(h, e, edge_known, diag_mask))

        e = 0.5 * (e + e.transpose(1, 2))   # keep the final edge repr symmetric
        return h, e


class GraphTower(_BaseGraphTower):
    """Discrete front-end: node/edge token embeddings + the block stack."""

    def __init__(self, cfg: Config):
        super().__init__(cfg, gene_dim=cfg.d_model)
        d = cfg.d_model
        self.tok_emb      = nn.Embedding(cfg.node_vocab + 1, d)
        self.edge_tok_emb = nn.Embedding(cfg.edge_vocab + 1, d)
        self.edge_mask_id = cfg.edge_vocab

    def gene_embeddings(self) -> Tensor:
        # the same table used for node_toks, minus the MASK row
        return self.tok_emb.weight[: self.node_vocab]

    def _embed_nodes(self, node_toks: Tensor) -> Tensor:
        return self.tok_emb(node_toks)

    def _placeholder_mask(self, node_toks: Tensor) -> Optional[Tensor]:
        # node_vocab is the MASK row of tok_emb. Under uniform-transition
        # diffusion no token ever equals it, so only PAD matches there —
        # which is the correct behaviour, not an oversight.
        return (node_toks == self.pad_id) | (node_toks == self.node_vocab)

    def _embed_edges(self, edge_toks: Tensor) -> Tensor:
        return self.edge_tok_emb(edge_toks)

    def _edge_known(self, edge_toks: Tensor) -> Tensor:
        # 1 where the edge token is a real class rather than MASK. Constant 1
        # under uniform-transition diffusion, which has no MASK token.
        return (edge_toks != self.edge_mask_id).float().unsqueeze(-1)


class ContinuousGraphTower(_BaseGraphTower):
    """
    Continuous front-end: linear projections of real-valued node state
    (B, N, node_in_dim) and edge state (B, N, N, edge_in_dim) into the same
    (d_model) space the GraphBlocks operate in.

    Nothing is ever "observed" during Gaussian diffusion, so `edge_known` is
    a constant 1 — the noise level is communicated through the time
    embedding instead.
    """

    def __init__(self, cfg: Config, node_in_dim: int, edge_in_dim: int = 2,
                 gene_embed_table: Optional[nn.Module] = None):
        # There is no token embedding table here, so the co-expression
        # encoder pools over the model's UnifiedEmbeddingModule table
        # instead; the owning model passes it in.
        super().__init__(cfg, gene_dim=cfg.embed_dim)
        d = cfg.d_model
        self.node_in_proj = nn.Sequential(nn.Linear(node_in_dim, d), nn.LayerNorm(d))
        self.edge_in_proj = nn.Sequential(nn.Linear(edge_in_dim, d), nn.LayerNorm(d))
        self._gene_embed_table = [gene_embed_table]   # list => not a submodule, no duplicate params

    def gene_embeddings(self) -> Tensor:
        table = self._gene_embed_table[0]
        if table is None:
            raise RuntimeError(
                "ContinuousGraphTower needs the model's entity embedding table "
                "to encode co-expression; pass gene_embed_table=..."
            )
        return table.weight[: self.node_vocab]

    def _embed_nodes(self, node_state: Tensor) -> Tensor:
        return self.node_in_proj(node_state)

    def _embed_edges(self, edge_state: Tensor) -> Tensor:
        return self.edge_in_proj(edge_state)

    def _edge_known(self, edge_state: Tensor) -> Tensor:
        return torch.ones(*edge_state.shape[:-1], 1,
                          device=edge_state.device, dtype=edge_state.dtype)
