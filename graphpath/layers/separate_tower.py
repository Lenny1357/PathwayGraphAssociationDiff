"""Sequential node-then-edge tower — the v7 architecture, as a GraphTower.

`GraphTower` interleaves the two streams: node and edge states update each
other at every layer. This one is the alternative that predates it and is
still in use in a sibling project:

    h_node = NodeTransformer(node_toks, edge_toks, t)   # edges enter only as
                                                        # an additive bias on
                                                        # the attention score
    h_edge = AdjacencyTransformer(edge_toks, h_node, t) # edges then read the
                                                        # FINISHED node states

so information flows nodes -> edges once, rather than both ways at every
layer. Whether that matters is an empirical question, which is the point of
having both behind one interface: `GraphDiffBERTUniform(cfg, V, tower=...)`
swaps them without touching anything downstream.

Edges reach the node stream ONCE here — through `EdgeGATAggregation` at the
input — against GraphTower's two injection points in every block (an attention
bias plus a gated message pass). That is the substantive difference, and it
matters most for uniform-transition diffusion, where the edge stream is
corrupted to ~50% density and the `edge_known` flag that would let the gate
discount it is constant.

The wrapper honours `cfg.use_random_node_features` and `cfg.n_node_types` so
the remaining differences from GraphTower are the topology. Set both off to
get the sibling project's model exactly:

    Config(..., use_random_node_features=False, n_node_types=0,
           use_type_bias=False, d_ff=256)

Conditioning (pooled co-expression, pairwise bias, neighbourhood attention)
is NOT implemented here and raises rather than being silently ignored.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphpath.config import Config
from graphpath.layers.common import SinusoidalTimeEmbedding


def _make_ff(d: int, d_ff: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(d_ff, d), nn.Dropout(dropout),
    )


class EdgeGATAggregation(nn.Module):
    """
    One round of edge-gated message passing, run once at the input:

        h_i <- LayerNorm(h_i + W_out( sum_j gate(h_i, h_j, e_ij, known_ij)
                                      * W_msg(h_j) ))

    Same formulation as GraphBlock's step 2 — the difference is WHERE it runs.
    GraphBlock does this inside every block, on an edge state that has itself
    been updated by the preceding blocks; here it happens once, on the raw
    edge token embedding, and the layer stack afterwards is pure node
    self-attention.

    `edge_known` is (edge_toks != edge_vocab), i.e. "not MASK". Under
    uniform-transition diffusion no token ever equals that, so the flag is
    constant 1 and the gate has no way to tell a corrupted edge from a real
    one. That is faithful to the sibling project, and worth knowing when
    reading a result from this backbone.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.mask_id = cfg.edge_vocab
        self.edge_emb = nn.Embedding(cfg.edge_vocab + 1, d)
        self.gate = nn.Sequential(
            nn.Linear(3 * d + 1, d), nn.SiLU(), nn.Linear(d, d), nn.Sigmoid())
        self.W_msg = nn.Linear(d, d, bias=False)
        self.W_out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h: Tensor, edge_toks: Tensor) -> Tensor:
        B, N, d = h.shape
        e_emb = self.edge_emb(edge_toks)                       # (B,N,N,d)
        hi = h.unsqueeze(2).expand(-1, -1, N, -1)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)
        known = (edge_toks != self.mask_id).to(h.dtype).unsqueeze(-1)
        gate = self.gate(torch.cat([hi, hj, e_emb, known], dim=-1))
        diag = torch.eye(N, device=h.device, dtype=torch.bool)
        gate = gate.masked_fill(diag[None, :, :, None], 0.0)
        agg = (gate * self.W_msg(hj)).sum(dim=2)               # (B,N,d)
        return self.norm(h + self.W_out(self.drop(agg)))


class _NodeAttnLayer(nn.Module):
    """
    One layer of the node stream: self-attention with the edge embedding
    added to the score, then a feed-forward — both post-norm.

    Written as a module rather than inline so `benchmarks.capture_attention`
    can switch on `store_attn` and read the same three tensors it reads from
    a GraphBlock: the post-softmax attention, the Q.K term alone, and the
    edge stream's additive contribution.
    """

    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.H, self.d_k = n_heads, d // n_heads
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.ff = _make_ff(d, d_ff, dropout)
        self.ff_norm = nn.LayerNorm(d)
        self.store_attn = False

    def forward(self, h: Tensor, adj_bias: Optional[Tensor],
                diag_mask: Tensor) -> Tensor:
        B, N, d = h.shape
        H, d_k = self.H, self.d_k
        q = self.W_q(h).view(B, N, H, d_k).transpose(1, 2)
        k = self.W_k(h).view(B, N, H, d_k).transpose(1, 2)
        v = self.W_v(h).view(B, N, H, d_k).transpose(1, 2)

        content = (q @ k.transpose(-1, -2)) / math.sqrt(d_k)   # (B,H,N,N)
        scores = content if adj_bias is None else content + adj_bias
        scores = scores.masked_fill(diag_mask[None, None], -1e4)
        attn = F.softmax(scores, dim=-1)

        if self.store_attn:
            self.last_attn = attn.detach()
            self.last_content = content.detach()
            if adj_bias is not None:
                self.last_edge_bias = adj_bias.detach().expand_as(content)

        o = (attn @ v).transpose(1, 2).reshape(B, N, H * d_k)
        h = self.norm(h + self.out(o))
        return self.ff_norm(h + self.ff(h))


class _AxialLayer(nn.Module):
    """Row or column attention over a (B, N, N, d) edge state."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.H, self.d_k = n_heads, d // n_heads
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d)

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        shape = x.shape
        S, d = shape[-2], shape[-1]
        Beff = x.numel() // (S * d)
        xf = x.reshape(Beff, S, d)
        H, d_k = self.H, self.d_k
        q = self.W_q(xf).view(Beff, S, H, d_k).transpose(1, 2)
        k = self.W_k(xf).view(Beff, S, H, d_k).transpose(1, 2)
        v = self.W_v(xf).view(Beff, S, H, d_k).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            s = s.masked_fill(mask[None, None], -1e4)
        o = (F.softmax(s, dim=-1) @ v).transpose(1, 2).reshape(Beff, S, d)
        return self.out(o).reshape(shape)


class NodeTransformer(nn.Module):
    """Node self-attention, with edges entering only as an attention bias."""

    def __init__(self, cfg: Config, node_vocab: int):
        super().__init__()
        d = cfg.d_model
        self.tok_emb = nn.Embedding(node_vocab + 1, d)   # +1 keeps the table
        self.node_vocab = node_vocab                     # shape-compatible
        self.time_emb = SinusoidalTimeEmbedding(d)       # with GraphTower
        self.time_proj = nn.Linear(d, d)
        # One round of edge-gated message passing at the input, as in the
        # sibling project. This is the ONLY place edges reach the node stream
        # there, unless sep_attn_edge_bias is turned on.
        self.gat_pre = EdgeGATAggregation(cfg) if cfg.sep_gat_pre else None
        self.use_attn_edge_bias = cfg.sep_attn_edge_bias
        self.edge_emb = nn.Embedding(cfg.edge_vocab + 1, d)
        self.edge_bias_proj = nn.Linear(d, cfg.n_heads, bias=False)
        self.layers = nn.ModuleList(
            _NodeAttnLayer(d, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, node_toks: Tensor, edge_toks: Tensor, t: Tensor,
                extra: Optional[Tensor] = None) -> Tensor:
        B, N = node_toks.shape
        h = self.tok_emb(node_toks)
        h = h + self.time_proj(self.time_emb(t))[:, None, :]
        if extra is not None:
            h = h + extra
        h = self.drop(h)
        if self.gat_pre is not None:
            h = self.gat_pre(h, edge_toks)

        adj_bias = None
        if self.use_attn_edge_bias:
            adj_bias = self.edge_bias_proj(self.edge_emb(edge_toks))  # (B,N,N,H)
            adj_bias = adj_bias.permute(0, 3, 1, 2)                   # (B,H,N,N)
        diag = torch.eye(N, device=h.device, dtype=torch.bool)
        for layer in self.layers:
            h = layer(h, adj_bias, diag)
        return h


class AdjacencyTransformer(nn.Module):
    """Alternating row/column attention over the edge state."""

    def __init__(self, cfg: Config):
        super().__init__()
        d, H = cfg.d_model, cfg.edge_n_heads
        assert d % H == 0
        self.tok_emb = nn.Embedding(cfg.edge_vocab + 1, d)
        self.pair_proj = nn.Linear(2 * d, d)
        self.time_emb = SinusoidalTimeEmbedding(d)
        self.time_proj = nn.Linear(d, d)
        self.row = nn.ModuleList(_AxialLayer(d, H) for _ in range(cfg.n_layers))
        self.col = nn.ModuleList(_AxialLayer(d, H) for _ in range(cfg.n_layers))
        self.row_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(cfg.n_layers))
        self.col_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(cfg.n_layers))
        self.ff = nn.ModuleList(_make_ff(d, cfg.d_ff, cfg.dropout)
                                for _ in range(cfg.n_layers))
        self.ff_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(cfg.n_layers))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, edge_toks: Tensor, h_nodes: Tensor, t: Tensor) -> Tensor:
        B, N, _ = edge_toks.shape
        d = h_nodes.shape[-1]
        e = self.tok_emb(edge_toks)
        hi = h_nodes.unsqueeze(2).expand(-1, -1, N, -1)
        hj = h_nodes.unsqueeze(1).expand(-1, N, -1, -1)
        e = e + self.pair_proj(torch.cat([hi, hj], dim=-1))
        e = e + self.time_proj(self.time_emb(t))[:, None, None, :]
        e = self.drop(e)

        diag = torch.eye(N, device=e.device, dtype=torch.bool)
        for i in range(len(self.row)):
            e = self.row_norm[i](e + self.row[i](e.reshape(B * N, N, d), diag)
                                 .reshape(B, N, N, d))
            e_col = e.transpose(1, 2).reshape(B * N, N, d)
            e = self.col_norm[i](e + self.col[i](e_col, diag)
                                 .reshape(B, N, N, d).transpose(1, 2))
            e = self.ff_norm[i](e + self.ff[i](e))
        return 0.5 * (e + e.transpose(1, 2))      # undirected inductive bias


class SeparateTower(nn.Module):
    """
    NodeTransformer -> AdjacencyTransformer behind GraphTower's interface.

    `node_vocab` must be passed explicitly. An injected tower is built BEFORE
    the model, and it is the model's __init__ that sets `cfg.node_vocab` — so
    reading it from cfg here would silently pick up the dataclass default.
    """

    use_coexpr = False

    def __init__(self, cfg: Config, node_vocab: int):
        super().__init__()
        self.cfg = cfg
        self.node_vocab = node_vocab
        self.node_tf = NodeTransformer(cfg, node_vocab)
        self.adj_tf = AdjacencyTransformer(cfg)

        d = cfg.d_model
        self.use_rand_feat = cfg.use_random_node_features
        self.rand_feat_dim = cfg.rand_feat_dim
        self.rand_feat_scope = getattr(cfg, "rand_feat_scope", "all")
        self.pad_id = getattr(cfg, "pad_id", 0)
        self.node_id_proj = (nn.Linear(cfg.rand_feat_dim, d)
                             if self.use_rand_feat else None)
        self.n_types = cfg.n_node_types
        self.type_emb = (nn.Embedding(cfg.n_node_types + 1, d)
                         if cfg.n_node_types > 0 else None)

    # ── the bits the models and benchmarks reach for ──────────────────────
    @property
    def tok_emb(self):
        return self.node_tf.tok_emb

    @property
    def blocks(self):
        """What `benchmarks.capture_attention` iterates over."""
        return self.node_tf.layers

    def gene_embeddings(self) -> Tensor:
        return self.node_tf.tok_emb.weight[: self.node_vocab]

    def post_init_zeros(self):
        """Start at the no-edge-bias baseline, as GraphTower does."""
        if self.node_tf.use_attn_edge_bias:
            nn.init.zeros_(self.node_tf.edge_bias_proj.weight)

    def forward(self, node_in: Tensor, edge_in: Tensor,
                t_node: Tensor, t_edge: Tensor,
                node_id_feat: Optional[Tensor] = None,
                type_ids: Optional[Tensor] = None,
                co_expr=None, cond_drop=None, cond_pooled=None,
                nbr_emb=None, nbr_w=None, nbr_slots=None
                ) -> Tuple[Tensor, Tensor]:
        if co_expr is not None or cond_pooled is not None or nbr_emb is not None:
            raise NotImplementedError(
                "SeparateTower has no conditioning path — build the model with "
                "the default GraphTower to use co-expression conditioning.")

        B, N = node_in.shape
        extra = None
        if self.use_rand_feat:
            if node_id_feat is None:
                node_id_feat = torch.randn(B, N, self.rand_feat_dim,
                                           device=node_in.device,
                                           dtype=self.node_tf.tok_emb.weight.dtype)
            extra = self.node_id_proj(node_id_feat)
            if self.rand_feat_scope == "placeholders":
                ph = (node_in == self.pad_id) | (node_in == self.node_vocab)
                extra = extra * ph.unsqueeze(-1).to(extra.dtype)
        if self.type_emb is not None and type_ids is not None:
            te = self.type_emb(type_ids)
            extra = te if extra is None else extra + te

        h_node = self.node_tf(node_in, edge_in, t_node, extra)
        h_edge = self.adj_tf(edge_in, h_node, t_edge)
        return h_node, h_edge
