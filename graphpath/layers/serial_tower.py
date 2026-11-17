"""Serial node-then-edge denoiser for the marginal transition model.

`NodeTransformer` runs all L layers of self-attention over node tokens with
the edge tokens entering only as an additive attention bias; the
`AdjacencyTransformer` then runs its own L layers of alternating row/column
attention over edge states initialised from the final node states. The two
towers are joined at a single seam, which is the limitation the joint tower
(`graphpath.layers.joint_tower`) removes.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphpath.config import MarginalConfig
from graphpath.layers.common import SinusoidalTimeEmbedding, _make_ff, _mha_block


class NodeTransformer(nn.Module):
    def __init__(self, cfg: MarginalConfig):
        super().__init__()
        self.cfg = cfg
        d, H = cfg.d_model, cfg.n_heads
        d_k  = d // H
        self.H, self.d_k = H, d_k

        self.tok_emb        = nn.Embedding(cfg.node_vocab, d)
        self.time_emb       = SinusoidalTimeEmbedding(d)
        self.time_proj      = nn.Linear(d, d)
        self.edge_emb       = nn.Embedding(cfg.edge_vocab, d)
        self.edge_bias_proj = nn.Linear(d, H, bias=False)
        self.in_norm        = nn.LayerNorm(d) if cfg.input_layernorm else nn.Identity()
        self.type_emb = (nn.Embedding(cfg.n_node_types + 1, d)
                         if cfg.n_node_types > 0 else None)

        self.layers    = nn.ModuleList()
        self.norms     = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        self.ff_norms  = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.layers.append(nn.ModuleDict({
                "W_q": nn.Linear(d, d, bias=False),
                "W_k": nn.Linear(d, d, bias=False),
                "W_v": nn.Linear(d, d, bias=False),
                "out": nn.Linear(d, d),
            }))
            self.norms.append(nn.LayerNorm(d))
            self.ff_layers.append(_make_ff(d, cfg.d_ff, cfg.dropout))
            self.ff_norms.append(nn.LayerNorm(d))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, node_toks: Tensor, edge_toks: Tensor, t: Tensor,
                type_ids: Optional[Tensor] = None,
                type_scale: Optional[Tensor] = None) -> Tensor:
        """
        node_toks : (B, N)    long  — noisy node tokens in {0..node_vocab-1}
        edge_toks : (B, N, N) long  — noisy edge tokens in {0..edge_vocab-1}
        t         : (B,)      long
        type_ids  : (B, N)    long  — node types, or None
        type_scale: (B,)      float — per-sample gate on the type signal
        """
        B, N = node_toks.shape
        H, d_k = self.H, self.d_k

        h = self.tok_emb(node_toks)
        if self.type_emb is not None and type_ids is not None:
            te = self.type_emb(type_ids)
            if type_scale is not None:
                te = te * type_scale[:, None, None]
            h = h + te
        h = h + self.time_proj(self.time_emb(t))[:, None, :]
        h = self.drop(self.in_norm(h))

        e_emb    = self.edge_emb(edge_toks)               # (B, N, N, d)
        adj_bias = self.edge_bias_proj(e_emb)             # (B, N, N, H)
        adj_bias = adj_bias.permute(0, 3, 1, 2)           # (B, H, N, N)

        # Forbidding a node from attending to itself makes the copy operation
        # that low-noise denoising needs go through the residual stream alone.
        diag_mask = (torch.eye(N, device=h.device, dtype=torch.bool)
                     if self.cfg.mask_attention_diagonal else None)

        for layer, norm, ff, ffnorm in zip(
            self.layers, self.norms, self.ff_layers, self.ff_norms
        ):
            B_, N_, d = h.shape
            Q = layer["W_q"](h).view(B_, N_, H, d_k).transpose(1, 2)
            K = layer["W_k"](h).view(B_, N_, H, d_k).transpose(1, 2)
            V = layer["W_v"](h).view(B_, N_, H, d_k).transpose(1, 2)
            scores = (Q @ K.transpose(-1, -2)) / math.sqrt(d_k)
            scores = scores + adj_bias
            if diag_mask is not None:
                scores = scores.masked_fill(diag_mask[None, None], -1e4)
            out = (F.softmax(scores, dim=-1) @ V).transpose(1, 2).reshape(B_, N_, H * d_k)
            out = layer["out"](out)
            h = norm(h + out)
            h = ffnorm(h + ff(h))
        return h


class AdjacencyTransformer(nn.Module):
    """
    Alternating row + column attention over (B, N, N, d) edge hidden states.
    Initialised from edge token embeddings + node pair context.
    """
    def __init__(self, cfg: MarginalConfig):
        super().__init__()
        self.cfg = cfg
        d, H = cfg.d_model, cfg.edge_n_heads
        d_k  = d // H
        assert d % H == 0
        self.H, self.d_k = H, d_k

        self.tok_emb   = nn.Embedding(cfg.edge_vocab, d)
        self.pair_proj = nn.Linear(2 * d, d)
        self.time_emb  = SinusoidalTimeEmbedding(d)
        self.time_proj = nn.Linear(d, d)
        self.in_norm   = nn.LayerNorm(d) if cfg.input_layernorm else nn.Identity()

        self.row_layers = nn.ModuleList()
        self.row_norms  = nn.ModuleList()
        self.col_layers = nn.ModuleList()
        self.col_norms  = nn.ModuleList()
        self.ff_layers  = nn.ModuleList()
        self.ff_norms   = nn.ModuleList()

        for _ in range(cfg.n_layers):
            for store_l, store_n in [(self.row_layers, self.row_norms),
                                     (self.col_layers, self.col_norms)]:
                store_l.append(nn.ModuleDict({
                    "W_q": nn.Linear(d, d, bias=False),
                    "W_k": nn.Linear(d, d, bias=False),
                    "W_v": nn.Linear(d, d, bias=False),
                    "out": nn.Linear(d, d),
                }))
                store_n.append(nn.LayerNorm(d))
            self.ff_layers.append(_make_ff(d, cfg.d_ff, cfg.dropout))
            self.ff_norms.append(nn.LayerNorm(d))

        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, edge_toks: Tensor, h_nodes: Tensor, t: Tensor) -> Tensor:
        """
        edge_toks : (B, N, N) long  — noisy edge tokens in {0..edge_vocab-1}
        h_nodes   : (B, N, d) float — node hidden states
        t         : (B,)      long
        """
        B, N, _ = edge_toks.shape
        d       = h_nodes.shape[-1]
        H, d_k  = self.H, self.d_k

        e = self.tok_emb(edge_toks)                       # (B, N, N, d)

        hi = h_nodes.unsqueeze(2).expand(-1, -1, N, -1)
        hj = h_nodes.unsqueeze(1).expand(-1, N, -1, -1)
        e  = e + self.pair_proj(torch.cat([hi, hj], dim=-1))

        t_emb = self.time_proj(self.time_emb(t))
        e = e + t_emb[:, None, None, :]
        e = self.drop(self.in_norm(e))

        # NOTE this masks the ATTENTION diagonal (edge cell (i,j) may not
        # attend to itself), which is not the same thing as masking the
        # graph's self-loop cells (i,i) — the latter is handled in
        # compute_loss and sample() by zeroing the diagonal directly.
        diag_mask = (torch.eye(N, device=e.device, dtype=torch.bool)
                     if self.cfg.mask_attention_diagonal else None)

        for row_l, row_n, col_l, col_n, ff, ffnorm in zip(
            self.row_layers, self.row_norms,
            self.col_layers, self.col_norms,
            self.ff_layers,  self.ff_norms,
        ):
            # Row-wise: attend over j for fixed i
            e_row = e.reshape(B * N, N, d)
            out_r = _mha_block(e_row, row_l, H, d_k, diag_mask)
            e = row_n(e + out_r.reshape(B, N, N, d))

            # Column-wise: attend over i for fixed j
            e_col = e.transpose(1, 2).reshape(B * N, N, d)
            out_c = _mha_block(e_col, col_l, H, d_k, diag_mask)
            e = col_n(e + out_c.reshape(B, N, N, d).transpose(1, 2))

            e = ffnorm(e + ff(e))

        # Symmetrise: undirected-graph inductive bias
        e = 0.5 * (e + e.transpose(1, 2))
        return e
