"""DiGress-style joint node/edge layers.

One quantity does both jobs — the attention score tensor:

    Y_ij = (Q_i · K_j / sqrt(d_k)) ⊙ (1 + W_mul E_ij) + W_add E_ij   (B,H,N,N)
    X' = softmax_j(Y) V          nodes attend, with edges modulating the map
    E' = W_out(Y)                edges are READ BACK OUT of the same tensor

So edges steer attention (edges -> nodes) and attention becomes the new edge
state (nodes -> edges), every layer, in one operation. Both streams get their
own residual + norm + FF, the time embedding FiLM-conditions every layer, and
E is symmetrised after every layer.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphpath.config import JointConfig
from graphpath.layers.common import SinusoidalTimeEmbedding, _make_ff


class JointLayer(nn.Module):
    """One DiGress-style XE block: X and E updated from the same tensor."""

    def __init__(self, cfg: JointConfig):
        super().__init__()
        d, H, de = cfg.d_model, cfg.n_heads, cfg.d_edge
        assert d % H == 0, "d_model must divide n_heads"
        self.H, self.d_k, self.de = H, d // H, de

        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.x_out = nn.Linear(d, d)

        # edges -> attention (FiLM on the score tensor), and attention -> edges
        self.e_mul = nn.Linear(de, H)
        self.e_add = nn.Linear(de, H)
        self.e_out = nn.Linear(H, de)

        self.norm_x, self.norm_e = nn.LayerNorm(d), nn.LayerNorm(de)
        self.ff_x, self.ff_e = _make_ff(d, cfg.d_ff, cfg.dropout), \
                               _make_ff(de, cfg.d_ff, cfg.dropout)
        self.ffn_x, self.ffn_e = nn.LayerNorm(d), nn.LayerNorm(de)

        self.film_x = nn.Linear(d, 2 * d) if cfg.film_time else None
        self.film_e = nn.Linear(d, 2 * de) if cfg.film_time else None
        self.cfg = cfg

    def forward(self, X: Tensor, E: Tensor, t_emb: Tensor,
                mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """X (B,N,d), E (B,N,N,de), t_emb (B,d), mask (B,N) bool or None."""
        B, N, d = X.shape
        H, d_k = self.H, self.d_k

        Q = self.W_q(X).view(B, N, H, d_k).transpose(1, 2)      # (B,H,N,dk)
        K = self.W_k(X).view(B, N, H, d_k).transpose(1, 2)
        V = self.W_v(X).view(B, N, H, d_k).transpose(1, 2)
        scores = (Q @ K.transpose(-1, -2)) / (d_k ** 0.5)       # (B,H,N,N)

        mul = self.e_mul(E).permute(0, 3, 1, 2)                 # (B,H,N,N)
        add = self.e_add(E).permute(0, 3, 1, 2)
        Y = scores * (1.0 + mul) + add        # edges modulate the attention map
        edge_delta = Y - scores               # kept for attention capture

        if mask is not None:
            keep = mask[:, None, None, :]                       # (B,1,1,N)
            Y = Y.masked_fill(~keep, -1e4)

        A = F.softmax(Y, dim=-1)

        # Opt-in attention capture, off unless benchmarks.attention.collect
        # switches it on. Three tensors, matching what GraphBlock exposes so
        # the same analysis code reads both backbones:
        #   last_attn      post-softmax attention
        #   last_content   the Q.K term alone, before edges touch it
        #   last_edge_bias what the edge stream contributed to the score.
        #                  GraphBlock adds a bias, so there it is one term;
        #                  here edges act as FiLM (scale AND shift), so the
        #                  contribution is Y - scores = scores*mul + add —
        #                  the part of the score that would vanish if the
        #                  edge stream were removed. Taken before the padding
        #                  mask, so the -1e4 fill does not leak into it.
        if getattr(self, "store_attn", False):
            self.last_attn = A.detach()
            self.last_content = scores.detach()
            self.last_edge_bias = edge_delta.detach()

        Xo = self.x_out((A @ V).transpose(1, 2).reshape(B, N, d))
        Eo = self.e_out(Y.permute(0, 2, 3, 1))   # ...and become the new edges

        if self.film_x is not None:
            gx, bx = self.film_x(t_emb).chunk(2, -1)
            Xo = Xo * (1 + gx[:, None, :]) + bx[:, None, :]
            ge, be = self.film_e(t_emb).chunk(2, -1)
            Eo = Eo * (1 + ge[:, None, None, :]) + be[:, None, None, :]

        X = self.norm_x(X + Xo)
        X = self.ffn_x(X + self.ff_x(X))
        E = self.norm_e(E + Eo)
        E = self.ffn_e(E + self.ff_e(E))
        E = 0.5 * (E + E.transpose(1, 2))     # undirected, at EVERY layer
        return X, E


class JointGraphTransformer(nn.Module):
    """L joint layers, returning both streams' hidden states."""

    def __init__(self, cfg: JointConfig):
        super().__init__()
        d, de = cfg.d_model, cfg.d_edge
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.node_vocab, d)
        self.edge_emb = nn.Embedding(cfg.edge_vocab, de)
        self.time_emb = SinusoidalTimeEmbedding(d)
        self.time_proj = nn.Linear(d, d)
        # the fix for "attention does nothing at init": normalise the
        # embedding sum before the first block, as BERT does
        self.in_norm_x = nn.LayerNorm(d) if cfg.input_layernorm else nn.Identity()
        self.in_norm_e = nn.LayerNorm(de) if cfg.input_layernorm else nn.Identity()
        self.type_emb = (nn.Embedding(cfg.n_node_types + 1, d)
                         if cfg.n_node_types > 0 else None)
        self.layers = nn.ModuleList(JointLayer(cfg) for _ in range(cfg.n_layers))
        self.drop = nn.Dropout(cfg.dropout)

    @property
    def blocks(self):
        """Alias so benchmarks.attention.capture_attention finds the layers.

        A property rather than `self.blocks = self.layers`: assigning the same
        ModuleList to a second attribute registers it twice, which duplicates
        every parameter in the state_dict and breaks load_state_dict on the
        checkpoints already on disk.
        """
        return self.layers

    def forward(self, node_toks: Tensor, edge_toks: Tensor, t: Tensor,
                mask: Optional[Tensor] = None,
                type_ids: Optional[Tensor] = None,
                type_scale: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        t_emb = self.time_proj(self.time_emb(t))                # (B,d)
        x0 = self.tok_emb(node_toks)
        if self.type_emb is not None and type_ids is not None:
            te = self.type_emb(type_ids)
            if type_scale is not None:
                te = te * type_scale[:, None, None]
            x0 = x0 + te
        X = self.drop(self.in_norm_x(x0 + t_emb[:, None, :]))
        E = self.drop(self.in_norm_e(self.edge_emb(edge_toks)))
        for layer in self.layers:
            X, E = layer(X, E, t_emb, mask)
        return X, E
