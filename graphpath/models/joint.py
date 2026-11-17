"""
GraphDiffBERT v9 — joint node/edge updates (DiGress-style)
==========================================================

Same diffusion process as graphpath.models.marginal (v8.1) — marginal transition kernel,
exact reverse posterior, symmetric edge corruption. Only the DENOISER changes.

What this fixes
───────────────
v8's denoiser is two towers in series:

    NodeTransformer (all L layers)  ->  h_nodes  ->  AdjacencyTransformer (all L)

so each direction of influence gets exactly one pass, and the two are very
unequal. Measured on the v8 stack at initialisation:

    nodes -> edges    permuting the node tokens moves the edge logits by
                      0.59 of a logit standard deviation      (strong)
    edges -> nodes    replacing the adjacency with a complete graph moves the
                      node logits by 1.5e-06 of a logit sd    (nothing)

Edges reach the node stream only as an additive bias on attention scores.
That reweights WHICH neighbours a node attends to; it never puts edge content
into the node representation, and a node's prediction can never be revised
after the edges are.

DiGress's central claim (Vignac et al. 2023, §3.2) is that X and E must be
updated JOINTLY at every layer. This file does that.

The layer
─────────
One quantity does both jobs — the attention score tensor. Following DiGress's
XEy block:

    Y_ij = (Q_i · K_j / sqrt(d_k)) ⊙ (1 + W_mul E_ij) + W_add E_ij   (B,H,N,N)

    X' = softmax_j(Y) V          nodes attend, with edges modulating the map
    E' = W_out(Y)                edges are READ BACK OUT of the same tensor

So edges steer attention (edges -> nodes) and attention becomes the new edge
state (nodes -> edges), every layer, in one operation. Both streams then get
their own residual + norm + FF.

Two further departures from v8, both of which the v8 audit turned up:

  input LayerNorm   v8 fed tok_emb + time_emb straight into attention. At
                    d_model=128 the token embedding has norm 0.23 against the
                    time embedding's 1.97 — the input was 90% clock, 10%
                    identity — and Q·K/sqrt(d_k) landed at 1e-3, leaving
                    softmax uniform to 0.13%. Normalised here.
  FiLM time         v8 added the time embedding once, at the input of each
                    tower, then passed it through L post-LN blocks. Here t
                    conditions every layer by FiLM on both streams, which is
                    what DiGress does with its global y.

E is symmetrised after every layer, not only at the end, so the undirected
constraint holds throughout rather than being imposed once on the way out.

`GraphDiffBERTJoint` subclasses `GraphDiffBERTMarginal` and overrides only the
denoiser. compute_loss, sample, set_marginals, the schedule and the exact
posterior are the SAME verified code — this file cannot change the diffusion
maths even by accident, which is the point of subclassing rather than forking.

This is the arm the comparison notebooks call "transition".
"""

from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from graphpath.config import JointConfig
from graphpath.layers.joint_tower import JointGraphTransformer
from graphpath.models.marginal import GraphDiffBERTMarginal
from graphpath.processes.marginal import estimate_marginals

__all__ = ["JointConfig", "GraphDiffBERTJoint", "estimate_marginals"]


class GraphDiffBERTJoint(GraphDiffBERTMarginal):
    """
    v8.1's diffusion with a joint denoiser.

    Only __init__ and forward are overridden. compute_loss, sample,
    set_marginals and the schedule are inherited unchanged, so the diffusion
    maths is bit-for-bit the version already verified against brute force.
    """

    def __init__(self, cfg: JointConfig, vocab_size: int,
                 m_node: Optional[Tensor] = None, m_edge: Optional[Tensor] = None):
        super().__init__(cfg, vocab_size, m_node=m_node, m_edge=m_edge)
        # replace the two serial towers with one joint stack
        del self.node_tf, self.adj_tf
        self.tower = JointGraphTransformer(cfg)
        self.edge_norm = nn.LayerNorm(cfg.d_edge)
        self.edge_head = nn.Linear(cfg.d_edge, self.edge_vocab)
        self._init_weights()

    def _init_weights(self):
        std = self.cfg.init_std or (self.cfg.d_model ** -0.5)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=std)

    def forward(self, node_toks: Tensor, edge_toks: Tensor,
                t_node: Tensor, t_edge: Tensor) -> Tuple[Tensor, Tensor]:
        # one t for both streams: the joint tower shares a representation, so
        # independent_t has no meaning here. t_node is used and t_edge ignored.
        type_ids = self._type_ids(node_toks)
        X, E = self.tower(node_toks, edge_toks, t_node, None, type_ids,
                          self._type_scale(t_node))
        node_logits = self.node_head(self.node_norm(X))
        edge_logits = self.edge_head(self.edge_norm(E))
        edge_logits = self._add_type_pair_bias(edge_logits, type_ids,
                                               self._type_scale(t_edge))
        return node_logits, edge_logits
