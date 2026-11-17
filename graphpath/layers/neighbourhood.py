"""Seed-node neighbourhood encoding for conditioned completion.

THE VARIANT. A few nodes of a pathway are revealed and the model generates the
rest. On top of that, the revealed nodes are *looked up* in an evidence
network — co-expression, Perturb-seq response, or both — their neighbourhood
is retrieved, and graph attention over that neighbourhood produces an UPDATED
representation of each revealed node. The model then conditions on those
updated vectors rather than on the bare token embeddings.

How this differs from the conditioning already in the codebase, since three
things here are easy to confuse:

  CoExpressionEncoder (conditioning.py)  pools each tissue's top-k GLOBAL hub
      genes into one vector per sample. Seeded on the tissue, not on the
      graph, and identical for every graph with the same label.
  use_coexpr_attn (pairwise_conditioning.py)  lets EVERY node attend over its
      co-expressed partners, inside the denoiser, at every layer.
  this module                            seeds on the REVEALED nodes only,
      runs outside the denoiser once per forward pass, and returns a
      representation of what those particular nodes sit next to.

INJECTION USES EXISTING CHANNELS, so no file in graphpath/ is modified:

  node_id_feat   (B, N, rand_feat_dim) — the tower already projects this per
                 node and adds it at the input. The neighbourhood vector is
                 concatenated onto the random identity features rather than
                 replacing them, so FIX 5's per-node identity survives.
  cond_pooled    (B, d_model) — the tower's existing pooled-condition path.

ZERO-INIT. Both output projections start at zero, so a freshly built model is
EXACTLY the base model and the variant can only help if training pays for it.
That is also what makes an untrained sanity check meaningful: if the outputs
are not identical at init, the wiring is wrong.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NeighbourhoodEncoder(nn.Module):
    """Graph attention over the evidence-network neighbourhood of seed nodes.

    lookup : a CoexprLookup over the (n_channels, V, V) evidence tensor. Its
             `neighbours(channel_ids, toks)` returns the top-k partners and
             weights per node, which is the retrieval step; this module adds
             the attention and the hop iteration on top.
    """

    def __init__(self, d_model: int, out_dim: int, cond_dim: int,
                 n_heads: int = 4, hops: int = 2, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must divide n_heads"
        self.h, self.dk, self.hops = n_heads, d_model // n_heads, hops
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        self.ff_norm = nn.LayerNorm(d_model)
        # the two injection heads, both zero so the module starts inert
        self.to_node = nn.Linear(d_model, out_dim)
        # cond_dim, NOT d_model: GraphConditioner.apply feeds this straight
        # into Linear(cfg.cond_d_model, d_model) and FiLM(cfg.cond_d_model, ...),
        # so the pooled vector has to arrive at the conditioner's width
        self.to_pooled = nn.Linear(d_model, cond_dim)
        nn.init.zeros_(self.to_node.weight);  nn.init.zeros_(self.to_node.bias)
        nn.init.zeros_(self.to_pooled.weight); nn.init.zeros_(self.to_pooled.bias)
        self.out_dim = out_dim

    def _attend(self, x: Tensor, nbr: Tensor, w: Tensor) -> Tensor:
        """One round: each node attends over its retrieved neighbours.

        `w` is the EVIDENCE STRENGTH, folded into the attention logits as an
        additive log-bias rather than multiplying the output. A strong
        co-expression partner should win attention against a weak one; scaling
        afterwards would instead let a node with many weak partners accumulate
        the same mass as one with a single strong partner.
        """
        B, N, K, D = nbr.shape
        q = self.q(x).view(B, N, self.h, self.dk).transpose(1, 2)        # B,h,N,dk
        k = self.k(nbr).view(B, N, K, self.h, self.dk).permute(0, 3, 1, 2, 4)
        v = self.v(nbr).view(B, N, K, self.h, self.dk).permute(0, 3, 1, 2, 4)
        att = (q.unsqueeze(-2) * k).sum(-1) / (self.dk ** 0.5)          # B,h,N,K
        att = att + torch.log(w.clamp(min=1e-9))[:, None]               # evidence
        att = att.masked_fill(w[:, None] <= 0, float("-inf"))
        # a node with NO retrieved partner would give an all -inf row and a NaN
        # softmax; those rows are zeroed instead and contribute nothing
        dead = torch.isinf(att).all(-1, keepdim=True)
        att = torch.where(dead, torch.zeros_like(att), att)
        a = F.softmax(att, dim=-1)
        a = torch.where(dead, torch.zeros_like(a), a)
        out = (a.unsqueeze(-1) * v).sum(-2)                             # B,h,N,dk
        out = out.transpose(1, 2).reshape(B, N, self.h * self.dk)
        x = self.norm(x + self.o(out))
        return self.ff_norm(x + self.ff(x))

    def forward(self, node_toks: Tensor, seed_mask: Tensor, tok_emb: nn.Embedding,
                lookup, channel_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """(per-node (B,N,out_dim), pooled (B,d_model)).

        Non-seed slots get exactly zero: the point of the variant is that the
        REVEALED nodes carry the evidence, and letting every slot contribute
        would make this the existing use_coexpr_attn instead.
        """
        B, N = node_toks.shape
        dev = node_toks.device
        x = tok_emb(node_toks.clamp(min=0))
        x = x * seed_mask.unsqueeze(-1).to(x.dtype)          # seeds only

        cur = node_toks.clamp(min=0)
        for _ in range(max(self.hops, 1)):
            idx, w = lookup.neighbours(channel_ids, cur)      # (B,N,k) each
            w = w * seed_mask.unsqueeze(-1).to(w.dtype)       # non-seeds inert
            x = self._attend(x, tok_emb(idx), w)
            # next hop expands from the strongest partner of each node, which
            # keeps the tensor rectangular; a full k-ary expansion would be
            # k^hops wide and is not worth the memory here
            cur = idx.gather(-1, w.argmax(-1, keepdim=True)).squeeze(-1)

        per_node = self.to_node(x) * seed_mask.unsqueeze(-1).to(x.dtype)
        n_seed = seed_mask.sum(1).clamp(min=1).unsqueeze(-1).to(x.dtype)
        pooled = self.to_pooled((x * seed_mask.unsqueeze(-1).to(x.dtype)).sum(1) / n_seed)
        return per_node, pooled


class NeighbourhoodConditioned(nn.Module):
    """Mixin: give a GraphDiffBERT-like model the seed-neighbourhood encoder.

    `attach` is called after construction so the base model is built exactly
    as it always is. `build_inputs` returns the two kwargs the base forward
    already accepts, so nothing in graphpath/ needs to change.
    """

    def attach_neighbourhood(self, hops: int = 2, n_heads: int = 4,
                             nbr_dim: int = 32):
        self.nbr_dim = nbr_dim
        self.nbr_encoder = NeighbourhoodEncoder(
            self.cfg.d_model, out_dim=nbr_dim,
            cond_dim=self.cfg.cond_d_model, n_heads=n_heads, hops=hops,
            dropout=self.cfg.dropout).to(next(self.parameters()).device)
        return self

    def build_inputs(self, node_toks: Tensor, known_nodes: Tensor,
                     channel_ids: Tensor, rand_feat: Optional[Tensor] = None):
        """(node_id_feat, cond_pooled) for the base model's forward."""
        seed = (known_nodes >= 0)
        per_node, pooled = self.nbr_encoder(
            torch.where(seed, known_nodes, torch.zeros_like(known_nodes)),
            seed, self.tower.tok_emb, self.coexpr_lookup, channel_ids)
        want = self.cfg.rand_feat_dim
        base = want - self.nbr_dim
        if rand_feat is None:
            rand_feat = torch.randn(node_toks.shape[0], node_toks.shape[1],
                                    base, device=node_toks.device)
        return torch.cat([rand_feat, per_node], dim=-1), pooled
