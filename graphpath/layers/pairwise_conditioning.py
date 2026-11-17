"""Relational co-expression conditioning.

The pooled-vector conditioner shifts every node slot by the same vector
before the block stack runs. It can say "this is a lung-ish context"; it
cannot say "given you placed gene A here, gene B belongs with it in lung".
That second statement is pairwise, and the two modules here give it a
channel:

  PairwiseEdgeBias            co-expression between the two endpoints of a
                              candidate edge, added to the EDGE logits —
                              it informs whether to connect.
  NeighbourhoodCrossAttention each node slot attends over the genes its
                              current token is co-expressed with, with the
                              attention modulated by co-expression strength
                              — it informs which entity to place.

Both read a `(n_tissues, V, V)` sparse co-expression tensor, registered once
via `model.set_coexpression(...)`. Both are zero-initialised at the output,
so an untrained model is exactly the unconditioned one.

Both are also keyed on the CURRENT node token, so they are inert while that
token is MASK — they sharpen as the trajectory decodes, like the type-pair
bias, and cannot drive the earliest decisions. `expected` mode on the edge
bias lifts that restriction; see its docstring.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CoexprLookup(nn.Module):
    """
    GPU-resident lookup over a sparse (n_tissues, V, V) co-expression tensor.

    Two access patterns, both needed every forward pass, so both are
    precomputed rather than searched at run time:

      pair(t, u, v)      value for an arbitrary pair — sorted keys per
                         tissue plus searchsorted
      neighbours(t, u)   the top-k strongest partners of u, with weights
    """

    def __init__(self, coexpr_sparse: Tensor, vocab_size: int, top_k: int = 16,
                 verbose: bool = True):
        super().__init__()
        sp = coexpr_sparse.coalesce()
        t_idx, u, v = sp.indices()
        w = sp.values()
        self.V = int(vocab_size)
        self.n_tissues = int(sp.shape[0])
        self.top_k = int(top_k)

        # ── one globally sorted key array, t*V*V + u*V + v ────────────────
        # A per-tissue key list would mean a Python loop and two device-to-host
        # syncs per tissue inside pair(); folding the tissue into the key makes
        # the whole lookup a single searchsorted with no syncs at all.
        t_np, u_np, v_np = (x.cpu().numpy().astype(np.int64)
                            for x in (t_idx, u, v))
        w_np = w.detach().cpu().numpy().astype(np.float32)
        V64 = np.int64(self.V)

        row = t_np * V64 + u_np                        # (tissue, node) group id
        gkey = row * V64 + v_np
        if gkey.size and not (np.diff(gkey) >= 0).all():
            order = np.argsort(gkey)                   # coalesce() normally
            gkey, w_np = gkey[order], w_np[order]      # leaves this sorted
            row, u_np, v_np = row[order], u_np[order], v_np[order]
        keys = torch.as_tensor(gkey)
        vals = torch.as_tensor(w_np)

        # ── top-k partners per (tissue, node) ─────────────────────────────
        # Sorted by (node, -weight), each group's strongest partners are its
        # first top_k rows and rank within a group is position minus the
        # group's start. Done per tissue: n log n with a small n, thirty
        # times, beats one sort over every pair and keeps the slices in cache.
        # (The obvious per-NODE Python loop is n_tissues * V iterations and
        # does not scale to a real vocabulary.)
        #
        # int32 table: it is (n_tissues, V, k) and int64 doubles it for
        # nothing — nn.Embedding takes int32 indices.
        nbr_idx = np.zeros((self.n_tissues, self.V, top_k), dtype=np.int32)
        nbr_w = np.zeros((self.n_tissues, self.V, top_k), dtype=np.float32)
        flat_idx, flat_w = nbr_idx.reshape(-1, top_k), nbr_w.reshape(-1, top_k)
        bounds = np.searchsorted(t_np, np.arange(self.n_tissues + 1))

        for t in range(self.n_tissues):
            lo, hi = int(bounds[t]), int(bounds[t + 1])
            if hi <= lo:
                continue
            rw, rv, rr = w_np[lo:hi], v_np[lo:hi], row[lo:hi]
            lex = np.lexsort((-rw, rr))
            rw, rv, rr = rw[lex], rv[lex], rr[lex]
            first = np.flatnonzero(np.concatenate(([True], rr[1:] != rr[:-1])))
            rank = np.arange(rr.size) - np.repeat(
                first, np.diff(np.append(first, rr.size)))
            sel = rank < top_k
            flat_idx[rr[sel], rank[sel]] = rv[sel]
            flat_w[rr[sel], rank[sel]] = rw[sel]

        self.register_buffer("keys", keys)
        self.register_buffer("vals", vals)
        self.register_buffer("nbr_idx", torch.as_tensor(nbr_idx))
        self.register_buffer("nbr_w", torch.as_tensor(nbr_w))
        if verbose:
            cov = float((nbr_w[:, :, 0] > 0).mean())
            mb = sum(b.element_size() * b.numel() for b in
                     (self.keys, self.vals, self.nbr_idx, self.nbr_w)) / 1e6
            print(f"co-expression lookup: {len(self.keys)} pairs, top-{top_k} "
                  f"neighbours for {cov:.0%} of (tissue, entity) slots, "
                  f"{mb:.0f} MB resident ({self.n_tissues} tissues x {self.V} "
                  f"entities x {top_k})")

    def pair(self, tissue_ids: Tensor, u: Tensor, v: Tensor) -> Tensor:
        """Values for (u, v) per batch element; `u`/`v` broadcast to (B, ...)."""
        if self.keys.numel() == 0:
            return torch.zeros(u.shape, device=u.device, dtype=self.vals.dtype)
        t = tissue_ids.long().view(-1, *([1] * (u.dim() - 1))).expand_as(u)
        q = ((t * self.V + u.long()) * self.V + v.long()).reshape(-1)
        pos = torch.searchsorted(self.keys, q).clamp(max=self.keys.numel() - 1)
        hit = self.keys[pos] == q
        return torch.where(hit, self.vals[pos], torch.zeros((), dtype=self.vals.dtype,
                                                            device=u.device)
                           ).reshape(u.shape)

    def neighbours(self, tissue_ids: Tensor, node_toks: Tensor
                   ) -> Tuple[Tensor, Tensor]:
        """(B, N, k) partner ids and their co-expression weights."""
        toks = node_toks.clamp(min=0, max=self.V - 1).long()
        t = tissue_ids.long()[:, None].expand_as(toks)
        return self.nbr_idx[t, toks], self.nbr_w[t, toks]


class PairwiseEdgeBias(nn.Module):
    """
    Co-expression between a candidate edge's two endpoints, added to the edge
    logits.

    The value is passed through a small MLP rather than used as a raw scalar,
    so the model can learn a non-monotone response (a weak correlation may
    mean something different from none at all) and can learn per-edge-class
    effects. Zero-initialised output, so it starts as a no-op.
    """

    def __init__(self, edge_vocab: int, hidden: int = 32):
        super().__init__()
        # features: value, log1p(value), present-flag
        self.mlp = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(),
                                 nn.Linear(hidden, edge_vocab))
        self.reset_zeros()

    def reset_zeros(self):
        """Re-apply the zero-init; a model's generic trunc_normal sweep over
        every nn.Linear runs after __init__ and would otherwise clobber it."""
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, edge_logits: Tensor, values: Tensor,
                keep: Optional[Tensor] = None) -> Tensor:
        v = values.unsqueeze(-1)
        feats = torch.cat([v, torch.log1p(v.abs()) * v.sign(),
                           (v != 0).to(v.dtype)], dim=-1)
        delta = self.mlp(feats)
        if keep is not None:
            # conditioning dropout: an all-zero `values` row is NOT the same as
            # no condition, since the hidden layer has a bias and mlp(0) != 0
            delta = delta * keep.view(-1, *([1] * (delta.dim() - 1))).to(delta.dtype)
        return edge_logits + delta


class NeighbourhoodCrossAttention(nn.Module):
    """
    Each node slot attends over the genes its current token is co-expressed
    with, modulated by co-expression strength.

    Attention is over the slot's OWN top-k partners rather than a shared
    memory bank of every gene: retrieval is the part co-expression already
    answers, so spending attention on it wastes capacity and O(N*V) compute.
    Strength enters as a learned additive bias on the attention logits, so
    the model can decide how much a weak correlation is worth rather than
    having the weighting fixed for it.

    Zero-initialised output gate, so it starts as a no-op.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.H, self.d_k = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.w_bias = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, n_heads))
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(d_model))
        self.reset_zeros()

    def reset_zeros(self):
        """Start as an exact no-op, so the model begins at its unconditioned
        behaviour and opens the branch only if the data pays for it. Zero the
        gate ONLY: zeroing the output projection as well makes the branch
        dead rather than closed, since then neither the gate nor the
        projection receives any gradient and it can never open."""
        nn.init.zeros_(self.gate)

    def forward(self, h: Tensor, nbr_emb: Tensor, nbr_w: Tensor,
                slots: Tensor) -> Tensor:
        """
        h        (B, N, d)   node states
        nbr_emb  (M, k, d)   partner embeddings, for the M slots that HAVE
                             partners — in a metabolic pathway most slots are
                             metabolites and reactions with none, and running
                             the projections on those only to mask the result
                             away costs several times what the real work does
        nbr_w    (M, k)      their co-expression strengths (0 = padding)
        slots    (M,)        flat index of each row into h.reshape(B*N, d)
        """
        B, N, d = h.shape
        M, k, _ = nbr_emb.shape
        H, dk = self.H, self.d_k

        flat = h.reshape(B * N, d)
        hs = flat.index_select(0, slots)                  # (M, d)

        q = self.q(hs).view(M, H, dk)
        kk = self.k(nbr_emb).view(M, k, H, dk)
        vv = self.v(nbr_emb).view(M, k, H, dk)

        logits = torch.einsum("mhd,mkhd->mkh", q, kk) / math.sqrt(dk)
        w = nbr_w.unsqueeze(-1)
        logits = logits + self.w_bias(torch.cat(
            [w, torch.log1p(w.abs()) * w.sign()], dim=-1))

        # a slot can have fewer than k partners; those entries are padding
        logits = logits.masked_fill((nbr_w == 0).unsqueeze(-1), -1e4)
        attn = self.drop(F.softmax(logits, dim=1))

        out = torch.einsum("mkh,mkhd->mhd", attn, vv).reshape(M, d)
        # normalise the UPDATE, not the sum: `LayerNorm(h + 0)` still rescales
        # h, so a zero gate would not leave the model untouched and the
        # "starts as the unconditioned model" guarantee would be false.
        # Slots absent from `slots` are never touched, so a partnerless slot
        # gets exactly nothing rather than out()'s bias through LayerNorm.
        upd = self.gate * self.norm(self.out(out))
        return flat.index_add(0, slots, upd).reshape(B, N, d)
