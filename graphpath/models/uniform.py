"""GraphDiffBERTUniform — joint uniform-transition-matrix diffusion model.

Both node and edge streams:
  • No MASK token — vocab is exactly the clean classes
  • Forward: x_t = x_0 w.p. ᾱ_t, else Uniform{classes}
  • Reverse: posterior computed analytically from predicted x_0
  • Loss: noise-weighted CE over ALL positions (not just "corrupted" ones)

Contrast with the absorbing variant in graphpath/models/masked.py: there, an
undecoded position is literally a distinct MASK token, decoding is
monotonic, and the model only ever supervises masked positions. Here every
position always holds a real class, so there is no "revealed vs masked"
state, no confidence-guided unmasking and no remasking — the reverse
posterior does that job implicitly, and a wrong token can always be
corrected on a later step.
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphpath.layers.graph_block import GraphTower
from graphpath.config import Config
from graphpath.layers.node_types import NodeTypeMixin
from graphpath.processes.uniform import UniformTransitionSchedule
from graphpath.samplers.uniform import sample_uniform
from graphpath.training.class_weights import inverse_freq_weights


class GraphDiffBERTUniform(NodeTypeMixin, nn.Module):
    """
    forward() returns:
        node_logits : (B, N, node_vocab)       — x_0 class predictions
        edge_logits : (B, N, N, edge_vocab)    — x_0 class predictions
    """

    def __init__(self, cfg: Config, vocab_size: int,
                 schedule: Optional[nn.Module] = None,
                 tower:    Optional[nn.Module] = None):
        """
        vocab_size = dataset.vocab_size + 1  (API compat with the absorbing
        model — the +1 MASK slot is unused here, internally
        node_vocab = vocab_size - 1 as before)

        `schedule` / `tower` are injectable; they default to
        `UniformTransitionSchedule(cfg)` and `GraphTower(cfg)`. Any
        replacement tower must have GraphTower's signature:
        (node_toks, edge_toks, t_node, t_edge, node_id_feat=, type_ids=)
        -> (h_node, h_edge).
        """
        super().__init__()
        self.cfg        = cfg
        self.vocab_size = vocab_size
        self.node_vocab = vocab_size - 1      # keep API compat
        self.edge_vocab = cfg.edge_vocab      # 2
        cfg.node_vocab  = self.node_vocab

        self.schedule = schedule if schedule is not None else UniformTransitionSchedule(cfg)
        self.tower    = tower    if tower    is not None else GraphTower(cfg)

        d = cfg.d_model
        self.node_norm = nn.LayerNorm(d)
        self.node_head = nn.Linear(d, self.node_vocab)
        self.edge_norm = nn.LayerNorm(d)
        self.edge_head = nn.Linear(d, self.edge_vocab)

        # FIX 7 — optional per-class node weights, shared API with the
        # absorbing model. Defaults to all-ones (no-op).
        self.register_buffer("node_class_weight", torch.ones(self.node_vocab))

        # Stationary distribution of the edge kernel. All-ones = uniform =
        # the original behaviour; set_edge_marginal() swaps in the data
        # marginal, which is the kernel D3PM defines for imbalanced classes.
        self.register_buffer("edge_marginal", torch.ones(self.edge_vocab))
        if cfg.edge_marginal_p is not None:
            self.set_edge_marginal(cfg.edge_marginal_p)

        # FIX 11 — typed connectivity. No MASK token here, so every node
        # always has a known type and the type bias/mask is active from the
        # very first reverse step (unlike the absorbing model, where it stays
        # inert until nodes decode).
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
        # gate/FiLM (only applies to GraphBlock-based towers).
        if hasattr(self.tower, "post_init_zeros"):
            self.tower.post_init_zeros()

    def set_node_class_weights(self, counts, power: Optional[float] = None):
        """See GraphDiffBERT.set_node_class_weights — same contract."""
        power = self.cfg.node_class_weight_power if power is None else power
        w = inverse_freq_weights(counts, power, self.node_vocab,
                                 self.node_class_weight.device)
        self.node_class_weight.copy_(w)

    def set_edge_marginal(self, p_edge: float):
        """
        p_edge : share of valid node pairs that carry an edge in the data
                 (e.g. float(adjcs[valid_pair].float().mean())).

        Switches the edge stream from the uniform kernel to the
        marginal-matched one, so x_T starts at the data's own density rather
        than at 50%.
        """
        if not 0.0 < p_edge < 1.0:
            raise ValueError(f"p_edge must be in (0,1), got {p_edge}")
        m = torch.zeros(self.edge_vocab, device=self.edge_marginal.device)
        m[0], m[1] = 1.0 - p_edge, p_edge
        self.edge_marginal.copy_(m)

    @property
    def _edge_marg(self):
        """None while the marginal is uniform, so the fast path is unchanged."""
        m = self.edge_marginal
        return None if bool(torch.allclose(m, torch.full_like(m, float(m[0])))) else m

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        node_toks:    Tensor,   # (B, N)     long — noisy, in {0..node_vocab-1}
        edge_toks:    Tensor,   # (B, N, N)  long — noisy, in {0..edge_vocab-1}
        t_node:       Tensor,   # (B,)       long
        t_edge:       Tensor,   # (B,)       long
        node_id_feat: Optional[Tensor] = None,   # see FIX 5
        co_expr:      Optional[dict]   = None,   # co-expression conditioning
        cond_drop:    Optional[Tensor] = None,
        cond_pooled:  Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        type_ids = self._type_ids(node_toks)

        h_node, h_edge = self.tower(node_toks, edge_toks, t_node, t_edge,
                                    node_id_feat=node_id_feat, type_ids=type_ids,
                                    co_expr=co_expr, cond_drop=cond_drop,
                                    cond_pooled=cond_pooled)
        node_logits = self.node_head(self.node_norm(h_node))        # (B,N,Kn)
        edge_logits = self.edge_head(self.edge_norm(h_edge))        # (B,N,N,Ke)

        # FIX 11: learned per-type-pair prior on the edge logits.
        edge_logits = self._add_type_pair_bias(edge_logits, type_ids)
        return node_logits, edge_logits

    # ── loss ─────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        node_feats:       Tensor,    # (B, N)    long — clean
        adjcs:            Tensor,    # (B, N, N) long — clean (0/1)
        pad_mask:         Tensor,    # (B, N)    bool — True = valid node
        loss_edge_weight: float = 1.0,
        co_expr:          Optional[dict] = None,
    ) -> dict:
        """
        Loss = noise-weighted CE on x_0 predictions at ALL positions.

        Noise weighting: (1 - ᾱ_t) per sample.
          At t≈0 the input is nearly clean so predicting x_0 is trivial —
          down-weight this.  At t≈T the input is nearly uniform so every
          prediction is hard — full weight.

        Edge class balancing: inverse-frequency weights within the batch.
        Node class balancing (FIX 7) is applied when
        cfg.use_node_class_weights is set and set_node_class_weights() has
        been called.
        """
        B, N   = node_feats.shape
        device = node_feats.device

        # Shared timestep for both streams
        t = torch.randint(0, self.cfg.T, (B,), device=device)

        # ── Forward diffusion (uniform corruption) ────────────────────────
        sup_pad = self.cfg.supervise_pad
        node_t = self.schedule.q_sample(node_feats, t, self.node_vocab)
        if not sup_pad:
            node_t = torch.where(pad_mask, node_t, node_feats)  # keep padding clean

        valid_pair = pad_mask[:, :, None] & pad_mask[:, None, :]
        edge_t     = self.schedule.q_sample(adjcs, t, self.edge_vocab,
                                            marginal=self._edge_marg)
        if not sup_pad:
            edge_t = torch.where(valid_pair, edge_t, adjcs)

        # CFG-style conditioning dropout (see GraphConditioner).
        cond_drop = None
        if self.tower.use_coexpr and co_expr is not None and self.training:
            cond_drop = self.tower.conditioner.train_drop_mask(B, device)

        # ── Predict x_0 ───────────────────────────────────────────────────
        node_logits, edge_logits = self.forward(node_t, edge_t, t, t,
                                                co_expr=co_expr, cond_drop=cond_drop)

        # ── Noise-level weights (one per batch element) ───────────────────
        # noise weight: (1 - ᾱ_t),  shape (B,)
        nw = (1.0 - self.schedule.ac[t])            # (B,)

        # ── Node CE — all valid positions, noise-weighted ─────────────────
        node_ce = F.cross_entropy(
            node_logits.reshape(B * N, self.node_vocab),
            node_feats.reshape(B * N),
            weight=self.node_class_weight if self.cfg.use_node_class_weights else None,
            reduction="none",
        ).reshape(B, N)                                        # (B, N)

        # Zero out padding, apply noise weight. With supervise_pad the padding
        # slots ARE the lesson — PAD is how the model says "this graph is
        # shorter than N" — so they stay in.
        if sup_pad:
            w = torch.where(pad_mask, torch.ones_like(node_ce),
                            torch.full_like(node_ce, self.cfg.pad_loss_weight))
            node_ce = node_ce * w
        else:
            node_ce = node_ce * pad_mask.float()               # (B, N)
        node_ce  = node_ce * nw[:, None]                       # (B, N)
        n_valid  = ((pad_mask.float() + (~pad_mask).float()
                     * self.cfg.pad_loss_weight).sum().clamp(min=1)
                    if sup_pad else pad_mask.float().sum().clamp(min=1))
        loss_n   = node_ce.sum() / n_valid

        # ── Edge CE — all valid non-diagonal pairs, noise-weighted ────────
        diag     = torch.eye(N, device=device, dtype=torch.bool)
        # every off-diagonal pair when padding is supervised: a pair touching
        # a PAD slot has target 0, which is exactly what we want it to learn
        vp = (~diag[None]).expand(B, N, N) if sup_pad else (valid_pair & ~diag[None])

        # Balanced class weights from this batch
        flat_tgts = adjcs[vp]                                  # (M,)
        n_pos = flat_tgts.sum().float().clamp(min=1)
        n_neg = (flat_tgts == 0).sum().float().clamp(min=1)
        total = n_pos + n_neg
        class_w = torch.stack([n_pos / total, n_neg / total])  # [w0, w1]

        # CE per edge position: (B, N, N)
        edge_ce = F.cross_entropy(
            edge_logits.reshape(B * N * N, self.edge_vocab),
            adjcs.reshape(B * N * N),
            weight=class_w,
            reduction="none",
        ).reshape(B, N, N)

        # Zero out invalid / diagonal, apply noise weight
        edge_ce  = edge_ce * vp.float()                        # (B, N, N)
        edge_ce  = edge_ce * nw[:, None, None]                 # (B, N, N)
        m_valid  = vp.float().sum().clamp(min=1)
        loss_e   = edge_ce.sum() / m_valid

        loss = loss_n + loss_edge_weight * loss_e
        return {
            "loss":      loss,
            "loss_node": loss_n.item(),
            "loss_edge": loss_e.item(),
        }

    # ── sampling ─────────────────────────────────────────────────────────────

    def sample(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """Reverse-time generation. Implemented in `graphpath.samplers.uniform`."""
        return sample_uniform(self, *args, **kwargs)
