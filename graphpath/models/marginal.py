"""GraphDiffBERTMarginal — marginal transition-matrix diffusion (v8.1).

Forward process Q_t = α_t I + (1-α_t) 1 mᵀ with m the data marginal, exact
reverse posterior, symmetric edge corruption; see
`graphpath.processes.marginal` for the maths. This module holds the network,
the loss and the typed-connectivity hooks. Sampling lives in
`graphpath.samplers.marginal`; the joint-tower variant that the main runs
call "transition" is `graphpath.models.joint.GraphDiffBERTJoint`.

THE LOSS WEIGHTING (v8.1) — read this before anything else
───────────────────────────────────────────────────────────
v7 and the first v8 weighted every training sample by (1 - ᾱ_t), and that
single line is why neither trained. The weight is ~0 where the input is
almost clean and 1 where it is pure noise, so the gradient is aimed at
exactly the regime where x_0 CANNOT be recovered — and there the
loss-minimising prediction is the class marginal itself. The model duly
learns the marginal and stops.

  t                   0       10       25       50       75       99
  ᾱ_t            0.9994   0.9667   0.8356   0.4783   0.1335   0.0000
  weight         0.0006   0.0333   0.1644   0.5217   0.8665   1.0000

The lower-noise half of the schedule carries 18.9% of the total weight, and
t=5 — where the input is 99.94% clean and denoising is nearly a copy —
carries 0.0006. But the last reverse steps and the final argmax run entirely
in that regime, so the sampler is driven by the part of the model that was
never trained. Two symptoms follow directly:

  every slot gets the same token   the final step is an argmax over a nearly
                                   flat logit vector
  density runs to ~0.5 or to 1.0   once ᾱ_{t-1} → 1 the posterior is
                                   dominated by p_θ, not by the kernel, so an
                                   edge head near chance sets the density
                                   whatever the prior at x_T was

Measured on a 16-graph memorisation task, 600 steps, same seed, reporting
UNWEIGHTED CE at fixed t (chance 8.775, marginal-entropy floor 7.907):

    t                    5      25      50      75      95
    (1-ᾱ) weighted   4.639   4.753   5.538   6.246   6.455
    unweighted       2.345   3.020   4.637   6.079   6.481

Half the error where it matters, no cost where it does not. An
x_0-parameterised discrete diffusion loss is a plain cross-entropy (D3PM
eq. 5; DiGress §3.3) — the weighting was never part of it.

`loss_noise_weighting="one_minus_alpha_bar"` restores the old behaviour for
ablation. The default is "none".

THE EDGE CLASS WEIGHTS (v8.1) — why the graphs were too dense
──────────────────────────────────────────────────────────────
Separate bug, same file. The edge CE carried inverse-frequency class
weights, which are wrong for a generative model.

Weighted cross-entropy does not have the data distribution as its optimum:
for two classes with weights (w0, w1) the minimiser is the REBALANCED
q = w1·p / (w1·p + w0·(1−p)). With p = 0.035 the weights come out
(0.076, 1.924) — a ratio of 25.5 — so the optimum is q ≈ 0.48. A classifier
does not care, because you threshold it. A diffusion model samples from its
predicted distribution, so it inherits the rebalancing wholesale.

Fitting a single logit to a Bernoulli(0.035), no network involved:

    no class weights      learned P(edge) 0.0341    1.0x true
    inverse-frequency     learned P(edge) 0.4834   13.8x true

and the densities measured off the trained model were 0.40–0.49, 11–14x the
data. The same number, arrived at from both directions.

The concern the weights were answering is real — at p = 0.035 a model can
score well by saying "no edge" everywhere — but the marginal kernel is
already the answer to it: under Q_t = α_t I + (1−α_t) 1 mᵀ the noisy input
carries the true density at every t, so predicting edges is genuinely how
the loss goes down. `edge_class_weights=True` restores the old behaviour.

Key differences from v7
────────────────────────
  • Q_t = α_t I + (1-α_t) 1 mᵀ  — noise walks toward the data marginal
  • Marginals estimated from data (estimate_marginals) and stored as buffers
  • Marginals are floored + renormalised: a class with m_k = 0 makes the
    posterior divide by zero, so smoothing is mandatory, not cosmetic
  • Exact reverse posterior (v7's approximation available via a flag)
  • Prior sample at t=T drawn from Cat(m), not Uniform{K}
  • FIX: forward edge corruption is now symmetric. v7 corrupted the full
    N×N matrix independently, so x_t was asymmetric during training but
    symmetric at sampling time — a train/test mismatch.
  • FIX: sampling symmetrisation was `OR` of both triangles, which inflates
    edge probability (p_or = 2p - p²) and fights the marginal you just
    installed. Now: sample the upper triangle, mirror it.
  • Inverse-frequency class weights generalised to K > 2 edge classes
  • Optional genuinely-independent per-stream timesteps (v7's docstring
    claimed this; the code passed the same t to both)
  • Optional clamped context in sample() — the hook for inpainting /
    gap-filling, where part of the graph is observed and held fixed
  • AdjacencyTransformer / NodeTransformer unchanged

"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphpath.config import MarginalConfig
from graphpath.layers.serial_tower import AdjacencyTransformer, NodeTransformer
from graphpath.processes.marginal import (MarginalTransitionSchedule, _mirror_upper,
                                          _smooth, estimate_marginals)
from graphpath.samplers.marginal import sample_marginal

# Names the experiment notebooks use through `import graphpath.models.marginal as MM`.
Config = MarginalConfig
__all__ = ["GraphDiffBERTMarginal", "GraphDiffBERT", "MarginalConfig", "Config",
           "MarginalTransitionSchedule", "estimate_marginals", "train_step"]


def __getattr__(name):
    # `MM.train_step` in the notebooks: the shared step in graphpath.training.loop
    if name == "train_step":
        from graphpath.training.loop import train_step
        return train_step
    raise AttributeError(name)


class GraphDiffBERTMarginal(nn.Module):
    """
    Joint marginal-transition-matrix diffusion model.

    Both node and edge streams:
      • No MASK token — vocab is exactly the clean classes
      • Forward: x_t = x_0 w.p. ᾱ_t, else draw from Cat(m)
      • Reverse: exact posterior computed analytically from predicted x_0
      • Loss: noise-weighted CE over ALL positions
      • Prior at t=T is Cat(m), so sampling starts at the right density

    forward() returns:
        node_logits : (B, N, node_vocab)       — x_0 class predictions
        edge_logits : (B, N, N, edge_vocab)    — x_0 class predictions
    """

    def __init__(
        self,
        cfg: MarginalConfig,
        vocab_size: int,
        m_node: Optional[Tensor] = None,
        m_edge: Optional[Tensor] = None,
    ):
        """
        vocab_size = dataset.vocab_size + 1  (API compat — +1 is ignored,
        internally node_vocab = vocab_size - 1 as before)

        m_node / m_edge : class marginals. If omitted the model falls back
        to uniform (i.e. v7 behaviour) and warns — call set_marginals()
        with values from estimate_marginals() before training.
        """
        super().__init__()
        self.cfg        = cfg
        self.node_vocab = vocab_size - 1      # keep API compat
        self.edge_vocab = cfg.edge_vocab
        cfg.node_vocab  = self.node_vocab

        self.schedule = MarginalTransitionSchedule(cfg)
        self.node_tf  = NodeTransformer(cfg)
        self.adj_tf   = AdjacencyTransformer(cfg)

        d = cfg.d_model
        self.node_norm = nn.LayerNorm(d)
        self.node_head = nn.Linear(d, self.node_vocab)
        self.edge_norm = nn.LayerNorm(d)
        self.edge_head = nn.Linear(d, self.edge_vocab)

        if m_node is None or m_edge is None:
            print("[GraphDiffBERT v8] WARNING: no marginals supplied — "
                  "falling back to UNIFORM (this is v7). Call "
                  "set_marginals(*estimate_marginals(loader, ...)) first.")
            m_node = torch.full((self.node_vocab,), 1.0 / self.node_vocab)
            m_edge = torch.full((self.edge_vocab,), 1.0 / self.edge_vocab)

        # token -> type lookup. Every entry starts at the reserved "unknown"
        # slot (= n_node_types) so an unset model behaves as if untyped.
        if cfg.n_node_types > 0:
            self.register_buffer("token_to_type",
                                 torch.full((self.node_vocab,), cfg.n_node_types,
                                            dtype=torch.long))
            # zero-init, so typing starts as an exact no-op and the model has
            # to earn any use of it
            self.type_pair_bias = (nn.Parameter(torch.zeros(
                cfg.n_node_types + 1, cfg.n_node_types + 1, self.edge_vocab))
                if cfg.use_type_bias else None)
        else:
            self.token_to_type = None
            self.type_pair_bias = None

        self.register_buffer("m_node", torch.empty(self.node_vocab))
        self.register_buffer("m_edge", torch.empty(self.edge_vocab))
        self.set_marginals(m_node, m_edge)

        self._init_weights()

    # ── node types ───────────────────────────────────────────────────────

    def set_node_types(self, token_to_type) -> None:
        """
        Register the node-token -> node-type lookup, as graphpath's
        NodeTypeMixin does. Requires cfg.n_node_types > 0.

            token_to_type = torch.tensor([...])   # len >= model.node_vocab
            model.set_node_types(token_to_type)
        """
        if self.cfg.n_node_types == 0:
            raise RuntimeError(
                "Config.n_node_types must be > 0 to use node types; "
                "rebuild the model with n_node_types set.")
        tt = torch.as_tensor(token_to_type, dtype=torch.long,
                             device=self.token_to_type.device)
        assert tt.numel() >= self.node_vocab, \
            f"expected at least {self.node_vocab} entries, got {tt.numel()}"
        self.token_to_type[: self.node_vocab] = tt[: self.node_vocab]

    def _type_ids(self, node_toks: Tensor) -> Optional[Tensor]:
        return (self.token_to_type[node_toks]
                if self.cfg.n_node_types > 0 else None)

    def _type_scale(self, t: Tensor) -> Optional[Tensor]:
        """alpha_bar_t gate, or None when the gate is off."""
        if not self.cfg.type_alpha_gate or self.cfg.n_node_types == 0:
            return None
        return self.schedule.ac[t]

    def _add_type_pair_bias(self, edge_logits: Tensor,
                            type_ids: Optional[Tensor],
                            scale: Optional[Tensor]) -> Tensor:
        if self.type_pair_bias is None or type_ids is None:
            return edge_logits
        N = type_ids.shape[1]
        ti = type_ids.unsqueeze(2).expand(-1, -1, N)
        tj = type_ids.unsqueeze(1).expand(-1, N, -1)
        b = self.type_pair_bias[ti, tj]                  # (B,N,N,edge_vocab)
        if scale is not None:
            b = b * scale[:, None, None, None]
        return edge_logits + b

    def set_marginals(self, m_node: Tensor, m_edge: Tensor) -> None:
        """Install (and smooth) the class marginals. Safe to call after
        construction — e.g. once you've swept the loader."""
        m_node = _smooth(torch.as_tensor(m_node, dtype=torch.float),
                         self.cfg.marginal_floor)
        m_edge = _smooth(torch.as_tensor(m_edge, dtype=torch.float),
                         self.cfg.marginal_floor)
        assert m_node.numel() == self.node_vocab, \
            f"m_node has {m_node.numel()} classes, expected {self.node_vocab}"
        assert m_edge.numel() == self.edge_vocab, \
            f"m_edge has {m_edge.numel()} classes, expected {self.edge_vocab}"
        self.m_node.copy_(m_node.to(self.m_node.device))
        self.m_edge.copy_(m_edge.to(self.m_edge.device))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        node_toks: Tensor,   # (B, N)     long — noisy, in {0..node_vocab-1}
        edge_toks: Tensor,   # (B, N, N)  long — noisy, in {0..edge_vocab-1}
        t_node:    Tensor,   # (B,)       long
        t_edge:    Tensor,   # (B,)       long
    ) -> Tuple[Tensor, Tensor]:
        type_ids = self._type_ids(node_toks)
        scale    = self._type_scale(t_node)
        h_node      = self.node_tf(node_toks, edge_toks, t_node, type_ids, scale)
        h_edge      = self.adj_tf(edge_toks, h_node, t_edge)
        node_logits = self.node_head(self.node_norm(h_node))        # (B,N,Kn)
        edge_logits = self.edge_head(self.edge_norm(h_edge))        # (B,N,N,Ke)
        edge_logits = self._add_type_pair_bias(edge_logits, type_ids,
                                               self._type_scale(t_edge))
        return node_logits, edge_logits

    # ── loss ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _inv_freq_weights(targets: Tensor, K: int) -> Tensor:
        """
        Inverse-frequency class weights, generalised to K classes.

        v7 hard-coded the 2-class case. With a skewed marginal and K>2 edge
        types (substrate×1, product×2, …) you need this, or the model
        collapses onto class 0 and the CE looks great.
        """
        counts = torch.bincount(targets.reshape(-1), minlength=K).float()
        w = 1.0 / counts.clamp(min=1.0)
        w = torch.where(counts > 0, w, torch.zeros_like(w))
        return w / w.sum().clamp(min=1e-12) * (counts > 0).sum().clamp(min=1)

    def compute_loss(
        self,
        node_feats:       Tensor,    # (B, N)    long — clean
        adjcs:            Tensor,    # (B, N, N) long — clean
        pad_mask:         Tensor,    # (B, N)    bool — True = valid node
        loss_edge_weight: float = 1.0,
        co_expr:          Optional[dict] = None,   # unused: this model is unconditioned
    ) -> dict:
        """
        Loss = noise-weighted CE on x_0 predictions at ALL positions.

        Noise weighting: (1 - ᾱ_t) per sample, per stream.
        Edge class balancing: inverse-frequency weights within the batch.
        """
        B, N   = node_feats.shape
        device = node_feats.device

        t_n = torch.randint(0, self.cfg.T, (B,), device=device)
        t_e = torch.randint(0, self.cfg.T, (B,), device=device) \
              if self.cfg.independent_t else t_n

        # ── Forward diffusion (marginal corruption) ───────────────────────
        sup_pad = getattr(self.cfg, "supervise_pad", False)
        node_t = self.schedule.q_sample(node_feats, t_n, self.m_node)
        if not sup_pad:
            node_t = torch.where(pad_mask, node_t, node_feats)  # padding clean

        valid_pair = pad_mask[:, :, None] & pad_mask[:, None, :]
        edge_t     = self.schedule.q_sample(adjcs, t_e, self.m_edge)
        if self.cfg.symmetric_edges:
            # FIX vs v7: corrupt symmetrically, so x_t at training time has
            # the same symmetry the sampler enforces at inference time.
            edge_t = _mirror_upper(edge_t)
        edge_t = torch.where(valid_pair, edge_t, adjcs)
        diag   = torch.eye(N, device=device, dtype=torch.bool)
        edge_t = edge_t.masked_fill(diag[None], 0)

        # ── Predict x_0 ───────────────────────────────────────────────────
        node_logits, edge_logits = self.forward(node_t, edge_t, t_n, t_e)

        # ── Noise-level weights ───────────────────────────────────────────
        #
        # THIS IS THE FIX. v7/v8 weighted every sample by (1 - alpha_bar_t),
        # which is ~0 where the input is nearly clean and 1 where it is pure
        # noise. That points the gradient at exactly the wrong regime: at high
        # t the input carries no information about x_0, so the loss-minimising
        # prediction IS the marginal, and the model converges to emitting it.
        # Meanwhile the low-noise regime — which is what the last reverse
        # steps and the final argmax actually use — gets almost no gradient
        # at all: t=5 has alpha_bar = 0.9994 and therefore weight 0.0006, and
        # the lower-noise half of the schedule carries 18.9% of the total.
        #
        # Measured, 16-graph memorisation, 600 steps, unweighted CE at fixed t:
        #
        #     t                    5      25      50      75      95
        #     (1-alpha_bar)    4.639   4.753   5.538   6.246   6.455
        #     unweighted       2.345   3.020   4.637   6.079   6.481
        #
        # Half the error at low t, no cost at high t. An x_0-parameterised
        # discrete diffusion loss should be a plain CE (D3PM eq. 5, DiGress
        # sec. 3.3); the weighting was the reason nothing trained.
        if self.cfg.loss_noise_weighting == "one_minus_alpha_bar":
            nw_n = (1.0 - self.schedule.ac[t_n])        # (B,)
            nw_e = (1.0 - self.schedule.ac[t_e])        # (B,)
        else:
            nw_n = torch.ones_like(self.schedule.ac[t_n])
            nw_e = torch.ones_like(self.schedule.ac[t_e])

        # ── Node CE — all valid positions, noise-weighted ─────────────────
        node_ce = F.cross_entropy(
            node_logits.reshape(B * N, self.node_vocab),
            node_feats.reshape(B * N),
            reduction="none",
        ).reshape(B, N)

        if sup_pad:
            # every slot is a target; padding carries its own weight so the
            # class balance can be corrected in whichever direction the
            # dataset needs
            w = torch.where(pad_mask, torch.ones_like(node_ce),
                            torch.full_like(node_ce,
                                            float(self.cfg.pad_loss_weight)))
            node_ce = node_ce * w
            n_valid = w.sum().clamp(min=1)
        else:
            node_ce = node_ce * pad_mask.float()
            n_valid = pad_mask.float().sum().clamp(min=1)
        node_ce  = node_ce * nw_n[:, None]
        loss_n   = node_ce.sum() / n_valid

        # ── Edge CE — all valid non-diagonal pairs, noise-weighted ────────
        vp = valid_pair & ~diag[None]                    # (B, N, N)

        # ── THE DENSITY FIX ───────────────────────────────────────────────
        #
        # Inverse-frequency class weights are wrong for a GENERATIVE model,
        # and they are why the samples came out ~13x too dense.
        #
        # Weighted cross-entropy does not have the data distribution as its
        # optimum. For two classes with weights (w0, w1) the minimiser is the
        # REBALANCED distribution
        #
        #     q = w1*p / (w1*p + w0*(1-p))
        #
        # With this data (p = 0.035) the weights come out (0.076, 1.924), a
        # ratio of 25.5, so the optimum is q = 0.48. A classifier is fine
        # with that — you rebalance to fight imbalance and then threshold.
        # But a diffusion model SAMPLES from its predicted distribution, so
        # the rebalancing is inherited directly by the graphs it draws.
        #
        # Fitting a single logit to a Bernoulli(0.035) confirms it exactly,
        # with no network involved:
        #
        #     no class weights        learned P(edge) 0.0341   1.0x true
        #     inverse-frequency       learned P(edge) 0.4834  13.8x true
        #
        # and the sampled densities measured off the trained model were 0.40
        # to 0.49 — 11 to 14x the data. Same number.
        #
        # The concern behind the weights is real: with p = 0.035 a model can
        # score well by predicting "no edge" everywhere. But the cure is
        # worse than the disease, and the marginal kernel is already the
        # right answer to it — under Q_t = a_t I + (1-a_t) 1 m^T the noisy
        # input x_t carries the true density at every t, so predicting the
        # edges is genuinely the way to lower the loss.
        if self.cfg.edge_class_weights:
            class_w = self._inv_freq_weights(adjcs[vp], self.edge_vocab).to(device)
        else:
            class_w = None

        edge_ce = F.cross_entropy(
            edge_logits.reshape(B * N * N, self.edge_vocab),
            adjcs.reshape(B * N * N),
            weight=class_w,
            reduction="none",
        ).reshape(B, N, N)

        edge_ce  = edge_ce * vp.float()
        edge_ce  = edge_ce * nw_e[:, None, None]
        m_valid  = vp.float().sum().clamp(min=1)
        loss_e   = edge_ce.sum() / m_valid

        loss = loss_n + loss_edge_weight * loss_e
        return {
            "loss":      loss,
            "loss_node": loss_n.item(),
            "loss_edge": loss_e.item(),
        }

    def _edge_logit_shift(self) -> Optional[Tensor]:
        """log(w) for the edge classes, or None when no correction is due.

        Derived from m_edge rather than from the batch, so the correction is a
        fixed property of the dataset and does not move with whatever happened
        to be sampled — `_inv_freq_weights` is computed per batch during
        training, but its EXPECTATION is the class frequency, which is m_edge.
        """
        if not (self.cfg.edge_class_weights and self.cfg.edge_logit_adjust):
            return None
        counts = self.m_edge.double().clamp(min=1e-12)
        w = 1.0 / counts
        w = w / w.sum() * (counts > 0).sum()          # matches _inv_freq_weights
        return w.log().to(self.m_edge.dtype)

    # ── sampling ─────────────────────────────────────────────────────────────

    def sample(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """Reverse-time generation. Implemented in `graphpath.samplers.marginal`."""
        return sample_marginal(self, *args, **kwargs)


GraphDiffBERT = GraphDiffBERTMarginal   # the class name the checkpoints were trained under
