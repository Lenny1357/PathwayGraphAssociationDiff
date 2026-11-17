"""GraphVAE — a one-shot baseline for the diffusion models.

Encode a graph to a single latent, decode the whole thing back in one pass.
No iterative refinement, no noise schedule: the point of comparison is
whether the diffusion process buys anything over sampling a latent and
decoding once.

The encoder and decoder are both `GraphTower` stacks, the same architecture
the diffusion models use. That is deliberate — swapping in a different
network would confound "VAE vs diffusion" with "this architecture vs that
one". What differs is only the generative process.

It exposes the same interface as the diffusion models — compute_loss(...)
and sample(...) — so `train_step`, `train` and every benchmark accept it
unchanged.

One caveat worth knowing before reading its scores. The reconstruction loss
compares slot i to slot i, so a one-shot decoder can only learn a per-slot
distribution; if the training loop permutes node order every batch (it does,
by design, for the diffusion models), the VAE cannot do better than the
marginal at each slot and its node accuracy will look poor for reasons that
have nothing to do with being a VAE. Train it with `train(..., permute=False)`
for a fair comparison, and say which setting a reported number came from.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphpath.layers.graph_block import ContinuousGraphTower, GraphTower
from graphpath.config import Config
from graphpath.layers.node_types import NodeTypeMixin
from graphpath.samplers.vae import sample_vae
from graphpath.training.class_weights import inverse_freq_weights


class GraphVAE(NodeTypeMixin, nn.Module):

    def __init__(self, cfg: Config, vocab_size: int, max_nodes: int = 256):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.node_vocab = vocab_size - 1        # API parity: the MASK slot is unused
        self.edge_vocab = cfg.edge_vocab
        cfg.node_vocab = self.node_vocab

        d, dz = cfg.d_model, cfg.latent_dim
        self.latent_dim = dz

        # ── encoder: the graph, as tokens, into a single latent ────────────
        self.encoder = GraphTower(cfg)
        self.to_mu     = nn.Linear(d, dz)
        self.to_logvar = nn.Linear(d, dz)

        # ── decoder: latent + a per-slot tag, back to nodes and edges ──────
        # Slots are exchangeable, so without a distinguishing feature every
        # slot decodes identically. The diffusion models break that symmetry
        # with RANDOM tags, because they are trained to be permutation
        # invariant. A VAE cannot use those: its reconstruction target is
        # slot-wise, and a tag redrawn on every forward pass is noise the
        # decoder can never invert — with random tags this model collapsed to
        # the prior and reconstructed at chance. Learned slot embeddings are
        # stable across calls, so the decoder can address a slot.
        self.max_nodes = max_nodes
        self.slot_emb = nn.Embedding(max_nodes, cfg.slot_dim)
        self.decoder = ContinuousGraphTower(cfg, node_in_dim=cfg.slot_dim,
                                            edge_in_dim=1)

        # The tower LayerNorms its input projection. A latent concatenated
        # there is constant across slots, so the norm squashes it against the
        # slot embeddings and the decoder learns to ignore it — measured:
        # reconstruction exactly equal to the slot-marginal baseline, and
        # zeroing z changing nothing. z is therefore injected through the
        # tower's ADDITIVE per-node path, which sits after that LayerNorm and
        # reaches h at full magnitude.
        self.encoder.use_rand_feat = False        # would add its own random tag
        self.decoder.use_rand_feat = True
        self.decoder.rand_feat_dim = dz
        self.decoder.node_id_proj = nn.Linear(dz, d)
        self.node_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, self.node_vocab))
        self.edge_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, self.edge_vocab))

        self.register_buffer("node_class_weight", torch.ones(self.node_vocab))
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))
        # optional BN on mu (cfg.vae_mu_batchnorm): removes what all graphs share and keeps
        # each latent dimension's spread across graphs at vae_bn_gamma
        self.mu_bn = nn.BatchNorm1d(dz, affine=False) if cfg.vae_mu_batchnorm else None
        self.mu_bn_shift = nn.Parameter(torch.zeros(dz)) if cfg.vae_mu_batchnorm else None

        # optional bag-of-words head (cfg.vae_bow_weight): z -> which entities the graph holds
        self.bow_head = nn.Linear(dz, self.node_vocab) if cfg.vae_bow_weight > 0 else None

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
        for tower in (self.encoder, self.decoder):
            tower.post_init_zeros()

        # Applied AFTER the generic sweep above, which would otherwise undo
        # them. Both are required — see Config.init_logvar / z_proj_std.
        nn.init.constant_(self.to_logvar.bias, self.cfg.init_logvar)
        nn.init.normal_(self.decoder.node_id_proj.weight, std=self.cfg.z_proj_std)
        nn.init.zeros_(self.decoder.node_id_proj.bias)

    def set_output_prior(self, node_counts, edge_counts=None) -> None:
        """
        Initialise the output biases to the log class frequencies.

        Why this is worth a method rather than a footnote: the decoder can
        reach the class-marginal using its output BIAS alone — no encoder, no
        latent, no slot embedding, nothing on the input path. That solution is
        a wide, easily-found basin, and gradient descent goes there first. On
        E. coli the whole of a 520-step run was spent travelling from chance
        (ln 6472 = 8.775) to the token-marginal (8.236), which is exactly that
        basin, and the run ended before anything else began.

        Setting the bias to log(p) hands the model that solution at step 0, so
        the budget goes to the part that needs it. The logits start at the
        class prior, and the weights only ever have to learn the DEVIATION
        from it. Same idea as the prior-init in the focal-loss paper.

        Call it once, after construction (the generic init zeroes biases) and
        before training:

            counts = torch.bincount(node_feats[pad_mask].flatten(),
                                    minlength=model.node_vocab)
            model.set_output_prior(counts, edge_counts)   # edge optional

        `edge_counts` is (n_no_edge, n_edge) over valid off-diagonal pairs.
        The edge head has the same problem in a sharper form — at ~4% density
        "no edge everywhere" already scores H(0.04) = 0.17 nats against
        ln 2 = 0.69.
        """
        def _log_p(counts, n):
            c = torch.as_tensor(counts, dtype=torch.float,
                                device=self.node_class_weight.device).clamp(min=1e-6)
            assert c.numel() == n, f"expected {n} counts, got {c.numel()}"
            return (c / c.sum()).log()

        with torch.no_grad():
            # node_out / edge_out are Sequential(LayerNorm, Linear); the
            # LayerNorm output is ~zero-mean, so at step 0 the logits ARE the
            # bias and the model emits exactly the prior
            self.node_out[-1].bias.copy_(_log_p(node_counts, self.node_vocab))
            if edge_counts is not None:
                self.edge_out[-1].bias.copy_(_log_p(edge_counts, self.edge_vocab))

    def set_node_class_weights(self, counts, power: Optional[float] = None):
        """See GraphDiffBERT.set_node_class_weights — same contract."""
        power = self.cfg.node_class_weight_power if power is None else power
        self.node_class_weight.copy_(inverse_freq_weights(
            counts, power, self.node_vocab, self.node_class_weight.device))

    # ── encode / decode ────────────────────────────────────────────────────

    def _zero_t(self, B, device):
        """The towers take a diffusion timestep; there isn't one here."""
        return torch.zeros(B, dtype=torch.long, device=device)

    def encode(self, node_feats: Tensor, adjcs: Tensor, pad_mask: Tensor,
               co_expr: Optional[dict] = None) -> Tuple[Tensor, Tensor]:
        B = node_feats.shape[0]
        t = self._zero_t(B, node_feats.device)
        h, _ = self.encoder(node_feats, adjcs, t, t,
                            type_ids=self._type_ids(node_feats), co_expr=co_expr)
        # mean over real nodes only — padding would otherwise dilute the
        # latent in proportion to how short the pathway is
        w = pad_mask.float().unsqueeze(-1)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        mu = self.to_mu(pooled)
        if self.mu_bn is not None:
            mu = self.cfg.vae_bn_gamma * self.mu_bn(mu) + self.mu_bn_shift
        return mu, self.to_logvar(pooled).clamp(-8.0, 8.0)

    def decode(self, z: Tensor, N: int, co_expr: Optional[dict] = None,
               node_toks: Optional[Tensor] = None, raw_edges: bool = False) -> Tuple[Tensor, Tensor]:
        """(node logits, edge logits). Edge logits get the type-pair bias from
        `node_toks` (default: the argmax node tokens); `raw_edges` returns them
        before it, for a sampler that picks the nodes first (see edge_head)."""
        B = z.shape[0]
        device = z.device
        if N > self.max_nodes:
            raise ValueError(f"N={N} exceeds max_nodes={self.max_nodes}; "
                             f"rebuild with max_nodes >= N")
        node_state = self.slot_emb(torch.arange(N, device=device))[None].expand(B, -1, -1)
        z_feat = z[:, None, :].expand(-1, N, -1)      # additive path, see __init__
        edge_state = torch.zeros(B, N, N, 1, device=device)

        t = self._zero_t(B, device)
        h, e = self.decoder(node_state, edge_state, t, t,
                            node_id_feat=z_feat, co_expr=co_expr)
        node_logits = self.node_out(h)
        edge_logits = self.edge_out(e)
        if raw_edges:
            return node_logits, edge_logits
        return node_logits, self.edge_head(edge_logits,
                                           node_logits.argmax(-1) if node_toks is None else node_toks)

    @torch.no_grad()
    def recalibrate_bn(self, loader, device, max_batches: int = 200) -> None:
        """Exact BN statistics for mu ("precise BN"), from the training set with
        the final encoder. Needed before encoding in eval mode: pre-BN mu varies
        little between graphs, so BN divides by a small spread and the lagging
        running averages turn into large latent shifts (held-out CE swung
        7.5-9.4 with them, a steady 7.36 with exact statistics). Generation from
        the prior never passes through BN and is unaffected."""
        if self.mu_bn is None:
            return
        was = self.training
        self.eval(); self.mu_bn.train()
        self.mu_bn.reset_running_stats(); self.mu_bn.momentum = None   # cumulative average
        for i, x in enumerate(loader):
            if i >= max_batches:
                break
            adj, ids, msk = x[0].long().to(device), x[1].to(device), x[2].bool().to(device)
            if self.cfg.vae_canonical_order:
                ids, adj, msk = self.canonical_order(ids, adj, msk)
            self.encode(ids, adj, msk)
        self.mu_bn.momentum = 0.1
        self.train(was)

    def edge_head(self, edge_raw: Tensor, node_toks: Tensor) -> Tensor:
        """Raw edge logits -> final: type-pair bias from `node_toks`, then
        (cfg.vae_symmetric_edges) averaged with the transpose."""
        out = self._add_type_pair_bias(edge_raw, self._type_ids(node_toks) if self.n_types else None)
        if self.cfg.vae_symmetric_edges:
            out = 0.5 * (out + out.transpose(1, 2))
        return out

    def forward(self, node_feats, adjcs, pad_mask, co_expr=None):
        mu, logvar = self.encode(node_feats, adjcs, pad_mask, co_expr)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        toks = node_feats if self.cfg.vae_teacher_types else None
        return self.decode(z, node_feats.shape[1], co_expr=co_expr, node_toks=toks), mu, logvar

    # ── loss ───────────────────────────────────────────────────────────────

    def _beta(self) -> float:
        w = self.cfg.kl_warmup_steps
        if w <= 0:
            return self.cfg.kl_weight
        return self.cfg.kl_weight * min(1.0, float(self._step) / w)

    @staticmethod
    def canonical_order(node_feats: Tensor, adjcs: Tensor, pad_mask: Tensor):
        """The graph with its slots sorted by token id, padding last: one fixed
        meaning per slot ("the i-th smallest entity"), so a slot-wise decoder
        has a target it can learn. Edges and the mask move with their nodes."""
        N = node_feats.shape[1]
        key = torch.where(pad_mask, node_feats, torch.full_like(node_feats, torch.iinfo(node_feats.dtype).max))
        p = key.argsort(dim=1, stable=True)
        bi = torch.arange(node_feats.shape[0], device=node_feats.device)[:, None]
        adjcs = adjcs.gather(1, p[..., None].expand(-1, -1, N)).gather(2, p[:, None, :].expand(-1, N, -1))
        return node_feats[bi, p], adjcs, pad_mask[bi, p]

    def compute_loss(self, node_feats: Tensor, adjcs: Tensor, pad_mask: Tensor,
                     loss_edge_weight: float = 1.0,
                     co_expr: Optional[dict] = None) -> dict:
        B, N = node_feats.shape
        device = node_feats.device
        if self.cfg.vae_canonical_order:
            node_feats, adjcs, pad_mask = self.canonical_order(node_feats, adjcs, pad_mask)
        (node_logits, edge_logits), mu, logvar = self.forward(
            node_feats, adjcs, pad_mask, co_expr)

        # with supervise_pad the PAD slots are targets too (token 0), so the
        # decoder learns graph size; canonical order puts them last
        node_tgt = torch.ones_like(pad_mask) if self.cfg.supervise_pad else pad_mask
        loss_n = torch.tensor(0.0, device=device)
        if node_tgt.any():
            loss_n = F.cross_entropy(
                node_logits[node_tgt], node_feats[node_tgt],
                weight=self.node_class_weight if self.cfg.use_node_class_weights else None,
                reduction="mean")

        diag = torch.eye(N, device=device, dtype=torch.bool)
        vp = (pad_mask[:, :, None] & pad_mask[:, None, :]) & ~diag[None]
        loss_e = torch.tensor(0.0, device=device)
        if vp.any():
            loss_e = F.cross_entropy(edge_logits[vp], adjcs[vp], reduction="mean")

        # KL with free bits: dimensions already below the floor are exempt, so
        # the penalty cannot squeeze every dimension to the prior at once
        kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)     # (B, dz)
        fb = self.cfg.kl_free_bits
        kl = kl_dim.mean(0).clamp(min=fb).sum() if fb > 0 else kl_dim.sum(1).mean()

        beta = self._beta()
        loss = loss_n + loss_edge_weight * loss_e + beta * kl

        # bag of words: every real token of the graph scored under one
        # distribution read from z, so z has to carry what is in the graph
        loss_bow = torch.tensor(0.0, device=device)
        if self.bow_head is not None and pad_mask.any():
            bow = F.log_softmax(self.bow_head(mu), dim=-1)                 # (B, V)
            tok = node_feats.clamp(max=self.node_vocab - 1)
            loss_bow = -(bow.gather(1, tok) * pad_mask).sum() / pad_mask.sum()
            loss = loss + self.cfg.vae_bow_weight * loss_bow
        if self.training:
            self._step += 1

        return {"loss": loss,
                "loss_node": loss_n.item(),
                "loss_edge": loss_e.item(),
                "loss_kl":   kl.item(),
                "loss_bow":  loss_bow.item(),
                "beta":      beta}

    # ── sampling ─────────────────────────────────────────────────────────────

    def sample(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """Reverse-time generation. Implemented in `graphpath.samplers.vae`."""
        return sample_vae(self, *args, **kwargs)

    @torch.no_grad()
    def reconstruct(self, node_feats: Tensor, adjcs: Tensor, pad_mask: Tensor,
                    greedy: bool = True):
        """Encode then decode — how well the latent preserves a real graph."""
        mu, _ = self.encode(node_feats, adjcs, pad_mask)
        node_logits, edge_logits = self.decode(mu, node_feats.shape[1])
        node_toks = node_logits.argmax(-1)
        adj = (edge_logits.argmax(-1) == 1).long()
        adj = adj | adj.transpose(-1, -2)
        diag = torch.eye(node_feats.shape[1], device=node_feats.device, dtype=torch.bool)
        return node_toks, adj.masked_fill(diag[None], 0)
