"""Model / diffusion hyper-parameters."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    node_vocab:   int   = 10       # clean node types; MASK = node_vocab
    edge_vocab:   int   = 2        # clean edge types (0,1); MASK = edge_vocab
    d_model:      int   = 128
    n_heads:      int   = 4
    n_layers:     int   = 4        # layers in the interleaved graph tower
    d_ff:         int   = 256
    dropout:      float = 0.1
    edge_n_heads: int   = 0        # 0 → same as n_heads
    T:            int   = 500
    schedule:     str   = "cosine"
    # FIX 1: exponent for timestep sampling; 2.0 → quadratic bias toward low t
    t_sample_exp: float = 2.0

    # ── FIX 5: random node-identity features (permutation symmetry breaking) ──
    # Node slots are NOT canonically ordered (the training loop explicitly
    # randomizes node order every epoch via iso_perm), so a learned
    # positional-embedding table would fight that augmentation and wouldn't
    # transfer to sampling anyway. Instead every node slot gets its own
    # random tag, which is enough to make otherwise-identical (fully masked)
    # node slots distinguishable to self-attention.
    # Where the random identity feature is applied.
    #   "all"           every slot, as FIX 5 originally did
    #   "placeholders"  only slots holding PAD or MASK — the ones that are
    #                   genuinely indistinguishable without it. A real token
    #                   keeps its embedding intact, which matters because the
    #                   projected random vector has ~5.7x the norm of the
    #                   token embedding it is added to and otherwise buries
    #                   the identity the model is being asked to predict.
    # Supervise the padding slots instead of holding them clean and excluding
    # them from the loss. Without this NOTHING is ever trained to emit PAD, so
    # at sampling time — where there is no pad_mask — the model has no way to
    # produce a graph shorter than N and must fill every slot. That is the
    # source of the duplicate-entity rates.
    supervise_pad:    bool = False
    # Weight on the padding slots' contribution when supervise_pad is on.
    # PAD is a majority class whenever graphs are much shorter than N, and
    # the final decode is an argmax — so an unweighted CE makes PAD win every
    # slot. Down-weight it until PAD stops dominating the argmax.
    pad_loss_weight:  float = 1.0

    # Distribution a corrupted EDGE token is drawn from under the uniform
    # transition kernel. None keeps the plain uniform kernel, whose stationary
    # distribution is 50% density against real pathways at ~5.5%. Set it to
    # the data density (or call model.set_edge_marginal) to use D3PM's
    # marginal-matched kernel instead.
    edge_marginal_p:  Optional[float] = None

    # How sample() makes the adjacency symmetric each reverse step.
    #   "or"    take the OR of both directions. What the sibling project does,
    #           and a defect: (i,j) and (j,i) are two INDEPENDENT draws from
    #           the same posterior, so OR maps a per-entry probability p to
    #           1-(1-p)^2, and the next step anchors on that inflated state.
    #           With the marginal kernel it equilibrates at ~43% density
    #           against the 5.5% the model is trained on — 8x out of
    #           distribution, which is why samples come out as hairballs
    #           however well the loss falls.
    #   "triu"  draw the upper triangle and mirror it. One draw per undirected
    #           edge, so the sampled density matches the training density at
    #           every step.
    # Default stays "or" so existing runs and the debug/ replica comparison
    # are unaffected; set "triu" for anything you intend to sample from.
    edge_symmetrise: str = "or"      # "or" | "triu"

    # ── SeparateTower (the sibling project's sequential architecture) ─────
    # Defaults reproduce it exactly. Both knobs exist so the two differences
    # from GraphTower can be ablated one at a time.
    sep_gat_pre:        bool = True    # one round of edge-gated message passing
                                       # at the INPUT, before the layer stack
    sep_attn_edge_bias: bool = False   # add a per-layer edge bias to the node
                                       # attention. GraphTower does this at
                                       # every layer; the sibling project
                                       # declares the projection and never
                                       # calls it, so its attention sees no
                                       # edge signal at all.

    rand_feat_scope:  str  = "all"       # "all" | "placeholders"
    pad_id:           int  = 0           # token treated as padding
    use_random_node_features: bool = True
    rand_feat_dim:            int  = 32

    # ── FIX 7: node class balancing ────────────────────────────────────────
    # Edge loss already balances positive/negative classes; node loss did
    # not, which let a few high-frequency node types (e.g. near-duplicate
    # lipid species) dominate the CE gradient and get over-sampled at
    # generation time. power=0 → off, 0.5 → sqrt-inverse-freq (gentle,
    # recommended default), 1.0 → full inverse-freq (aggressive).
    use_node_class_weights:   bool  = True
    node_class_weight_power:  float = 0.5

    # ── FIX 8: don't starve the high-t ("cold start") training regime ─────
    # sample_t's low-t^2 bias means ~50% of training batches have t < T/4
    # (mostly-revealed, easy). Sampling *starts* at t = T-1 (fully masked),
    # which is exactly the regime this under-trains. Mix in some uniform-t
    # batches so the model still sees plenty of near-fully-masked examples.
    uniform_t_mix_p: float = 0.15

    # ── FIX 9 / 10: sampling-time confidence-guided decoding ──────────────
    use_confidence_unmask:  bool  = True   # reveal most-confident masked positions first, not random ones
    # Gumbel noise scale for the reveal/remask ordering. Scores are LOG
    # probabilities, so this is in log-prob units: 0 = greedy (deterministic
    # ordering), ~0.3-1.0 = meaningfully stochastic, >>1 = effectively random
    # order. (Adding the same noise to raw probabilities instead makes any
    # value above ~0.01 equivalent to random selection, since competing
    # candidates differ by far less than the noise.)
    confidence_temperature: float = 0.3
    remask_frac:            float = 0.06   # fraction of revealed, non-fixed positions to reconsider each step
    remask_stop_frac:       float = 0.15   # stop remasking for the last 15% of steps so sampling converges

    # Reveal budgets are carried as fractions across steps rather than
    # rounded per step. Rounding drops every budget below 0.5, which with
    # only N node slots leaves the first ~30% of the trajectory revealing
    # nothing at all (edges, having N^2 candidates, are unaffected).
    # Set False for the old round-per-step behaviour.
    carry_reveal_budget:    bool = True

    # Optionally let confident predictions run AHEAD of the schedule: any
    # still-masked position whose top-1 probability exceeds the threshold is
    # revealed even if the step's budget is already spent. None disables it.
    # This trades exactness for speed — the state then contains fewer masks
    # than q(x_t | x_0) would produce, i.e. slightly off the training
    # distribution — but the positions it commits early are precisely the
    # ones the model is most sure about. Note a binary edge distribution has
    # max-probability >= 0.5 by construction, so the edge threshold must be
    # set far higher than the node one (try 0.99+) or it reveals everything.
    confidence_reveal_thresh:      Optional[float] = None   # nodes
    edge_confidence_reveal_thresh: Optional[float] = None   # edges

    # ── FIX 11: typed connectivity (gene→enzyme→reaction→metabolite, etc.) ─
    # Nothing in the base architecture is told that node identities fall
    # into a small number of TYPES with hard rules about which types may be
    # adjacent. The model has to rediscover that abstraction purely from
    # co-occurrence statistics over specific entity tokens, which is much
    # harder to generalize than an explicit categorical feature — this is
    # usually why loss/graph-shape can look fine while type-adjacency is
    # still wrong. n_node_types=0 disables all of this (fully backward
    # compatible); set it to your number of node types (e.g. 4) to enable.
    n_node_types: int  = 0
    use_type_bias: bool = True   # learned per-type-pair bias added to edge logits

    # ── co-expression conditioning (all three variants) ───────────────────
    # Pathway generation is conditioned on a tissue's gene co-expression
    # graph, supplied as a sparse (n_tissues, V, V) tensor. The signal is
    # reduced to one pooled vector per sample via degree-weighted hub
    # pooling and injected once at the tower input (additive gate + FiLM),
    # not per-layer. use_coexpr=False disables all of it.
    use_coexpr:     bool  = False
    cond_d_model:   int   = 64    # width of the pooled vector, not a memory bank
    n_tissues:      int   = 1     # must match the number of tissues in your data
    top_k_hubs:     Optional[int] = 8   # per-tissue hub genes to pool over; None → all
    cond_dropout_p: float = 0.1   # CFG-style conditioning dropout during training

    # ── backbone ──────────────────────────────────────────────────────────
    # Which block the tower stacks. Everything else — embeddings, time
    # conditioning, type embedding, identity features, the co-expression
    # conditioner — is shared, so switching this compares the attention
    # mechanism and nothing else.
    #   "tower"             the interleaved GraphBlock (FIX 12)
    #   "graph_transformer" Dwivedi & Bresson (2021), edge-gated attention
    #                       with the pre-softmax scores recycled as edges
    backbone:     str  = "tower"
    gt_ffn_mult:  int  = 4       # graph_transformer FFN width multiplier
    gt_time_film: bool = False   # per-layer timestep FiLM from the paper.
                                 # Off by default: GraphBlock has no
                                 # equivalent, so enabling it for one
                                 # backbone only confounds the comparison.

    # ── Graph VAE baseline (graphpath/models/vae.py) ─────────────────────────────
    # Ignored by the diffusion variants.
    latent_dim:      int   = 64     # graph-level latent
    slot_dim:        int   = 32     # learned per-slot embedding for the decoder
    # Two initialisations that decide whether the latent gets used at all.
    # Measured on a toy set of 8 graphs: with either at its "natural" value
    # the model reconstructs at exactly the slot-marginal baseline (node CE
    # pinned at ln 8) and zeroing z changes nothing; with both, node CE 0.12
    # and reconstruction accuracy 1.00.
    init_logvar:     float = -3.0   # start the posterior narrow (std ~0.22) so
                                    # mu is not drowned by sampling noise
    z_proj_std:      float = 0.5    # z enters through an additive projection;
                                    # at the generic 0.02 init its contribution
                                    # is ~10% of the slot signal and the
                                    # decoder never learns to read it
    kl_weight:       float = 1.0    # beta; <1 trades KL for reconstruction
    kl_free_bits:    float = 0.05   # nats per latent dim exempt from the KL
                                    # penalty. Discrete-graph VAEs collapse to
                                    # the prior readily; this is the cheapest
                                    # guard. 0 disables it.
    kl_warmup_steps: int   = 2000   # linearly ramp beta from 0 over this many
                                    # steps, so reconstruction is learned before
                                    # the KL starts pulling
    vae_canonical_order: bool = False  # reconstruct each graph with its nodes sorted
                                    # by token id (PAD last). Training graphs come in
                                    # the chunk sampler's arbitrary order, so slot i
                                    # means something different in every graph and the
                                    # slot-wise target is unlearnable: the oct2 VAEs sat
                                    # at the token marginal for all 20k steps
    vae_mu_batchnorm: bool = False  # BN-VAE: batch-normalise mu (fixed gain vae_bn_gamma,
                                    # learned shift). mu is a mean over a graph's node
                                    # states, so what every graph shares (time embedding
                                    # at t=0, type embeddings) dominates it: its graph-to-
                                    # graph spread starts at ~3% of its size, the decoder
                                    # learns slot/marginal biases instead, and the latent
                                    # collapses to one value for all graphs (debug_vae_towers.py)
    vae_bn_gamma:    float = 1.0    # spread BN holds each mu dimension at across graphs
    vae_symmetric_edges: bool = False  # symmetrise edge logits and sample each pair ONCE
                                    # (upper triangle, mirrored). Off reproduces the old
                                    # sampler, which drew (i,j) and (j,i) independently and
                                    # OR-ed them: density ~2p, e.g. E. coli 0.075 vs 0.040
    vae_teacher_types: bool = False # type-pair edge bias from the TRUE node tokens in
                                    # training and the SAMPLED ones at generation, as the
                                    # diffusion models see real tokens. Off reproduces the
                                    # old argmax-of-logits types, which near the marginal
                                    # are one type everywhere and carry nothing
    vae_bow_weight:  float = 0.0    # bag-of-words loss: predict the graph's entity set
                                    # from z alone, which forces content into the latent
                                    # (the standard cure for an unused VAE latent). 0 = off

    # ── relational co-expression conditioning ─────────────────────────────
    # The pooled conditioner applies one vector to every slot, so it can
    # express "this is a lung-ish context" but not "A and B belong together
    # in lung". These give that statement a channel. Both need
    # model.set_coexpression(...) and are inert until then.
    use_pairwise_bias:  bool = False   # co-expression -> edge logits
    pairwise_hidden:    int  = 32
    use_coexpr_attn:    bool = False   # nodes attend over their co-expressed partners
    coexpr_attn_k:      int  = 16      # partners per entity; memory is T*V*k
    coexpr_attn_heads:  int  = 4
    coexpr_attn_layers: str  = "all"   # "all" | "first" | "last"

    # ── continuous (Gaussian) diffusion in embedding space ────────────────
    # Only read by ContinuousGraphDiffusion; ignored by the two discrete
    # variants. Node tokens are encoded to `bottleneck_dim` vectors, diffused
    # with Gaussian noise alongside a real-valued adjacency, and decoded back
    # to tokens at the end.
    embed_dim:        int   = 0      # 0 → d_model; UnifiedEmbeddingModule width
    bottleneck_dim:   int   = 64     # dimension actually diffused
    loss_mse_weight:  float = 1.0    # x̂₀ reconstruction in embedding space
    loss_ce_weight:   float = 1.0    # CE after decoding x̂₀ back to tokens
    loss_adj_weight:  float = 1.0    # adjacency BCE
    adj_pos_weight:   float = 20.0   # BCE pos_weight — graphs are sparse
    self_cond_p:      float = 0.5    # probability of a self-conditioning pass
    # Node types are not directly observable in embedding space, so they are
    # estimated by decoding the current noisy state; anything the decoder is
    # less sure about than this falls back to the "unknown type" slot.
    type_conf_thresh: float = 0.5

    # ── "separate" continuous backend (NodeDenoiser + AdjacencyHead) ──────
    # Only used when ContinuousGraphDiffusion is built with backend="separate"
    # instead of the interleaved GraphTower. 0 → inherit the shared setting.
    adj_hidden_dim:   int  = 0       # 0 → d_model
    adj_n_heads:      int  = 0       # 0 → n_heads
    adj_n_layers:     int  = 0       # 0 → n_layers
    adj_edge_hidden:  int  = 0       # 0 → d_ff
    use_prev_adj:     bool = True    # feed self-conditioned adjacency to the edge MLP

    def __post_init__(self):
        if self.edge_n_heads == 0:
            self.edge_n_heads = self.n_heads
        if self.embed_dim == 0:
            self.embed_dim = self.d_model
        if self.adj_hidden_dim == 0:
            self.adj_hidden_dim = self.d_model
        if self.adj_n_heads == 0:
            self.adj_n_heads = self.n_heads
        if self.adj_n_layers == 0:
            self.adj_n_layers = self.n_layers
        if self.adj_edge_hidden == 0:
            self.adj_edge_hidden = self.d_ff


# ══════════════════════════════════════════════════════════════════════════════
# Marginal transition-matrix diffusion (graphpath.models.marginal / .joint).
# A separate dataclass: the process has its own knobs and none of the
# absorbing/continuous ones apply.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarginalConfig:
    node_vocab:   int   = 10      # number of node classes (no MASK token)
    edge_vocab:   int   = 2       # number of edge classes: 0=no-edge, 1=edge
    d_model:      int   = 128
    n_heads:      int   = 4
    n_layers:     int   = 4
    d_ff:         int   = 256
    dropout:      float = 0.1
    edge_n_heads: int   = 0       # 0 → same as n_heads
    T:            int   = 500
    schedule:     str   = "cosine"   # "cosine" | "linear"

    # ── marginal-diffusion specific ──────────────────────────────────────
    marginal_floor:      float = 1e-4   # min prob per class after smoothing
    exact_posterior:     bool  = True   # False → v7's approximation
    independent_t:       bool  = False  # separate timestep per stream
    symmetric_edges:     bool  = True   # corrupt/sample edges symmetrically

    # ── loss weighting over the noise level ──────────────────────────────
    # "none"      unweighted CE on x_0. The D3PM / DiGress default for an
    #             x_0-parameterised loss, and what this model needs.
    # "one_minus_alpha_bar"  v7's weighting, kept for ablation. See the
    #             module docstring for why it does not train.
    loss_noise_weighting: str = "none"

    # Inverse-frequency class weights on the edge CE. See compute_loss for why
    # they bias the SAMPLES, and edge_logit_adjust for the correction that
    # makes them usable.
    edge_class_weights: bool = False

    # Undo the class weighting at SAMPLING time.
    #
    # Weighted CE is minimised at the rebalanced q = w1*p / (w1*p + w0*(1-p)),
    # not at p. A classifier does not care — you threshold it — but a
    # diffusion model SAMPLES from what it predicts, so it inherits the
    # rebalancing and the graphs come out ~13x too dense.
    #
    # The bias is exactly log(w) on the logits, so it is exactly removable:
    #     adjusted = logits - log(w),   w from the class frequencies in m_edge
    # This is standard logit adjustment (Menon et al. 2021). It gives the
    # LEARNING benefit of the weights — the 96/4 imbalance no longer lets the
    # model score well by predicting "no edge" everywhere — while sampling
    # from the unbiased distribution.
    #
    # Only meaningful when edge_class_weights is on; a no-op otherwise.
    edge_logit_adjust: bool = True

    # LayerNorm on the embedding sum, before the first attention block.
    #
    # Diagnostically it looks essential: without it the d_model=128 token
    # embedding has norm 0.23 against the time embedding's 1.97 (the input is
    # 90% clock, 10% identity) and Q·K/sqrt(d_k) lands at 1e-3, leaving
    # softmax uniform to 0.13%. BERT uses the same std=0.02 but normalises the
    # embedding sum first.
    #
    # It does not help HERE, so the default stays at v8's behaviour. Measured,
    # 16-graph memorisation, 600 steps, unweighted CE at fixed t:
    #
    #     input_layernorm     t=5     t=25     t=50     density
    #     False             2.024    2.749    4.503     1.5x data
    #     True              2.394    3.078    4.675     1.6x data
    #
    # Slightly worse on every measure. The serial tower has a direct
    # pair_proj path from nodes into edges that does not go through attention,
    # so suppressed attention costs it less than the diagnostic suggests.
    # graphpath.models.joint, where attention is the ONLY path, turns it on.
    input_layernorm: bool = False

    # ── node types (same convention as graphpath.layers.node_types) ─────────────
    # 0 disables everything below. With n_node_types > 0 the model takes a
    # token -> type lookup via set_node_types(), adds a learned type embedding
    # to the node stream, and (if use_type_bias) a learned per-type-pair bias
    # to the edge logits. gene/metabolite/enzyme/reaction is n_node_types=4.
    # ── padding as a predictable token ────────────────────────────────────
    #
    # Off by default, which reproduces every existing checkpoint exactly.
    #
    # OFF: padding slots are held clean and excluded from the loss, so nothing
    # teaches the model to stop, and at sampling — where there is no pad_mask
    # — it can only fill all N slots.
    # ON: PAD becomes a token the model can learn to emit, which is how it
    # says "this graph is shorter than N".
    #
    # `pad_loss_weight` cuts both ways and which way depends on the DATA.
    # graphpath's Config warns about PAD being a MAJORITY class when graphs
    # are much shorter than N; on this project's pathway chunks the opposite
    # holds — E. coli held-out graphs average 63.22 of 64 nodes, so PAD is
    # 1.21% of slots and the real risk is that it is never predicted at all.
    # Weight ABOVE 1 here; below 1 on a dataset of short graphs.
    supervise_pad:    bool  = False
    pad_loss_weight:  float = 1.0

    n_node_types:  int  = 0
    use_type_bias: bool = True

    # Damp the type signal by alpha_bar_t.
    #
    # READ THIS BEFORE TURNING TYPES ON. Under a marginal (or uniform) kernel
    # a corrupted node still holds a REAL token, so it resolves to a confident
    # but WRONG type — measured elsewhere in this project at 61% of pairs
    # mistyped at t=100. That is the opposite of the absorbing model, where a
    # corrupted node is MASK and maps to the reserved "unknown" slot, which is
    # inert. Untyped runs of the uniform variant beat typed ones for exactly
    # this reason.
    #
    # With the gate on, the type embedding and the pair bias are scaled by
    # alpha_bar_t, so they carry weight only where the tokens are mostly
    # clean and fade out where they are mostly noise. Off by default to match
    # the rest of the project; turn it on if you turn types on.
    type_alpha_gate: bool = False

    # Mask the attention diagonal in both towers, i.e. forbid a node (or an
    # edge cell) from attending to itself. Left at v7's True: the ablation
    # showed it makes no measurable difference either way (fixed-t CE 2.338
    # with the mask vs 2.365 without; sampled density 12.5x vs 12.7x data),
    # so there is no evidence for changing it. Exposed because it is a
    # surprising thing for a denoiser to do and worth being able to switch.
    mask_attention_diagonal: bool = True

    def __post_init__(self):
        if self.edge_n_heads == 0:
            self.edge_n_heads = self.n_heads


@dataclass
class JointConfig(MarginalConfig):
    d_edge:    int  = 64     # width of the edge stream
    film_time: bool = True   # FiLM-condition every layer on t

    # Attention is the ONLY path between the streams here, so its scale at
    # initialisation matters far more than it does in the serial tower, which
    # has a direct pair_proj shortcut from nodes into edges.
    #
    # v8's _init_weights uses trunc_normal(std=0.02) for every Linear whatever
    # d_model is. That is BERT's constant, and it assumes d=768 with a
    # LayerNorm on the embedding sum. At d=128 it leaves Q·K/sqrt(d_k) an
    # order of magnitude too small. Measured on this tower:
    #
    #     init std   attn logit std   edges->nodes   nodes->edges
    #     0.0200          3.86e-02         0.0001         0.0006
    #     0.0884          8.62e-01         0.0012         0.0166
    #
    # None -> 1/sqrt(d_model), the scale-correct choice. Set 0.02 to reproduce
    # v8's initialisation.
    init_std: Optional[float] = None

    # on by default here, unlike the serial model: see MarginalConfig.input_layernorm
    input_layernorm: bool = True
