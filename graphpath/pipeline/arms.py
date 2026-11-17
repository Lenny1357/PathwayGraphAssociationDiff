"""The experimental arms: how each model is configured, built and rebuilt.

One place for the hyper-parameters every comparison shared, the per-arm
settings (typing, pad supervision, the continuous arm's adjacency weight)
and the rules for rebuilding a model from a saved checkpoint, so a scoring
script never has to guess what a training script did.
"""

from __future__ import annotations

import collections
import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from graphpath.config import Config, JointConfig
from graphpath.models import ARMS, build_model
from graphpath.processes.marginal import estimate_marginals

# the backbone every arm shared in every run
SHARED = dict(T=200, n_layers=4, d_model=128, n_heads=4, edge_n_heads=4, d_ff=256, dropout=0.1)
TRANSITION = dict(d_edge=64, edge_vocab=2, schedule="cosine", marginal_floor=1e-4,
                  exact_posterior=True, symmetric_edges=True,
                  loss_noise_weighting="none", edge_class_weights=False)
VAE = dict(latent_dim=64, slot_dim=32, kl_weight=1.0, kl_free_bits=0.05,
           kl_warmup_steps=2000, init_logvar=-3.0, z_proj_std=0.5,
           # the fixes (see Config): a latent that varies between graphs, a learnable
           # slot order, one draw per edge, and edges that see the real node types.
           # Checkpoints saved before these keep their old behaviour (fields default off).
           vae_mu_batchnorm=True, vae_canonical_order=True,
           vae_symmetric_edges=True, vae_teacher_types=True)

# The integer order the typed runs used for set_node_types(). Kept as the
# checkpoints were trained; a different order would make the type-pair bias
# mean something else.
TYPE_ORDER = ["gene", "enzyme", "reaction", "metabolite"]
DEFAULT_ARMS = ["masked", "transition", "continuous", "vae"]
# The VAE's reconstruction compares slot i to slot i, so permuting node order
# every batch makes its target unlearnable beyond the per-slot marginal.
PERMUTE = {"masked": True, "transition": True, "marginal": True, "uniform": True,
           "continuous": True, "vae": False}


@dataclass
class ArmOptions:
    """What varied between the comparison runs."""
    typed: bool = False
    supervise_pad: bool = False
    pad_loss_weight: float = 1.0
    pad_target: Optional[float] = None    # None -> weight 1 (match the data); else solve w for this P(PAD)
    adj_pos_weight: Optional[float] = None   # None -> Config default (20)
    size_clamp: bool = False              # sampling-time: draw graph size once per graph
    note: str = ""

    def pad_weight(self, pad_frac: float) -> float:
        """w such that weighted CE is minimised at P(PAD) = pad_target.

        Weighted CE is optimal at q = w f / (1 + f (w - 1)); solving for w at
        a target q gives w = q (1 - f) / (f (1 - q)). The September typed run
        used a shared w = 4.0, which collapsed the human arm; `pad_target`
        derives it per dataset instead.
        """
        if self.pad_target is None:
            return self.pad_loss_weight
        f, q = pad_frac, float(self.pad_target)
        return float(q * (1 - f) / (f * (1 - q))) if 0 < f < 1 else 1.0


PRESETS = {
    "sept":  ArmOptions(note="untyped, padding excluded from the loss; size fixed at 64"),
    "typed": ArmOptions(typed=True, supervise_pad=True, pad_loss_weight=4.0,
                        note="n_node_types=4, padding supervised with a shared weight of 4"),
    "nopad": ArmOptions(typed=True, supervise_pad=False, adj_pos_weight=1.0,
                        note="typed, padding excluded from the loss; size fixed at 64"),
    "pad":   ArmOptions(typed=True, supervise_pad=True, adj_pos_weight=1.0, size_clamp=True,
                        note="typed, padding supervised at the data rate; size drawn once per graph"),
}


def type_table(id_to_node: dict, ntype, n_rows: int) -> torch.Tensor:
    """token id -> type id in TYPE_ORDER; unknown names and PAD fall back to 0."""
    t_of = {t: i for i, t in enumerate(TYPE_ORDER)}
    tab = torch.zeros(n_rows, dtype=torch.long)
    for tok, name in id_to_node.items():
        if 0 <= int(tok) < n_rows:
            tab[int(tok)] = t_of.get(ntype(name), 0)
    return tab


def arm_config(kind: str, opts: ArmOptions, pad_frac: float = 0.0):
    typed = dict(n_node_types=len(TYPE_ORDER), use_type_bias=True) if opts.typed else {}
    pad = dict(supervise_pad=opts.supervise_pad, pad_loss_weight=opts.pad_weight(pad_frac))
    if kind == "transition":
        return JointConfig(**SHARED, **TRANSITION, **typed, **pad)
    if kind == "marginal":
        from graphpath.config import MarginalConfig
        return MarginalConfig(**SHARED, **{k: v for k, v in TRANSITION.items() if k != "d_edge"},
                              **typed, **pad)
    extra = dict(adj_pos_weight=opts.adj_pos_weight) if opts.adj_pos_weight is not None else {}
    if kind == "vae":
        return Config(**SHARED, **typed, **pad, **extra, **VAE)
    return Config(**SHARED, **typed, **pad, **extra)


def build_arm(kind: str, bundle, opts: ArmOptions, device, max_nodes: int = 64,
              batch_size: int = 32, overrides: Optional[dict] = None):
    """A fresh model for `kind`, with marginals / output prior / types installed.
    `overrides` replaces config fields (e.g. the VAE fixes) before the model is built."""
    V = bundle.vocab_size
    cfg = arm_config(kind, opts, bundle.pad_frac(max_nodes))
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    kw = {"max_nodes": max_nodes} if kind == "vae" else {}
    model = build_model(kind, cfg, vocab_size=V + 1, **kw).to(device)
    if kind in ("transition", "marginal"):
        loader = torch.utils.data.DataLoader(bundle.train, batch_size=batch_size)
        model.set_marginals(*estimate_marginals(loader, node_vocab=V, edge_vocab=cfg.edge_vocab))
    if kind == "vae":
        # start at the class prior: the decoder reaches the marginal with its
        # output bias alone, and an earlier run spent its whole budget getting there
        counts = bundle.token_counts()
        if opts.supervise_pad:     # PAD is a target then: give it its data share at step 0
            counts[0] = len(bundle.train_graphs) * max_nodes - counts[1:].sum()
        model.set_output_prior(counts, torch.tensor([1 - bundle.density, bundle.density]))
    if opts.typed:
        n_rows = model.token_to_type.numel()
        model.set_node_types(type_table(bundle.id_to_node, bundle.ntype, n_rows).to(device))
    return model, cfg


def config_from_dict(kind: str, d: dict):
    """Rebuild the config class an arm uses from a saved `cfg` dict, dropping
    fields the class no longer knows."""
    cls = JointConfig if kind == "transition" else Config
    if kind == "marginal":
        from graphpath.config import MarginalConfig
        cls = MarginalConfig
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def build_from_checkpoint(kind: str, ck: dict, device="cpu"):
    """The model a checkpoint was trained as, weights loaded. Needs no data:
    vocabulary, marginals, type table and class weights all live in the
    state dict."""
    sd = ck["state_dict"]
    cfg = config_from_dict(kind, ck.get("cfg", {}))
    V = (sd["embedding.embed.weight"].shape[0] if "embedding.embed.weight" in sd else
         sd["node_out.1.weight"].shape[0] if "node_out.1.weight" in sd else
         sd["node_head.weight"].shape[0])
    kw = {"max_nodes": sd["slot_emb.weight"].shape[0]} if "slot_emb.weight" in sd else {}
    model = build_model(kind, cfg, vocab_size=V + 1, **kw)
    model.load_state_dict(sd)
    return model.to(device).eval()


def load_arm(run, dataset: str, arm: str, variant=None, device="cpu"):
    """Checkpoint -> (model, checkpoint dict)."""
    ck = run.load_checkpoint(dataset, arm, variant)
    kind = ck.get("kind", arm)
    return build_from_checkpoint(kind, ck, device), ck


def pad_clamp(size_prior: np.ndarray, B: int, max_nodes: int, rng) -> tuple:
    """(B, N) long with PAD (0) on the slots a graph does not use and -1 free.

    One size per graph from the training distribution. PAD slots are
    scattered, not parked at the tail: training permutes every slot, so the
    model's slots are exchangeable and scattering keeps the sampler in the
    distribution it was fitted on.
    """
    want = rng.choice(size_prior, size=B, replace=True)
    kn = torch.full((B, max_nodes), -1, dtype=torch.long)
    for b, s in enumerate(want):
        if s < max_nodes:
            kn[b, torch.from_numpy(rng.permutation(max_nodes)[: max_nodes - s])] = 0
    return kn, want


FREE_PAD_CAP = 0.10   # share of a clamped graph's free slots the masked arm may still turn into PAD


def size_clamped(variant) -> bool:
    """Whether a run variant draws graph size before sampling. Variant names
    are preset names (`CompareConfig.variants` maps one to the other)."""
    opts = PRESETS.get(variant or "")
    return bool(opts and opts.size_clamp)


def pad_scale_for(run, dataset: str) -> Optional[dict]:
    """The PAD damping for a run's pad masked arm on `dataset`, as sampler
    kwargs ({pad_scale, pad_schedule}), from <run>/pad_scale.json
    (scripts/calibrate_pad_scale.py; `schedule` defaults to constant), or None."""
    import json
    p = run.path / "pad_scale.json"
    rec = json.loads(p.read_text()).get(dataset, {}) if p.exists() else {}
    if rec.get("scale") is None:
        return None
    return dict(pad_scale=rec["scale"], pad_schedule=rec.get("schedule", "constant"))


def clamp_kwargs(kind: str, variant, size_prior: np.ndarray, B: int, max_nodes: int, seed: int,
                 pad_scale: Optional[dict] = None) -> tuple:
    """(model.sample kwargs, requested sizes) for one draw. Every sampling site
    goes through here, so the scored samples, the trajectories and the
    ablations of a size-clamped variant are all drawn the same way.

    The masked arm with a calibrated `pad_scale` sizes its own graphs: no
    clamp, PAD emittable everywhere at damped probability. Without one it
    falls back to the size clamp with a capped share of PAD on free slots."""
    if not (size_clamped(variant) and kind in ("masked", "transition")):
        return {}, None
    if kind == "masked" and pad_scale is not None:
        return dict(pad_scale) if isinstance(pad_scale, dict) else dict(pad_scale=pad_scale), None
    kn, want = pad_clamp(size_prior, B, max_nodes, np.random.default_rng(seed))
    if kind == "masked":
        return dict(known_nodes=kn, free_pad_cap=FREE_PAD_CAP), want
    return dict(node_clamp=(kn >= 0, kn.clamp(min=0))), want


def _to(kw: dict, device) -> dict:
    """clamp_kwargs with its tensors moved to the model's device."""
    mv = lambda x: x.to(device) if torch.is_tensor(x) else tuple(mv(y) for y in x) if isinstance(x, tuple) else x
    return {k: mv(v) for k, v in kw.items()}


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_state_dict(sd: dict, buffer_names=("node_class_weight", "m_node", "m_edge", "_step"),
                     buffer_prefix=("schedule.",)) -> tuple:
    """(learned, buffers, {top-level module: n}) — buffers are not capacity."""
    learned, buf, mods = 0, 0, collections.Counter()
    for k, v in sd.items():
        if not hasattr(v, "numel"):
            continue
        n = int(v.numel())
        if k in buffer_names or k.startswith(buffer_prefix):
            buf += n
        else:
            learned += n
            mods[k.split(".")[0]] += n
    return learned, buf, dict(mods)
