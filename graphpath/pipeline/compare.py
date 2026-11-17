"""Train the arms on every dataset, sample from each, keep everything.

The pipeline behind compare_models_{sept,typed,oct,oct2}.ipynb. A run is

    outputs/runs/<name>/[<variant>/]<dataset>/weights/<arm>.pt   state_dict + cfg + history
    outputs/runs/<name>/[<variant>/]<dataset>/samples/<arm>.pt   the generated graphs
    outputs/runs/<name>/history.json                             per-step losses, every arm
    outputs/runs/<name>/samples.json                             density / size / PAD diagnostics
    outputs/runs/<name>/manifest.json                            what was run, with what

Resumable: an arm whose weights exist is skipped. Steps, not epochs — the
datasets differ in size ~6x, so equal epochs would give very different budgets.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch

from graphpath.pipeline import arms as A
from graphpath.pipeline.common import DatasetBundle, RunLayout, dump_json, free, key, load_bundle, load_json
from graphpath.training.loop import train_step


@dataclass
class CompareConfig:
    datasets: list = field(default_factory=lambda: ["ecoli", "human"])
    arms: list = field(default_factory=lambda: list(A.DEFAULT_ARMS))
    variants: dict = field(default_factory=lambda: {"": "sept"})  # variant name -> preset
    steps: int = 20_000
    batch_size: int = 32
    max_nodes: int = 64
    split: float = 0.9
    seed: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-2
    n_samples: int = 96
    sample_seed: int = 0
    log_every: int = 100

    def options(self, variant: str) -> A.ArmOptions:
        preset = self.variants[variant]
        return preset if isinstance(preset, A.ArmOptions) else A.PRESETS[preset]


def permute_batch(ids, adj, msk, max_nodes, device):
    """Random node order per graph; the mask moves with the data it describes."""
    p = torch.stack([torch.randperm(max_nodes, device=device) for _ in range(ids.shape[0])])
    bi = torch.arange(ids.shape[0], device=device)[:, None]
    ids = ids[bi, p]
    adj = adj.gather(1, p[..., None].expand(-1, -1, max_nodes)) \
             .gather(2, p[:, None, :].expand(-1, max_nodes, -1))
    return ids, adj, msk[bi, p]


class CompareExperiment:
    def __init__(self, cfg: CompareConfig, run: RunLayout, device):
        self.cfg, self.run, self.device = cfg, run, device
        self.data: dict[str, DatasetBundle] = {}

    # ── data ──────────────────────────────────────────────────────────────
    def load_data(self, names: Optional[list] = None):
        for n in names or self.cfg.datasets:
            try:
                self.data[n] = load_bundle(n, self.cfg.max_nodes, self.cfg.split,
                                           self.cfg.seed, device=self.device)
            except FileNotFoundError as e:
                print(f"  {n}: MISSING ({e}) — skipped")
        if not self.data:
            raise FileNotFoundError("no dataset available; set PT_FILES for Human-GEM")
        for n, b in self.data.items():
            print(f"  {n:<7} train {len(b.train_graphs):,} / test {len(b.test_graphs):,}"
                  f"   vocab {b.vocab_size:,}   density {b.density:.4f}"
                  f"   PAD {100 * b.pad_frac(self.cfg.max_nodes):.2f}% of slots")
        return self.data

    def _variants(self):
        return [v or None for v in self.cfg.variants]

    # ── training ──────────────────────────────────────────────────────────
    def train_arm(self, dname: str, kind: str, variant=None) -> list:
        cfg, run, device = self.cfg, self.run, self.device
        bundle = self.data[dname]
        wpath = run.weights(dname, kind, variant)
        if wpath.exists():
            ck = torch.load(wpath, map_location="cpu", weights_only=False)
            print(f"  {kind:<11} weights exist, skipping ({ck.get('steps', '?')} steps)")
            return ck.get("history", [])
        run.arm_dir(dname, variant, make=True)
        opts = cfg.options(variant or "")
        model, mcfg = A.build_arm(kind, bundle, opts, device, cfg.max_nodes, cfg.batch_size)
        if kind in ("transition", "marginal") and device.type == "mps":
            model = model.to("cpu")      # the marginal posterior is float64, which MPS lacks
        device = next(model.parameters()).device
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loader = torch.utils.data.DataLoader(bundle.train, batch_size=cfg.batch_size, shuffle=True)
        permute, V = A.PERMUTE.get(kind, True), bundle.vocab_size
        hist, n, t0 = [], 0, time.time()
        it = iter(loader)
        while n < cfg.steps:
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader); b = next(it)
            adj, ids, msk = b[0].long().to(device), b[1].to(device), b[2].bool().to(device)
            if permute:
                ids, adj, msk = permute_batch(ids, adj, msk, cfg.max_nodes, device)
            r = train_step(model, opt, ids, adj, V, V + 1, mask=msk)
            n += 1
            if n % cfg.log_every == 0 or n == cfg.steps:
                hist.append({"step": n, **{k: float(v) for k, v in r.items()
                                           if isinstance(v, (int, float))}})
                if n % (cfg.log_every * 10) == 0 or n == cfg.steps:
                    el = time.time() - t0
                    print(f"  {kind:<11} {n:>6}/{cfg.steps}  loss {hist[-1]['loss']:.4f}"
                          f"   {el / n:.3f}s/step  eta {(cfg.steps - n) * el / n / 60:.0f}m",
                          flush=True)
        if kind == "vae":       # exact BN statistics for eval-mode encoding (see recalibrate_bn)
            model.recalibrate_bn(torch.utils.data.DataLoader(bundle.train, batch_size=cfg.batch_size,
                                                             shuffle=True), device)
        torch.save({"state_dict": model.state_dict(), "cfg": asdict(mcfg), "history": hist,
                    "steps": cfg.steps, "kind": kind, "dataset": dname,
                    "variant": variant, "options": asdict(opts),
                    "n_params": A.count_params(model)}, wpath)
        print(f"  {kind:<11} done in {(time.time() - t0) / 60:.1f}m -> {wpath.name}")
        free(model, opt)
        return hist

    def train_all(self) -> dict:
        history = {}
        for variant in self._variants():
            for dname in self.data:
                print(f"\n{'=' * 74}\n{(variant + '/') if variant else ''}{dname}\n{'=' * 74}")
                for kind in self.cfg.arms:
                    history[key(dname, kind, variant)] = self.train_arm(dname, kind, variant)
        # merge: parallel jobs on one run each train a subset of the arms
        hp = self.run.path / "history.json"
        dump_json({**(load_json(hp) if hp.exists() else {}), **history}, hp)
        self.run.write_manifest(config=asdict(self.cfg), datasets=list(self.data),
                                variants={(v or ""): asdict(self.cfg.options(v or ""))
                                          for v in self._variants()})
        return history

    # ── sampling ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample_arm(self, dname: str, kind: str, variant=None, n: Optional[int] = None,
                   seed: Optional[int] = None) -> dict:
        cfg, bundle, device = self.cfg, self.data[dname], self.device
        n = n or cfg.n_samples
        seed = cfg.sample_seed if seed is None else seed
        opts = cfg.options(variant or "")
        model, _ = A.load_arm(self.run, dname, kind, variant, device)
        # the marginal posterior is float64, which MPS lacks
        if kind in ("transition", "marginal") and device.type == "mps":
            model = model.to("cpu")
        dev = next(model.parameters()).device
        torch.manual_seed(seed)
        clamp = opts.size_clamp and kind in ("masked", "transition")
        kw, want = A.clamp_kwargs(kind, variant, bundle.sizes(), n, cfg.max_nodes, seed,
                                       A.pad_scale_for(self.run, dname)) if clamp else ({}, None)
        nodes, adj = model.sample(N=cfg.max_nodes, B=n, device=dev, **A._to(kw, dev))
        nodes, adj = nodes.long().cpu(), (adj > 0).long().cpu()
        diag = self.sample_diagnostics(nodes, adj, bundle, clamp, want)
        pad = nodes == 0
        padpair = pad[:, :, None] | pad[:, None, :]
        adj = adj * (~padpair).long()        # no edges on PAD slots (reported above)
        self.run.save_samples(dname, kind, nodes, adj, variant,
                              want=(None if want is None else torch.as_tensor(want)), seed=seed)
        free(model)
        return diag

    @staticmethod
    def sample_diagnostics(nodes, adj, bundle, clamp=False, want=None) -> dict:
        N = nodes.shape[1]
        iu = torch.triu_indices(N, N, 1)
        pad = nodes == 0
        sizes = (~pad).sum(1).float()
        padpair = pad[:, :, None] | pad[:, None, :]
        prior = bundle.sizes()
        d = dict(
            n_graphs=int(len(nodes)),
            density=float(adj[:, iu[0], iu[1]].float().mean()),
            data_density=bundle.density,
            distinct_per_graph=float(np.mean([nodes[i].unique().numel() for i in range(len(nodes))])),
            pad_frac=float(pad.float().mean()), data_pad_frac=bundle.pad_frac(N),
            graphs_with_pad=int(pad.any(1).sum()),
            nodes_mean=float(sizes.mean()), nodes_min=int(sizes.min()), nodes_max=int(sizes.max()),
            train_size_min=int(prior.min()), train_size_mean=float(prior.mean()),
            below_train_min=int((sizes.numpy() < prior.min()).sum()),
            edges_on_pad_slots=int((adj * padpair.long()).sum() // 2),
            size_clamped=bool(clamp),
        )
        if want is not None:
            d["clamp_requested_mean"] = float(np.mean(want))
            d["clamp_exact"] = int((sizes.numpy().astype(int) == np.asarray(want)).sum())
        return d

    def sample_all(self) -> dict:
        out = {}
        for variant in self._variants():
            for dname in self.data:
                for kind in self.cfg.arms:
                    if not self.run.weights(dname, kind, variant).exists():
                        continue
                    d = self.sample_arm(dname, kind, variant)
                    out[key(dname, kind, variant)] = d
                    print(f"  {key(dname, kind, variant):<22} density {d['density']:.4f} "
                          f"(data {d['data_density']:.4f})  distinct/graph "
                          f"{d['distinct_per_graph']:.1f}  PAD {100 * d['pad_frac']:.2f}%"
                          + (f"  !! {d['below_train_min']} graphs below the training minimum"
                             if d["below_train_min"] else ""))
        sp = self.run.path / "samples.json"
        dump_json({**(load_json(sp) if sp.exists() else {}), **out}, sp)
        return out
