"""Score a finished run with every benchmark its data supports.

    suite = BenchmarkSuite(bundle, device)          # built once per dataset
    scorer = ScoreRun(run, device)
    scorer.load_data(); scorer.load_samples()
    scorer.score()                                   # -> analysis/<run>/scores.json

scores.json:
    {"held-out": {dataset: {bench: result}},
     "arms":     {"dataset|arm": {bench: {"generated"|"shuffled"|"random": result}}},
     "notes":    {dataset: [...]}, "n_graphs": int, "resampled": bool}

where `result` is `common.result_to_dict(BenchmarkResult)`. Four arms per
benchmark — generated / shuffled / random / held-out — because a raw number
says nothing on its own. The held-out arm is computed once per dataset.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from graphpath.benchmarks import (CoExpressionBenchmark, KnownStructureBenchmark,
                                  NearestTrainingBenchmark, NovelEdgeBenchmark, ReferenceNetwork,
                                  TopologyBenchmark, random_like, shuffled_like)
from graphpath.paths import DATA
from graphpath.pipeline import arms as A
from graphpath.pipeline.common import (DatasetBundle, RunLayout, dump_json, free, key,
                                       load_bundle, result_to_dict)


def reference_network(bundle: DatasetBundle, device="cpu") -> tuple:
    """(ReferenceNetwork, {node -> [subsystems]}) from the bundle's edge table."""
    ref = ReferenceNetwork(bundle.train.node_to_id, device=device, num_nodes=bundle.vocab_size)
    ref.load_edges(bundle.edges)
    ref.build_adjacency()
    cols = list(pd.read_csv(bundle.edges, nrows=0).columns)
    sub = next((c for c in cols if c.lower() == "subsystem"), None)
    subs = ref.build_pathway_associations(subsystem=sub) if sub else ref.build_pathway_associations()
    return ref, subs


def expression_source(bundle: DatasetBundle, device="cpu"):
    """(loader, default tissue) for whichever co-expression tables this dataset has."""
    if bundle.expr and Path(bundle.expr, "meta.json").exists():
        from graphpath.data.expression_ecoli import EcoliExpressionLoader
        ex = EcoliExpressionLoader(bundle.expr, device=device)
        return ex, ex.tissues[0]
    if bundle.expr_path and Path(bundle.expr_path).is_dir():
        from graphpath.data.expression import ExpressionLoader
        ex = ExpressionLoader(path=bundle.expr_path, device=device)
        return ex, ex.tissues[0]
    return None, None


class BenchmarkSuite:
    """Every benchmark a dataset supports, built once. Missing inputs are
    recorded in `notes`, never silently dropped."""

    def __init__(self, bundle: DatasetBundle, device="cpu", heavy: bool = True):
        # sparse tensors (reference network, co-expression tables) have no MPS kernels
        device = torch.device("cpu") if torch.device(device).type == "mps" else torch.device(device)
        self.bundle, self.device, self.notes = bundle, device, []
        self.ref, self.subsystems = reference_network(bundle, device)
        ntype = bundle.ntype
        self.benches = {
            "known": KnownStructureBenchmark(self.ref, node_to_subsystems=self.subsystems, device=device),
            "topo":  TopologyBenchmark(bundle.id_to_node, node_type_fn=ntype, heavy=heavy),
            "memo":  NearestTrainingBenchmark(bundle.train, device=device),
            "novel": NovelEdgeBenchmark(bundle.train, self.ref, device=device, node_type_fn=ntype),
        }
        self.expr, self.tissue = None, None
        try:
            self.expr, self.tissue = expression_source(bundle, device)
        except Exception as e:  # noqa: BLE001 - reported, not fatal
            self.notes.append(f"expression source failed ({type(e).__name__}: {e})")
        if self.expr is not None and bundle.expr:
            try:
                self.benches["coexpr"] = CoExpressionBenchmark(
                    bundle.train, expression_loader=self.expr, device=device,
                    node_type_fn=ntype, name_map_path=bundle.name_map, node_filter="gene")
            except Exception as e:  # noqa: BLE001
                self.notes.append(f"coexpr unavailable ({type(e).__name__}: {e})")
        else:
            self.notes.append("coexpr: no node-space expression tables for this dataset")
        if bundle.perturb and Path(bundle.perturb, "meta.json").exists():
            try:
                from graphpath.benchmarks import PerturbationBenchmark
                for d in ("row", "column"):
                    self.benches[f"perturb-{d}"] = PerturbationBenchmark(
                        bundle.perturb, bundle.train, direction=d, mask_self=True)
                self.notes.append(f"perturbation: both directions; "
                                  f"{self.benches['perturb-row'].n_self_masked} self-knockdown cells masked")
            except Exception as e:  # noqa: BLE001
                self.notes.append(f"perturbation unavailable ({type(e).__name__}: {e})")
        else:
            self.notes.append("perturbation: no perturb_ref for this dataset")
        gem = DATA / "gem" / ("gem_ref_ecoli" if bundle.name == "ecoli" else "gem_ref")
        if bundle.name == "ecoli" and (gem / "gem.npz").exists():
            try:
                from graphpath.benchmarks.flux import FluxValidityBenchmark
                from graphpath.fba import StoichModel
                self.benches["flux"] = FluxValidityBenchmark(StoichModel(str(gem)), bundle.train,
                                                             mode="asserted")
            except Exception as e:  # noqa: BLE001
                self.notes.append(f"flux unavailable ({type(e).__name__}: {e})")
        else:
            self.notes.append("flux: E. coli only (gem_ref is Human-GEM)")
        seen = collections.Counter(ntype(n) for n in bundle.id_to_node.values())
        self.notes.append(f"node types: {dict(seen)}")
        if seen.get("gene", 0) == 0 and "coexpr" in self.benches:
            self.notes.append("! no node is typed 'gene' — coexpr will score 0/N; check node_type_fn")

    def heldout(self, n_graphs: int, seed: int = 0) -> dict:
        out = {}
        for name, b in self.benches.items():
            try:
                out[name] = b.evaluate_dataset(self.bundle.test, n_graphs=n_graphs,
                                               generator=torch.Generator().manual_seed(seed))
            except Exception as e:  # noqa: BLE001
                self.notes.append(f"held-out {name} FAILED ({type(e).__name__}: {e})")
        return out

    def evaluate(self, nodes, adj, benches: Optional[list] = None) -> dict:
        """{bench: {generated, shuffled, random: BenchmarkResult}}."""
        sn, sa = shuffled_like(nodes, adj)
        rn, ra = random_like(nodes, adj, self.bundle.vocab_size)
        arms = {"generated": (nodes, adj), "shuffled": (sn, sa), "random": (rn, ra)}
        out = {}
        for name, b in self.benches.items():
            if benches and name not in benches:
                continue
            try:
                out[name] = {a: b.evaluate(x, y) for a, (x, y) in arms.items()}
            except Exception as e:  # noqa: BLE001
                self.notes.append(f"{name} FAILED ({type(e).__name__}: {e})")
        return out


class ScoreRun:
    def __init__(self, run: RunLayout, device="cpu", n_graphs: int = 96, max_nodes: int = 64,
                 split: float = 0.9, seed: int = 0):
        self.run, self.device, self.n_graphs = run, device, n_graphs
        self.max_nodes, self.split, self.seed = max_nodes, split, seed
        self.data: dict[str, DatasetBundle] = {}
        self.suites: dict[str, BenchmarkSuite] = {}
        self.samples: dict[str, tuple] = {}
        self.out = run.analysis_dir()

    def load_data(self):
        for d in sorted({d for d, _, _ in self.run.all_weights()}):
            try:
                self.data[d] = load_bundle(d, self.max_nodes, self.split, self.seed, device=self.device)
            except FileNotFoundError as e:
                print(f"  {d}: data MISSING ({e}) — skipped")
        return self.data

    def load_samples(self, resample=False, sample_seed: int = 0):
        """The graphs the run produced; re-sampled only on request, so the
        numbers match the ones the training stage printed. `resample` is True
        for every arm, or a collection of keys ('pad/human|masked') for some."""
        for d, arm, v in self.run.all_weights():
            if d not in self.data:
                continue
            k = key(d, arm, v)
            redo = resample is True or (bool(resample) and k in resample)
            s = None if redo else self.run.load_samples(d, arm, v)
            if s is None:
                model, _ = A.load_arm(self.run, d, arm, v, self.device)
                if arm in ("transition", "marginal") and self.device.type == "mps":
                    model = model.to("cpu")
                dev = next(model.parameters()).device
                torch.manual_seed(sample_seed)
                kw, want = A.clamp_kwargs(arm, v, self.data[d].sizes(), self.n_graphs, self.max_nodes, sample_seed,
                                           A.pad_scale_for(self.run, d))
                with torch.no_grad():
                    n_, a_ = model.sample(N=self.max_nodes, B=self.n_graphs, device=dev, **A._to(kw, dev))
                n_, a_ = n_.long().cpu(), (a_ > 0).long().cpu()
                pad = n_ == 0
                a_ = a_ * (~(pad[:, :, None] | pad[:, None, :])).long()   # no edges on PAD slots, as in compare
                s = (n_, a_)
                self.run.save_samples(d, arm, *s, variant=v, seed=sample_seed,
                                      want=(None if want is None else torch.as_tensor(want)))
                free(model)
            self.samples[k] = s
        self.resampled = resample
        return self.samples

    def suite(self, dname: str) -> BenchmarkSuite:
        if dname not in self.suites:
            self.suites[dname] = BenchmarkSuite(self.data[dname], self.device)
        return self.suites[dname]

    def score(self, benches: Optional[list] = None) -> dict:
        held, arms = {}, {}
        for dname in self.data:
            s = self.suite(dname)
            held[dname] = {b: result_to_dict(r) for b, r in s.heldout(self.n_graphs).items()}
            for k, (nodes, adj) in self.samples.items():
                if k.split("|")[0].split("/")[-1] != dname:
                    continue
                res = s.evaluate(nodes, adj, benches)
                arms[k] = {b: {a: result_to_dict(r) for a, r in rr.items()} for b, rr in res.items()}
                print(f"  {k:<24}{sorted(res)}")
        out = {"held-out": held, "arms": arms, "notes": {d: self.suite(d).notes for d in self.data},
               "n_graphs": self.n_graphs, "resampled": getattr(self, "resampled", False),
               "run": self.run.name}
        dump_json(out, self.out / "scores.json")
        return out

    def architecture(self) -> dict:
        """Read off the checkpoints, not a config literal: what was trained."""
        rows = {}
        for d, arm, v in self.run.all_weights():
            ck = self.run.load_checkpoint(d, arm, v)
            learned, buf, mods = A.split_state_dict(ck["state_dict"])
            hist = ck.get("history") or []
            rows[key(d, arm, v)] = dict(
                dataset=d, arm=arm, variant=v, cfg=ck.get("cfg", {}), learned=learned,
                buffers=buf, modules=mods, steps=ck.get("steps"),
                first_loss=(hist[0].get("loss") if hist else None),
                final_loss=(hist[-1].get("loss") if hist else None),
                ckpt_mb=self.run.weights(d, arm, v).stat().st_size / 1e6,
                vocab=(self.data[d].vocab_size if d in self.data else None))
        dump_json(rows, self.out / "architecture.json")
        return rows
