"""Shared plumbing for the experiment pipelines: JSON I/O, result
serialisation, dataset bundles and the on-disk layout of a run.

Every pipeline stage writes JSON (plus .pt for weights and samples) and every
figure is drawn from that JSON, so a plot can be redrawn without re-running
the experiment and an experiment can be re-run without touching the plots.
"""

from __future__ import annotations

import collections
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

from graphpath.paths import OUTPUTS, RUNS

ANALYSIS = OUTPUTS / "analysis"      # one folder per run: *.json, tables/, figures/


# ── JSON ──────────────────────────────────────────────────────────────────

class JSONEncoder(json.JSONEncoder):
    """numpy / torch / pandas / Path aware. NaN is written as null."""

    def default(self, o):
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if (isinstance(o, float) and math.isnan(o)) or np.isnan(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="split")
        if isinstance(o, pd.Series):
            return o.to_dict()
        return super().default(o)


def _clean(o):
    """NaN -> None everywhere (json's default hook never sees plain floats)."""
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.ndarray, torch.Tensor)):
        return _clean(o.tolist() if isinstance(o, np.ndarray) else o.detach().cpu().tolist())
    if isinstance(o, np.floating):
        return _clean(float(o))
    return o


def dump_json(obj, path, indent: int = 1) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(obj), cls=JSONEncoder, indent=indent, allow_nan=False)
    path.write_text(text)
    return path


def load_json(path):
    return json.loads(Path(path).read_text())


def nanfloat(v) -> float:
    """A JSON value back to a float, with null -> nan."""
    return float("nan") if v is None else float(v)


def result_to_dict(res) -> dict:
    """A `BenchmarkResult` as plain JSON-able data (tensors -> lists)."""
    def _arr(v):
        if v is None:
            return None
        if hasattr(v, "detach"):
            v = v.detach().cpu()
        if isinstance(v, dict):
            return {str(k): _arr(x) for k, x in v.items()}
        return np.round(np.asarray(v, dtype=float).ravel(), 6).tolist() if np.ndim(v) else v
    return dict(
        name=res.name,
        summary={k: (v if isinstance(v, (int, float, str, bool, type(None)))
                     else _arr(v)) for k, v in res.summary.items()},
        n_graphs=int(res.n_graphs), n_scored=int(res.n_scored),
        coverage=float(res.coverage), notes=list(res.notes),
        per_graph={k: _arr(v) for k, v in (res.per_graph or {}).items()},
        distributions={k: _arr(v) for k, v in (res.distributions or {}).items()},
    )


# ── datasets ──────────────────────────────────────────────────────────────

@dataclass
class DatasetBundle:
    """Everything an experiment needs about one dataset, loaded once."""
    name: str
    train: object                     # PathwayGraphDataset
    test: object
    train_graphs: list
    test_graphs: list
    node_map: dict
    common: Callable                  # node name -> display name
    ntype: Callable                   # node name -> entity type
    density: float                    # mean train edge density
    edges: Optional[str]              # reference edge table
    expr: Optional[str]               # expression directory (E. coli node-space tables)
    expr_path: Optional[str]          # human tissue co-expression directory
    perturb: Optional[str]            # Perturb-seq reference directory
    name_map: Optional[str]

    @property
    def vocab_size(self) -> int:
        return self.train.vocab_size

    @property
    def id_to_node(self) -> dict:
        return self.train.id_to_node

    def sizes(self) -> np.ndarray:
        return np.array([g["num_nodes"] for g in self.train_graphs], dtype=int)

    def pad_frac(self, max_nodes: int) -> float:
        return float(1 - self.sizes().mean() / max_nodes)

    def heldout_tensors(self, n: Optional[int] = None, max_nodes: int = 64,
                        seed: Optional[int] = None):
        """Held-out graphs packed like samples: (ids (B,N), adj (B,N,N))."""
        graphs = self.test_graphs
        if seed is not None and n is not None:
            g = torch.Generator().manual_seed(seed)
            pick = torch.randperm(len(graphs), generator=g)[:n].tolist()
            graphs = [graphs[i] for i in pick]
        elif n is not None:
            graphs = graphs[:n]
        return pack_graphs(graphs, max_nodes)

    def subsystems_of(self) -> dict:
        """{node name -> set(subsystems)} from the edge table (case-insensitive column)."""
        acc = collections.defaultdict(set)
        if not self.edges:
            return acc
        df = pd.read_csv(self.edges)
        col = next((c for c in df.columns if c.lower() == "subsystem"), None)
        if col is None:
            return acc
        for c in ("source", "target"):
            for n, s in zip(df[c].astype(str).str.strip(), df[col]):
                acc[n].add(str(s))
        return acc

    def token_counts(self) -> torch.Tensor:
        cnt = torch.zeros(self.vocab_size)
        for g in self.train_graphs:
            for t in g["node_ids"][:g["num_nodes"]]:
                cnt[int(t)] += 1
        return cnt


def pack_graphs(graphs, max_nodes: int = 64):
    ids = torch.zeros(len(graphs), max_nodes, dtype=torch.long)
    adj = torch.zeros(len(graphs), max_nodes, max_nodes, dtype=torch.long)
    for i, g in enumerate(graphs):
        k = g["num_nodes"]
        ids[i, :k] = torch.tensor(g["node_ids"][:k])
        adj[i, :k, :k] = torch.as_tensor(np.asarray(g["adj_matrix"])[:k, :k]).long()
    return ids, adj


def load_bundle(name: str, max_nodes: int = 64, split: float = 0.9, seed: int = 0,
                device="cpu") -> DatasetBundle:
    """The project's standard split for a registered dataset, as a bundle."""
    from graphpath import paths
    from graphpath.data.loaders import REGISTRY, load_dataset
    train_ds, test_ds, info = load_dataset(name, max_nodes, split, seed, device=device)
    spec = REGISTRY[name]
    tr = info["train_graphs"]
    dens = float(np.mean([np.asarray(g["adj_matrix"])[:g["num_nodes"], :g["num_nodes"]].sum()
                          / max(g["num_nodes"] * (g["num_nodes"] - 1), 1) for g in tr]))
    expr = info["expr_dir"] if name != "human" else None
    expr_path = str(paths.PT_FILES) if name == "human" and paths.PT_FILES.is_dir() else None
    perturb = str(paths.PERTURB_REF) if name == "human" and paths.PERTURB_REF.is_dir() else None
    return DatasetBundle(
        name=name, train=train_ds, test=test_ds, train_graphs=tr,
        test_graphs=info["test_graphs"], node_map=info["node_map"],
        common=info["get_common_name"], ntype=info["node_type_fn"], density=dens,
        edges=info["edges"], expr=expr, expr_path=expr_path, perturb=perturb,
        name_map=str(spec.name_map) if spec.name_map and spec.name_map.exists() else None)


# ── run layout ────────────────────────────────────────────────────────────

class RunLayout:
    """A training run on disk.

        outputs/runs/<name>/[<variant>/]<dataset>/{weights,samples,figures}/<arm>.pt
        outputs/runs/<name>/history.json, manifest.json

    Variants are optional (the October runs have `nopad/` and `pad/`); every
    accessor takes `variant=None` for runs without them.
    """

    SUBDIRS = ("weights", "samples", "figures")

    def __init__(self, name_or_path):
        p = Path(name_or_path)
        self.path = p if p.is_absolute() else RUNS / p
        self.name = self.path.name

    @classmethod
    def new(cls, prefix: str) -> "RunLayout":
        return cls(RUNS / f"{prefix}_{datetime.now():%Y%m%d_%H%M}")

    @classmethod
    def latest(cls, prefix: str, complete_steps: Optional[int] = None) -> "RunLayout":
        """Newest run with weights (and, if asked, at least `complete_steps`)."""
        cands = sorted(p for p in RUNS.glob(f"{prefix}_*") if p.is_dir())
        for p in reversed(cands):
            r = cls(p)
            ws = r.all_weights()
            if not ws:
                continue
            if complete_steps is not None:
                steps = [r.load_checkpoint(*k).get("steps", 0) for k in ws]
                if min(steps) < complete_steps:
                    continue
            return r
        raise FileNotFoundError(f"no complete {prefix}_* run under {RUNS}")

    # layout discovery
    def variants(self) -> list:
        if not self.path.exists():
            return [None]
        kids = [p for p in self.path.iterdir() if p.is_dir()]
        if any((k / "weights").is_dir() for k in kids):
            return [None]
        vs = sorted(k.name for k in kids if any((d / "weights").is_dir()
                                                for d in k.iterdir() if d.is_dir()))
        return vs or [None]

    def _base(self, variant):
        return self.path / variant if variant else self.path

    def datasets(self, variant=None) -> list:
        b = self._base(variant)
        if not b.exists():
            return []
        return sorted(p.name for p in b.iterdir() if (p / "weights").is_dir())

    def arms(self, dataset: str, variant=None) -> list:
        d = self._base(variant) / dataset / "weights"
        return sorted(p.stem for p in d.glob("*.pt")) if d.is_dir() else []

    def all_weights(self) -> list:
        """[(dataset, arm, variant), ...] for every checkpoint present."""
        out = []
        for v in self.variants():
            for d in self.datasets(v):
                for a in self.arms(d, v):
                    out.append((d, a, v))
        return out

    # paths
    def arm_dir(self, dataset: str, variant=None, make: bool = False) -> Path:
        d = self._base(variant) / dataset
        if make:
            for sub in self.SUBDIRS:
                (d / sub).mkdir(parents=True, exist_ok=True)
        return d

    def weights(self, dataset, arm, variant=None) -> Path:
        return self.arm_dir(dataset, variant) / "weights" / f"{arm}.pt"

    def samples(self, dataset, arm, variant=None) -> Path:
        return self.arm_dir(dataset, variant) / "samples" / f"{arm}.pt"

    # I/O
    def load_checkpoint(self, dataset, arm, variant=None, device="cpu") -> dict:
        return torch.load(self.weights(dataset, arm, variant), map_location=device,
                          weights_only=False)

    def load_samples(self, dataset, arm, variant=None):
        p = self.samples(dataset, arm, variant)
        if not p.exists():
            return None
        d = torch.load(p, map_location="cpu", weights_only=False)
        return d["nodes"].long(), (d["adj"] > 0).long()

    def save_samples(self, dataset, arm, nodes, adj, variant=None, **extra) -> Path:
        self.arm_dir(dataset, variant, make=True)
        p = self.samples(dataset, arm, variant)
        torch.save({"nodes": nodes.cpu(), "adj": adj.cpu(), **extra}, p)
        return p

    def manifest(self) -> dict:
        p = self.path / "manifest.json"
        return load_json(p) if p.exists() else {}

    def write_manifest(self, **fields) -> Path:
        m = self.manifest()
        m.update(fields, updated=datetime.now().isoformat(timespec="seconds"))
        return dump_json(m, self.path / "manifest.json")

    def analysis_dir(self) -> Path:
        d = ANALYSIS / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def __repr__(self):
        return f"RunLayout({self.path})"


def key(dataset: str, arm: str, variant=None) -> str:
    """The flat key used inside JSON files: 'ecoli|masked' or 'pad/ecoli|masked'."""
    return f"{variant + '/' if variant else ''}{dataset}|{arm}"


def split_key(k: str):
    variant, _, rest = k.rpartition("/")
    if not variant and rest.count("|") == 2:      # older history.json: 'pad|ecoli|masked'
        variant, _, rest = rest.partition("|")
    dataset, _, arm = rest.partition("|")
    return dataset, arm, (variant or None)


def arm_label(k: str) -> str:
    """The display name of an arm: 'masked', or 'masked (pad)' in a run with
    variants, so the two variants of one arm never share a table column."""
    _, arm, variant = split_key(k)
    return arm + (f" ({variant})" if variant else "")


def free(*objs):
    """Drop references and clear the accelerator cache."""
    import gc
    del objs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
