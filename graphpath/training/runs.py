"""Run directories and checkpoints — the bookkeeping every notebook re-wrote.

    run = RunDir.new("compare_sept")             # outputs/runs/compare_sept_<stamp>/
    run = RunDir.latest("compare_typed")         # newest existing one
    d = run.arm_dir("ecoli", "masked")           # .../ecoli/masked/{weights,figures,samples}
    save_checkpoint(model, cfg, d / "weights/masked.pt", history=hist)
    model = load_checkpoint(path, build_fn)

Checkpoints hold `state_dict`, the config as a dict, and anything else you
pass; they never pickle the model object, so code can move without
invalidating them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch

from graphpath.paths import RUNS


def guard(path, overwrite: bool = False) -> str:
    """Return `path`, creating its parent; refuse to clobber unless `overwrite`."""
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} exists — pass overwrite=True to replace it")
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


class RunDir:
    """One experiment's output tree under outputs/runs/<prefix>_<stamp>/."""

    SUBDIRS = ("weights", "figures", "samples", "tables")

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def new(cls, prefix: str, root: Path = RUNS) -> "RunDir":
        return cls(root / f"{prefix}_{datetime.now():%Y%m%d_%H%M}")

    @classmethod
    def latest(cls, prefix: str, root: Path = RUNS) -> "RunDir":
        runs = sorted(p for p in root.glob(f"{prefix}_*") if p.is_dir())
        if not runs:
            raise FileNotFoundError(f"no {root}/{prefix}_* run exists yet")
        return cls(runs[-1])

    def arm_dir(self, *parts: str) -> Path:
        d = self.path.joinpath(*parts)
        for sub in self.SUBDIRS:
            (d / sub).mkdir(parents=True, exist_ok=True)
        return d

    def write_json(self, name: str, obj, overwrite: bool = True) -> Path:
        p = Path(guard(self.path / name, overwrite))
        p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    def __truediv__(self, other):
        return self.path / other

    def __repr__(self):
        return f"RunDir({self.path})"


def _cfg_dict(cfg) -> dict:
    return asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))


def save_checkpoint(model: torch.nn.Module, cfg, path, overwrite: bool = True, **extra) -> str:
    """state_dict + config dict + whatever else (history, arm name, ...)."""
    p = guard(path, overwrite)
    torch.save({"state_dict": model.state_dict(), "cfg": _cfg_dict(cfg), **extra}, p)
    return p


def load_checkpoint(path, build: Callable[[dict], torch.nn.Module],
                    device: Optional[torch.device] = None, strict: bool = True):
    """
    `build(cfg_dict) -> model` constructs the right class from the saved
    config; the weights are then loaded into it. Returns (model, checkpoint).
    """
    ck = torch.load(path, map_location=device or "cpu", weights_only=False)
    model = build(ck["cfg"])
    model.load_state_dict(ck["state_dict"], strict=strict)
    if device is not None:
        model.to(device)
    return model, ck
