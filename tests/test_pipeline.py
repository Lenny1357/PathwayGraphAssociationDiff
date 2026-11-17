"""The pipeline plumbing without data: JSON round-trips, run layout
discovery, result serialisation, rank tables and figures from synthetic JSON."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from graphpath.pipeline import common as C
from graphpath.pipeline import tables as T


def test_json_encoder_roundtrip(tmp_path):
    obj = {"a": np.arange(3), "b": torch.tensor([1.5, 2.5]), "c": np.float32(2.0), "d": float("nan"),
           "e": {1, 2}, "f": np.int64(7), "g": pd.Series({"x": 1.0})}
    p = C.dump_json(obj, tmp_path / "x.json")
    back = json.loads(p.read_text())
    assert back["a"] == [0, 1, 2] and back["b"] == [1.5, 2.5] and back["c"] == 2.0
    assert back["d"] is None and back["e"] == [1, 2] and back["f"] == 7 and back["g"] == {"x": 1.0}


def test_result_to_dict():
    res = SimpleNamespace(name="r", summary={"m": 0.5, "k": 3}, n_graphs=4, n_scored=2, coverage=0.5,
                          notes=["n"], per_graph={"m": torch.tensor([0.1, 0.9])},
                          distributions={"deg": np.array([1, 2, 3]), "types": {"gene": 2}})
    d = C.result_to_dict(res)
    assert d["per_graph"]["m"] == [pytest.approx(0.1), pytest.approx(0.9)]
    assert d["distributions"]["deg"] == [1.0, 2.0, 3.0] and d["distributions"]["types"] == {"gene": 2}
    json.dumps(d, cls=C.JSONEncoder)


def _fake_run(root, variants=(None,)):
    for v in variants:
        for d in ("ecoli", "human"):
            base = root / (v or "") / d if v else root / d
            for arm in ("masked", "vae"):
                (base / "weights").mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": {}, "cfg": {}, "steps": 5, "kind": arm}, base / "weights" / f"{arm}.pt")
    return C.RunLayout(root)


def test_run_layout_flat(tmp_path):
    run = _fake_run(tmp_path / "compare_x_1")
    assert run.variants() == [None]
    assert run.datasets() == ["ecoli", "human"] and run.arms("ecoli") == ["masked", "vae"]
    assert len(run.all_weights()) == 4
    assert run.load_checkpoint("ecoli", "masked")["steps"] == 5
    run.save_samples("ecoli", "masked", torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, 4, 4, dtype=torch.long))
    n, a = run.load_samples("ecoli", "masked")
    assert n.shape == (2, 4) and a.shape == (2, 4, 4)


def test_run_layout_variants(tmp_path):
    run = _fake_run(tmp_path / "compare_oct2_1", variants=("nopad", "pad"))
    assert run.variants() == ["nopad", "pad"]
    assert run.datasets("pad") == ["ecoli", "human"]
    assert {v for _, _, v in run.all_weights()} == {"nopad", "pad"}
    assert C.split_key(C.key("ecoli", "masked", "pad")) == ("ecoli", "masked", "pad")
    assert C.split_key(C.key("ecoli", "masked")) == ("ecoli", "masked", None)


def _fake_scores():
    def res(**summary):
        return {"name": "x", "summary": summary, "n_graphs": 4, "n_scored": 4, "coverage": 1.0, "notes": [],
                "per_graph": {k: [v] * 4 for k, v in summary.items()}, "distributions": {}}
    arms = {}
    for arm, ep, dens in (("masked", 0.1, 0.03), ("transition", 0.3, 0.05), ("vae", 0.0, 0.08)):
        arms[f"ecoli|{arm}"] = {"known": {a: res(edge_precision=ep) for a in ("generated", "shuffled", "random")},
                                "topo": {a: res(density=dens, clustering=0.01) for a in ("generated", "shuffled", "random")}}
    held = {"ecoli": {"known": res(edge_precision=1.0), "topo": res(density=0.042, clustering=0.0)}}
    return {"held-out": held, "arms": arms, "notes": {}, "n_graphs": 4}


def test_rank_table_directions():
    raw, rank, broken = T.rank_table(_fake_scores(), "ecoli")
    assert broken == []
    assert rank.loc["known.edge_precision", "transition"] == 1.0           # up: highest wins
    assert rank.loc["topo.density", "transition"] == 1.0                   # target: closest to 0.042
    assert rank.loc["topo.clustering"].tolist() == [2.0, 2.0, 2.0]         # tied
    assert rank["transition"].mean() < rank["vae"].mean()


def test_metric_table_and_tables(tmp_path):
    scores = _fake_scores()
    tab, cov = T.metric_table(scores, "ecoli", "known")
    assert "edge_precision" in tab.index and cov["held-out"] == "4/4"
    C.dump_json(scores, tmp_path / "scores.json")
    written = T.write_tables(tmp_path / "scores.json", tmp_path / "tables")
    assert any(p.name == "rank_ecoli.csv" for p in written)


def test_figures_from_synthetic_json(tmp_path):
    from graphpath.pipeline import figures as F
    scores = _fake_scores()
    paths = F.score_bars(scores, tmp_path) + F.rank_figures(scores, tmp_path)
    assert paths and all(p.exists() for p in paths)
    hist = {"ecoli|masked": [{"step": s, "loss": 1 / (s + 1)} for s in range(1, 20)]}
    assert F.loss_curves(hist, tmp_path)[0].exists()


def test_compare_runs():
    new, old = _fake_scores(), _fake_scores()
    old["arms"]["ecoli|masked"]["known"]["generated"]["summary"]["edge_precision"] = 0.05
    df = T.compare_runs(new, old)
    row = df[(df.arm == "masked") & (df.metric == "known.edge_precision")].iloc[0]
    assert row.closer > 0      # 0.1 is closer to held-out 1.0 than 0.05 was
