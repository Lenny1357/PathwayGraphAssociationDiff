"""Tables derived from scores.json: per-benchmark metric tables, coverage,
rank aggregates, and the comparison of two runs. Pure functions of the JSON."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from graphpath.pipeline.common import arm_label, load_json, nanfloat, split_key

# direction per metric: "target" = match held-out, "down" / "up" = lower / higher is better
RANK_METRICS = [
    ("known", "edge_precision", "up"), ("known", "edge_f1", "up"),
    ("known", "subsystem_purity", "up"), ("known", "unique_node_frac", "up"),
    ("coexpr", "coexpr_hit_rate_pooled", "up"), ("coexpr", "coexpr_best_tissue_mean", "up"),
    ("coexpr", "duplicate_gene_rate", "down"),
    ("memo", "near_duplicate_rate", "down"), ("memo", "nn_node_jaccard_mean", "target"),
    ("memo", "distinct_matches", "up"),
    ("topo", "degree", "target"), ("topo", "clustering", "down"),      # triangles = type violations
    ("topo", "square_clustering", "target"), ("topo", "density", "target"),
    ("topo", "n_components", "target"), ("topo", "isolated_node_frac", "target"),
    ("novel", "type_possible_frac", "up"), ("novel", "novel_frac", "target"),
    ("flux", "participation_precision", "up"), ("flux", "mass_balanced_frac", "up"),
    ("perturb-row", "pooled_d", "up"), ("perturb-column", "pooled_d", "up"),
]


def arms_of(scores: dict, dataset: str) -> list:
    return [k for k in scores["arms"] if split_key(k)[0] == dataset]


def datasets_of(scores: dict) -> list:
    return sorted(scores["held-out"])


def _num(v):
    return nanfloat(v) if isinstance(v, (int, float, type(None))) else float("nan")


def metric_table(scores: dict, dataset: str, bench: str) -> tuple:
    """(DataFrame metrics x [arms..., held-out], coverage dict). Rows identical
    everywhere are dropped and counted."""
    keys = [k for k in arms_of(scores, dataset) if bench in scores["arms"][k]]
    if not keys:
        return pd.DataFrame(), {}
    cols = [arm_label(k) for k in keys]
    ho = scores["held-out"][dataset].get(bench)
    metrics = []
    for k in keys:
        for m, v in scores["arms"][k][bench]["generated"]["summary"].items():
            if isinstance(v, (int, float, type(None))) and m not in metrics:
                metrics.append(m)
    tab = pd.DataFrame(index=metrics, columns=cols + ["held-out"], dtype=float)
    for k, c in zip(keys, cols):
        s = scores["arms"][k][bench]["generated"]["summary"]
        for m in metrics:
            tab.loc[m, c] = _num(s.get(m))
    for m in metrics:
        tab.loc[m, "held-out"] = _num(ho["summary"].get(m)) if ho else float("nan")
    keep = [m for m in metrics if not (tab.loc[m].dropna().size
                                       and np.allclose(tab.loc[m].dropna().values,
                                                       tab.loc[m].dropna().values[0]))]
    tab = tab.loc[keep]
    tab.attrs["dropped_constant"] = len(metrics) - len(keep)
    cov = {c: f"{scores['arms'][k][bench]['generated']['n_scored']}/"
              f"{scores['arms'][k][bench]['generated']['n_graphs']}" for k, c in zip(keys, cols)}
    cov["held-out"] = f"{ho['n_scored']}/{ho['n_graphs']}" if ho else "-"
    return tab, cov


def coverage_table(scores: dict, dataset: str) -> pd.DataFrame:
    keys = arms_of(scores, dataset)
    benches = sorted({b for k in keys for b in scores["arms"][k]})
    cov = pd.DataFrame({arm_label(k): {
        b: (f"{scores['arms'][k][b]['generated']['n_scored']}/{scores['arms'][k][b]['generated']['n_graphs']}"
            if b in scores["arms"][k] else "-") for b in benches} for k in keys})
    cov["held-out"] = {b: (f"{scores['held-out'][dataset][b]['n_scored']}/"
                           f"{scores['held-out'][dataset][b]['n_graphs']}"
                           if b in scores["held-out"][dataset] else "-") for b in benches}
    return cov


def rank_table(scores: dict, dataset: str, metrics=RANK_METRICS) -> tuple:
    """(raw DataFrame, rank DataFrame, broken benches). A benchmark on which
    the held-out arm scored nothing is misconfigured and excluded."""
    keys = arms_of(scores, dataset)
    arms = [arm_label(k) for k in keys]
    held = scores["held-out"][dataset]
    broken = {b for b, r in held.items() if r["n_scored"] == 0}
    rows = []
    for bench, metric, direction in metrics:
        if bench in broken or not any(bench in scores["arms"][k] for k in keys):
            continue
        row = {"benchmark": bench, "metric": metric, "read": direction}
        for k, a in zip(keys, arms):
            row[a] = (_num(scores["arms"][k][bench]["generated"]["summary"].get(metric))
                      if bench in scores["arms"][k] else float("nan"))
        row["held-out"] = _num(held[bench]["summary"].get(metric)) if bench in held else float("nan")
        if all(not np.isfinite(row[a]) for a in arms):
            continue
        rows.append(row)
    raw = pd.DataFrame(rows)
    rank = pd.DataFrame(index=[f"{r['benchmark']}.{r['metric']}" for r in rows], columns=arms, dtype=float)
    for r, idx in zip(rows, rank.index):
        vals = {a: r[a] for a in arms}
        if r["read"] == "target" and np.isfinite(r["held-out"]):
            good = {a: -abs(v - r["held-out"]) for a, v in vals.items()}
        elif r["read"] == "down":
            good = {a: -v for a, v in vals.items()}
        else:
            good = dict(vals)
        rank.loc[idx] = pd.Series(good).rank(ascending=False, na_option="bottom")
    return raw, rank, sorted(broken)


def write_tables(scores_path, out_dir) -> list:
    """Every table scores.json supports, as CSV. Returns the paths written."""
    scores = load_json(scores_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    written = []
    for d in datasets_of(scores):
        benches = sorted({b for k in arms_of(scores, d) for b in scores["arms"][k]})
        for b in benches:
            tab, cov = metric_table(scores, d, b)
            if tab.empty:
                continue
            tab.loc["graphs scored"] = pd.Series(cov)
            p = out / f"table_{d}_{b}.csv"; tab.to_csv(p); written.append(p)
        p = out / f"coverage_{d}.csv"; coverage_table(scores, d).to_csv(p); written.append(p)
        raw, rank, _ = rank_table(scores, d)
        if not raw.empty:
            p = out / f"raw_{d}.csv"; raw.to_csv(p, index=False); written.append(p)
            p = out / f"rank_{d}.csv"; rank.to_csv(p); written.append(p)
            rank.mean().rename("mean_rank").to_csv(out / f"mean_rank_{d}.csv")
    return written


def architecture_table(arch: dict) -> pd.DataFrame:
    rows = []
    for k, a in sorted(arch.items()):
        top = max(a["modules"].items(), key=lambda kv: kv[1]) if a["modules"] else ("—", 0)
        rows.append({"dataset": a["dataset"], "arm": a["arm"], "variant": a["variant"] or "",
                     "vocab": a["vocab"], "params (M)": a["learned"] / 1e6,
                     "buffers (M)": a["buffers"] / 1e6,
                     "largest module": f"{top[0]} {top[1] / 1e6:.2f}M", "steps": a["steps"],
                     "loss first→last": (f"{a['first_loss']:.2f}→{a['final_loss']:.2f}"
                                         if a["first_loss"] is not None else "—"),
                     "ckpt MB": a["ckpt_mb"]})
    return pd.DataFrame(rows)


# ── two runs against each other (typed vs untyped) ────────────────────────

COMPARE_METRICS = [
    ("topo", "clustering", "down", "type", "triangles = edges that skip the bipartition"),
    ("topo", "square_clustering", "target", "type", "closure that survives a bipartite graph"),
    ("known", "edge_precision", "up", "type", "edges that exist in the GEM"),
    ("known", "edge_f1", "up", "type", ""),
    ("flux", "mass_balanced_frac", "up", "type", "reactions that conserve atoms"),
    ("flux", "participation_precision", "up", "type", ""),
    ("novel", "type_possible_frac", "up", "type", "novel edges that are at least type-legal"),
    ("topo", "n_nodes", "target", "pad", "graph size — fixed at 64 without supervise_pad"),
]


def compare_runs(scores_new: dict, scores_old: dict, label_new="typed", label_old="untyped",
                 metrics=COMPARE_METRICS) -> pd.DataFrame:
    """Per (dataset, arm, metric): both values, held-out, and whether the new
    run moved closer to held-out (symmetric bounded distance |v-h|/(|v|+|h|))."""
    def dist(v, h):
        if not (np.isfinite(v) and np.isfinite(h)):
            return float("nan")
        den = abs(v) + abs(h)
        return 0.0 if den == 0 else abs(v - h) / den
    rows = []
    for k, arms in scores_new["arms"].items():
        d, arm, _ = split_key(k)
        old = scores_old["arms"].get(f"{d}|{arm}") or scores_old["arms"].get(k)
        if old is None:
            continue
        for bench, metric, read, family, note in metrics:
            if bench not in arms or bench not in old:
                continue
            new_v = _num(arms[bench]["generated"]["summary"].get(metric))
            old_v = _num(old[bench]["generated"]["summary"].get(metric))
            ho = scores_new["held-out"].get(d, {}).get(bench)
            ho_v = _num(ho["summary"].get(metric)) if ho else float("nan")
            if not (np.isfinite(new_v) and np.isfinite(old_v)):
                continue
            rows.append(dict(dataset=d, arm=arm_label(k), metric=f"{bench}.{metric}", read=read,
                             family=family, note=note, **{label_old: old_v, label_new: new_v},
                             held_out=ho_v, d_old=dist(old_v, ho_v), d_new=dist(new_v, ho_v)))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["closer"] = df.d_old - df.d_new      # > 0: the new run moved toward held-out
    return df
