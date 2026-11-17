"""Every figure, drawn from the JSON an analysis wrote. Nothing here loads a
model or a dataset, so a plot can be redrawn in seconds after a style change.

    draw_all(run_name)                    everything whose JSON exists
    loss_curves(history, out, ...)        one function per figure family

Colour follows the model NAME, never its position, so filtering an arm out
never repaints the survivors; every multi-arm line also carries a marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from graphpath.pipeline import tables as T  # noqa: E402
from graphpath.pipeline.common import ANALYSIS, RUNS, arm_label, load_json, nanfloat, split_key  # noqa: E402
from graphpath.plotting.style import TYPE_COLORS, _color  # noqa: E402

INK, SOFT, GRID, GREY = "#1a1a19", "#6b6a66", "#e4e3de", "#b9b7ae"
NULL_COLOR = "#c25a10"
MODEL_ORDER = ["masked", "transition", "continuous", "vae", "marginal", "uniform"]
MODEL_COLOR = {k: _color(k, i) for i, k in enumerate(MODEL_ORDER)}
MODEL_COLOR["held-out"] = _color("held-out", len(MODEL_ORDER))
MODEL_MARKER = dict(zip(MODEL_ORDER, ["o", "s", "^", "D", "v", "P"]))
MODEL_MARKER["held-out"] = "v"
DATASET_COLOR = {"ecoli": "#2a78d6", "human": "#c25a10"}
RC = {"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8880",
      "grid.color": GRID, "font.size": 9, "figure.facecolor": "white"}


def model_color(k, default="#2a78d6"):
    return MODEL_COLOR.get(k.split()[0] if isinstance(k, str) else k, default)


def model_marker(k, default="o"):
    return MODEL_MARKER.get(k.split()[0] if isinstance(k, str) else k, default)


def _fname(label: str) -> str:
    """'masked (pad)' -> 'masked_pad', for file names."""
    return label.replace(" (", "_").replace(")", "").replace(" ", "_")


def _vsuffix(v) -> str:
    return f"_{v}" if v else ""


def _save(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return path


def _bars(ax, names, vals, colors, fmt="{:.3g}", ref=None, ref_label="held-out", ylabel=""):
    x = np.arange(len(names))
    bars = ax.bar(x, [0 if v is None or not np.isfinite(v) else v for v in vals], color=colors,
                  edgecolor="white", linewidth=2, width=.68, zorder=2)
    for b, v in zip(bars, vals):
        if v is not None and np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom",
                    fontsize=8, color=INK, zorder=4)
    if ref is not None and np.isfinite(ref):
        ax.axhline(ref, ls="--", lw=1.3, color=SOFT, zorder=1)
        ax.text(0.99, ref, ref_label + " ", ha="right", va="bottom", fontsize=7.5, color=SOFT,
                transform=ax.get_yaxis_transform(), bbox=dict(fc="white", ec="none", pad=1.2, alpha=.85))
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.grid(alpha=.35, axis="y"); ax.set_axisbelow(True)


def _roll(a, k=11):
    v = np.asarray(a, float).copy(); m = np.isfinite(v)
    if m.sum() < k:
        return v
    v[~m] = np.interp(np.flatnonzero(~m), np.flatnonzero(m), v[m])
    return np.convolve(v, np.ones(k) / k, mode="same")


def _arr(v):
    a = np.asarray([np.nan if x is None else x for x in (v or [])], dtype=float)
    return a[np.isfinite(a)]


# ── training ─────────────────────────────────────────────────────────────

def loss_curves(history: dict, out: Path, steps: Optional[int] = None) -> list:
    """Each arm in its own panel: four objectives on four scales. What IS
    comparable is the end slope."""
    paths, groups = [], {}
    for k, h in history.items():
        d, arm, v = split_key(k)
        groups.setdefault((d, v), {})[arm] = h
    with plt.rc_context(RC):
        for (d, v), hs in groups.items():
            hs = {k: h for k, h in hs.items() if h}
            if not hs:
                continue
            fig, ax = plt.subplots(1, len(hs), figsize=(3.5 * len(hs), 3.0), squeeze=False)
            for i, (k, h) in enumerate(hs.items()):
                a_ = ax[0][i]; st = [x["step"] for x in h]; ls = [x["loss"] for x in h]
                a_.plot(st, ls, lw=1.8, color=model_color(k)); a_.set_title(k, loc="left", color=INK)
                a_.set_xlabel("optimiser step"); a_.grid(alpha=.35); a_.set_axisbelow(True)
                w = max(2, len(h) // 5)
                sl = np.polyfit(np.arange(w), ls[-w:], 1)[0] if len(h) >= 2 else np.nan
                a_.text(0.98, 0.95, f"end slope {sl:+.2e}/log", transform=a_.transAxes, ha="right", va="top",
                        fontsize=7.5, color=SOFT)
                if i == 0:
                    a_.set_ylabel("training loss (own objective)")
            fig.suptitle(f"{(v + '/') if v else ''}{d} — loss per arm (comparable WITHIN a panel, never across)",
                         x=0.006, ha="left", fontsize=11, color=INK)
            fig.tight_layout(rect=[0, 0, 1, 0.90])
            paths.append(_save(fig, out / d / f"loss_curves{('_' + v) if v else ''}.png"))
    return paths


def sample_diagnostics(samples: dict, out: Path) -> list:
    paths, by = [], {}
    for k, s in samples.items():
        by.setdefault(split_key(k)[0], {})[arm_label(k)] = s
    with plt.rc_context(RC):
        for d, arms in by.items():
            names = list(arms)
            for key_, lbl, ref in (("density", "edge density", "data_density"),
                                    ("pad_frac", "PAD share of slots", "data_pad_frac"),
                                    ("nodes_mean", "nodes per graph", None)):
                fig, ax = plt.subplots(figsize=(5.4, 3.5))
                _bars(ax, names, [arms[n][key_] for n in names], [model_color(n) for n in names],
                      ref=(arms[names[0]][ref] if ref else None), ref_label="data", ylabel=lbl)
                paths.append(_save(fig, out / d / f"samples_{key_}.png"))
    return paths


# ── benchmark scores ─────────────────────────────────────────────────────

FIG_METRICS = {("known", "edge_precision"), ("known", "subsystem_purity"), ("topo", "density"),
               ("topo", "square_clustering"), ("memo", "nn_node_jaccard_mean"),
               ("coexpr", "coexpr_hit_rate_pooled"), ("flux", "participation_precision"),
               ("perturb-row", "pooled_d"), ("perturb-column", "pooled_d")}
FIG_DISTS = {"degree", "clustering_node", "square_clustering_node", "path_length", "shared_neighbour",
             "pairs", "similarity", "per_tissue_coexpr", "node_jaccard"}
CHANCE = {"pooled_d": (0.0, "random genes (d = 0)"), "pooled_auc": (0.5, "random genes (AUC = 0.5)")}


def score_bars(scores: dict, out: Path, metrics=FIG_METRICS) -> list:
    """Bars per arm, held-out dashed, interquartile range across graphs."""
    paths = []
    with plt.rc_context(RC):
        for d in T.datasets_of(scores):
            keys = T.arms_of(scores, d)
            for bench, metric in sorted(metrics):
                have = [k for k in keys if bench in scores["arms"][k] and metric in scores["arms"][k][bench]["generated"]["summary"]]
                if not have:
                    continue
                names = [arm_label(k) for k in have]
                vals = [nanfloat(scores["arms"][k][bench]["generated"]["summary"][metric]) for k in have]
                ho = scores["held-out"][d].get(bench)
                hv = nanfloat(ho["summary"].get(metric)) if ho else np.nan
                fig, ax = plt.subplots(figsize=(5.2, 3.6))
                _bars(ax, names, vals, [model_color(n) for n in names], ref=hv)
                for xi, k in enumerate(have):
                    pg = _arr(scores["arms"][k][bench]["generated"]["per_graph"].get(metric))
                    if pg.size > 2:
                        q1, q3 = np.percentile(pg, [25, 75])
                        ax.plot([xi, xi], [q1, q3], lw=2, color=INK, alpha=.55, zorder=3)
                    nv = nanfloat(scores["arms"][k][bench]["generated"]["summary"].get(f"{metric}_null"))
                    if np.isfinite(nv):
                        ax.plot([xi - .34, xi + .34], [nv, nv], lw=2.2, color=NULL_COLOR, zorder=5)
                if metric in CHANCE:
                    cv, cl = CHANCE[metric]
                    ax.axhline(cv, ls=":", lw=1.5, color=NULL_COLOR, zorder=1)
                    ax.text(0.01, cv, cl, ha="left", va="bottom", fontsize=7.5, color=NULL_COLOR,
                            transform=ax.get_yaxis_transform())
                cov = "  ".join(f"{n} {scores['arms'][k][bench]['generated']['n_scored']}/"
                                f"{scores['arms'][k][bench]['generated']['n_graphs']}" for n, k in zip(names, have))
                ax.set_title(f"{d} — {bench}: {metric}\nscored: {cov}", loc="left", fontsize=9, color=INK)
                paths.append(_save(fig, out / d / f"m_{bench}_{metric}.png"))
    return paths


def score_distributions(scores: dict, out: Path, dists=FIG_DISTS) -> list:
    """Pooled distributions as ECDFs, every arm on one axis; grey = references."""
    paths = []
    with plt.rc_context(RC):
        for d in T.datasets_of(scores):
            keys = T.arms_of(scores, d)
            for bench in sorted({b for k in keys for b in scores["arms"][k]}):
                have = [k for k in keys if bench in scores["arms"][k]]
                names = sorted({m for k in have for m in scores["arms"][k][bench]["generated"]["distributions"] if m in dists})
                for metric in names:
                    fig, ax = plt.subplots(figsize=(6.2, 3.9)); drew, nulls = 0, []
                    for k in have:
                        dd = scores["arms"][k][bench]["generated"]["distributions"]
                        arr = _arr(dd.get(metric))
                        if arr.size <= 5:
                            continue
                        xs = np.sort(arr); arm = arm_label(k)
                        ax.step(xs, np.arange(1, xs.size + 1) / xs.size, where="post", lw=1.9, color=model_color(arm),
                                label=f"{arm} (n={xs.size:,})", marker=model_marker(arm), markevery=max(xs.size // 10, 1), ms=4.5)
                        drew += 1
                        nl = _arr(dd.get(f"{metric}_null"))
                        if nl.size > 5:
                            nulls.append(nl)
                    if not drew:
                        plt.close(fig); continue
                    ho = scores["held-out"][d].get(bench)
                    hx = _arr(ho["distributions"].get(metric)) if ho else np.array([])
                    if hx.size > 5:
                        hx = np.sort(hx)
                        ax.step(hx, np.arange(1, hx.size + 1) / hx.size, where="post", lw=2.4, color=SOFT, ls="--",
                                label=f"held-out (n={hx.size:,})")
                    if nulls:
                        nx_ = np.sort(np.concatenate(nulls))
                        ax.step(nx_, np.arange(1, nx_.size + 1) / nx_.size, where="post", lw=2.0, color=GREY, ls=":",
                                label=f"random genes, pooled (n={nx_.size:,})")
                    ax.set_xlabel(metric); ax.set_ylabel("cumulative share")
                    ax.legend(fontsize=7, frameon=False, loc="lower right"); ax.grid(alpha=.35); ax.set_axisbelow(True)
                    ax.set_title(f"{d} — {bench}: {metric}", loc="left", fontsize=9, color=INK)
                    paths.append(_save(fig, out / d / f"d_{bench}_{metric}.png"))
    return paths


def rank_figures(scores: dict, out: Path) -> list:
    paths = []

    def ink_on(cmap, frac):
        if not np.isfinite(frac):
            return INK
        r, g, b = plt.get_cmap(cmap)(float(np.clip(frac, 0, 1)))[:3]
        return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else INK

    with plt.rc_context(RC):
        for d in T.datasets_of(scores):
            cov = T.coverage_table(scores, d)
            raw, rank, _ = T.rank_table(scores, d)
            if cov.empty:
                continue
            frac = cov.map(lambda s: float(s.split("/")[0]) / float(s.split("/")[1])
                           if isinstance(s, str) and "/" in s and float(s.split("/")[1]) > 0 else np.nan)
            fig, ax = plt.subplots(figsize=(1.15 * len(frac.columns) + 2.4, 0.42 * len(frac) + 1.9))
            im = ax.imshow(frac.values.astype(float), cmap="Greens", vmin=0, vmax=1, aspect="auto")
            for i in range(frac.shape[0]):
                for j in range(frac.shape[1]):
                    ax.text(j, i, cov.values[i, j], ha="center", va="center", fontsize=7.5, color=ink_on("Greens", frac.values[i, j]))
            ax.set_xticks(range(len(frac.columns))); ax.set_xticklabels(frac.columns, fontsize=8, rotation=20, ha="right")
            ax.set_yticks(range(len(frac.index))); ax.set_yticklabels(frac.index, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=.03, pad=.02, label="share of graphs scored")
            ax.set_title(f"{d} — how much of each benchmark ran (pale = missing, not poor)", loc="left", fontsize=9)
            paths.append(_save(fig, out / d / "coverage_map.png"))
            if rank.empty:
                continue
            kinds = list(rank.columns); rk = rank.astype(float)
            fig, ax = plt.subplots(figsize=(1.15 * len(kinds) + 3.4, 0.34 * len(rk) + 1.9))
            im = ax.imshow(rk.values, cmap="Blues_r", vmin=1, vmax=len(kinds), aspect="auto")
            for i in range(rk.shape[0]):
                for j in range(rk.shape[1]):
                    ax.text(j, i, f"{rk.values[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                            color=ink_on("Blues_r", (rk.values[i, j] - 1) / max(len(kinds) - 1, 1)))
            ax.set_xticks(range(len(kinds))); ax.set_xticklabels(kinds, fontsize=8, rotation=20, ha="right")
            ax.set_yticks(range(len(rk.index))); ax.set_yticklabels(rk.index, fontsize=7.5)
            fig.colorbar(im, ax=ax, fraction=.03, pad=.02, label="rank (1 = best)")
            ax.set_title(f"{d} — rank per metric (a metric a model could not be scored on ranks last)", loc="left", fontsize=9)
            paths.append(_save(fig, out / d / "rank_map.png"))
            order = sorted(kinds, key=lambda k: rk[k].mean()); means = [rk[k].mean() for k in order]
            fig, ax = plt.subplots(figsize=(6.0, 0.55 * len(order) + 2.1)); y = np.arange(len(order))[::-1]
            ax.barh(y, means, color=[model_color(k) for k in order], edgecolor="white", linewidth=2, height=.62, zorder=2)
            for yi, k, m in zip(y, order, means):
                q1, q3 = np.percentile(rk[k].dropna().values, [25, 75])
                ax.plot([q1, q3], [yi, yi], lw=2, color=INK, alpha=.55, zorder=3)
                ax.text(max(m, q3) + .06, yi, f"{m:.2f}", ha="left", va="center", fontsize=8.5, color=INK)
            ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9)
            ax.set_xlabel("mean rank across metrics — lower is better"); ax.set_xlim(0, len(kinds) + .55)
            ax.grid(alpha=.35, axis="x"); ax.set_axisbelow(True)
            ax.set_title(f"{d} — mean rank over {len(rk)} metrics (whiskers: interquartile range of the ranks)", loc="left", fontsize=9)
            paths.append(_save(fig, out / d / "mean_rank.png"))
    return paths


# ── trajectories ─────────────────────────────────────────────────────────

TRACES = [
    ("density", "confirmed edges / all pairs", True, False),
    ("density_resolved", "P(edge | pair decided)", True, False),
    ("edge_masked", "share of pairs still <MASK>", False, True),
    ("edge_reveal_present", "share of newly-decided pairs decided PRESENT", True, True),
    ("node_churn", "share of node slots whose token changed", False, False),
    ("edge_churn", "share of node pairs that flipped", True, False),
    ("masked", "share of node slots still <MASK>", False, True),
]


def trajectory_figures(traj: dict, out: Path, data_density: dict) -> list:
    paths, by = [], {}
    for k, tr in traj.items():
        by.setdefault(split_key(k)[0], {})[arm_label(k)] = tr
    with plt.rc_context(RC):
        for d, arms in by.items():
            dens = data_density.get(d)
            for key_, ylabel, hline, drop_zero in TRACES:
                have = [a for a, tr in arms.items() if _arr(tr[key_]).size
                        and not (drop_zero and np.allclose(_arr(tr[key_]), 0))]
                if not have:
                    continue
                fig, ax = plt.subplots(figsize=(6.2, 3.6))
                for a in have:
                    t = np.asarray(arms[a]["t"], float); y = np.asarray([np.nan if x is None else x for x in arms[a][key_]], float)
                    smooth = key_ in ("node_churn", "edge_churn", "edge_reveal_present")
                    if smooth:
                        ax.plot(t, y, lw=0.7, alpha=.25, color=model_color(a))
                        ax.plot(t, _roll(y), lw=1.9, color=model_color(a), label=a, marker=model_marker(a), markevery=max(len(t) // 12, 1), ms=4)
                    else:
                        ax.plot(t, y, lw=1.9, color=model_color(a), label=a, marker=model_marker(a), markevery=max(len(t) // 12, 1), ms=4)
                if hline and dens is not None:
                    ax.axhline(dens, ls="--", lw=1.3, color=SOFT)
                    ax.text(0.99, dens, "data density", ha="right", va="bottom", fontsize=7.5, color=SOFT,
                            transform=ax.get_yaxis_transform(), bbox=dict(fc="white", ec="none", pad=1.2, alpha=.85))
                ax.set_xlim(max(max(arms[a]["t"]) for a in have), 0)
                ax.set_xlabel("t   (sampling runs left to right)"); ax.set_ylabel(ylabel)
                ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.35); ax.set_axisbelow(True)
                ax.set_title(f"{d} — {key_}", loc="left", fontsize=9, color=INK)
                paths.append(_save(fig, out / d / f"trajectory_{key_}.png"))
            # the density ratio: how far off-manifold the trajectory strays
            if dens:
                fig, ax = plt.subplots(figsize=(6.2, 3.6))
                for a, tr in arms.items():
                    y = np.asarray([np.nan if x is None else x for x in tr["density_resolved"]], float) / dens
                    ax.plot(tr["t"], y, lw=1.9, color=model_color(a), label=a, marker=model_marker(a), markevery=max(len(tr["t"]) // 12, 1), ms=4)
                ax.axhline(1.0, ls="--", lw=1.3, color=SOFT); ax.set_xlim(max(max(tr["t"]) for tr in arms.values()), 0)
                ax.set_xlabel("t   (sampling runs left to right)"); ax.set_ylabel("decided-pair density / data density")
                ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.35); ax.set_axisbelow(True)
                paths.append(_save(fig, out / d / "trajectory_density_ratio.png"))
            # masked arm: do edge slots resolve before node slots?
            for a, tr in arms.items():
                if not _arr(tr["masked"]).size:
                    continue
                fig, ax = plt.subplots(figsize=(6.2, 3.6))
                for key_, lbl, col, mk in (("masked", "node slots still <MASK>", "#2a78d6", "o"),
                                           ("edge_masked", "node pairs still <MASK>", "#c25a10", "s")):
                    ax.plot(tr["t"], [np.nan if x is None else x for x in tr[key_]], lw=1.9, color=col, label=lbl,
                            marker=mk, markevery=max(len(tr["t"]) // 12, 1), ms=4)
                ax.set_xlim(max(tr["t"]), 0); ax.set_ylim(-0.02, 1.02)
                ax.set_xlabel("t   (sampling runs left to right)"); ax.set_ylabel("share still undecided")
                ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.35); ax.set_axisbelow(True)
                paths.append(_save(fig, out / d / f"trajectory_resolution_{_fname(a)}.png"))
    return paths


def trajectory_graphs(traj: dict, out: Path, bundles: dict) -> list:
    """The one figure that needs the dataset (for names): a single graph at
    N points along the trajectory, layout pinned to the final graph."""
    import networkx as nx
    from graphpath.plotting.graphs import create_graph_objs, plot_graph_samples
    paths = []
    for k, tr in traj.items():
        d, arm = split_key(k)[0], arm_label(k)
        if d not in bundles or "panels" not in tr:
            continue
        b, mask_id = bundles[d], tr["mask_id"]
        N = len(tr["panels"][0]["nodes"]); V = b.vocab_size; PLACE = V + 1
        id_to_node = dict(b.id_to_node)
        for i in range(N):
            id_to_node[PLACE + i] = f"<MASK {i}>"
        disp = type("DisplayDataset", (), {"id_to_node": id_to_node})()
        ntype = lambda nm: "mask" if str(nm).startswith("<MASK") else b.ntype(nm)
        cname = lambda nm: "" if str(nm).startswith("<MASK") else b.common(nm)
        import torch
        slot_ids = torch.arange(N) + PLACE
        final_n = torch.tensor(tr["panels"][-1]["nodes"])
        slot_final = [id_to_node.get(int(x), "<PAD>") for x in final_n]
        nodes, adjs, titles = [], [], []
        for p in tr["panels"]:
            n = torch.tensor(p["nodes"]); e = torch.tensor(p["edges"]).long()
            nodes.append(torch.where(n == mask_id, slot_ids, n) if mask_id is not None else n)
            adjs.append(e.float())
            nm = int((n == mask_id).sum()) if mask_id is not None else 0
            n_und = int((e == 2).sum()) // 2
            titles.append(f"t = {p['t']}\n{int((e == 1).sum()) // 2} edges" + (f", {n_und} undecided" if n_und else "")
                          + (f"\n{nm}/{N} nodes masked" if mask_id is not None else ""))
        graphs, _, labels, types, id_maps = create_graph_objs(torch.stack(nodes), torch.stack(adjs), disp,
                                                              get_common_name=cname, get_node_type=ntype)
        base = nx.spring_layout(graphs[-1], seed=42)
        name_pos = {id_maps[-1][n]: p for n, p in base.items()}
        rng = np.random.default_rng(0)
        def panel_pos(i):
            pos = {}
            for n, name in id_maps[i].items():
                if str(name).startswith("<MASK "):
                    name = slot_final[int(str(name)[6:-1])]
                pos[n] = name_pos.setdefault(name, rng.uniform(-1, 1, size=2)) if name not in name_pos else name_pos[name]
            return pos
        fig = plot_graph_samples(graphs, types, labels, titles=titles, verbose=False,
                                 pos_list=[panel_pos(i) for i in range(len(graphs))],
                                 suptitle=f"{d} — {arm}: reverse trajectory, one graph (layout pinned to the final graph)")
        paths.append(_save(fig, out / d / f"trajectory_graphs_{_fname(arm)}.png"))
    return paths


def remask_figures(remask: dict, out: Path) -> list:
    paths = []
    with plt.rc_context(RC):
        for k, r in remask.items():
            d, arm, v = split_key(k)
            steps, B = r["per_step"], r["B"]
            t_ = np.array([s["t"] for s in steps], float)
            rev = np.array([s["revealed"] for s in steps], float) / B
            rm = np.array([s["remasked"] for s in steps], float) / B
            cut = r["cut"]
            fig, ax = plt.subplots(figsize=(6.2, 3.6))
            for y, col, lbl in ((rev, model_color("masked"), "revealed"), (rm, "#d627b9", "remasked")):
                ax.plot(t_, y, lw=0.6, alpha=.25, color=col); ax.plot(t_, _roll(y), lw=2.0, color=col, label=lbl)
            ax.axvline(cut, ls="--", lw=1.2, color=SOFT)
            ax.text(cut, ax.get_ylim()[1], f" remask cutoff t={cut:.0f}", ha="left", va="top", fontsize=7.5, color=SOFT)
            ax.set_xlim(t_.max(), 0); ax.set_xlabel("t   (sampling runs left to right)"); ax.set_ylabel("node slots per step, per graph")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.35); ax.set_axisbelow(True)
            paths.append(_save(fig, out / d / f"remask_reveal_vs_remask{_vsuffix(v)}.png"))
            fig, ax = plt.subplots(figsize=(6.2, 3.6))
            ax.plot(t_, np.cumsum(rm), lw=2.0, color="#d627b9"); ax.axvline(cut, ls="--", lw=1.2, color=SOFT)
            ax.set_xlim(t_.max(), 0); ax.set_xlabel("t"); ax.set_ylabel("cumulative remasks, per graph"); ax.grid(alpha=.35)
            paths.append(_save(fig, out / d / f"remask_cumulative{_vsuffix(v)}.png"))
            counts = np.asarray(r["node_remask_count"])
            fig, ax = plt.subplots(figsize=(6.2, 3.6))
            ax.hist(counts.ravel(), bins=np.arange(counts.max() + 2) - 0.5, color=model_color("masked"), edgecolor="white", linewidth=1.5)
            ax.set_xlabel("times a slot was sent back to <MASK>"); ax.set_ylabel("node slots"); ax.grid(alpha=.35, axis="y")
            paths.append(_save(fig, out / d / f"remask_per_slot{_vsuffix(v)}.png"))
    return paths


# ── analyses ─────────────────────────────────────────────────────────────

def pathway_stat_figures(ps: dict, out: Path) -> list:
    paths = []
    with plt.rc_context(RC):
        for d, rows in ps.items():
            arms = list(rows)
            for key_, lbl in (("distinct", "distinct entities / N"), ("subsystems", "subsystems per graph"), ("density", "edge density")):
                fig, ax = plt.subplots(figsize=(5.2, 3.6))
                _bars(ax, arms, [rows[a]["mean"][key_] for a in arms], [GREY if a == "held-out" else model_color(a) for a in arms],
                      ref=rows.get("held-out", {}).get("mean", {}).get(key_), ylabel=lbl)
                paths.append(_save(fig, out / d / f"pathway_stat_{key_}.png"))
    return paths


def clustering_figures(cl: dict, out: Path) -> list:
    paths = []
    coeffs = [("clustering", "mean local triangle coefficient"), ("transitivity", "global transitivity"),
              ("square", "mean local 4-cycle coefficient"), ("n_edges", "edges per scored graph"), ("n_nodes", "nodes per scored graph")]
    with plt.rc_context(RC):
        for d, rows in cl.items():
            arms = list(rows)
            for conv, tag in (("isolates_dropped", "drop"), ("with_isolates", "all")):
                for coeff, what in coeffs:
                    vals = [rows[a].get(conv, {}).get(coeff) for a in arms]
                    if all(v is None for v in vals):
                        continue
                    fig, ax = plt.subplots(figsize=(5.4, 3.5))
                    _bars(ax, arms, vals, [GREY if a == "held-out" else model_color(a) for a in arms], fmt="{:.4g}", ylabel=what)
                    ax.set_title(f"{d} — {coeff}, {conv.replace('_', ' ')} (no type information enters this)", loc="left", fontsize=9)
                    paths.append(_save(fig, out / d / f"clustering_{coeff}_{tag}.png"))
    return paths


def coexpr_support_figures(cs: dict, out: Path) -> list:
    paths = []
    with plt.rc_context(RC):
        for d, rec in cs.items():
            if "arms" not in rec:
                continue
            rows = {a: r for a, r in rec["arms"].items() if "auc" in r}
            if not rows:
                continue
            arms = list(rows)
            for key_, lbl, ref in (("auc", "AUC: drawn edges vs pairs left unconnected", 0.5),
                                   ("supported", "share of edges above the 90th pct of non-edges", 0.1)):
                fig, ax = plt.subplots(figsize=(5.4, 3.6))
                _bars(ax, arms, [rows[a][key_] for a in arms], [GREY if a == "held-out" else "#1baf7a" for a in arms],
                      fmt="{:.3f}", ref=ref, ref_label="chance", ylabel=lbl)
                ax.set_title(f"{d} — tissue {rec['tissue']}", loc="left", fontsize=9)
                paths.append(_save(fig, out / d / f"coexpr_support_{key_}.png"))
    return paths


def discovery_figures(disc: dict, out: Path) -> list:
    paths = []
    with plt.rc_context(RC):
        for d, rec in disc.items():
            rows = {a: r for a, r in rec.get("arms", {}).items() if r.get("n")}
            if not rows:
                continue
            arms = list(rows); x = np.arange(len(arms))
            fig, ax = plt.subplots(figsize=(5.4, 3.2))
            ax.bar(x - .18, [rows[a]["supported"] for a in arms], width=.36, color="#1baf7a", edgecolor="white", linewidth=2, label="supported")
            ax.bar(x + .18, [rows[a]["control"] for a in arms], width=.36, color=GREY, edgecolor="white", linewidth=2, label="random non-edges")
            for i, a in enumerate(arms):
                ax.text(i, max(rows[a]["supported"], rows[a]["control"]), f"{rows[a]['excess']:+.1%}", ha="center", va="bottom", fontsize=8, color=INK)
            ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=8, rotation=20, ha="right")
            ax.set_ylabel("share with any propagated support"); ax.legend(fontsize=7.5, frameon=False)
            ax.set_title(f"{d} — novel links with co-expression support vs matched control (label = excess)", loc="left", fontsize=9)
            ax.grid(alpha=.35, axis="y"); ax.set_axisbelow(True)
            paths.append(_save(fig, out / d / "discovery_support.png"))
            for key_, lbl, ref in (("excess", "supported − control", 0.0), ("fold", "supported / control", 1.0)):
                fig, ax = plt.subplots(figsize=(5.8, 3.6))
                _bars(ax, arms, [rows[a].get(key_) for a in arms], ["#2a78d6"] * len(arms),
                      fmt="{:+.1%}" if key_ == "excess" else "{:.2f}x", ref=ref, ref_label="chance", ylabel=lbl)
                paths.append(_save(fig, out / d / f"discovery_{key_}_by_model.png"))
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            _bars(ax, arms, [rows[a]["n"] for a in arms], ["#2a78d6"] * len(arms), fmt="{:,.0f}", ylabel="novel links found")
            paths.append(_save(fig, out / d / "discovery_counts.png"))
            fig, ax = plt.subplots(figsize=(6.4, 3.9)); nulls = []
            for a in arms:
                sc = _arr(rows[a]["scores"])
                if sc.size > 5:
                    xs = np.sort(sc); z = int((xs == 0).sum())
                    ax.step(xs, np.arange(1, xs.size + 1) / xs.size, where="post", lw=1.9, color=model_color(a),
                            label=f"{a} (n={xs.size:,})", marker=model_marker(a), markevery=(z, max((xs.size - z) // 8, 1)), ms=4.5)
                nl = _arr(rows[a]["control_scores"])
                if nl.size > 5:
                    nulls.append(nl)
            if nulls:
                nx_ = np.sort(np.concatenate(nulls))
                ax.step(nx_, np.arange(1, nx_.size + 1) / nx_.size, where="post", lw=2.2, color=SOFT, ls=":", label=f"random non-edges, pooled (n={nx_.size:,})")
            ax.set_xlabel("propagated co-expression support"); ax.set_ylabel("cumulative share of links")
            ax.legend(fontsize=7.5, frameon=False, loc="lower right"); ax.grid(alpha=.35); ax.set_axisbelow(True)
            paths.append(_save(fig, out / d / "discovery_scores_all_models.png"))
        if len([d for d in disc if disc[d].get("arms")]) > 1:
            arms_all = list(dict.fromkeys(a for d in disc.values() for a, r in d.get("arms", {}).items() if r.get("n")))
            for key_, lbl, ref, fname in (("excess", "supported − control", 0.0, "discovery_excess_by_organism.png"),
                                          ("fold", "supported / control", 1.0, "discovery_fold_by_organism.png")):
                fig, ax = plt.subplots(figsize=(6.2, 3.8))
                for d, rec in disc.items():
                    ys = [rec["arms"].get(a, {}).get(key_) for a in arms_all]
                    ys = [np.nan if v is None else v for v in ys]
                    if np.isfinite(ys).any():
                        ax.plot(np.arange(len(arms_all)), ys, marker="o" if d == "ecoli" else "s", ms=6, lw=2,
                                color=DATASET_COLOR.get(d, INK), label=f"{d} (tissue {rec.get('tissue', '?')})")
                ax.axhline(ref, lw=1.3, ls=":", color=SOFT); ax.set_xticks(np.arange(len(arms_all)))
                ax.set_xticklabels(arms_all, fontsize=8, rotation=20, ha="right"); ax.set_ylabel(lbl)
                ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.35); ax.set_axisbelow(True)
                paths.append(_save(fig, out / fname))
    return paths


def cluster_figures(cl: dict, out: Path) -> list:
    paths, by = [], {}
    for k, r in cl.items():
        if "skipped" in r:
            continue
        by.setdefault(r["dataset"], {})[arm_label(k)] = r
    with plt.rc_context(RC):
        for d, arms in by.items():
            names = list(arms)
            for a, r in arms.items():
                emb = r.get("embedding", {})
                if "xy" not in emb:
                    continue
                xy = np.asarray(emb["xy"]); pos = {int(n): i for i, n in enumerate(emb["ids"])}
                sel = [(pos[int(n)], l) for n, l in zip(r["ids"], r["labels"]) if int(n) in pos]
                if not sel:
                    continue
                pi = np.array([s[0] for s in sel]); pl = np.array([s[1] for s in sel]); sig = set(r["significant_clusters"])
                fig, ax = plt.subplots(figsize=(6.6, 6.2))
                ax.scatter(xy[:, 0], xy[:, 1], s=3, c="#e2e0d9", linewidths=0)
                pal = plt.cm.tab20(np.linspace(0, 1, 20))
                for ci in sorted(set(pl.tolist())):
                    m = pl == ci
                    ax.scatter(xy[pi[m], 0], xy[pi[m], 1], s=7, color=pal[ci % 20], linewidths=0,
                               alpha=.95 if ci in sig else .30, label=(f"cluster {ci}" if ci in sig else None))
                ax.set_xticks([]); ax.set_yticks([])
                if sig:
                    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left", markerscale=2)
                ax.set_title(f"{d} / {a}: {r['n_clusters']} k-means clusters of the full space on its UMAP (solid = enriched)", loc="left", fontsize=9)
                paths.append(_save(fig, out / d / f"embed_clusters_{_fname(a)}.png"))
            def _ctrl(r, which):
                # clusters with an enriched subsystem under the control; older runs stored only the
                # number of enriched (cluster, subsystem) pairs, which bounds it and equals it at 0 or 1
                v = r.get(f"n_clusters_sig_{which}")
                if v is None:
                    k = r[f"n_sig_{which}"]
                    v = k if k <= 1 else np.nan
                return v
            fig, ax = plt.subplots(figsize=(6.6, 3.8))
            x = np.arange(len(names)); w = 0.26
            for off, vals, col, lab, hatch in (
                    (-w, [arms[a]["n_clusters_sig"] for a in names], [model_color(a) for a in names], "learned embedding", None),
                    (0, [_ctrl(arms[a], "random") for a in names], ["#b9b6ad"] * len(names), "random embedding", None),
                    (w, [_ctrl(arms[a], "shuffled") for a in names], ["#7d7a72"] * len(names), "shuffled cluster labels", "//")):
                bars = ax.bar(x + off, [0 if not np.isfinite(v) else v for v in vals], w, color=col,
                              edgecolor="white", hatch=hatch, label=lab, zorder=2)
                for b_, v in zip(bars, vals):
                    ax.text(b_.get_x() + b_.get_width() / 2, (v if np.isfinite(v) else 0) + 0.3,
                            "{:.0f}".format(v) if np.isfinite(v) else "n/a", ha="center", va="bottom", fontsize=7.5, color=INK)
            ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
            ax.set_ylabel(f"clusters with an enriched subsystem (of {arms[names[0]]['n_clusters']})", fontsize=8)
            ax.set_ylim(0, arms[names[0]]["n_clusters"] * 1.15)
            ax.grid(alpha=.35, axis="y"); ax.set_axisbelow(True)
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(facecolor="white", edgecolor=INK, label="learned embedding (model colour)"),
                               Patch(facecolor="#b9b6ad", label="random embedding"),
                               Patch(facecolor="#7d7a72", hatch="//", edgecolor="white", label="shuffled cluster labels")],
                      frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
            paths.append(_save(fig, out / d / "embed_cluster_enrichment.png"))
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            _bars(ax, names, [arms[a].get("motif_median_fold") for a in names], [model_color(a) for a in names], fmt="{:.1f}x",
                  ref=1.0, ref_label="unconnected", ylabel="median internal edge density / random")
            paths.append(_save(fig, out / d / "embed_cluster_motif.png"))
    return paths


def attention_figures(att: dict, out: Path, main_fraction=0.25) -> list:
    paths, by = [], {}
    for k, r in att.items():
        if "probes" in r:
            by.setdefault(split_key(k)[0], {})[arm_label(k)] = r
    with plt.rc_context(RC):
        for d, arms in by.items():
            for a, r in arms.items():
                main = min(r["probes"], key=lambda p: abs(p["frac"] - main_fraction))
                for l, m in enumerate(main.get("map_head_mean", [])):
                    fig, ax = plt.subplots(figsize=(5.6, 4.8))
                    im = ax.imshow(np.asarray(m), cmap="magma", aspect="auto")
                    fig.colorbar(im, ax=ax, fraction=.046, pad=.02, label="attention (head mean)")
                    ax.set_xlabel("attended slot"); ax.set_ylabel("attending slot")
                    ax.set_title(f"{d} / {a} — layer {l}, t={main['t']} (across-head sd {main['map_head_sd'][l]:.4f})", loc="left", fontsize=9)
                    paths.append(_save(fig, out / d / f"attn_map_{_fname(a)}_L{l}.png"))
                if main.get("flow"):
                    names = r["type_names"]
                    for l, fl in enumerate(main["flow"]):
                        fl = np.asarray(fl); n = fl.shape[0]
                        fig, ax = plt.subplots(figsize=(4.6, 4.0))
                        im = ax.imshow(fl, cmap="Blues", vmin=0, vmax=1, aspect="auto")
                        for i in range(n):
                            for j in range(n):
                                ax.text(j, i, f"{fl[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if fl[i, j] > .55 else INK)
                        ax.set_xticks(range(n)); ax.set_xticklabels(names[:n], fontsize=8, rotation=20, ha="right")
                        ax.set_yticks(range(n)); ax.set_yticklabels(names[:n], fontsize=8)
                        fig.colorbar(im, ax=ax, fraction=.046, pad=.02, label="share of the row's attention")
                        ax.set_title(f"{d} / {a} — type to type, layer {l}, t={main['t']}", loc="left", fontsize=9)
                        paths.append(_save(fig, out / d / f"attn_types_{_fname(a)}_L{l}.png"))
            fig, ax = plt.subplots(figsize=(6.0, 3.6))
            for a, r in arms.items():
                main = min(r["probes"], key=lambda p: abs(p["frac"] - main_fraction))
                if main["shared"]:
                    ax.plot([x["layer"] for x in main["shared"]], [x["ratio"] for x in main["shared"]], "o-", lw=1.9, ms=5,
                            color=model_color(a), label=f"{a} (t={main['t']})")
            ax.axhline(1.0, ls="--", lw=1.3, color=SOFT); ax.set_xlabel("layer")
            ax.set_ylabel("attention on shared-reaction pairs\n/ attention on other pairs"); ax.legend(fontsize=8, frameon=False)
            ax.grid(alpha=.35); ax.set_axisbelow(True)
            ax.set_title(f"{d} — do nodes sharing a reaction attend to each other? (direct edges excluded)", loc="left", fontsize=9)
            paths.append(_save(fig, out / d / "attn_shared_reaction_by_layer.png"))
            for key_, fname, ylabel in (("shared", "attn_shared_reaction_by_t.png", "best layer's shared-reaction ratio"),
                                        ("align", "attn_edge_following_by_t.png", "attention on believed-present edges / on absent pairs")):
                fig, ax = plt.subplots(figsize=(6.0, 3.6)); drew = False
                for a, r in arms.items():
                    pts = []
                    for p in sorted(r["probes"], key=lambda p: -p["t"]):
                        if key_ == "shared" and p["shared"]:
                            pts.append((p["t"], max(x["ratio"] for x in p["shared"])))
                        elif key_ == "align" and p.get("align"):
                            vals = [x.get("ratio") for x in p["align"]["per_layer"] if x.get("ratio") is not None]
                            if vals:
                                pts.append((p["t"], max(vals)))
                    if pts:
                        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", lw=1.9, ms=5, color=model_color(a), label=a); drew = True
                if not drew:
                    plt.close(fig); continue
                ax.axhline(1.0, ls="--", lw=1.3, color=SOFT); ax.set_xlim(max(r["T"] for r in arms.values()), 0)
                ax.set_xlabel("t   (generation runs left to right)"); ax.set_ylabel(ylabel); ax.legend(fontsize=8, frameon=False)
                ax.grid(alpha=.35); ax.set_axisbelow(True)
                paths.append(_save(fig, out / d / fname))
    return paths


def ablation_figures(abl: dict, out: Path) -> list:
    paths = []
    with plt.rc_context(RC):
        for k, rec in abl.items():
            d, _, v = split_key(k); arm = arm_label(k)
            runs = rec["runs"]; labels = [l for l in runs if "(seed" not in l]
            ruler = next((l for l in runs if "(seed" in l), None)
            for bench, metric, read in rec["metrics"]:
                vals = [nanfloat(runs[l]["results"].get(bench, {}).get("summary", {}).get(metric)) for l in labels]
                if all(not np.isfinite(x) for x in vals):
                    continue
                ho = nanfloat(rec["held_out"].get(bench, {}).get("summary", {}).get(metric))
                fig, ax = plt.subplots(figsize=(6.4, 3.6))
                _bars(ax, labels, vals, [model_color(arm)] * len(labels), fmt="{:.4g}", ref=ho, ylabel=f"{metric} ({read})")
                if ruler:
                    rv = nanfloat(runs[ruler]["results"].get(bench, {}).get("summary", {}).get(metric))
                    if np.isfinite(rv):
                        ax.axhline(rv, ls=":", lw=1.2, color=NULL_COLOR)
                        ax.text(0.01, rv, " baseline, second seed", ha="left", va="bottom", fontsize=7.5, color=NULL_COLOR, transform=ax.get_yaxis_transform())
                ax.set_title(f"{d} / {arm} — same weights, different reveal rule: {bench}.{metric}", loc="left", fontsize=9)
                paths.append(_save(fig, out / d / f"ablation_{bench}_{metric}{_vsuffix(v)}.png"))
    return paths


def threshold_figures(th: dict, out: Path) -> list:
    paths = []
    with plt.rc_context(RC):
        for k, rec in th.items():
            d, _, v = split_key(k); arm = arm_label(k); rows = rec["rows"]; ts = [r["thresh"] for r in rows]
            fig, ax = plt.subplots(figsize=(6.4, 3.9))
            ax.plot(ts, [r["density"] for r in rows], lw=2, marker="o", ms=5, color=model_color(arm))
            ax.axhline(rec["data_density"], ls="--", lw=1.4, color=SOFT)
            ax.text(0.01, rec["data_density"], " data density", ha="left", va="bottom", fontsize=7.5, color=SOFT, transform=ax.get_yaxis_transform())
            for xt, lab, col in ((0.5, "default 0.5", "#c2402a"), (0.952, "calibrated 0.952", "#1baf7a")):
                ax.axvline(xt, ls=":", lw=1.4, color=col)
                ax.text(xt, ax.get_ylim()[1], f" {lab}", rotation=90, ha="right", va="top", fontsize=7.5, color=col)
            ax.set_yscale("log"); ax.set_xlabel("edge_thresh"); ax.set_ylabel("generated edge density"); ax.grid(alpha=.35)
            ax.set_title(f"{d} / {arm} — density against the decision threshold (adj_pos_weight {rec['adj_pos_weight']:g})", loc="left", fontsize=9)
            paths.append(_save(fig, out / d / f"threshold_density{_vsuffix(v)}.png"))
            for key_, lbl in (("edge_precision", "edge precision vs the GEM"), ("transitivity", "transitivity — bipartite violations (0 = legal)")):
                ys = [np.nan if r[key_] is None else r[key_] for r in rows]
                if not np.isfinite(ys).any():
                    continue
                fig, ax = plt.subplots(figsize=(6.4, 3.6))
                ax.plot(ts, ys, lw=2, marker="s", ms=5, color=model_color(arm))
                for xt, col in ((0.5, "#c2402a"), (0.952, "#1baf7a")):
                    ax.axvline(xt, ls=":", lw=1.4, color=col)
                ax.set_xlabel("edge_thresh"); ax.set_ylabel(lbl); ax.grid(alpha=.35)
                paths.append(_save(fig, out / d / f"threshold_{key_}{_vsuffix(v)}.png"))
    return paths


def map_figures(maps: dict, out: Path) -> list:
    """Tissue co-expression map (shared layout per dataset) and each model's
    own embedding space, one set per figure."""
    paths = []
    with plt.rc_context(RC):
        for d, rec in maps.items():
            t = rec.get("tissue") or {}
            if "xy" in t:
                xy = np.asarray(t["xy"]); pos = {int(g): i for i, g in enumerate(t["ids"])}
                for label, s in t["sets"].items():
                    fig, ax = plt.subplots(figsize=(5.6, 5.4))
                    ax.scatter(xy[:, 0], xy[:, 1], s=4, c=GREY, alpha=.45, linewidths=0, zorder=1)
                    pts = [pos[int(g)] for ss in s["sets"] for g in ss if int(g) in pos]
                    for a, b in s.get("coexpressed_pairs", []):       # co-expressed pairs within one set
                        if int(a) in pos and int(b) in pos:
                            ax.plot(*zip(xy[pos[int(a)]], xy[pos[int(b)]]), color=model_color(label),
                                    lw=1.0, alpha=.55, zorder=2)
                    if pts:
                        ax.scatter(xy[pts, 0], xy[pts, 1], s=34, c=model_color(label), zorder=3, edgecolors="white", linewidths=1.0)
                    hr = s.get("pairs_coexpressed")
                    ax.set_title(f"{d} — {label} on the {t['name']} co-expression network\n{len(pts)} genes"
                                 + (f" · {hr:.0%} of pairs co-expressed" if hr is not None else ""), loc="left", fontsize=9)
                    ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
                    paths.append(_save(fig, out / d / f"tissue_map_{label}.png"))
            for arm, ms in rec.get("model_space", {}).items():
                if "xy" not in ms:
                    continue
                xy = np.asarray(ms["xy"]); pos = {int(g): i for i, g in enumerate(ms["ids"])}; types = np.asarray(ms["types"])
                for label, s in ms["sets"].items():
                    fig, ax = plt.subplots(figsize=(5.6, 5.4))
                    ax.scatter(xy[:, 0], xy[:, 1], s=4, c=GREY, alpha=.40, linewidths=0, zorder=1)
                    n_hl = 0
                    for ss in s["sets"]:
                        for g in ss:
                            if int(g) in pos:
                                i = pos[int(g)]
                                ax.scatter(xy[i, 0], xy[i, 1], s=42, c=TYPE_COLORS.get(types[i], "#7f7f7f"), edgecolors="white", linewidths=1.0, zorder=3)
                                n_hl += 1
                    c = s.get("compactness")
                    ax.set_title(f"{d} / {arm} — {label} in the model's own embedding space\n{n_hl} entities"
                                 + (f" · compactness {c:.2f}" if c is not None else ""), loc="left", fontsize=9)
                    ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
                    paths.append(_save(fig, out / d / f"model_space_{_fname(arm)}_{label}.png"))
    return paths


def completion_figures(comp: dict, out: Path) -> list:
    paths = []
    with plt.rc_context(RC):
        for d, rec in comp.items():
            tab = pd.DataFrame({k: v["mean"] for k, v in rec["arms"].items()}).T
            base = tab.loc["freq", "node_recall"] if "freq" in tab.index else np.nan
            for metric, lbl, ref in (("node_recall", "share of hidden nodes recovered", base),
                                     ("edge_f1", "edge F1 among recovered nodes", np.nan),
                                     ("novel_frac", "share of completed nodes not in the true graph", np.nan)):
                if metric not in tab.columns:
                    continue
                sub = tab[metric].dropna()
                fig, ax = plt.subplots(figsize=(7.4, 3.8))
                _bars(ax, list(sub.index), sub.values.tolist(), [GREY if k == "freq" else model_color(k) for k in sub.index],
                      ref=ref, ref_label="frequency floor", ylabel=lbl)
                ax.set_title(f"{d} — completion from {rec['k_revealed']} revealed nodes: {metric}", loc="left", fontsize=9)
                paths.append(_save(fig, out / d / f"completion_{metric}.png"))
            sweep = {}
            for k in tab.index:
                parts = k.split()
                if len(parts) == 3 and parts[0] in ("masked-cond", "masked-nbr"):
                    sweep.setdefault((parts[0], parts[1]), []).append((float(parts[2][1:]), tab.loc[k, "node_recall"]))
            if sweep:
                fig, ax = plt.subplots(figsize=(6.2, 3.8))
                for i, ((arm, ch), pts) in enumerate(sorted(sweep.items())):
                    pts.sort()
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=2, ms=5, marker=["o", "s", "^", "D"][i % 4],
                            color=_color(ch, i), ls="-" if arm == "masked-cond" else "--", label=f"{arm} {ch}")
                for k, st, lb in (("masked-plain", "--", "masked-plain"), ("freq", ":", "frequency floor")):
                    if k in tab.index:
                        ax.axhline(tab.loc[k, "node_recall"], ls=st, lw=1.4, color=SOFT)
                        ax.text(0.99, tab.loc[k, "node_recall"], f"{lb} ", ha="right", va="bottom", fontsize=7.5, color=SOFT, transform=ax.get_yaxis_transform())
                ax.set_xlabel("guidance scale"); ax.set_ylabel("hidden nodes recovered"); ax.legend(fontsize=7.5, frameon=False)
                ax.grid(alpha=.35); ax.set_axisbelow(True)
                ax.set_title(f"{d} — does turning the conditioning up help? (one set of weights)", loc="left", fontsize=9)
                paths.append(_save(fig, out / d / "guidance_sweep.png"))
    return paths


def validation_figures(val: dict, out: Path) -> list:
    paths = []
    with plt.rc_context(RC):
        for d, rows in val.get("ablation", {}).items():
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            for on, col, mk, lbl in ((True, "#2a78d6", "o", "encoder on"), (False, "#6b6a66", "s", "encoder off")):
                pts = sorted((r["k"], r["node_recall"]) for r in rows.values() if r["encoder"] == on)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=2, ms=6, marker=mk, color=col, label=lbl)
            ax.set_xlabel("revealed nodes"); ax.set_ylabel("share of hidden nodes recovered"); ax.legend(fontsize=8, frameon=False)
            ax.grid(alpha=.35); ax.set_axisbelow(True)
            ax.set_title(f"{d} — the same trained weights with the seed-neighbourhood encoder on and off", loc="left", fontsize=9)
            paths.append(_save(fig, out / d / "nbr_ablation.png"))
    return paths


def signal_figures(sig: dict, out: Path) -> list:
    S = sig["sources"]; covs = np.asarray(sig["covs"]); paths = []
    panels = [("E. coli", ["E. coli correlation", "E. coli iModulon"], ["#2a78d6", "#eb6834"]),
              ("human", ["human co-expression", "human Perturb-seq"], ["#1baf7a", "#eda100"])]
    have = [(t, [k for k in ks if k in S], cs) for t, ks, cs in panels]; have = [h for h in have if h[1]]
    if not have:
        return paths
    with plt.rc_context(RC):
        fig, ax = plt.subplots(1, len(have), figsize=(5.4 * len(have), 3.6), sharey=True, squeeze=False)
        for j, (title, ks, cs) in enumerate(have):
            a_ = ax[0][j]
            for k, c in zip(ks, cs):
                r = S[k]
                if r["enrichment"] is None:
                    a_.plot([], [], color=c, label=f"{k} (recorded — no curve)"); continue
                a_.plot(r["percentile"], [np.nan if e is None else e for e in r["enrichment"]], marker="o", ms=3.5, lw=2, color=c,
                        ls="-" if r["computed"] else "--", label=f"{k}{'' if r['computed'] else ' (recorded)'}  AUC {r['auc']:.3f}")
            a_.axhline(1, ls=":", lw=1.2, color=SOFT); a_.set_xlabel("percentile of the score"); a_.set_title(title, loc="left", fontsize=10)
            a_.legend(fontsize=7.5, frameon=False); a_.grid(alpha=.35); a_.set_axisbelow(True)
        ax[0][0].set_ylabel("enrichment for shared subsystem")
        fig.suptitle("Enrichment across the score range — 1.0 is chance", x=0.006, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout(rect=[0, 0, 1, 0.9]); paths.append(_save(fig, out / "enrichment_by_percentile.png"))
        fig, ax = plt.subplots(1, len(have), figsize=(5.4 * len(have), 3.6), sharey=True, squeeze=False)
        for j, (title, ks, cs) in enumerate(have):
            a_ = ax[0][j]
            for k, c in zip(ks, cs):
                r = S[k]
                a_.plot(covs * 100, r["precision"], marker="o", ms=4, lw=2, color=c, ls="-" if r["computed"] else "--",
                        label=f"{k}{'' if r['computed'] else ' (recorded)'}")
            a_.axhline(1, ls=":", lw=1.2, color=SOFT); a_.set_xscale("log"); a_.set_xlabel("% of pairs retained")
            a_.set_title(title, loc="left", fontsize=10); a_.legend(fontsize=7.5, frameon=False); a_.grid(alpha=.35)
        ax[0][0].set_ylabel("enrichment for shared subsystem")
        fig.suptitle("Precision at matched coverage — the sparse tail", x=0.006, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout(rect=[0, 0, 1, 0.9]); paths.append(_save(fig, out / "precision_at_coverage.png"))
        names = [k for _, ks, _ in have for k in ks]; cols = [c for _, ks, cs in have for k, c in zip(ks, cs)]
        fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
        for a_, key_, lbl, ref in ((ax[0], "auc", "AUC  (0.5 = chance)", 0.5), (ax[1], "d", "Cohen's d  (0 = no separation)", 0.0)):
            vals = [nanfloat(S[k][key_]) for k in names]
            _bars(a_, [n.replace(" ", "\n", 1) for n in names], vals, cols, fmt="{:.3f}", ref=ref, ref_label="", ylabel=lbl)
        fig.suptitle("Per-pair signal strength, all sources", x=0.006, ha="left", fontsize=11, color=INK)
        fig.tight_layout(rect=[0, 0, 1, 0.9]); paths.append(_save(fig, out / "headline_auc_d.png"))
        have_raw = [(k, r) for k, r in S.items() if r.get("sample_score")]
        if have_raw:
            ncol = min(3, len(have_raw)); nrow = int(np.ceil(len(have_raw) / ncol))
            fig, ax = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.3 * nrow), squeeze=False)
            for j, (k, r) in enumerate(have_raw):
                a_ = ax[j // ncol][j % ncol]; s = np.asarray(r["sample_score"]); lab = np.asarray(r["sample_label"], bool)
                lo, hi = np.percentile(s, [0.5, 99.5]); bins = np.linspace(lo, hi if hi > lo else lo + 1e-9, 70)
                a_.hist(s[~lab], bins=bins, density=True, color=GREY, label="different subsystem")
                a_.hist(s[lab], bins=bins, density=True, color="#2a78d6", alpha=.65, label="same subsystem")
                a_.axvline(s[~lab].mean(), lw=1.3, color=SOFT, ls="--"); a_.axvline(s[lab].mean(), lw=1.3, color="#2a78d6", ls="--")
                a_.set_title(f"{k}\nd = {nanfloat(r['d']):+.3f}   AUC = {r['auc']:.3f}", loc="left", fontsize=8.5)
                a_.set_xlabel("score"); a_.grid(alpha=.3)
                if j == 0:
                    a_.legend(fontsize=7, frameon=False)
            for j in range(len(have_raw), nrow * ncol):
                ax[j // ncol][j % ncol].set_visible(False)
            fig.suptitle("Raw score distributions — same vs different subsystem", x=0.006, ha="left", fontsize=10.5, color=INK)
            fig.tight_layout(rect=[0, 0, 1, 0.9]); paths.append(_save(fig, out / "score_distributions.png"))
    return paths


def coexpr_information_figures(ci: dict, out: Path) -> list:
    paths = []; A_, S_, C_ = ci["activity"], ci["similarity"], ci["ceiling"]; tissues = ci["tissues"]; T_ = len(tissues)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.4))
        x, y = np.asarray(A_["scatter"]["x"]), np.asarray(A_["scatter"]["y"])
        ax[0].scatter(x, y, s=4, alpha=0.10, color="#2a78d6", edgecolors="none")
        qs = np.quantile(x, np.linspace(0, 1, 13)); mid = 0.5 * (qs[1:] + qs[:-1])
        means = [y[(x >= lo) & (x < hi)].mean() if ((x >= lo) & (x < hi)).any() else np.nan for lo, hi in zip(qs[:-1], qs[1:])]
        ax[0].plot(mid, means, color="#eb6834", lw=2, marker="o", ms=4, label="binned mean"); ax[0].axhline(0, color=SOFT, lw=0.8, ls="--")
        ax[0].set_xlabel("relative co-expression (z within pathway)"); ax[0].set_ylabel("relative activity (z within pathway)")
        ax[0].set_title("a  The relationship itself", loc="left"); ax[0].legend(fontsize=8, frameon=False)
        ax[1].hist(A_["null"], bins=30, color=SOFT, alpha=0.65, label=f"permutation null (n={len(A_['null'])})")
        ax[1].axvline(A_["mi"], color="#eb6834", lw=2.2, label=f"observed {A_['mi']:.3f}")
        ax[1].set_xlabel("mutual information (nats)"); ax[1].set_title("b  Observed MI vs null", loc="left"); ax[1].legend(fontsize=8, frameon=False)
        o = np.argsort(A_["per_tissue_mi"])[::-1]
        ax[2].bar(range(T_), np.asarray(A_["per_tissue_mi"])[o], color="#2a78d6", zorder=2)
        ax[2].plot(range(T_), np.asarray(A_["per_tissue_null"])[o], color="#eb6834", lw=1.6, marker="o", ms=3, label="null", zorder=3)
        ax[2].set_xticks(range(T_)); ax[2].set_xticklabels([tissues[i] for i in o], rotation=45, ha="right", fontsize=7)
        ax[2].set_ylabel("MI (nats)"); ax[2].set_title("c  Per tissue", loc="left"); ax[2].legend(fontsize=8, frameon=False)
        for a in ax:
            a.grid(alpha=0.3, zorder=0); a.set_axisbelow(True)
        fig.suptitle("Does a tissue's co-expression predict how active a pathway is there?", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.9]); paths.append(_save(fig, out / "mi_coexpr_activity.png"))
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        TOPK, WR = np.asarray(S_["topk_jaccard"]), np.asarray(S_["weight_r"]); off = ~np.eye(T_, dtype=bool)
        d = 1 - TOPK; np.fill_diagonal(d, 0.0)
        o = dendrogram(linkage(squareform(d, checks=False), method="average"), no_plot=True)["leaves"]
        fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0), gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})
        for a, M, ttl in ((ax[0], TOPK, "a  Top-k pair overlap (Jaccard)"), (ax[1], WR, "b  Weight correlation on shared pairs")):
            Mo = M[np.ix_(o, o)]; im = a.imshow(Mo, cmap="viridis", vmin=np.nanmin(M[off]), vmax=1.0)
            a.set_xticks(range(T_)); a.set_yticks(range(T_))
            a.set_xticklabels([tissues[i] for i in o], rotation=45, ha="right", fontsize=7); a.set_yticklabels([tissues[i] for i in o], fontsize=7)
            a.set_title(ttl, loc="left", fontsize=10); fig.colorbar(im, ax=a, fraction=0.046)
        ax[2].bar(range(1, T_ + 1), np.asarray(S_["pairs_in_k_tissues"]) * 100, color="#2a78d6")
        ax[2].set_xlabel("number of tissues a pair appears in"); ax[2].set_ylabel("% of all pairs"); ax[2].set_title("c  Shared vs tissue-specific", loc="left", fontsize=10)
        fig.suptitle(f"Similarity between the {T_} tissue co-expression networks", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.91]); paths.append(_save(fig, out / "network_similarity.png"))
    return paths


# ── the driver ────────────────────────────────────────────────────────────

def draw_all(run_name: str, only: Optional[list] = None, bundles: Optional[dict] = None) -> list:
    """Every figure whose JSON exists under outputs/analysis/<run>/."""
    ana = ANALYSIS / run_name; out = ana / "figures"; run = RUNS / run_name
    paths = []
    def want(name):
        return only is None or name in only
    def have(fname):
        return (ana / fname).exists()
    if want("loss") and (run / "history.json").exists():
        paths += loss_curves(load_json(run / "history.json"), out)
    if want("samples") and (run / "samples.json").exists():
        paths += sample_diagnostics(load_json(run / "samples.json"), out)
    scores = load_json(ana / "scores.json") if have("scores.json") else None
    if scores and want("scores"):
        paths += score_bars(scores, out) + score_distributions(scores, out) + rank_figures(scores, out)
    if have("trajectories.json") and want("trajectories"):
        traj = load_json(ana / "trajectories.json")
        dens = {d: ps["held-out"]["mean"]["density"] for d, ps in load_json(ana / "pathway_stats.json").items()} \
            if have("pathway_stats.json") else {}
        if not dens and scores:
            dens = {d: nanfloat(scores["held-out"][d].get("topo", {}).get("summary", {}).get("density")) for d in scores["held-out"]}
        paths += trajectory_figures(traj, out, dens)
        if bundles:
            paths += trajectory_graphs(traj, out, bundles)
    for fname, fn in (("remask.json", remask_figures), ("pathway_stats.json", pathway_stat_figures),
                      ("clustering.json", clustering_figures), ("coexpr_support.json", coexpr_support_figures),
                      ("discovery.json", discovery_figures), ("clusters.json", cluster_figures),
                      ("attention.json", attention_figures), ("ablation.json", ablation_figures),
                      ("threshold.json", threshold_figures), ("maps.json", map_figures),
                      ("completion.json", completion_figures), ("validation.json", validation_figures),
                      ("signal_comparison.json", signal_figures), ("coexpr_information.json", coexpr_information_figures)):
        if have(fname) and want(fname.split(".")[0]):
            paths += fn(load_json(ana / fname), out)
    return paths
