"""Novel-link support and the best evidence routes, one figure per model and organism.

    sbatch run_debug_vae.sh scripts/discovery_routes.py --run compare_oct2pad_20261112_1238

Redraws the two-panel "co-exp_discovery" layout from outputs/analysis/<run>/discovery.json;
nothing is re-sampled or re-scored.

    left    share of novel generated links with propagated co-expression support against the
            matched random non-edge control (counts inside the bars)
    right   the n best-supported novel links, each as a linear route:
            entity u -> anchor gene a == co-expression == anchor gene b <- entity v,
            hop counts on the outer legs, the generated link u--v dashed above

Writes figures/<dataset>/discovery_routes_<arm>.png and, at the figures root,
discovery_routes_all.png (every model and organism stacked).
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graphpath.pipeline.common import RunLayout, load_bundle, load_json

TYPE_COLOR = {"gene": "#b48c1f", "metabolite": "#0effdb", "enzyme": "#2c39a0", "reaction": "#d627b9"}
ARMS = ("masked", "transition", "continuous", "vae")
NAMES = {"masked": "absorbing mask", "transition": "marginal transition", "continuous": "continuous", "vae": "VAE"}
ORG = {"ecoli": "E. coli", "human": "human"}


def short(s, n=24):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def draw(ax_bar, ax_routes, b, rec, title, n_top=8):
    n = rec["n"]
    sup, ctl = rec["supported"], rec["control"]
    k_sup, k_ctl = int(round(sup * n)), int(round(ctl * len(rec.get("control_scores", [])) or ctl * n))
    bars = ax_bar.bar([0, 1], [sup, ctl], color=["#2a78d6", "#8a8780"], width=0.62)
    top_y = max(sup, ctl) * 1.25 + 0.01
    for x, v, k, tot in ((0, sup, k_sup, n), (1, ctl, k_ctl, len(rec.get("control_scores", [])) or n)):
        ax_bar.text(x, v + 0.015 * top_y, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
        ax_bar.text(x, 0.02 * top_y, f"{k:,} of {tot:,}", ha="center", va="bottom", fontsize=7, color="white")
    ax_bar.set_xticks([0, 1], ["novel\ngenerated", "random\nnon-edges"], fontsize=8)
    ax_bar.set_ylabel("share with support", fontsize=8)
    ax_bar.set_ylim(0, max(sup, ctl) * 1.25 + 0.01)
    fold = rec.get("fold")
    ax_bar.set_title(f"{title}\nfold {fold:.2f}" if fold else title, fontsize=9)
    ax_bar.spines[["top", "right"]].set_visible(False)

    def name(i):
        nid = b.id_to_node.get(int(i), str(i))
        comp = f" [{nid[-1]}]" if str(nid).startswith("MAM") else ""     # Human-GEM metabolites: show compartment
        return short(b.common(nid), 22 - len(comp)) + comp
    color = lambda i: TYPE_COLOR.get(b.ntype(b.id_to_node.get(int(i), "")), "0.5")
    seen, top = set(), []
    for d in rec.get("top", []):              # the same link can be found in several generated graphs
        key = (min(d["u"], d["v"]), max(d["u"], d["v"]))
        if key not in seen:
            seen.add(key); top.append(d)
    top = top[:n_top]
    ax_routes.set_xlim(-0.3, 3.3); ax_routes.set_ylim(-0.6, max(len(top), 1) - 0.2)
    ax_routes.axis("off")
    for r, d in enumerate(top):
        y = len(top) - 1 - r
        xs = [0, 1, 2, 3]
        ids = [d["u"], d["gene_a"], d["gene_b"], d["v"]]
        ax_routes.plot([0, 1], [y, y], color="0.55", lw=1, zorder=1)
        ax_routes.plot([2, 3], [y, y], color="0.55", lw=1, zorder=1)
        ax_routes.plot([1, 2], [y, y], color="#1aa36f", lw=2.6, zorder=1)
        ax_routes.plot([0, 3], [y + 0.32, y + 0.32], color="#2a78d6", lw=1.1, ls="--", zorder=1)
        ax_routes.text(1.5, y + 0.36, "generated link", ha="center", va="bottom", fontsize=6, color="#2a78d6")
        ax_routes.text(1.5, y + 0.05, f"support {d['support']:.2f}", ha="center", va="bottom", fontsize=6.5, color="#1aa36f")
        hop = lambda h: f"{h} hop" + ("" if h == 1 else "s")
        ax_routes.text(0.5, y - 0.05, hop(d["hops_a"]), ha="center", va="top", fontsize=6, color="0.45")
        ax_routes.text(2.5, y - 0.05, hop(d["hops_b"]), ha="center", va="top", fontsize=6, color="0.45")
        for x, i in zip(xs, ids):
            ax_routes.scatter([x], [y], s=55, color=color(i), edgecolor="0.25", lw=0.6, zorder=3)
            ax_routes.text(x, y - 0.12, name(i), ha="center", va="top", fontsize=6.3)
    if not top:
        ax_routes.text(1.5, 0, "no supported novel link", ha="center", fontsize=8)


def legend(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=c, markeredgecolor="0.25", label=t) for t, c in TYPE_COLOR.items()]
    h += [Line2D([], [], color="#1aa36f", lw=2.6, label="co-expression of anchor genes"),
          Line2D([], [], color="#2a78d6", lw=1.1, ls="--", label="generated novel link")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=7.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="compare_oct2pad_20261112_1238")
    a = ap.parse_args()
    run = RunLayout(a.run)
    fig_dir = run.path.parents[1] / "analysis" / run.path.name / "figures"
    disc = load_json(fig_dir.parent / "discovery.json")
    panels = []
    for d in ("ecoli", "human"):
        if d not in disc or "arms" not in disc[d]:
            continue
        b = load_bundle(d)
        for arm in ARMS:
            rec = next((r for k, r in disc[d]["arms"].items() if k.split(" (")[0] == arm), None)
            if rec is None or "n" not in rec or rec.get("n", 0) == 0:
                continue
            title = f"{ORG[d]}, {NAMES[arm]} ({disc[d]['tissue']})"
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1, 2.6]))
            draw(ax0, ax1, b, rec, title)
            legend(fig); fig.tight_layout(rect=(0, 0.06, 1, 1))
            p = fig_dir / d / f"discovery_routes_{arm}.png"
            fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); print(f"-> {p}", flush=True)
            panels.append((b, rec, title))
    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 4.0 * len(panels)),
                             gridspec_kw=dict(width_ratios=[1, 2.6]))
    for (b, rec, title), (ax0, ax1) in zip(panels, np.atleast_2d(axes)):
        draw(ax0, ax1, b, rec, title)
    legend(fig); fig.tight_layout(rect=(0, 0.015, 1, 1))
    p = fig_dir / "discovery_routes_all.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"-> {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
