"""Shared figure style: palette, ink colours, rc context and the bar helper.

Colour identity follows the CONDITION, never the metric, and the order is
fixed — so `generated` is the same hue in every panel of every figure and a
reader can carry the mapping across pages. Hues are the validated
categorical defaults, taken in slot order.

Every panel shows the per-graph distribution as points behind the summary
bar: benchmark means over ~32 graphs are noisy, and a bar alone hides that.
"""

from typing import Optional, Sequence

import numpy as np

# categorical slots 1-4, light surface
PALETTE = {
    "generated": "#2a78d6",   # blue
    "shuffled":  "#eb6834",   # orange
    "random":    "#1baf7a",   # aqua
    "training":  "#eda100",   # yellow
}
FALLBACK = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]

INK       = "#0b0b0b"
INK_SOFT  = "#52514e"
INK_MUTED = "#8a8880"
GRID      = "#e4e3de"
SURFACE   = "#ffffff"


def _rc():
    return {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": INK_SOFT, "axes.titlecolor": INK,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "font.size": 9, "legend.fontsize": 8.5, "legend.frameon": False,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "savefig.facecolor": SURFACE, "savefig.bbox": "tight",
    }


def _color(name, i=0):
    return PALETTE.get(name, FALLBACK[i % len(FALLBACK)])


def _as_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _bars(ax, names, means, points=None, ylabel="", title="", fmt="{:.3f}",
          ymax=None):
    """Thin bars + the per-graph cloud behind them + one direct label each."""
    x = np.arange(len(names))
    tops = []
    for i, n in enumerate(names):
        ax.bar(x[i], means[i], width=0.62, color=_color(n, i),
               edgecolor=SURFACE, linewidth=2, zorder=2)
        top = means[i]
        if points is not None and points[i] is not None and len(points[i]):
            p = _as_np(points[i])
            jitter = (np.random.default_rng(i).uniform(-0.13, 0.13, len(p)))
            ax.scatter(x[i] + jitter, p, s=9, color=_color(n, i), alpha=0.35,
                       linewidths=0.5, edgecolors=SURFACE, zorder=3)
            top = max(top, float(np.nanmax(p)))
        tops.append(top)

    # labels sit clear of the point cloud, horizontal — a rotated number
    # printed through the data is harder to read than no number at all
    head = max(tops) if tops else 1.0
    for i, n in enumerate(names):
        ax.text(x[i], tops[i] + head * 0.035, fmt.format(means[i]),
                ha="center", va="bottom", fontsize=8, color=INK_SOFT, zorder=4)
    ax.set_ylim(0, head * 1.20 if ymax is None else ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=8)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)



# ── node and edge styling for the network figures ─────────────────────────
# the project's node styling, matching graphpath/plotting/graphs.py so a node looks
# the same here as in the sample plots
TYPE_COLORS = {
    "gene":       "#b48c1f",
    "metabolite": "#0effdb",
    "enzyme":     "#2c39a0",
    "reaction":   "#d627b9",
    "mask":       "#5a5a55",
}
TYPE_SHAPES = {"gene": "o", "metabolite": "s", "enzyme": "^",
               "reaction": "D", "mask": "X"}
TYPE_HATCHES = {"gene": "", "metabolite": "//", "enzyme": "xx",
                "reaction": "..", "mask": "++"}

# edge roles in the context figure — each answers a different question
EDGE_STYLE = {
    "confirmed": dict(color="#4a4a46", width=1.9, style="solid"),
    "novel":     dict(color="#eb6834", width=1.7, style=(0, (5, 3))),
    "focus":     dict(color="#d1341a", width=3.2, style=(0, (6, 2.5))),
    "missed":    dict(color="#c2c1ba", width=1.3, style=(0, (1, 2.5))),
    "route":     dict(color="#4a3aa7", width=1.6, style=(0, (6, 2, 1, 2))),
    "coexpr":    dict(color="#0f8f5f", width=3.0, style="solid"),
}
