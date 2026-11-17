"""Bar-and-distribution figures for benchmark results: one panel per
question, points behind every bar, colour by condition."""

from typing import Optional, Sequence

import numpy as np

from graphpath.plotting.style import (EDGE_STYLE, FALLBACK, GRID, INK, INK_MUTED,
                                      INK_SOFT, PALETTE, SURFACE, TYPE_COLORS,
                                      TYPE_HATCHES, TYPE_SHAPES, _as_np, _bars,
                                      _color, _rc)

from graphpath.benchmarks.topology import POOLED, topology_distances


def coexpression_figure(results: dict, save: Optional[str] = None,
                        tissues: Optional[Sequence[str]] = None):
    """
    Three panels: the two independent factors of the co-expression score,
    then the score itself.

        coexpr_mean = hit rate x mean value when present

    Splitting them matters — a set can score by touching a few pairs very
    strongly or many pairs weakly, and only the first panel distinguishes
    "these genes are related at all" from "related how strongly".
    """
    import matplotlib.pyplot as plt

    names = [n for n in results if results[n].per_graph.get("hit_rate_best") is not None]
    if not names:
        raise ValueError("no result carries per-graph co-expression values")

    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))

        _bars(axes[0], names,
              [results[n].summary["coexpr_hit_rate_best"] for n in names],
              [results[n].per_graph["hit_rate_best"] for n in names],
              ylabel="fraction of gene pairs", title="a  Hit rate",
              fmt="{:.1%}")
        _bars(axes[1], names,
              [results[n].summary["coexpr_value_when_present"] for n in names],
              [results[n].per_graph["value_when_present"] for n in names],
              ylabel="co-expression", title="b  Strength where present")
        _bars(axes[2], names,
              [results[n].summary["coexpr_best_tissue_mean"] for n in names],
              [results[n].per_graph["coexpr_best"] for n in names],
              ylabel="co-expression", title="c  Score  (a $\\times$ b)")

        fig.suptitle("Gene co-expression within a generated pathway, best-matching tissue",
                     x=0.02, ha="left", fontsize=11, color=INK)
        fig.text(0.02, -0.04,
                 "Points: individual graphs. Bars: mean. "
                 "The score is the product of the two factors, so panels a and b "
                 "say how it was earned.",
                 fontsize=8, color=INK_MUTED, ha="left")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        if save:
            fig.savefig(save, dpi=300)
        return fig


def known_structure_figure(results: dict, save: Optional[str] = None):
    """
    Panel a groups by metric and colours by condition, so precision, recall
    and F1 are read against the same four references without a second axis.
    Panel b is the duplication defect, which the other metrics inherit.
    """
    import matplotlib.pyplot as plt

    names = list(results)
    metrics = [("edge_precision", "precision"), ("edge_recall", "recall"),
               ("edge_f1", "F1")]

    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2),
                                 gridspec_kw={"width_ratios": [1.75, 1]})

        ax = axes[0]
        gx = np.arange(len(metrics))
        w = 0.78 / len(names)
        for i, n in enumerate(names):
            vals = [results[n].summary.get(k, np.nan) for k, _ in metrics]
            off = (i - (len(names) - 1) / 2) * w
            ax.bar(gx + off, vals, width=w * 0.86, color=_color(n, i),
                   edgecolor=SURFACE, linewidth=2, label=n, zorder=2)
            # the control bars round to nothing at this scale; without a
            # number the reader cannot tell 0.011 from 0.0007
            for gi, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(gx[gi] + off, v + 0.02, f"{v:.2f}", ha="center",
                            va="bottom", fontsize=7, color=INK_SOFT, zorder=4)
        ax.set_xticks(gx)
        ax.set_xticklabels([lbl for _, lbl in metrics])
        ax.set_ylabel("fraction of edges")
        ax.set_ylim(0, 1.05)
        ax.set_title("a  Generated edges that exist in the curated network",
                     loc="left", pad=8)
        ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
        ax.legend(ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, -0.16))

        _bars(axes[1], names,
              [results[n].summary.get("unique_node_frac", np.nan) for n in names],
              [results[n].per_graph.get("unique_node_frac") for n in names],
              ylabel="unique nodes / nodes",
              title="b  Entity duplication", fmt="{:.2f}", ymax=1.15)
        axes[1].axhline(1.0, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        # left-aligned: the right end of the line is where the training bar
        # and its value label already are
        axes[1].text(-0.42, 1.012, "no repeats", fontsize=7.5,
                     color=INK_MUTED, ha="left", va="bottom")

        fig.suptitle("Structural fidelity against the curated reference network",
                     x=0.02, ha="left", fontsize=11, color=INK)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        if save:
            fig.savefig(save, dpi=300)
        return fig


def tissue_heatmap(results: dict, tissues: Sequence[str],
                   save: Optional[str] = None):
    """
    Co-expression by tissue. Magnitude, so a single-hue sequential ramp —
    a coherent set of pathways should light up a few tissues, not all.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    names = [n for n in results if "per_tissue_coexpr" in results[n].per_graph]
    mat = np.stack([_as_np(results[n].per_graph["per_tissue_coexpr"]).mean(0)
                    for n in names])

    ramp = LinearSegmentedColormap.from_list("seq", ["#f4f7fd", "#2a78d6", "#123a6b"])
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(1.1 + 0.42 * len(tissues), 0.55 * len(names) + 1.5))
        im = ax.imshow(mat, cmap=ramp, aspect="auto", vmin=0)
        ax.set_xticks(range(len(tissues)))
        ax.set_xticklabels(tissues, rotation=45, ha="right")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        # 2px surface gap between cells, so adjacent steps stay separable
        ax.set_xticks(np.arange(-0.5, len(tissues), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.outline.set_visible(False)
        cb.set_label("mean co-expression", fontsize=8.5, color=INK_SOFT)
        ax.set_title("Co-expression by tissue", loc="left", pad=8)
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=300)
        return fig


def memorisation_figure(results: dict, duplicate_threshold: float = 0.9,
                        save: Optional[str] = None):
    """
    Nearest-training-example similarity. Panel a is the distribution rather
    than a bar: the mean is the wrong summary here, because memorisation
    shows up as a TAIL — a handful of near-copies among otherwise novel
    graphs moves the mean barely at all.
    """
    import matplotlib.pyplot as plt

    names = [n for n in results if "node_jaccard" in results[n].per_graph]
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

        ax = axes[0]
        bins = np.linspace(0, 1, 21)
        for i, n in enumerate(names):
            v = _as_np(results[n].per_graph["node_jaccard"])
            ax.hist(v, bins=bins, histtype="step", linewidth=2,
                    color=_color(n, i), label=n, zorder=2)
        ax.axvline(duplicate_threshold, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)))
        # along the line at mid-height: the top of this panel is where the
        # legend sits
        ax.text(duplicate_threshold - 0.015, ax.get_ylim()[1] * 0.45,
                "near-duplicate", fontsize=7.5, color=INK_MUTED,
                ha="right", va="center", rotation=90)
        ax.set_xlabel("node Jaccard with nearest training pathway")
        ax.set_ylabel("graphs")
        ax.set_title("a  Similarity to the closest training example",
                     loc="left", pad=8)
        ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
        ax.legend(loc="upper right")

        _bars(axes[1], names,
              [results[n].summary.get("nn_edge_jaccard_mean", np.nan) for n in names],
              [results[n].per_graph.get("edge_jaccard") for n in names],
              ylabel="edge Jaccard",
              title="b  Edge agreement with that match", fmt="{:.2f}")

        fig.suptitle("Memorisation: distance from the training set",
                     x=0.02, ha="left", fontsize=11, color=INK)
        fig.text(0.02, -0.04,
                 "Identical nodes wired differently is recombination; identical "
                 "nodes wired identically is a copy — panel b separates them.",
                 fontsize=8, color=INK_MUTED, ha="left")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        if save:
            fig.savefig(save, dpi=300)
        return fig


def conditioning_figure(arms: dict, heldout: Optional[str] = None,
                        save: Optional[str] = None):
    """
    Does the conditioning reach the output?

    Panel a compares tissue-recovery accuracy across arms against chance —
    the conditioned arm has to beat the shuffled-condition and unconditional
    arms, not merely beat chance, since some tissues are easier to hit by
    accident than others.

    Panel b is the confusion matrix for the conditioned arm. Accuracy alone
    hides the usual failure: recovering two dominant tissues and guessing on
    the rest looks respectable in aggregate.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    names = list(arms)
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4),
                                 gridspec_kw={"width_ratios": [1, 1.45]})

        ax = axes[0]
        acc = [arms[n]["accuracy"] for n in names]
        for i, (n, a) in enumerate(zip(names, acc)):
            ax.bar(i, a, width=0.62, color=_color(n, i), edgecolor=SURFACE,
                   linewidth=2, zorder=2)
            ax.text(i, a + 0.012, f"{a:.0%}", ha="center", va="bottom",
                    fontsize=8.5, color=INK_SOFT)
        chance = arms[names[0]]["chance"]
        ax.axhline(chance, ls=(0, (4, 3)), lw=1, color=INK_MUTED, zorder=3)
        ax.text(-0.45, chance + 0.008, f"chance ({chance:.0%})", fontsize=7.5,
                color=INK_MUTED, ha="left", va="bottom")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in names])
        ax.set_ylabel("conditioning tissue recovered")
        ax.set_ylim(0, max(max(acc), chance) * 1.35 + 0.02)
        ax.set_title("a  Is the condition readable from the output?",
                     loc="left", pad=8, fontsize=9.5)
        ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)

        first = arms[names[0]]
        conf = first["confusion"].astype(float)
        row = conf.sum(1, keepdims=True)
        conf = conf / np.clip(row, 1, None)
        tis = first["tissues"]
        ramp = LinearSegmentedColormap.from_list("seq", ["#f4f7fd", "#2a78d6", "#123a6b"])
        ax = axes[1]
        im = ax.imshow(conf, cmap=ramp, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(tis))); ax.set_xticklabels(tis, rotation=45, ha="right")
        ax.set_yticks(range(len(tis))); ax.set_yticklabels(tis)
        ax.set_xlabel("recovered"); ax.set_ylabel("conditioned on")
        ax.set_xticks(np.arange(-0.5, len(tis), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(tis), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0); ax.tick_params(length=0)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        if heldout is not None and heldout in tis:
            h = tis.index(heldout)
            ax.add_patch(plt.Rectangle((-0.5, h - 0.5), len(tis), 1, fill=False,
                                       edgecolor="#d1341a", lw=2.0, zorder=5))
            ax.text(len(tis) - 0.4, h, " held out", color="#d1341a", fontsize=8,
                    va="center", ha="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.outline.set_visible(False)
        cb.set_label("share of graphs", fontsize=8.5, color=INK_SOFT)
        ax.set_title(f"b  {names[0]}: which tissue was recovered",
                     loc="left", pad=8, fontsize=9.5)

        fig.suptitle("Conditioning on a tissue co-expression network",
                     x=0.02, ha="left", fontsize=11, color=INK)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        if save:
            fig.savefig(save, dpi=300)
        return fig


def guidance_sweep_figure(sweep: dict, save: Optional[str] = None):
    """
    Guidance dose-response. Two panels rather than two y-axes: tissue
    recovery should rise with guidance while structural fidelity holds. A
    flat recovery curve means the condition is inert; recovery rising as
    precision collapses means guidance is buying the metric with biology.
    """
    import matplotlib.pyplot as plt

    scales = sorted(sweep)
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), sharex=True)
        for ax, key, label, col in (
                (axes[0], "accuracy", "conditioning tissue recovered", PALETTE["generated"]),
                (axes[1], "edge_precision", "edge precision", PALETTE["training"])):
            y = [sweep[s].get(key, np.nan) for s in scales]
            ax.plot(scales, y, marker="o", ms=6, lw=2, color=col,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax.set_xlabel("guidance scale")
            ax.set_ylabel(label)
            ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
        if "chance" in sweep[scales[0]]:
            axes[0].axhline(sweep[scales[0]]["chance"], ls=(0, (4, 3)), lw=1,
                            color=INK_MUTED)
        axes[0].set_title("a  Specificity", loc="left", pad=8, fontsize=9.5)
        axes[1].set_title("b  Structural fidelity", loc="left", pad=8, fontsize=9.5)
        fig.suptitle("Classifier-free guidance dose-response",
                     x=0.02, ha="left", fontsize=11, color=INK)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        if save:
            fig.savefig(save, dpi=300)
        return fig


def topology_figure(results: dict, reference: str = "training",
                    save: Optional[str] = None):
    """
    Structure, not identity. Panels a-c overlay the pooled distributions the
    means in the summary table flatten away — two conditions can share a mean
    degree and have entirely different shapes. Panel d collapses each
    distribution to its distance from the reference so the panels above can
    be ranked rather than squinted at.

    Distributions are drawn as step CCDFs on a log y-axis: pathway degree
    distributions are heavy-tailed, and a linear histogram puts every bin of
    interest in the last 5% of the axis.
    """
    import matplotlib.pyplot as plt

    names = list(results)
    panels = [("degree", "degree", "a"),
              ("clustering_node", "clustering coefficient", "b"),
              ("path_length", "shortest path length", "c")]

    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.1))

        for ax, (key, label, tag) in zip(axes, panels):
            for i, n in enumerate(names):
                v = np.asarray(results[n].distributions.get(key, []), dtype=float)
                if v.size == 0:
                    continue
                x = np.sort(v)
                ccdf = 1.0 - np.arange(x.size) / x.size
                ax.step(x, ccdf, where="post", color=_color(n, i), lw=1.6,
                        label=n, zorder=3 if n == reference else 2,
                        alpha=1.0 if n == reference else 0.9)
            ax.set_yscale("log")
            ax.set_xlabel(label)
            ax.set_ylabel("P(X > x)" if tag == "a" else "")
            ax.set_title(f"{tag}  {label.capitalize()}", loc="left", pad=8)
            ax.grid(zorder=0); ax.set_axisbelow(True)

        # d: distance to the reference, so lower is closer and the bars are
        # directly comparable across metrics
        ax = axes[3]
        others = [n for n in names if n != reference]
        metrics = [(k, k.replace("_node", "").replace("_", " ")) for k in POOLED]
        gx = np.arange(len(metrics))
        w = 0.78 / max(len(others), 1)
        if reference in results:
            for i, n in enumerate(others):
                d = topology_distances(results[n], results[reference])
                vals = [d.get(f"{k}_w1", np.nan) for k, _ in metrics]
                off = (i - (len(others) - 1) / 2) * w
                ax.bar(gx + off, vals, width=w * 0.86,
                       color=_color(n, names.index(n)), edgecolor=SURFACE,
                       linewidth=2, label=n, zorder=2)
            ax.set_xticks(gx)
            ax.set_xticklabels([lbl for _, lbl in metrics], rotation=20,
                               ha="right", fontsize=7.5)
            ax.set_ylabel(f"W1 distance to {reference}")
            ax.axhline(0, color=INK_MUTED, lw=0.8, zorder=1)
            ax.set_title("d  Distance from the real distribution", loc="left", pad=8)
            ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
        else:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=len(names), loc="lower center",
                   bbox_to_anchor=(0.5, -0.06))
        fig.suptitle("Generated topology against real pathway topology",
                     x=0.02, ha="left", fontsize=11, color=INK)
        fig.tight_layout(rect=[0, 0.02, 1, 0.92])
        if save:
            fig.savefig(save, dpi=300)
        return fig
