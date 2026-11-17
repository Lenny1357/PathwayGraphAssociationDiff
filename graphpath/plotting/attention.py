"""Attention diagnostics: per-head maps, type-to-type flow, edge alignment."""

from typing import Optional, Sequence

import numpy as np

from graphpath.plotting.style import (EDGE_STYLE, FALLBACK, GRID, INK, INK_MUTED,
                                      INK_SOFT, PALETTE, SURFACE, TYPE_COLORS,
                                      TYPE_HATCHES, TYPE_SHAPES, _as_np, _bars,
                                      _color, _rc)


def attention_maps_figure(cap: dict, type_ids=None, graph: int = 0,
                          layers=None, heads=None, save: Optional[str] = None):
    """
    Per-layer, per-head attention over the node slots for one graph.

    Slots are reordered by entity type when `type_ids` is given, with
    dividers between the blocks — a type grammar is invisible in slot order
    and obvious once the rows are grouped.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    attn = cap["attn"]
    L, B, H, N, _ = attn.shape
    layers = list(range(L)) if layers is None else list(layers)
    heads = list(range(H)) if heads is None else list(heads)

    order = np.arange(N)
    bounds, labels = [], []
    if type_ids is not None:
        ti = type_ids[graph].cpu().numpy()
        order = np.argsort(ti, kind="stable")
        srt = ti[order]
        bounds = list(np.flatnonzero(np.diff(srt)) + 1)
        for t in np.unique(srt):
            idx = np.flatnonzero(srt == t)
            labels.append((idx.mean(), int(t)))

    ramp = LinearSegmentedColormap.from_list("attn", ["#ffffff", "#2a78d6", "#0b2545"])
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(len(layers), len(heads),
                                 figsize=(1.85 * len(heads) + 1.0,
                                          1.85 * len(layers) + 0.9),
                                 squeeze=False)
        vmax = float(attn[layers][:, graph].quantile(0.99))
        for r, l in enumerate(layers):
            for c, hd in enumerate(heads):
                ax = axes[r][c]
                a = attn[l, graph, hd].numpy()[order][:, order]
                ax.imshow(a, cmap=ramp, vmin=0, vmax=vmax, aspect="equal")
                for b in bounds:
                    ax.axhline(b - 0.5, color="#d1341a", lw=0.6)
                    ax.axvline(b - 0.5, color="#d1341a", lw=0.6)
                ax.set_xticks([]); ax.set_yticks([])
                for sp_ in ax.spines.values():
                    sp_.set_color(GRID)
                if r == 0:
                    ax.set_title(f"head {hd}", fontsize=8.5, pad=4)
                if c == 0:
                    ax.set_ylabel(f"layer {l}", fontsize=8.5)
        fig.suptitle(f"Node attention, graph {graph}"
                     + ("  (slots grouped by entity type)" if type_ids is not None else ""),
                     x=0.02, ha="left", fontsize=11, color=INK)
        fig.text(0.02, 0.005, "row = attending slot, column = attended slot; "
                              "red lines separate entity types",
                 fontsize=8, color=INK_MUTED, ha="left")
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        if save:
            fig.savefig(save, dpi=300)
        return fig


def type_attention_figure(flow: np.ndarray, type_names: Sequence[str],
                          save: Optional[str] = None):
    """
    Attention mass from one entity type to another, per layer.

    The model is never told the type grammar — it gets a type embedding and
    a bias on the edge logits, nothing that says "genes attend to enzymes".
    If that structure appears here, it was inferred.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    L, T, _ = flow.shape
    ramp = LinearSegmentedColormap.from_list("seq", ["#f4f7fd", "#2a78d6", "#123a6b"])
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, L, figsize=(2.5 * L + 1.4, 3.1), squeeze=False)
        for l in range(L):
            ax = axes[0][l]
            im = ax.imshow(flow[l], cmap=ramp, vmin=0, vmax=float(flow.max()))
            ax.set_xticks(range(T)); ax.set_xticklabels(type_names, rotation=45, ha="right")
            ax.set_yticks(range(T))
            ax.set_yticklabels(type_names if l == 0 else [""] * T)
            ax.set_title(f"layer {l}", loc="left", fontsize=9)
            ax.set_xticks(np.arange(-0.5, T, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, T, 1), minor=True)
            ax.grid(which="minor", color=SURFACE, linewidth=2)
            ax.tick_params(which="minor", length=0); ax.tick_params(length=0)
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            if l == 0:
                ax.set_ylabel("attending type")
        cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.02)
        cb.outline.set_visible(False)
        cb.set_label("share of attention", fontsize=8.5, color=INK_SOFT)
        fig.suptitle("Where each entity type sends its attention",
                     x=0.02, ha="left", fontsize=11, color=INK)
        # below the tick labels; bbox_inches="tight" keeps it in frame
        fig.text(0.02, -0.13, "rows sum to 1; the attended type is on the x axis",
                 fontsize=8, color=INK_MUTED, ha="left")
        if save:
            fig.savefig(save, dpi=300, bbox_inches="tight")
        return fig


def attention_edges_figure(align: dict, entropy: Optional[np.ndarray] = None,
                           n_slots: Optional[int] = None,
                           save: Optional[str] = None):
    """
    Does attention follow the edges the model believes in, and how much of
    the attention score comes from the edge stream rather than node content?

    Panel a: attention on believed-present pairs over absent ones. 1.0 means
    the edge state is not steering attention at all.
    Panel b: the edge bias's share of the raw score — the direct measure of
    how much the model trusts its own edge representation, layer by layer.
    """
    import matplotlib.pyplot as plt

    rows = align["per_layer"]
    L = [r["layer"] for r in rows]
    with plt.rc_context(_rc()):
        has_share = "edge_share" in rows[0]
        n = 2 + (entropy is not None)
        fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 3.2))
        axes = np.atleast_1d(axes)

        ax = axes[0]
        ax.bar(L, [r["ratio"] for r in rows], width=0.6,
               color=PALETTE["generated"], edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.axhline(1.0, ls=(0, (4, 3)), lw=1, color=INK_MUTED, zorder=3)
        ax.text(L[0] - 0.45, 1.02, "no preference", fontsize=7.5, color=INK_MUTED,
                ha="left", va="bottom")
        ax.set_xlabel("layer"); ax.set_ylabel("attention  present / absent")
        ax.set_title("a  Does attention follow believed edges?", loc="left",
                     pad=8, fontsize=9.5)
        ax.grid(axis="y", zorder=0); ax.set_axisbelow(True); ax.set_xticks(L)

        ax = axes[1]
        if has_share:
            ax.bar(L, [r["edge_share"] for r in rows], width=0.6,
                   color=PALETTE["training"], edgecolor=SURFACE, linewidth=2, zorder=2)
            ax.set_ylabel("edge bias share of the score")
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, "this backbone has no\nadditive edge bias",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color=INK_MUTED)
            ax.set_yticks([])
        ax.set_xlabel("layer"); ax.set_xticks(L)
        ax.set_title("b  How much comes from the edge stream?", loc="left",
                     pad=8, fontsize=9.5)
        ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)

        if entropy is not None:
            ax = axes[2]
            for hd in range(entropy.shape[1]):
                ax.plot(range(entropy.shape[0]), entropy[:, hd], marker="o", ms=5,
                        lw=1.6, label=f"head {hd}", markeredgecolor=SURFACE,
                        markeredgewidth=1.0)
            if n_slots:
                ax.axhline(np.log(n_slots), ls=(0, (4, 3)), lw=1, color=INK_MUTED)
                ax.text(0, np.log(n_slots) * 0.99, " uniform", fontsize=7.5,
                        color=INK_MUTED, va="top")
            ax.set_xlabel("layer"); ax.set_ylabel("entropy (nats)")
            ax.set_title("c  How focused is each head?", loc="left", pad=8, fontsize=9.5)
            ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
            ax.legend(fontsize=7.5, ncol=2)

        fig.suptitle("How the attention decision is made",
                     x=0.02, ha="left", fontsize=11, color=INK)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        if save:
            fig.savefig(save, dpi=300)
        return fig
