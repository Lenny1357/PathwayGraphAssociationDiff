"""Figures for supported novel links: evidence chains and one discovery in situ."""

from typing import Optional, Sequence

import numpy as np

from graphpath.plotting.style import (EDGE_STYLE, FALLBACK, GRID, INK, INK_MUTED,
                                      INK_SOFT, PALETTE, SURFACE, TYPE_COLORS,
                                      TYPE_HATCHES, TYPE_SHAPES, _as_np, _bars,
                                      _color, _rc)


def discovery_figure(finder, discoveries: list, control: Optional[np.ndarray] = None,
                     tissue: str = "", save: Optional[str] = None,
                     max_rows: int = 8, label_fn=None):
    """
    Novel generated links, each drawn as the evidence chain that supports it.

    One row per discovery, read left to right:

        endpoint ──hops──▶ gene A  ≈≈ co-expression ≈≈  gene B ◀──hops── endpoint
        └──────────────── the generated edge, absent from the reference ─────┘

    The left panel puts the same scores against random non-edges. That
    control is the whole argument: hub metabolites reach a large part of the
    network in two hops, so a high support value only means something if
    arbitrary pairs do not also get one.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    label_fn = label_fn or finder.name_fn
    rows = [d for d in discoveries if d["support"] > 0][:max_rows]
    if not rows:
        raise ValueError("no supported novel links to draw")

    def tcol(i):
        return TYPE_COLORS.get(finder.type_fn(finder.dataset.id_to_node.get(int(i), "")),
                               "#7f7f7f")

    with plt.rc_context(_rc()):
        ncols = 2 if control is not None else 1
        fig, axes = plt.subplots(
            1, ncols, figsize=(6.2 + 5.2 * (ncols - 1), 0.62 * len(rows) + 1.9),
            gridspec_kw={"width_ratios": [1, 2.2] if ncols == 2 else [1]})
        axes = np.atleast_1d(axes)
        ax_chain = axes[-1]

        if control is not None:
            ax = axes[0]
            sup = np.array([d["support"] for d in discoveries])
            # the comparison is a RATE, at matched n. A histogram of 8 novel
            # links against 300 random ones compares nothing — the random
            # spike at zero is just the larger sample.
            n = min(len(sup), len(control))
            rates = [float((sup > 0).mean()), float((control > 0).mean())]
            counts = [int((sup > 0).sum()), int((control > 0).sum())]
            names = ["novel\ngenerated", "random\nnon-edges"]
            cols = [PALETTE["generated"], INK_MUTED]
            for i, (r, c, nm, col) in enumerate(zip(rates, counts, names, cols)):
                ax.bar(i, r, width=0.6, color=col, edgecolor=SURFACE, linewidth=2, zorder=2)
                ax.text(i, r + 0.02, f"{r:.0%}", ha="center", va="bottom",
                        fontsize=9, color=INK)
                ax.text(i, 0.02, f"{c} of {len(sup) if i == 0 else len(control)}",
                        ha="center", va="bottom", fontsize=7.5, color=SURFACE, zorder=3)
            ax.set_xticks([0, 1]); ax.set_xticklabels(names)
            ax.set_xlim(-0.65, 1.65)              # keep the bars beside each other
            ax.set_ylabel("share with support")
            ax.set_ylim(0, max(rates) * 1.42 + 0.02)
            fold = rates[0] / rates[1] if rates[1] > 0 else float("inf")
            ax.set_title(f"a  How often support exists\n"
                         + (f"{fold:.1f}$\\times$ more often than chance"
                            if rates[1] > 0 else "no random pair had any"),
                         loc="left", pad=8, fontsize=9.5)
            ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
            # inside the axes, clear of the two-line tick labels
            med = np.median(sup[sup > 0]) if (sup > 0).any() else 0.0
            medc = np.median(control[control > 0]) if (control > 0).any() else 0.0
            ax.text(0.5, 0.97, f"median where present: {med:.2f} vs {medc:.2f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7.5, color=INK_MUTED)

        # ── evidence chains ────────────────────────────────────────────────
        xs = [0.0, 0.30, 0.70, 1.0]
        for r, d in enumerate(rows):
            y = len(rows) - r - 1
            ids = [d["u"], d["gene_a"], d["gene_b"], d["v"]]

            # the generated edge itself, spanning the whole row
            ax_chain.annotate("", xy=(xs[3], y + 0.30), xytext=(xs[0], y + 0.30),
                              arrowprops=dict(arrowstyle="-", lw=1.4,
                                              color=PALETTE["generated"],
                                              linestyle=(0, (5, 3))))
            ax_chain.text((xs[0] + xs[3]) / 2, y + 0.36, "generated link",
                          fontsize=6.5, color=PALETTE["generated"], ha="center")

            # hop segments, then the co-expression bridge
            for a, b, lab in ((0, 1, f"{d['hops_a']} hop" + ("s" if d['hops_a'] != 1 else "")),
                              (2, 3, f"{d['hops_b']} hop" + ("s" if d['hops_b'] != 1 else ""))):
                ax_chain.plot([xs[a], xs[b]], [y, y], color=INK_MUTED, lw=1.0, zorder=2)
                ax_chain.text((xs[a] + xs[b]) / 2, y - 0.20, lab, fontsize=6.5,
                              color=INK_MUTED, ha="center", va="top")
            ax_chain.plot([xs[1], xs[2]], [y, y], color="#1baf7a", lw=2.2, zorder=2)
            ax_chain.text((xs[1] + xs[2]) / 2, y + 0.13, f"co-expr {d['support']:.2f}",
                          fontsize=7, color="#0f7a55", ha="center")

            for x, i in zip(xs, ids):
                ax_chain.scatter([x], [y], s=110, c=tcol(i), edgecolors=SURFACE,
                                 linewidths=1.4, zorder=4)
                ax_chain.text(x, y - 0.34, str(label_fn(i))[:18], fontsize=7,
                              color=INK, ha="center", va="top")

        ax_chain.set_xlim(-0.16, 1.16)
        ax_chain.set_ylim(-0.75, len(rows) - 0.25)
        ax_chain.set_axis_off()
        ax_chain.set_title(("b  " if control is not None else "") +
                           "Evidence chains for the best-supported novel links",
                           loc="left", pad=8, fontsize=9.5)

        present = [t for t in TYPE_COLORS]
        fig.legend(handles=[Line2D([0], [0], marker="o", linestyle="", color=TYPE_COLORS[t],
                                   label=t, markersize=7) for t in present],
                   loc="lower center", ncol=len(present), bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Novel generated links with propagated co-expression support"
                     + (f"  ({tissue})" if tissue else ""),
                     x=0.02, ha="left", fontsize=11, color=INK)
        fig.tight_layout(rect=[0, 0.06, 1, 0.92])
        if save:
            fig.savefig(save, dpi=300)
        return fig


def discovery_context_figure(finder, discovery: dict, node_toks, adjacency,
                             label_fn=None, save: Optional[str] = None,
                             seed: int = 0, max_label: int = 16,
                             show_missed: bool = True):
    """
    One discovery in situ, with every edge classified by where it came from.

    The generated pathway is not one thing — part of it reproduces the
    curated network and part of it does not, and those two parts are what a
    reader needs to tell apart:

        confirmed   generated, and present in the reference network
        novel       generated, absent from the reference
        the link    the novel edge under scrutiny
        missed      in the reference between these nodes, but NOT generated
        route       reference edges leading out to an anchor gene
        co-expr     the gene-gene correlation that supports the link

    Nodes use the project's type styling (shape + hatch + colour, as in
    graphpath/plotting/graphs.py). Filled = in the generated pathway; hollow =
    reached only by the evidence route.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as _np
    import torch
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    label_fn = label_fn or finder.name_fn
    d = discovery
    b = d["graph"]
    ids = node_toks[b].cpu()
    adj = adjacency[b].cpu().bool()
    N = ids.shape[0]

    pathway = sorted({int(ids[i]) for i in range(N) if int(ids[i]) != finder.pad_id})
    focus = (int(d["u"]), int(d["v"]))
    bridge = (int(d["gene_a"]), int(d["gene_b"]))

    def known(a, c):
        return bool(finder.ref.has_edge(finder._id(a), finder._id(c))[0])

    G = nx.Graph()
    G.add_nodes_from(pathway)
    roles = {}

    # generated edges, split by whether the reference agrees
    gen_pairs = set()
    for i in range(N):
        for j in range(i + 1, N):
            if not adj[i, j]:
                continue
            a, c = int(ids[i]), int(ids[j])
            if a == c or finder.pad_id in (a, c):
                continue
            gen_pairs.add((min(a, c), max(a, c)))
    for a, c in gen_pairs:
        G.add_edge(a, c)
        roles[(a, c)] = ("focus" if {a, c} == set(focus)
                         else "confirmed" if known(a, c) else "novel")

    # reference edges among the same nodes that the model did NOT draw
    if show_missed:
        for x, a in enumerate(pathway):
            for c in pathway[x + 1:]:
                if (a, c) in roles or not known(a, c):
                    continue
                G.add_edge(a, c); roles[(a, c)] = "missed"

    # the evidence route, then the co-expression bridge
    for path in (d["path_a"], d["path_b"]):
        for a, c in zip(path, path[1:]):
            a, c = int(a), int(c)
            key = (min(a, c), max(a, c))
            G.add_edge(a, c)
            roles.setdefault(key, "route")
    G.add_edge(*bridge)
    roles[(min(bridge), max(bridge))] = "coexpr"

    # spring_layout pulls harder on heavier edges. The co-expression bridge
    # is the thing to read, so give it the least pull and let it stretch;
    # otherwise the two anchor genes collapse together and the key evidence
    # becomes a two-pixel line.
    for (a, c), r in roles.items():
        G[a][c]["weight"] = {"coexpr": 0.18, "route": 0.55}.get(r, 1.0)
    pos = nx.spring_layout(G, seed=seed, k=2.4 / max(len(G) ** 0.5, 1),
                           iterations=300, weight="weight")
    evidence_only = set(G.nodes()) - set(pathway)

    def type_of(n):
        return finder.type_fn(finder.dataset.id_to_node.get(int(n), ""))

    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(7.8, 6.4))

        for role in ("missed", "confirmed", "novel", "route", "coexpr", "focus"):
            el = [e for e, r in roles.items() if r == role]
            if not el:
                continue
            st = EDGE_STYLE[role]
            kw = {}
            if role == "coexpr":
                # curved: co-expression is not an edge of either graph, and
                # drawing it straight makes it look like one
                kw["connectionstyle"] = "arc3,rad=0.28"
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=el, edge_color=st["color"],
                                   width=st["width"], style=st["style"],
                                   alpha=0.95 if role != "missed" else 0.9, **kw)

        for group, filled in ((pathway, True), (sorted(evidence_only), False)):
            by_type = {}
            for n in group:
                by_type.setdefault(type_of(n), []).append(n)
            for t, nodes_ in by_type.items():
                col = TYPE_COLORS.get(t, "#7f7f7f")
                coll = nx.draw_networkx_nodes(
                    G, pos, ax=ax, nodelist=nodes_,
                    node_shape=TYPE_SHAPES.get(t, "o"),
                    node_size=300 if filled else 260,
                    node_color=col if filled else SURFACE,
                    edgecolors=col if not filled else "#3d3d3a",
                    linewidths=1.0 if filled else 2.2)
                if coll is not None and filled and TYPE_HATCHES.get(t):
                    coll.set_hatch(TYPE_HATCHES[t])

        # co-expression weight, written on its own edge
        p0, p1 = _np.asarray(pos[bridge[0]]), _np.asarray(pos[bridge[1]])
        mid = (p0 + p1) / 2
        seg = p1 - p0
        perp = _np.array([-seg[1], seg[0]])
        norm = _np.linalg.norm(perp) or 1.0
        # sit on the outside of the arc's bulge
        lab = mid + (perp / norm) * 0.16 * _np.linalg.norm(seg)
        ax.text(lab[0], lab[1], f"co-expressed  r={d['support']:.2f}", fontsize=8,
                color=EDGE_STYLE["coexpr"]["color"], ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.22", fc=SURFACE, ec="none", alpha=0.9),
                zorder=6)

        ys = _np.array([p[1] for p in pos.values()])
        dy = 0.06 * ((ys.max() - ys.min()) or 1.0)
        lpos = {n: (p[0], p[1] - dy) for n, p in pos.items()}
        key_nodes = {focus[0], focus[1], bridge[0], bridge[1]}
        for subset, size, colr in ((key_nodes, 8.5, INK),
                                   (set(G.nodes()) - key_nodes, 6.5, INK_MUTED)):
            nx.draw_networkx_labels(
                G, lpos, ax=ax,
                labels={n: str(label_fn(n))[:max_label] for n in G.nodes() if n in subset},
                font_size=size, font_color=colr, verticalalignment="top")

        ax.set_axis_off(); ax.margins(0.14)
        n_conf = sum(1 for r in roles.values() if r == "confirmed")
        n_nov = sum(1 for r in roles.values() if r == "novel")
        n_mis = sum(1 for r in roles.values() if r == "missed")
        ax.set_title(f"{label_fn(focus[0])} — {label_fn(focus[1])}   "
                     f"co-expression support {d['support']:.2f}\n"
                     f"pathway: {n_conf} confirmed, {n_nov + 1} novel"
                     + (f", {n_mis} known edges missed" if show_missed else ""),
                     loc="left", pad=10, fontsize=10)

        order = [("focus", "the novel link"), ("novel", "other novel edges"),
                 ("confirmed", "confirmed by the reference"),
                 ("missed", "known but not generated"),
                 ("route", "route to the anchor gene"),
                 ("coexpr", "gene co-expression")]
        handles = [Line2D([0], [0], color=EDGE_STYLE[r]["color"],
                          lw=EDGE_STYLE[r]["width"], ls=EDGE_STYLE[r]["style"], label=lab)
                   for r, lab in order if r != "missed" or show_missed]
        types_present = sorted({type_of(n) for n in G.nodes()})
        handles += [Line2D([0], [0], marker=TYPE_SHAPES.get(t, "o"), lw=0,
                           markerfacecolor=TYPE_COLORS.get(t, "#7f7f7f"),
                           markeredgecolor="#3d3d3a", markersize=8, label=t)
                    for t in types_present]
        handles += [Line2D([0], [0], marker="o", lw=0, markerfacecolor=SURFACE,
                           markeredgecolor="#3d3d3a", markeredgewidth=1.8,
                           markersize=8, label="evidence only (not generated)")]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
                   bbox_to_anchor=(0.5, -0.015))
        fig.tight_layout(rect=[0, 0.16, 1, 0.97])
        if save:
            fig.savefig(save, dpi=300)
        return fig
