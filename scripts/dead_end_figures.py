"""Figures and tables for the dead-end hypotheses (reads outputs of dead_end_hypotheses.py). GPU-job launcher.

    sbatch run_debug_vae.sh scripts/dead_end_figures.py [--run R]

Writes to outputs/analysis/<run>/dead_ends/figures/:
    calibration.png         probe probability for hidden true edges vs other reactions (held-out graphs)
    overview.png            dead ends with candidates per tier; tier-1 hypotheses by kind and gap direction
    network_<metabolite>.png  one context subgraph per showcase dead end: existing edges, the dead end's
                            masked row, and its top predicted links (dashed, labelled with P)
    hypotheses_top.tex      LaTeX table of the top tier-1 hypotheses (one per dead end)
Cofactor dead ends (graphpath.data.cofactors) are dropped from tables and counts.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from graphpath.data.cofactors import is_cofactor
from graphpath.pipeline.common import RunLayout, load_bundle

TYPE_COLOR = {"gene": "#b48c1f", "metabolite": "#0effdb", "enzyme": "#2c39a0", "reaction": "#d627b9"}
TYPE_SHAPE = {"gene": "o", "metabolite": "s", "enzyme": "^", "reaction": "D"}
SHOWCASE = ["MAM01806m", "MAM02892c", "MAM01355r", "MAM03863r"]   # farnesyl-PP[m], selenomethionine-Se-oxide[c], apoC3[r], phytoceramide[r]


def tex(s):
    return str(s).replace("\\", "/").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="compare_oct2pad_20261112_1238"); a = ap.parse_args()
    run = RunLayout(a.run)
    D = run.path.parents[1] / "analysis" / run.path.name / "dead_ends"; F = D / "figures"; F.mkdir(exist_ok=True)
    b = load_bundle("human"); n2i = {n: i for i, n in b.id_to_node.items()}
    rows = list(csv.DictReader(open(D / "candidates.csv")))
    cof = lambda r: is_cofactor(r["metabolite"], r["metabolite_name"])
    rows = [r for r in rows if not cof(r)]
    for r in rows:
        for k in ("p_mean", "support"): r[k] = float(r[k]) if r[k] not in ("", "nan") else np.nan
        for k in ("tier", "n_contexts", "transport_link", "rank"): r[k] = int(r[k])
    per = [p for p in csv.DictReader(open(D / "per_metabolite.csv")) if not is_cofactor(p["metabolite"], p["name"])]
    cal = json.load(open(D / "calibration.json"))

    # ── calibration with degree-deficit controls (probe_controls.json) ──
    pc = json.load(open(D / "probe_controls.json"))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[3, 4, 3]))
    def bars(ax, items, title, chance=None):
        xs = np.arange(len(items))
        for x, (lab, cond, key, col) in zip(xs, items):
            m, lo, hi, n = pc[cond][key]
            ax.bar(x, m, color=col, width=0.66)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="0.2", capsize=2, lw=0.9)
            ax.text(x, hi + 0.02, f"{m:.2f}", ha="center", fontsize=7)
        if chance is not None:
            ax.axhline(chance, ls=":", color="0.4", lw=1); ax.text(-0.45, chance + 0.012, "chance", fontsize=6.5, ha="left", color="0.4")
        ax.set_xticks(xs, [it[0] for it in items], fontsize=6.8); ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=8, loc="left"); ax.spines[["top", "right"]].set_visible(False)
    BL, BN, GR = "#2a78d6", "#7aa9e6", "#b9b6ad"
    bars(axes[0], [("probe\nstandard", "A", "probe_auc", BL), ("probe\nneutral", "C", "probe_auc", BN),
                   ("degree\ndeficit", "A", "deficit_frac_auc", GR)], "AUC, all hidden edges", chance=0.5)
    bars(axes[1], [("probe\nstandard", "A", "probe_top1", BL), ("probe\nneutral", "C", "probe_top1", BN),
                   ("degree\ndeficit", "A", "deficit_frac_top1", GR), ("degree\ndeficit\nneutral", "C", "deficit_frac_top1", GR)],
         "top-1 accuracy", chance=pc["A"]["chance_top1"][0])
    bars(axes[2], [("probe\nstandard", "A", "probe_auc_unseen", BL), ("probe\nneutral", "C", "probe_auc_unseen", BN),
                   ("degree\ndeficit", "A", "deficit_frac_auc_unseen", GR)],
         f"AUC, unseen edges (n = {pc['A']['probe_auc_unseen'][3]})", chance=0.5)
    fig.tight_layout(); fig.savefig(F / "calibration.png", dpi=220); plt.close(fig)

    # ── overview ──
    best = {}
    for r in rows:
        if r["metabolite"] not in best or r["tier"] < best[r["metabolite"]]["tier"]:
            best[r["metabolite"]] = r
    n_ctx = sum(int(p["n_contexts"]) > 0 for p in per)
    tiers = [sum(1 for r in best.values() if r["tier"] == t) for t in (1, 2, 3, 4)]
    t1 = [r for r in rows if r["tier"] == 1]
    kinds = {"same species,\nother compartment": sum(r["transport_link"] for r in t1),
             "co-expression\nsupport only": sum(1 - r["transport_link"] for r in t1)}
    gaps = {g: sum(1 for r in t1 if r["gap"] == g) for g in ("needs_sink", "needs_source")}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axes[0]
    ax.bar(range(4), tiers, color=["#2a78d6", "#7aa9e6", "#b9b6ad", "#e4e3de"])
    for x, v in enumerate(tiers): ax.text(x, v + 3, str(v), ha="center", fontsize=8)
    ax.set_xticks(range(4), ["tier 1", "tier 2", "tier 3", "tier 4"], fontsize=8)
    ax.set_ylabel("dead ends (best candidate)", fontsize=8)
    ax.set_title(f"{len(per)} non-cofactor dead ends, {n_ctx} with a context", fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    lab = list(kinds) ; ax.barh([0, 1], [kinds[k] for k in lab], color=["#1aa36f", "#8a8780"])
    for y, k in enumerate(lab): ax.text(kinds[k] + 0.5, y, str(kinds[k]), va="center", fontsize=8)
    ax.set_yticks([0, 1], lab, fontsize=8); ax.invert_yaxis()
    ax.set_title(f"{len(t1)} tier-1 links: {gaps['needs_sink']} sinks, {gaps['needs_source']} sources", fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(F / "overview.png", dpi=220); plt.close(fig)

    # ── showcase networks ──
    ctx = torch.load(D / "contexts.pt", weights_only=False)
    typ = lambda t: b.ntype(b.id_to_node.get(int(t), ""))
    name = lambda t: b.common(b.id_to_node.get(int(t), str(t)))
    eq_of = {(r["metabolite"], r["reaction"]): r for r in rows}
    for m in SHOWCASE:
        if m not in n2i: continue
        t = n2i[m]
        cand = sorted([r for r in rows if r["metabolite"] == m], key=lambda r: -r["p_mean"])[:3]
        if not cand: continue
        best_i, best_p = None, -1
        for i, (tt, split, gi) in enumerate(ctx["meta"]):
            if tt != t: continue
            ids, adj, s = ctx["items"][i]; p = ctx["probs"][i]
            score = sum(float(p[j]) for j in range(64) if int(ids[j]) in {n2i["rxn_" + c["reaction"]] for c in cand if "rxn_" + c["reaction"] in n2i})
            if score > best_p: best_i, best_p = i, score
        ids, adj, s = ctx["items"][best_i]; p = ctx["probs"][best_i]
        k = int((ids > 0).sum())
        G = nx.Graph(); G.add_nodes_from(range(k))
        for i, j in zip(*np.nonzero(np.triu(adj[:k, :k].numpy(), 1))): G.add_edge(int(i), int(j))
        known = [j for j in G.neighbors(s)]
        rx = [j for j in range(k) if typ(ids[j]) == "reaction" and j not in known]
        top = sorted(rx, key=lambda j: -float(p[j]))[:3]
        pos = nx.spring_layout(G, seed=2, k=0.9 / np.sqrt(max(k, 2)))
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.7", width=0.8)
        from matplotlib.patches import FancyArrowPatch
        for q, j in enumerate(top):
            rad = (0.25, -0.25, 0.4)[q % 3]
            ax.add_patch(FancyArrowPatch(pos[s], pos[j], connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                                         ls="--", color="#d62728", lw=1.8, zorder=3))
            a_, b2 = np.asarray(pos[s]), np.asarray(pos[j]); mid = (a_ + b2) / 2
            nrm = np.array([-(b2 - a_)[1], (b2 - a_)[0]]) * rad / 2
            ax.text(*(mid + nrm), f"{float(p[j]):.2f}", color="#d62728", fontsize=7.5, ha="center", va="center",
                    bbox=dict(fc="white", ec="none", pad=0.6), zorder=4)
        for ty, sh in TYPE_SHAPE.items():
            nl = [j for j in range(k) if typ(ids[j]) == ty]
            nx.draw_networkx_nodes(G, pos, nodelist=nl, node_shape=sh, node_color=TYPE_COLOR[ty], node_size=55,
                                   edgecolors="0.25", linewidths=0.5, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=[s], node_shape="s", node_color=TYPE_COLOR["metabolite"], node_size=260,
                               edgecolors="#d62728", linewidths=2.2, ax=ax)
        lab = {s: f"{name(ids[s])} [{m[-1]}]"} | {j: name(ids[j])[:30] for j in top} | {j: name(ids[j])[:24] for j in known}
        for j, l in lab.items():
            ax.text(pos[j][0], pos[j][1] + 0.06, l, fontsize=6.5 if j != s else 8, ha="center",
                    fontweight="bold" if j == s else None, zorder=5)
        import textwrap
        info = eq_of.get((m, b.id_to_node[int(ids[top[0]])].replace("rxn_", "")))
        sub = "\n".join(textwrap.wrap(f"top link: {info['reaction_equation']}", 75)) if info else ""
        ax.set_title(f"{name(ids[s])} [{m[-1]}]: dead end, {'needs a sink' if cand[0]['gap'] == 'needs_sink' else 'needs a source'}\n{sub}",
                     fontsize=7.5, loc="left")
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color="0.7", lw=1, label="Human-GEM edge"),
                           Line2D([], [], color="#d62728", lw=1.8, ls="--", label="predicted link (P)"),
                           Line2D([], [], marker="s", ls="", markerfacecolor=TYPE_COLOR["metabolite"], markeredgecolor="#d62728",
                                  markeredgewidth=2, markersize=9, label="dead end (edges masked)")]
                  + [Line2D([], [], marker=TYPE_SHAPE[t], ls="", color=TYPE_COLOR[t], markeredgecolor="0.25", label=t) for t in TYPE_SHAPE],
                  loc="lower right", fontsize=6.5, frameon=False, ncol=2)
        ax.axis("off"); fig.tight_layout(); fig.savefig(F / f"network_{m}.png", dpi=220); plt.close(fig)
        print(f"-> network_{m}.png  (context {ctx['meta'][best_i][1]} graph {ctx['meta'][best_i][2]}, {k} nodes)")

    # ── LaTeX table: best tier-1 link per dead end ──
    seen, lines = set(), []
    for r in [r for r in rows if r["tier"] == 1]:
        if r["metabolite"] in seen: continue
        seen.add(r["metabolite"])
        comp = r["same_species_in"].split(";")[0][-1:] if r["transport_link"] else ""
        kind = f"{r['metabolite'][-1]}$\\rightarrow${comp}" if r["transport_link"] else "co-expr."
        named = r["reaction_name"] and not r["reaction_name"].startswith(("MAR", "rxn_"))
        rn = r["reaction_name"] if named else f"{r['reaction']} ({r['reaction_subsystem']})"
        sup = f"{r['support']:.2f}" if np.isfinite(r["support"]) else "--"
        lines.append(f"{tex(r['metabolite_name'][:34])} [{r['metabolite'][-1]}] & {'sink' if r['gap'] == 'needs_sink' else 'source'} & "
                     f"{tex(rn[:58])} & {r['p_mean']:.2f} & {r['n_contexts']} & {sup} & {kind} \\\\")
    (F / "hypotheses_top.tex").write_text("\n".join(lines))
    print(f"tier-1 dead ends (non-cofactor): {len(seen)}; table rows {len(lines)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
