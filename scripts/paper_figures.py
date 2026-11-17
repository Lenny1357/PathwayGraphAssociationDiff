"""Paper figures the pipeline does not draw: named pathway panels and novel-link evidence chains.

    sbatch run_debug_vae.sh scripts/paper_figures.py --run compare_oct2pad_20261112_1238

Reads only saved outputs (samples, pathway_stats.json, discovery.json); nothing is
re-sampled or re-scored. Writes to outputs/analysis/<run>/figures/<dataset>/:

    pathways_<arm>.png        the 6 graphs pathway_stats.json scored for that arm (and
                              pathways_held-out.png), titled with distinct share, edges
                              and subsystems touched -- the numbers in pathway_stats.json
    discovery_<arm>_<i>.png   the i-th best-supported novel link of discovery.json in its
                              generated pathway, with the co-expression route that supports it

and, at the figures root, node_type_fractions.png (type composition of node slots per arm
against held-out, from graph_stats.json). The type-violation rate of every arm and the
held-out set (share of edges joining a type pair that never occurs in a training graph)
goes to outputs/analysis/<run>/type_violations.json.
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from graphpath.pipeline.common import RunLayout, dump_json, load_bundle, load_json
from graphpath.plotting.graphs import create_graph_objs, plot_graph_samples


def arm_name(label):                       # "masked (pad)" -> "masked"
    return label.split(" (")[0]


def pathways(run, out, d, b, stats, max_nodes=64):
    disp = type("D", (), {"id_to_node": dict(b.id_to_node)})()
    hid, hadj = b.heldout_tensors(6 * 4, max_nodes)
    for label, rec in stats.items():
        if label == "held-out":
            nodes, adj = hid, (hadj > 0).long()
        else:
            s = run.load_samples(d, arm_name(label), "pad")
            if s is None:
                print(f"  {d} {label}: no samples"); continue
            nodes, adj = s
        idx = [g["index"] for g in rec["graphs"]]
        titles = [f"{label}: distinct {g['distinct']:.0%}, {g['edges']} edges,\n{g['subsystems']} subsystems"
                  for g in rec["graphs"]]
        graphs, _, labels, types, _ = create_graph_objs(nodes[idx], adj[idx].float(), disp,
                                                        get_common_name=b.common, get_node_type=b.ntype)
        fig = plot_graph_samples(graphs, types, labels, titles=titles, verbose=False,
                                 suptitle=f"{d}: {label}, 6 graphs (duplicates merged, isolated nodes hidden)")
        p = out / d / f"pathways_{arm_name(label)}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"-> {p}", flush=True)


def chains(run, out, d, b, disc, arms, n=1):
    from graphpath.benchmarks import DiscoveryFinder
    from graphpath.pipeline.score import BenchmarkSuite, expression_source
    from graphpath.plotting.discovery import discovery_context_figure
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = BenchmarkSuite(b, device)
    expr, _ = expression_source(b, device)
    from graphpath.data.cofactors import cofactor_ids
    finder = DiscoveryFinder(suite.ref, expr, b.train, node_type_fn=b.ntype, max_hops=disc["max_hops"],
                             exclude=cofactor_ids(b),
                             name_fn=lambda i: b.common(b.id_to_node.get(int(i), i)))
    for label, rec in disc["arms"].items():
        if arm_name(label) not in arms or not rec.get("top"):
            continue
        nodes, adj = run.load_samples(d, arm_name(label), "pad")
        for i, top in enumerate(rec["top"][:n]):
            p = out / d / f"discovery_{arm_name(label)}_{i}.png"
            discovery_context_figure(finder, top, nodes, adj, save=str(p))
            plt.close("all"); print(f"-> {p}  (support {top['support']:.3f})", flush=True)


def legal_type_pairs(b):
    pairs = set()
    for g in b.train_graphs:
        k = g["num_nodes"]; ids = g["node_ids"][:k]
        adj = np.asarray(g["adj_matrix"])[:k, :k]
        ts = [b.ntype(b.id_to_node.get(int(i), "")) for i in ids]
        for i, j in zip(*np.nonzero(np.triu(adj, 1))):
            pairs.add(tuple(sorted((ts[i], ts[j]))))
    return pairs


def violations(b, nodes, adj, legal):
    tmap = {}
    bad = tot = 0
    for n, a in zip(nodes.tolist(), adj.numpy()):
        iu, ju = np.nonzero(np.triu(a > 0, 1))
        for i, j in zip(iu, ju):
            ti = tmap.setdefault(n[i], b.ntype(b.id_to_node.get(int(n[i]), "")))
            tj = tmap.setdefault(n[j], b.ntype(b.id_to_node.get(int(n[j]), "")))
            tot += 1; bad += tuple(sorted((ti, tj))) not in legal
    return dict(violating=bad, edges=tot, rate=bad / max(tot, 1))


def type_fractions(gstats, out):
    types = ("gene", "enzyme", "reaction", "metabolite")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, d in zip(axes, ("ecoli", "human")):
        rows = gstats[d]; names = list(rows)
        x = np.arange(len(types)); w = 0.8 / len(names)
        for i, nm in enumerate(names):
            ax.bar(x + (i - (len(names) - 1) / 2) * w, [rows[nm]["mean"][f"frac_{t}"] for t in types], w,
                   label=nm, color="0.3" if nm == "held-out" else None)
        ax.set_xticks(x, types); ax.set_title("E. coli" if d == "ecoli" else "human")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("share of node slots"); axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); p = out / "node_type_fractions.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); print(f"-> {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="compare_oct2pad_20261112_1238")
    a = ap.parse_args()
    run = RunLayout(a.run)
    out = run.path.parents[1] / "analysis" / run.path.name / "figures"
    ana = out.parent
    pstats, disc = load_json(ana / "pathway_stats.json"), load_json(ana / "discovery.json")
    type_fractions(load_json(ana / "graph_stats.json"), out)
    viol = {}
    want = {"ecoli": ("masked", "transition", "continuous", "vae"), "human": ("masked", "transition", "continuous", "vae")}
    for d in ("ecoli", "human"):
        b = load_bundle(d)
        legal = legal_type_pairs(b)
        viol[d] = dict(legal_pairs=sorted(map(list, legal)))
        hid, hadj = b.heldout_tensors(None, 64)
        viol[d]["held-out"] = violations(b, hid, (hadj > 0).long(), legal)
        for arm in ("masked", "transition", "continuous", "vae"):
            s_ = run.load_samples(d, arm, "pad")
            if s_ is not None:
                viol[d][f"{arm} (pad)"] = violations(b, s_[0], s_[1], legal)
        print(d, {k: round(v["rate"], 4) for k, v in viol[d].items() if k != "legal_pairs"}, flush=True)
        dump_json(viol, ana / "type_violations.json")
        pathways(run, out, d, b, pstats[d])
        chains(run, out, d, b, disc[d], want[d])
    return 0


if __name__ == "__main__":
    sys.exit(main())
