"""Draw graphs the fixed GraphVAE sampled, next to held-out and diffusion graphs. Run as a job.

    sbatch run_debug_vae.sh scripts/draw_vae_samples.py [--run compare_oct2fix_20261111_1637]

Uses the samples the run saved (no re-sampling), the first 12 per arm, so the
pictures are the same graphs the benchmark scored. Writes to
outputs/analysis/<run>/figures/<dataset>/:

    vae_samples_<variant>.png   12 VAE graphs (nopad and pad)
    vae_vs_models.png           3 graphs each: held-out data, then VAE, masked, transition, continuous
                                of one variant (nopad when the run has it, else pad)

Panel titles give the raw slot counts (before duplicate entities are merged and
isolated nodes dropped for the drawing): real nodes, edges and the share of
edges joining a type pair that never occurs in a training graph.
"""

import argparse
import sys

import numpy as np
import torch

from graphpath.pipeline.common import RunLayout, load_bundle
from graphpath.plotting.graphs import create_graph_objs, plot_graph_samples


def legal_type_pairs(b):
    pairs = set()
    for g in b.train_graphs:
        k = g["num_nodes"]; ids = g["node_ids"][:k]
        adj = np.asarray(g["adj_matrix"])[:k, :k]
        ts = [b.ntype(b.id_to_node.get(int(i), "")) for i in ids]
        for i, j in zip(*np.nonzero(np.triu(adj, 1))):
            pairs.add(tuple(sorted((ts[i], ts[j]))))
    return pairs


def title(b, nodes, adj, legal, tag):
    n = nodes.tolist(); a = adj.numpy()
    ts = [b.ntype(b.id_to_node.get(int(t), "")) for t in n]
    real = sum(t != 0 for t in n)
    iu = np.triu_indices(len(n), 1)
    e = [(i, j) for i, j in zip(*iu) if a[i, j]]
    bad = sum(tuple(sorted((ts[i], ts[j]))) not in legal for i, j in e)
    distinct = len({t for t in n if t != 0})
    return f"{tag}\n{real} nodes ({distinct} distinct)\n{len(e)} edges, {bad / max(len(e), 1):.0%} type-violating"


def draw(b, nodes, adj, titles, suptitle, path):
    disp = type("D", (), {"id_to_node": dict(b.id_to_node)})()
    graphs, _, labels, types, _ = create_graph_objs(nodes, adj.float(), disp,
                                                    get_common_name=b.common, get_node_type=b.ntype)
    fig = plot_graph_samples(graphs, types, labels, titles=titles, suptitle=suptitle, verbose=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"-> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="compare_oct2fix_20261111_1637")
    a = ap.parse_args()
    run = RunLayout(a.run)
    out = run.path.parents[1] / "analysis" / run.path.name / "figures"
    for d in ("ecoli", "human"):
        b = load_bundle(d)
        legal = legal_type_pairs(b)
        print(f"{d}: legal type pairs {sorted(legal)}", flush=True)
        for v in ("nopad", "pad"):
            s = run.load_samples(d, "vae", v)
            if s is None:
                print(f"  no vae ({v}) samples"); continue
            n_, a_ = s[0][:12], s[1][:12]
            draw(b, n_, a_, [title(b, n_[i], a_[i], legal, f"VAE ({v}) sample {i + 1}") for i in range(len(n_))],
                 f"{d} — GraphVAE ({v}): 12 sampled graphs (duplicates merged, isolated nodes hidden)",
                 out / d / f"vae_samples_{v}.png")
        hid, hadj = b.heldout_tensors(3, 64, seed=0)
        rows = [("held-out", hid, (hadj > 0).long())]
        cmp_v = "nopad" if run.load_samples(d, "vae", "nopad") is not None else "pad"
        for arm in ("vae", "masked", "transition", "continuous"):
            s = run.load_samples(d, arm, cmp_v)
            if s is not None and len(rows) < 4:      # the grid holds 4 columns of 3
                rows.append((f"{arm} ({cmp_v})", s[0][:3], s[1][:3]))
        nodes = torch.cat([r[1][:3].long() for r in rows]); adj = torch.cat([r[2][:3].long() for r in rows])
        titles = [title(b, r[1][i], r[2][i], legal, f"{r[0]} {i + 1}") for r in rows for i in range(3)]
        # plot_graph_samples lays out a 3x4 grid row-major; reorder so each model is one column
        order = [r * 3 + i for i in range(3) for r in range(len(rows))]
        draw(b, nodes[order], adj[order], [titles[k] for k in order],
             f"{d} — columns: " + " | ".join(r[0] for r in rows),
             out / d / "vae_vs_models.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
