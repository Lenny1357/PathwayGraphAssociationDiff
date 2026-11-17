"""iML1515 -> pathway graphs, in the format PathwayGraphDataset expects.

The piece data/ecoli was missing. Human-GEM arrives with
`human_gem_sampled_graphs.pkl` already built; there is no E. coli equivalent,
so `compare_models_ecoli.ipynb` had nothing to train on.

Emits the same structure as the human pickle:

    {"node_ids": [...], "adj_matrix": (n,n) float, "num_nodes": n,
     "original_group_id": <subsystem>, "center_node": <node name>}

plus `node_id_map.pkl` mapping every entity name to a token id.

    python build_pathways.py --ref-dir ecoli_ref --out-dir ecoli_ref

Two decisions worth knowing about, both visible in the printed stats:

THE HIGHEST-DEGREE ENTITIES ARE STRIPPED. In iML1515 these turn out to be
promiscuous ENZYMES rather than currency metabolites — ompF and friends sit in
286 reactions each, against 70 for pyruvate. Either way they locate nothing:
left in, every chunk is the same hub neighbourhood. `--n-hubs 0` disables it.

CHUNKS ARE GROWN PER SUBSYSTEM, from a seed reaction outward, and are allowed
to OVERLAP by default. Overlap is what the human dataset does and it is
usually a leak — held-out human chunks sit at ~0.85 Jaccard to a training
chunk. Here it is not: 2,242 overlapping chunks have mean pairwise Jaccard
0.011 (p95 0.041), because the network is sparse enough that different seeds
inside one subsystem land in genuinely different places. It buys 5x the graphs
for almost nothing. `--disjoint` enforces non-overlap anyway (458 graphs,
Jaccard 0.005).
"""

import argparse
import collections
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def load_network(ref_dir: Path):
    adj = collections.defaultdict(set)
    subs = collections.defaultdict(set)
    with open(ref_dir / "ecoli_gem_reaction_node_edges.csv") as f:
        for r in csv.DictReader(f):
            u, v = r["source"], r["target"]
            adj[u].add(v); adj[v].add(u)
            s = (r.get("subsystem") or "").strip()
            if s:
                subs[u].add(s); subs[v].add(s)
    return adj, subs


def grow(seed, adj, banned, max_nodes, taken):
    """BFS from `seed`, skipping hubs and anything already used in this pass."""
    seen, frontier = {seed}, collections.deque([seed])
    while frontier and len(seen) < max_nodes:
        for nb in sorted(adj[frontier.popleft()]):
            if nb in banned or nb in seen or nb in taken:
                continue
            seen.add(nb); frontier.append(nb)
            if len(seen) >= max_nodes:
                break
    return seen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", default="ecoli_ref")
    ap.add_argument("--out-dir", default="ecoli_ref")
    ap.add_argument("--max-nodes", type=int, default=64)
    ap.add_argument("--min-nodes", type=int, default=8)
    ap.add_argument("--n-hubs", type=int, default=30,
                    help="highest-degree entities to strip (0 = keep all)")
    ap.add_argument("--disjoint", action="store_true",
                    help="forbid chunks within a subsystem from sharing nodes; "
                         "yields ~458 graphs instead of ~2,242")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    ref = here / a.ref_dir if not Path(a.ref_dir).is_absolute() else Path(a.ref_dir)
    out = here / a.out_dir if not Path(a.out_dir).is_absolute() else Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    adj, subs = load_network(ref)
    deg = {k: len(v) for k, v in adj.items()}
    hubs = set(sorted(deg, key=lambda k: -deg[k])[:a.n_hubs]) if a.n_hubs else set()
    print(f"network: {len(adj):,} entities, "
          f"{sum(len(v) for v in adj.values()) // 2:,} edges")
    if hubs:
        top = sorted(hubs, key=lambda k: -deg[k])[:5]
        print(f"  stripping the {len(hubs)} highest-degree entities, e.g. "
              f"{[(t, deg[t]) for t in top]}")
        print(f"  (mostly promiscuous enzymes here, not currency metabolites)")

    by_sub = collections.defaultdict(list)
    for n, ss in subs.items():
        if n in hubs:
            continue
        for s in ss:
            by_sub[s].append(n)
    print(f"  {len(by_sub)} subsystems")

    rng = np.random.default_rng(a.seed)
    graphs = []
    for sub, members in sorted(by_sub.items()):
        pool = sorted(set(members))
        seeds = [n for n in pool if n.startswith("rxn_")] or pool
        rng.shuffle(seeds)
        taken = set()
        for s in seeds:
            if not (not a.disjoint) and s in taken:
                continue
            nodes = grow(s, adj, hubs, a.max_nodes, taken if not (not a.disjoint) else set())
            nodes = sorted(nodes)
            if len(nodes) < a.min_nodes:
                continue
            if not (not a.disjoint):
                taken |= set(nodes)
            idx = {n: i for i, n in enumerate(nodes)}
            A = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
            for u in nodes:
                for v in adj[u]:
                    if v in idx:
                        A[idx[u], idx[v]] = A[idx[v], idx[u]] = 1.0
            graphs.append({"nodes": nodes, "adj_matrix": A,
                           "num_nodes": len(nodes),
                           "original_group_id": sub, "center_node": s})

    # Deduplicate by node set BEFORE anything else. A node can belong to
    # several subsystems, so the same neighbourhood gets grown more than once
    # and lands in more than one group — 20.7% of graphs were duplicates, and
    # 168 distinct node sets spanned multiple subsystems, which defeats a
    # subsystem-held-out split outright. Keep the first occurrence and record
    # every subsystem it belonged to, so the split can still avoid it.
    seen_sets, deduped = {}, []
    for gph in graphs:
        key = frozenset(gph["nodes"])
        if key in seen_sets:
            seen_sets[key]["also_in"].add(gph["original_group_id"])
            continue
        gph["also_in"] = {gph["original_group_id"]}
        seen_sets[key] = gph
        deduped.append(gph)
    n_dup = len(graphs) - len(deduped)
    cross = sum(1 for gph in deduped if len(gph["also_in"]) > 1)
    graphs = deduped
    for gph in graphs:
        gph["also_in"] = sorted(gph["also_in"])
    print(f"  dropped {n_dup:,} duplicate node sets; {cross:,} of the survivors "
          f"belong to more than one subsystem")

    # one vocabulary over every entity that made it into a graph
    vocab = sorted({n for g in graphs for n in g["nodes"]})
    node_id_map = {n: i for i, n in enumerate(vocab)}
    for g in graphs:
        g["node_ids"] = [node_id_map[n] for n in g.pop("nodes")]

    with open(out / "ecoli_sampled_graphs.pkl", "wb") as f:
        pickle.dump(graphs, f)
    with open(out / "ecoli_node_id_map.pkl", "wb") as f:
        pickle.dump(node_id_map, f)

    sz = np.array([g["num_nodes"] for g in graphs])
    dens = np.array([g["adj_matrix"].sum() / (n * (n - 1))
                     for g, n in zip(graphs, sz) if n > 1])
    kinds = collections.Counter()
    for n in vocab:
        kinds["reaction" if n.startswith("rxn_") else
              "enzyme" if n.startswith("enz_") else
              "gene" if n.startswith("b") and n[1:].isdigit() else "metabolite"] += 1
    print(f"\n  {len(graphs):,} graphs, vocabulary {len(vocab):,} entities")
    print(f"  nodes per graph: median {int(np.median(sz))}, "
          f"range {sz.min()}-{sz.max()}")
    print(f"  edge density over valid pairs: median {np.median(dens):.4f}")
    print(f"  composition: " + "  ".join(f"{k} {v:,}" for k, v in kinds.most_common()))
    print(f"  subsystems represented: "
          f"{len({g['original_group_id'] for g in graphs})}")
    print(f"\n  wrote {out/'ecoli_sampled_graphs.pkl'} and "
          f"{out/'ecoli_node_id_map.pkl'}")
    (out / "pathways_meta.json").write_text(json.dumps(
        {"n_graphs": len(graphs), "vocab_size": len(vocab),
         "max_nodes": a.max_nodes, "min_nodes": a.min_nodes,
         "n_hubs_stripped": a.n_hubs, "overlap": (not a.disjoint), "seed": a.seed}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
