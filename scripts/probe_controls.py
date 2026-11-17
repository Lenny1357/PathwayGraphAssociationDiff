"""Controls for the in-context link-prediction calibration (GPU job).

    sbatch run_debug_vae.sh scripts/probe_controls.py [--n 3000] [--boot 1000]

Concern: masking a metabolite m's edges leaves each of its true reactions one participant
short, so a model that knows typical reaction degrees can find them without chemistry. Two
controls, on the same held-out cases (metabolite slots in held-out human subgraphs with >= 1
reaction neighbour and >= 3 other reactions):

    A  standard          m's edge row masked (as in dead_end_hypotheses.py)
    B  deficit-neutral   additionally, every decoy reaction has one of its own edges (to a node
                         other than m) masked, so every candidate is one participant short
                         (but decoys now carry two masked pairs: a mask-count cue)
    C  fully neutral     instead, that decoy edge is deleted (set absent): every candidate is one
                         participant short AND only m's row is masked -- neither cue is left

Scorers, each evaluated in A and B:
    probe          P(edge m-j) from the absorbing-mask model (t_node = 0, t_edge = 100)
    deficit        Human-GEM degree of j minus its decided edges in the input graph
    deficit_frac   the same, divided by the Human-GEM degree

Metrics: per-case AUC of true reactions vs the graph's other reactions, and top-1 accuracy
(chance = 1 / #reactions), each with a 95% cluster-bootstrap interval (resampling held-out
graphs), overall and for true edges that occur in no training graph.
Writes outputs/analysis/<run>/dead_ends/probe_controls.json.
"""
import argparse
import collections
import json
import sys

import numpy as np
import torch

from graphpath.pipeline import arms as A
from graphpath.pipeline.common import RunLayout, dump_json, load_bundle
from graphpath.pipeline.score import BenchmarkSuite

T_EDGE = 100


def as_tensors(g):
    k = g["num_nodes"]
    ids = torch.zeros(64, dtype=torch.long); ids[:k] = torch.as_tensor(g["node_ids"][:k])
    adj = torch.zeros(64, 64, dtype=torch.long); adj[:k, :k] = torch.as_tensor(np.asarray(g["adj_matrix"])[:k, :k]).long()
    return ids, adj


def auc_top1(score, nb, non):
    s = np.asarray(score, float)
    pn, pf = s[nb], s[non]
    auc = float((pn[:, None] > pf[None, :]).mean() + 0.5 * (pn[:, None] == pf[None, :]).mean())
    cand = list(nb) + list(non); best = s[cand].max()
    tied = [c for c in cand if s[c] == best]
    top1 = sum(c in nb for c in tied) / len(tied)          # ties split fairly
    return auc, top1


def boot(vals, groups, n_boot, rng):
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    ug = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in ug}
    est = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug), replace=True)
        est.append(vals[np.concatenate([idx[g] for g in pick])].mean())
    return [float(vals.mean()), float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="compare_oct2pad_20261112_1238")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    device = torch.device("cuda")
    run = RunLayout(a.run)
    out = run.path.parents[1] / "analysis" / run.path.name / "dead_ends"; out.mkdir(parents=True, exist_ok=True)
    b = load_bundle("human")
    typ = lambda t: b.ntype(b.id_to_node.get(int(t), ""))
    model, _ = A.load_arm(run, "human", "masked", "pad", device); model.eval()
    MASK = model.edge_mask
    ref = BenchmarkSuite(b, device).ref
    ix = ref.adj_sparse.coalesce().indices().cpu().numpy()
    refdeg = collections.Counter(int(u) for u in ix[0])
    train_edges = set()
    for g in b.train_graphs:
        ids, adj = as_tensors(g)
        for i, j in zip(*np.nonzero(np.triu(adj.numpy(), 1))):
            train_edges.add((min(int(ids[i]), int(ids[j])), max(int(ids[i]), int(ids[j]))))

    cases = []
    for gi, g in enumerate(b.test_graphs):
        ids, adj = as_tensors(g); k = g["num_nodes"]
        rx = [j for j in range(k) if typ(ids[j]) == "reaction"]
        for s in range(k):
            if typ(ids[s]) != "metabolite":
                continue
            nb = [j for j in rx if adj[s, j]]; non = [j for j in rx if not adj[s, j]]
            if nb and len(non) >= 3:
                cases.append((gi, s, nb, non))
    rng = np.random.default_rng(0)
    cases = [cases[i] for i in rng.choice(len(cases), min(a.n, len(cases)), replace=False)]
    print(f"cases {len(cases)} from {len({c[0] for c in cases})} held-out graphs", flush=True)

    def build(case, neutral, rng, delete=False):
        gi, s, nb, non = case
        ids, adj = as_tensors(b.test_graphs[gi]); e = adj.clone()
        e[s, :] = MASK; e[:, s] = MASK; e[s, s] = 0
        if neutral:
            for j in non:                       # hide one of each decoy's own edges (not to m)
                others = [x for x in np.flatnonzero(adj[j].numpy()) if x != s]
                if others:
                    x = int(rng.choice(others)); v = 0 if delete else MASK
                    e[j, x] = v; e[x, j] = v
        pad = ids == 0; e[pad, :] = 0; e[:, pad] = 0
        return ids, e

    res = {cond: collections.defaultdict(list) for cond in ("A", "B", "C")}
    groups = {cond: [] for cond in res}; unseen_mask = {cond: [] for cond in res}
    for cond, neutral, delete in (("A", False, False), ("B", True, False), ("C", True, True)):
        crng = np.random.default_rng(1)
        built = [build(c, neutral, crng, delete) for c in cases]
        for c0 in range(0, len(cases), 128):
            chunk = built[c0:c0 + 128]; cc = cases[c0:c0 + 128]
            ids = torch.stack([x[0] for x in chunk]); e = torch.stack([x[1] for x in chunk])
            B = len(chunk)
            tn = torch.zeros(B, dtype=torch.long, device=device); te = torch.full((B,), T_EDGE, dtype=torch.long, device=device)
            with torch.no_grad():
                _, el = model.forward(ids.to(device), e.to(device), tn, te)
            pe = torch.softmax(el[..., :2].float(), -1)[..., 1].cpu()
            for bi, (gi, s, nb, non) in enumerate(cc):
                p = ((pe[bi, s] + pe[bi, :, s]) / 2).numpy()
                dec = (e[bi] != MASK) & (e[bi] > 0)               # decided-present edges
                vis = dec.sum(1).numpy()
                rd = np.array([refdeg.get(int(t), 0) for t in ids[bi]], float)
                deficit = rd - vis
                frac = np.where(rd > 0, deficit / np.maximum(rd, 1), 0)
                for name, sc in (("probe", p), ("deficit", deficit), ("deficit_frac", frac)):
                    au, t1 = auc_top1(sc, nb, non)
                    res[cond][f"{name}_auc"].append(au); res[cond][f"{name}_top1"].append(t1)
                res[cond]["chance_top1"].append(len(nb) / (len(nb) + len(non)))
                m = int(ids[bi, s])
                unseen = [j for j in nb if (min(m, int(ids[bi, j])), max(m, int(ids[bi, j]))) not in train_edges]
                # per-case AUC restricted to unseen true edges (nan if none)
                if unseen:
                    au_u = {name: auc_top1(sc, unseen, non)[0] for name, sc in (("probe", p), ("deficit", deficit), ("deficit_frac", frac))}
                else:
                    au_u = {name: np.nan for name in ("probe", "deficit", "deficit_frac")}
                for name, v in au_u.items():
                    res[cond][f"{name}_auc_unseen"].append(v)
                groups[cond].append(gi)
        print(f"condition {cond} done", flush=True)

    rngb = np.random.default_rng(2)
    summary = dict(n_cases=len(cases), n_graphs=len({c[0] for c in cases}), t_edge=T_EDGE, n_boot=a.boot)
    for cond in res:
        g = np.array(groups[cond]); rows = {}
        for key, v in res[cond].items():
            v = np.asarray(v, float); ok = np.isfinite(v)
            rows[key] = boot(v[ok], g[ok], a.boot, rngb) + [int(ok.sum())]
        summary[cond] = rows
    dump_json(summary, out / "probe_controls.json")
    lab = {"A": "standard (only m's edges masked)", "B": "deficit-neutral, decoy edge masked (masked-count cue)",
           "C": "deficit- and mask-neutral: one real edge of every decoy deleted, only m's row masked"}
    for cond in ("A", "B", "C"):
        print(f"\n== {lab[cond]}")
        for key in ("probe_auc", "deficit_auc", "deficit_frac_auc", "probe_top1", "deficit_top1", "deficit_frac_top1",
                    "chance_top1", "probe_auc_unseen", "deficit_auc_unseen", "deficit_frac_auc_unseen"):
            m, lo, hi, n = summary[cond][key]
            print(f"  {key:26} {m:.3f}  [{lo:.3f}, {hi:.3f}]  n={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
