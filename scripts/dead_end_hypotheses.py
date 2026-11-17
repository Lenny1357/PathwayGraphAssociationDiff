"""Connection hypotheses for Human-GEM dead-end metabolites, from in-context link prediction. GPU job.

    sbatch --time=02:00:00 run_debug_vae.sh scripts/dead_end_hypotheses.py [--run R]

Why in context. Pinning a dead end into an otherwise empty graph does not steer an
unconditionally trained diffusion model (it builds a typical pathway elsewhere and attaches the
pinned node with one arbitrary edge), and scoring a metabolite against isolated reactions is at
chance (AUC ~0.5). Edge predictions need pathway context. The absorbing-mask model was trained
with independent node and edge noise levels, so a graph with every node visible and only some
edges masked is a state it has seen. The probe:

    for each real subgraph (training or held-out) that contains dead end m:
        mask every edge of m, keep all nodes and all other edges, t_edge = 100
        read P(edge m-j) for every reaction j in the graph

Calibration (held-out subgraphs, ordinary metabolites, one edge row hidden): how well the probe
ranks the true reaction among the graph's other reactions -- overall, and for edges that never
occur in any training graph (the closest available test of generalisation).

Candidates: reactions in m's subgraphs that Human-GEM does not link to m. Per (m, r):
    p_mean / p_max   probe probability over the subgraphs containing both
    n_contexts       independent subgraphs in which r was scored for m
    rank_in_graph    r's rank among the graph's reactions (1 = top)
    support          propagated co-expression support (3 hops, cofactors excluded), against two
                     controls: the model's own low-probability reactions in the same subgraphs,
                     and random reactions
    transport_link   r carries the same species in another compartment (a transport fix)

Writes outputs/analysis/<run>/dead_ends/: calibration.json, candidates.csv (ranked),
per_metabolite.csv, summary.json, contexts.pt (graphs + probabilities, for drawing).
"""

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch

from graphpath.data.cofactors import cofactor_ids
from graphpath.pipeline import arms as A
from graphpath.pipeline.common import RunLayout, dump_json, load_bundle
from graphpath.pipeline.score import BenchmarkSuite, expression_source

sys.path.insert(0, "data/gem")
from build_gem import as_str_list  # noqa: E402

GEM_MAT = Path("data/gem/gem_cache/Human-GEM-v1.19.0.mat")
DEAD_CSV = Path("data/gem/gem_gaps_metabolites.csv")
T_EDGE = 100


def gem_reactions():
    g = sio.loadmat(GEM_MAT, struct_as_record=False, squeeze_me=True)
    g = g[[k for k in g if not k.startswith("__")][0]]
    S = sp.csc_matrix(g.S)
    mets, rxns = list(as_str_list(g.mets)), list(as_str_list(g.rxns))
    subs = list(as_str_list(g.subSystems)) if hasattr(g, "subSystems") else [""] * len(rxns)
    gr = list(as_str_list(g.grRules)) if hasattr(g, "grRules") else [""] * len(rxns)
    rn = list(as_str_list(g.rxnNames)) if hasattr(g, "rxnNames") else rxns
    mn = list(as_str_list(g.metNames)) if hasattr(g, "metNames") else mets
    lb = np.asarray(g.lb, float)
    def eq(j):
        cs = [(mets[i], float(v)) for i, v in zip(S.indices[S.indptr[j]:S.indptr[j + 1]], S.data[S.indptr[j]:S.indptr[j + 1]])]
        side = lambda sg: " + ".join(f"{'' if abs(v) == 1 else f'{abs(v):g} '}{mn[mets.index(m)]}[{m[-1]}]" for m, v in cs if (v < 0) == sg)
        return f"{side(True)} {'<=>' if lb[j] < 0 else '=>'} {side(False)}"
    return {r: dict(mets={mets[i]: float(v) for i, v in zip(S.indices[S.indptr[j]:S.indptr[j + 1]],
                                                            S.data[S.indptr[j]:S.indptr[j + 1]])},
                    subsystem=subs[j], gpr=gr[j], name=rn[j] or r, equation=eq(j))
            for j, r in enumerate(rxns)}


def as_tensors(g):
    k = g["num_nodes"]
    ids = torch.zeros(64, dtype=torch.long); ids[:k] = torch.as_tensor(g["node_ids"][:k])
    adj = torch.zeros(64, 64, dtype=torch.long)
    adj[:k, :k] = torch.as_tensor(np.asarray(g["adj_matrix"])[:k, :k]).long()
    return ids, adj


@torch.no_grad()
def probe_batch(model, items, device):
    """items: [(ids, adj, slot)] -> list of (64,) P(edge slot-j)."""
    ids = torch.stack([x[0] for x in items]); adj = torch.stack([x[1] for x in items]).clone()
    for b_, (_, _, s) in enumerate(items):
        adj[b_, s, :] = model.edge_mask; adj[b_, :, s] = model.edge_mask; adj[b_, s, s] = 0
    pad = ids == 0
    adj[pad[:, :, None].expand_as(adj)] = 0; adj[pad[:, None, :].expand_as(adj)] = 0
    B = len(items)
    tn = torch.zeros(B, dtype=torch.long, device=device); te = torch.full((B,), T_EDGE, dtype=torch.long, device=device)
    _, el = model.forward(ids.to(device), adj.to(device), tn, te)
    pe = torch.softmax(el[..., :2].float(), -1)[..., 1]
    return [((pe[b_, s] + pe[b_, :, s]) / 2).cpu() for b_, (_, _, s) in enumerate(items)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="compare_oct2pad_20261112_1238")
    ap.add_argument("--max-contexts", type=int, default=30)
    ap.add_argument("--n-calib", type=int, default=1500)
    a = ap.parse_args()
    device = torch.device("cuda")
    run = RunLayout(a.run)
    out = run.path.parents[1] / "analysis" / run.path.name / "dead_ends"
    out.mkdir(parents=True, exist_ok=True)

    b = load_bundle("human")
    n2i = {n: i for i, n in b.id_to_node.items()}
    typ = lambda t: b.ntype(b.id_to_node.get(int(t), ""))
    model, _ = A.load_arm(run, "human", "masked", "pad", device); model.eval()
    suite = BenchmarkSuite(b, device); ref = suite.ref
    ix = ref.adj_sparse.coalesce().indices().cpu().numpy()
    nbr = collections.defaultdict(set)
    for u, v in zip(ix[0], ix[1]):
        nbr[int(u)].add(int(v))

    # edges present in at least one training graph
    train_edges = set()
    for g in b.train_graphs:
        ids, adj = as_tensors(g)
        for i, j in zip(*np.nonzero(np.triu(adj.numpy(), 1))):
            train_edges.add((min(int(ids[i]), int(ids[j])), max(int(ids[i]), int(ids[j]))))

    # ── calibration on held-out subgraphs ─────────────────────────────────
    cases = []
    for g in b.test_graphs:
        ids, adj = as_tensors(g); k = g["num_nodes"]
        rx = [j for j in range(k) if typ(ids[j]) == "reaction"]
        for s in range(k):
            if typ(ids[s]) != "metabolite":
                continue
            nb = [j for j in rx if adj[s, j]]; non = [j for j in rx if not adj[s, j]]
            if nb and len(non) >= 3:
                cases.append((ids, adj, s, nb, non))
    rng = np.random.default_rng(0)
    pick = [cases[i] for i in rng.choice(len(cases), min(a.n_calib, len(cases)), replace=False)]
    cal = collections.defaultdict(list)
    for c0 in range(0, len(pick), 128):
        chunk = pick[c0:c0 + 128]
        ps = probe_batch(model, [(x[0], x[1], x[2]) for x in chunk], device)
        for (ids, adj, s, nb, non), p in zip(chunk, ps):
            pn, pnon = p[nb].numpy(), p[non].numpy()
            auc = float((pn[:, None] > pnon[None, :]).mean() + 0.5 * (pn[:, None] == pnon[None, :]).mean())
            cand = nb + non; top = cand[int(np.argmax(p[cand].numpy()))]
            unseen = [j for j in nb if (min(int(ids[s]), int(ids[j])), max(int(ids[s]), int(ids[j]))) not in train_edges]
            cal["auc"].append(auc); cal["hit1"].append(top in nb); cal["chance"].append(len(nb) / len(cand))
            cal["p_true"] += pn.tolist(); cal["p_false"] += pnon.tolist()
            for j in unseen:     # generalisation: this hidden edge never occurs in a training graph
                cal["auc_unseen"].append(float((p[j].item() > pnon).mean() + 0.5 * (p[j].item() == pnon).mean()))
    pt, pf = np.array(cal["p_true"]), np.array(cal["p_false"])
    thr = {q: float(np.quantile(pt, q)) for q in (0.25, 0.5)}
    prec_at = {f"p>={v:.2f}": float((pt >= v).sum() / max((pt >= v).sum() + (pf >= v).sum(), 1)) for v in thr.values()}
    calib = dict(n_cases=len(pick), t_edge=T_EDGE, auc=float(np.mean(cal["auc"])), hit1=float(np.mean(cal["hit1"])),
                 hit1_chance=float(np.mean(cal["chance"])), n_unseen=len(cal["auc_unseen"]),
                 auc_unseen=float(np.mean(cal["auc_unseen"])) if cal["auc_unseen"] else None,
                 p_true_quartiles=thr, precision_at_threshold=prec_at,
                 p_true_median=float(np.median(pt)), p_false_median=float(np.median(pf)))
    dump_json(calib, out / "calibration.json")
    print("calibration:", json.dumps(calib), flush=True)

    # ── dead ends in context ──────────────────────────────────────────────
    dead = [r for r in csv.DictReader(open(DEAD_CSV)) if r["metabolite"] in n2i]
    tok_of = {n2i[r["metabolite"]]: r for r in dead}
    ctx = collections.defaultdict(list)
    for split, graphs in (("train", b.train_graphs), ("test", b.test_graphs)):
        for gi, g in enumerate(graphs):
            k = g["num_nodes"]
            for s, t in enumerate(g["node_ids"][:k]):
                if int(t) in tok_of and len(ctx[int(t)]) < a.max_contexts:
                    ctx[int(t)].append((split, gi, s))
    items, meta = [], []
    for t, lst in ctx.items():
        for split, gi, s in lst:
            g = (b.train_graphs if split == "train" else b.test_graphs)[gi]
            ids, adj = as_tensors(g); items.append((ids, adj, s)); meta.append((t, split, gi))
    probs = []
    for c0 in range(0, len(items), 128):
        probs += probe_batch(model, items[c0:c0 + 128], device)
    print(f"dead ends with a context: {len(ctx)} / {len(dead)}; contexts probed {len(items)}", flush=True)

    rx = gem_reactions()
    expr, _ = expression_source(b, device)
    from graphpath.benchmarks import DiscoveryFinder
    cof = cofactor_ids(b)
    finder = DiscoveryFinder(ref, expr, b.train, node_type_fn=b.ntype, max_hops=3, exclude=cof,
                             name_fn=lambda i: b.common(b.id_to_node.get(int(i), i)))
    tissue = "breast"

    agg = collections.defaultdict(lambda: dict(p=[], rank=[], ctx=[]))
    low = collections.defaultdict(set)       # the model's own low-probability picks (bottom half)
    known_rank = collections.defaultdict(list)
    for (ids, adj, s), p, (t, split, gi) in zip(items, probs, meta):
        k = int((ids > 0).sum())
        rxs = [j for j in range(k) if typ(ids[j]) == "reaction"]
        if not rxs:
            continue
        pv = p[rxs].numpy(); order = (-pv).argsort(); rank = np.empty_like(order); rank[order] = np.arange(1, len(order) + 1)
        for jj, j in enumerate(rxs):
            r = int(ids[j])
            if r in nbr[t]:
                known_rank[t].append(rank[jj] / len(rxs))
                continue
            d = agg[(t, r)]; d["p"].append(float(pv[jj])); d["rank"].append(int(rank[jj])); d["ctx"].append((split, gi))
            if rank[jj] > len(rxs) / 2:
                low[t].add(r)

    rows = []
    for (t, r), d in agg.items():
        rr = tok_of[t]; m = rr["metabolite"]; core = m[:-1]
        rn = b.id_to_node[r]; info = rx.get(rn.replace("rxn_", ""), {})
        other_comp = [x for x in info.get("mets", {}) if x[:-1] == core and x != m]
        rows.append(dict(metabolite=m, metabolite_name=rr["name"], compartment=rr["compartment"], gap=rr["gap"],
                         transport_would_fix=int(rr["transport_would_fix"]), current_subsystems=rr["subsystems"],
                         reaction=rn.replace("rxn_", ""), reaction_name=info.get("name") or b.common(rn),
                         reaction_equation=info.get("equation", "")[:160],
                         reaction_subsystem=info.get("subsystem", ""), reaction_gpr=info.get("gpr", "")[:80],
                         p_mean=float(np.mean(d["p"])), p_max=float(np.max(d["p"])),
                         best_rank_in_graph=int(min(d["rank"])), n_contexts=len(set(d["ctx"])),
                         transport_link=int(bool(other_comp)), same_species_in=";".join(other_comp), _t=t, _r=r))
    # support for the top candidates of each dead end and for the two controls
    by_t = collections.defaultdict(list)
    for d in rows:
        by_t[d["_t"]].append(d)
    rng = np.random.default_rng(1)
    rxn_tokens = np.array([i for n, i in n2i.items() if b.ntype(n) == "reaction"])
    ctl_low, ctl_rand = [], []
    for t, lst in by_t.items():
        lst.sort(key=lambda d: -d["p_mean"])
        for d in lst[:10]:
            sc = finder.score_pair(t, d["_r"], tissue)
            d["support"] = sc["support"]
            d["anchor_genes"] = (f"{b.common(b.id_to_node.get(sc['gene_a'], ''))}~{b.common(b.id_to_node.get(sc['gene_b'], ''))}"
                                 if sc["gene_a"] is not None else "")
        for r in list(low[t])[:5]:
            ctl_low.append(finder.score_pair(t, r, tissue)["support"])
        for r in rng.choice(rxn_tokens, 5, replace=False):
            if int(r) not in nbr[t]:
                ctl_rand.append(finder.score_pair(t, int(r), tissue)["support"])
    ctl_low, ctl_rand = np.array(ctl_low), np.array(ctl_rand)
    q95 = float(np.quantile(np.concatenate([ctl_low, ctl_rand]), 0.95)) if len(ctl_low) + len(ctl_rand) else 0.0
    p50 = thr[0.5]
    for d in rows:
        d.setdefault("support", np.nan); d.setdefault("anchor_genes", "")
        conf = d["p_mean"] >= p50
        evid = (d["support"] > q95 if np.isfinite(d["support"]) else False) or d["transport_link"]
        d["tier"] = 1 if (conf and evid and d["n_contexts"] >= 2) else 2 if (conf and evid) else 3 if conf else 4
    rows.sort(key=lambda d: (d["tier"], -d["p_mean"], -(d["support"] if np.isfinite(d["support"]) else -1)))
    for i, d in enumerate(rows):
        d["rank"] = i + 1
    cols = ["rank", "tier"] + [k for k in rows[0] if k not in ("rank", "tier", "_t", "_r")]
    with open(out / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    per = []
    for r_ in dead:
        t = n2i[r_["metabolite"]]; lst = sorted(by_t.get(t, []), key=lambda d: (d["tier"], -d["p_mean"]))
        per.append(dict(metabolite=r_["metabolite"], name=r_["name"], gap=r_["gap"], n_contexts=len(ctx.get(t, [])),
                        known_reaction_rank_frac=float(np.mean(known_rank[t])) if known_rank[t] else np.nan,
                        n_candidates=len(lst), best_tier=lst[0]["tier"] if lst else None,
                        top_reaction=lst[0]["reaction_name"] if lst else "", top_p=lst[0]["p_mean"] if lst else np.nan))
    with open(out / "per_metabolite.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0])); w.writeheader(); w.writerows(per)

    top = [d for t in by_t for d in sorted(by_t[t], key=lambda d: -d["p_mean"])[:10]]
    sup_top = np.array([d["support"] for d in top if np.isfinite(d["support"])])
    kr = [x for v in known_rank.values() for x in v]
    summ = dict(run=a.run, model="masked (pad)", t_edge=T_EDGE, tissue=tissue, support_hops=3, cofactors_excluded=len(cof),
                calibration=calib, n_dead_ends=len(dead), n_with_context=len(ctx), n_contexts=len(items),
                known_reaction_rank_frac_median=float(np.median(kr)) if kr else None,
                n_candidates=len(rows), tiers={t: sum(d["tier"] == t for d in rows) for t in (1, 2, 3, 4)},
                dead_ends_with_tier1=len({d["metabolite"] for d in rows if d["tier"] == 1}),
                dead_ends_with_tier12=len({d["metabolite"] for d in rows if d["tier"] <= 2}),
                support_top10_supported=float((sup_top > 0).mean()) if len(sup_top) else None,
                support_top10_above_q95=float((sup_top > q95).mean()) if len(sup_top) else None,
                support_low_control_supported=float((ctl_low > 0).mean()) if len(ctl_low) else None,
                support_random_control_supported=float((ctl_rand > 0).mean()) if len(ctl_rand) else None,
                control_q95=q95, p_threshold=p50,
                transport_links_top10=int(sum(d["transport_link"] for d in top)), n_top10=len(top),
                top25=[{k: d[k] for k in ("rank", "tier", "metabolite_name", "compartment", "gap", "reaction_name",
                                         "reaction_subsystem", "p_mean", "n_contexts", "support", "anchor_genes",
                                         "transport_link", "same_species_in")} for d in rows[:25]])
    dump_json(summ, out / "summary.json")
    torch.save(dict(items=[(x[0], x[1], x[2]) for x in items], probs=probs, meta=meta), out / "contexts.pt")
    print(json.dumps({k: v for k, v in summ.items() if k not in ("top25", "calibration")}, indent=1, default=str))
    for d in summ["top25"]:
        print(f"  #{d['rank']:4d} T{d['tier']} {d['metabolite_name'][:28]:28} [{d['compartment'][:10]:10} {d['gap']:12}] -> "
              f"{d['reaction_name'][:42]:42} p {d['p_mean']:.2f} ctx {d['n_contexts']} sup {d['support']:.2f} tr {d['transport_link']} {d['same_species_in'][:20]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
