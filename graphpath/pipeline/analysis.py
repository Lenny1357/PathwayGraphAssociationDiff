"""Analyses of a scored run beyond the benchmarks. Each function reads the
run's samples / weights, writes one JSON file into outputs/analysis/<run>/,
and returns what it wrote. Figures are drawn from those files by
`graphpath.pipeline.figures`.

    pathway_stats     distinct / edges / subsystems / composition per arm, vs held-out
    graph_stats       every structural statistic from the adjacency alone
    clustering        triangle / transitivity / square clustering with no type information
    coexpr_support    are drawn edges better supported by propagated co-expression than non-edges
    discovery         novel links with propagated support, against a random control
    embedding_clusters  k-means over each model's entity table, subsystem enrichment, motifs
    attention         where the models attend: shared-reaction pairs, edge following, type flow
    ablation          the masked arm re-sampled under different reveal rules
    threshold_sweep   the continuous arm's edge threshold vs density and precision
    maps              2-D layouts (tissue network, model embedding) with sets marked
"""

from __future__ import annotations

import collections
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch

from graphpath.pipeline import arms as A
from graphpath.pipeline.common import DatasetBundle, RunLayout, arm_label, dump_json, free, key, split_key
from graphpath.pipeline.score import BenchmarkSuite, expression_source


def _by_dataset(samples: dict, dname: str):
    return [(k, s) for k, s in samples.items() if split_key(k)[0] == dname]


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg]); o = allv.argsort()
    rk = np.empty(len(allv)); rk[o] = np.arange(1, len(allv) + 1)
    return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ── 1. pathway statistics (what the arms actually emit) ───────────────────

def pathway_stats(run, data: dict, samples: dict, max_nodes=64, n_draw=6, seed=0) -> dict:
    out = {}
    iu = torch.triu_indices(max_nodes, max_nodes, 1)
    for dname, b in data.items():
        subs = b.subsystems_of()
        rng = np.random.default_rng(seed)
        srcs = _by_dataset(samples, dname) + [("held-out", b.heldout_tensors(n_draw * 4, max_nodes))]
        rows = {}
        for k, (nodes, adj) in srcs:
            sel = rng.choice(len(nodes), size=min(n_draw, len(nodes)), replace=False)
            per = []
            for i in sel:
                real = nodes[i][nodes[i] != 0]
                names = [b.id_to_node[int(t)] for t in real]
                per.append(dict(
                    index=int(i), distinct=int(real.unique().numel()) / max(len(real), 1),
                    edges=int(adj[i][iu[0], iu[1]].sum()),
                    density=float(adj[i][iu[0], iu[1]].float().mean()),
                    subsystems=len({s for n in names for s in subs.get(n, ())}),
                    composition=dict(collections.Counter(b.ntype(n) for n in names))))
            tot = sum(sum(p["composition"].values()) for p in per)
            rows[k if k == "held-out" else arm_label(k)] = dict(
                graphs=per,
                mean={m: float(np.mean([p[m] for p in per])) for m in ("distinct", "edges", "density", "subsystems")},
                composition={t: sum(p["composition"].get(t, 0) for p in per) / max(tot, 1)
                             for t in ("gene", "enzyme", "reaction", "metabolite")})
        out[dname] = rows
    dump_json(out, run.analysis_dir() / "pathway_stats.json")
    return out


# ── 2. graph statistics from the adjacency alone ─────────────────────────

def _graph_stats_one(tok, adj, ntype, id_to_node) -> Optional[dict]:
    keep = np.flatnonzero(np.asarray(tok) != 0)
    n = len(keep)
    if n < 2:
        return None
    Am = (np.asarray(adj)[np.ix_(keep, keep)] > 0); Am = Am | Am.T
    np.fill_diagonal(Am, False)
    iu = np.triu_indices(n, 1)
    e = int(Am[iu].sum()); deg = Am.sum(1)
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(zip(iu[0][Am[iu]], iu[1][Am[iu]]))
    comps = list(nx.connected_components(G))
    H = G.subgraph([v for v in G if G.degree(v) > 0]).copy()
    ids = [int(t) for t in np.asarray(tok)[keep]]
    ta = np.array([ntype(id_to_node[i]) if i in id_to_node else "unknown" for i in ids])
    same = int(sum(ta[a] == ta[c] for a, c in zip(iu[0][Am[iu]], iu[1][Am[iu]])))
    out = dict(n_nodes=n, n_distinct=len(set(ids)), duplicate_frac=1.0 - len(set(ids)) / n,
               n_edges=e, density=e / (n * (n - 1) / 2), degree_mean=float(deg.mean()),
               degree_max=float(deg.max()),
               degree_cv=float(deg.std() / deg.mean()) if deg.mean() > 0 else np.nan,
               isolated_frac=float((deg == 0).mean()), n_components=len(comps),
               largest_comp_frac=max(len(c) for c in comps) / n,
               same_type_edge_frac=(same / e) if e else np.nan)
    if H.number_of_nodes() >= 3:
        out["clustering"] = float(np.mean(list(nx.clustering(H).values())))
        out["transitivity"] = float(nx.transitivity(H))
        sq = nx.square_clustering(H)
        out["square_clustering"] = float(np.mean(list(sq.values()))) if sq else np.nan
    else:
        out["clustering"] = out["transitivity"] = out["square_clustering"] = np.nan
    for t in ("gene", "reaction", "metabolite", "enzyme"):
        out[f"frac_{t}"] = float((ta == t).mean())
    return out


def graph_stats(run, data: dict, samples: dict, max_nodes=64) -> dict:
    out = {}
    for dname, b in data.items():
        srcs = _by_dataset(samples, dname) + [("held-out", b.heldout_tensors(None, max_nodes))]
        rows = {}
        for k, (nodes, adj) in srcs:
            per = [r for r in (_graph_stats_one(nodes[i], adj[i], b.ntype, b.id_to_node)
                               for i in range(len(nodes))) if r is not None]
            if not per:
                continue
            df = pd.DataFrame(per)
            rows[k if k == "held-out" else arm_label(k)] = dict(
                n=len(per), mean=df.mean().to_dict(), sd=df.std().to_dict())
        out[dname] = rows
    dump_json(out, run.analysis_dir() / "graph_stats.json")
    return out


# ── 3. clustering with no reference to type ──────────────────────────────

def _clustering_batch(nodes, adj, pad_id=0) -> dict:
    """{"with_isolates": {...}, "isolates_dropped": {...}} means over graphs."""
    sinks = {"with_isolates": collections.defaultdict(list), "isolates_dropped": collections.defaultdict(list)}
    for b in range(len(nodes)):
        keep = np.flatnonzero(np.asarray(nodes[b]) != pad_id)
        if len(keep) < 3:
            continue
        Am = (np.asarray(adj[b])[np.ix_(keep, keep)] > 0); Am = Am | Am.T
        np.fill_diagonal(Am, False)
        G = nx.Graph(); G.add_nodes_from(range(len(keep)))
        iu = np.triu_indices(len(keep), 1)
        G.add_edges_from(zip(iu[0][Am[iu]], iu[1][Am[iu]]))
        for tag, H in (("with_isolates", G),
                       ("isolates_dropped", G.subgraph([n for n in G if G.degree(n) > 0]).copy())):
            if H.number_of_nodes() < 3:
                continue
            s = sinks[tag]
            s["clustering"].append(float(np.mean(list(nx.clustering(H).values()))))
            s["transitivity"].append(float(nx.transitivity(H)))
            sq = nx.square_clustering(H)
            if sq:
                s["square"].append(float(np.mean(list(sq.values()))))
            s["n_nodes"].append(H.number_of_nodes()); s["n_edges"].append(H.number_of_edges())
    return {tag: {k: float(np.mean(v)) for k, v in s.items()} for tag, s in sinks.items()}


def clustering(run, data: dict, samples: dict, scores: Optional[dict] = None, max_nodes=64) -> dict:
    out = {}
    for dname, b in data.items():
        rows = {arm_label(k): _clustering_batch(n_, a_) for k, (n_, a_) in _by_dataset(samples, dname)}
        rows["held-out"] = _clustering_batch(*b.heldout_tensors(None, max_nodes))
        # cross-check against TopologyBenchmark, which reaches the same number through the typed path
        if scores:
            for k, _ in _by_dataset(samples, dname):
                topo = scores["arms"].get(k, {}).get("topo")
                if topo:
                    rows[arm_label(k)]["topology_benchmark_clustering"] = topo["generated"]["summary"].get("clustering")
        out[dname] = rows
    dump_json(out, run.analysis_dir() / "clustering.json")
    return out


# ── 4. co-expression support for drawn edges ─────────────────────────────

def _genes_of(b: DatasetBundle, toks) -> list:
    return sorted({int(n) for n in toks.flatten().tolist() if n != 0
                   and b.id_to_node.get(int(n)) is not None and b.ntype(b.id_to_node[int(n)]) == "gene"})


def best_tissue(expr, b: DatasetBundle, max_nodes=64, n_graphs=48) -> tuple:
    """The tissue under which the REAL held-out pathways co-express most, so
    the tissue is a property of the data and no arm picks the one that flatters it."""
    ids, _ = b.heldout_tensors(n_graphs, max_nodes)
    pu, pv = [], []
    for row in ids:
        g = [x for x in _genes_of(b, row) if x < expr.vocab_size]
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                pu.append(g[i]); pv.append(g[j])
    if not pu:
        return expr.tissues[0], {}
    u, v = torch.tensor(pu, device=expr.device), torch.tensor(pv, device=expr.device)
    rates = {}
    for t in expr.tissues:
        with torch.no_grad():
            rates[t] = float((expr._lookup(t, u, v) != 0).float().mean())
    return max(rates, key=rates.get), rates


def coexpr_support(run, data: dict, samples: dict, suites: dict, device, max_nodes=64,
                   hops=3, decay=0.5, n_pairs=20_000) -> dict:
    """Every drawn edge against a size-matched sample of generated NON-edges
    from the same graphs, scored by co-expression propagated over the
    metabolic network. AUC and the share above the non-edge 90th percentile."""
    from graphpath.data.propagation import CoExpressionPropagator
    out = {}
    iu = torch.triu_indices(max_nodes, max_nodes, 1)
    for dname, b in data.items():
        device = suites[dname].device          # sparse ops: CPU on Apple silicon
        expr, tissue = expression_source(b, device)
        if expr is None:
            out[dname] = dict(skipped="no co-expression source"); continue
        rates = None
        if dname == "human":
            tissue, rates = best_tissue(expr, b, max_nodes)
        V = b.vocab_size
        from graphpath.data.cofactors import cofactor_ids
        cof = cofactor_ids(b)
        adj = suites[dname].ref.adj_sparse.coalesce()
        if cof:          # no evidence routed through currency metabolites
            ix = adj.indices(); bad = torch.tensor(sorted(cof), device=ix.device)
            keep = ~(torch.isin(ix[0], bad) | torch.isin(ix[1], bad))
            adj = torch.sparse_coo_tensor(ix[:, keep], adj.values()[keep], adj.shape).coalesce()
        prop = CoExpressionPropagator(adj, expr.co_expression[tissue],
                                      vocab_size=V, hops=hops, decay=decay, normalize="sym").fit()
        rows = {}
        srcs = _by_dataset(samples, dname) + [("held-out", b.heldout_tensors(None, max_nodes))]
        for k, (nodes, a_) in srcs:
            eu, ev, nu, nv = [], [], [], []
            for g in range(len(nodes)):
                t = nodes[g]; real = (t != 0)
                if cof:      # cofactor endpoints are not scored, in edges or in the non-edge control
                    real = real & ~torch.isin(t, torch.tensor(sorted(cof), dtype=t.dtype))
                m = (a_[g][iu[0], iu[1]] > 0) & real[iu[0]] & real[iu[1]]
                nm = (~(a_[g][iu[0], iu[1]] > 0)) & real[iu[0]] & real[iu[1]]
                eu += t[iu[0][m]].tolist(); ev += t[iu[1][m]].tolist()
                k_ = int(m.sum())
                if k_ and int(nm.sum()):
                    pick = torch.randperm(int(nm.sum()))[:k_]
                    nu += t[iu[0][nm][pick]].tolist(); nv += t[iu[1][nm][pick]].tolist()
            label = k if k == "held-out" else arm_label(k)
            if len(eu) < 20 or len(nu) < 20:
                rows[label] = dict(too_few_pairs=True, n=len(eu)); continue
            with torch.no_grad():
                se = prop.score(torch.tensor(eu[:n_pairs], device=device),
                                torch.tensor(ev[:n_pairs], device=device)).cpu().numpy()
                sn = prop.score(torch.tensor(nu[:n_pairs], device=device),
                                torch.tensor(nv[:n_pairs], device=device)).cpu().numpy()
            thr = float(np.percentile(sn, 90))
            rows[label] = dict(auc=_auc(se, sn), supported=float((se > thr).mean()),
                               edge_mean=float(se.mean()), nonedge_mean=float(sn.mean()), n=len(se))
        out[dname] = dict(tissue=tissue, tissue_rates=rates, hops=hops, decay=decay, arms=rows,
                          cofactors_excluded=len(cof))
        print(f"  {dname}: tissue {tissue}; " + "  ".join(
            f"{a} AUC {r['auc']:.3f}" for a, r in rows.items() if "auc" in r))
    dump_json(out, run.analysis_dir() / "coexpr_support.json")
    return out


# ── 5. novel links with propagated support ───────────────────────────────

def discovery(run, data: dict, samples: dict, suites: dict, device, max_hops=2, n_keep=8,
              support_json: Optional[dict] = None) -> dict:
    from graphpath.benchmarks import DiscoveryFinder
    out = {}
    for dname, b in data.items():
        device = suites[dname].device
        expr, tissue = expression_source(b, device)
        if expr is None:
            out[dname] = dict(skipped="no co-expression source"); continue
        if support_json and dname in support_json and "tissue" in support_json[dname]:
            tissue = support_json[dname]["tissue"]      # the same network as the support analysis
        from graphpath.data.cofactors import cofactor_ids
        cof = cofactor_ids(b)
        finder = DiscoveryFinder(suites[dname].ref, expr, b.train, node_type_fn=b.ntype,
                                 max_hops=max_hops, exclude=cof,
                                 name_fn=lambda i: b.common(b.id_to_node.get(int(i), i)))
        rows = {}
        for k, (nodes, adj) in _by_dataset(samples, dname):
            arm = arm_label(k)
            try:
                found = finder.find(nodes.to(device), adj.to(device), tissue)
            except Exception as e:  # noqa: BLE001
                rows[arm] = dict(failed=f"{type(e).__name__}: {e}"); continue
            if not found:
                rows[arm] = dict(n=0); continue
            control = np.asarray(finder.control_scores(len(found), tissue, seed=0))
            scores = np.array([d["support"] for d in found])
            s_rate, c_rate = float((scores > 0).mean()), float((control > 0).mean()) if len(control) else float("nan")
            top = sorted([d for d in found if d["support"] > 0], key=lambda d: -d["support"])[:n_keep]
            rows[arm] = dict(n=len(found), supported=s_rate, control=c_rate, excess=s_rate - c_rate,
                             fold=(s_rate / c_rate if c_rate > 0 else None),
                             scores=scores.tolist(), control_scores=control.tolist(),
                             top=[{kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                                   for kk, vv in d.items()} for d in top])
            print(f"  {k:<22} novel {len(found):>6,}  supported {s_rate:.1%}  control {c_rate:.1%}")
        out[dname] = dict(tissue=tissue, max_hops=max_hops, arms=rows, cofactors_excluded=len(cof))
    dump_json(out, run.analysis_dir() / "discovery.json")
    return out


# ── 6. what each model organised its vocabulary into ──────────────────────

def _bh(p, q=0.05):
    p = np.asarray(p, float)
    if p.size == 0:
        return np.zeros(0, bool)
    order = p.argsort(); ranked = p[order]
    passed = ranked <= q * (np.arange(1, p.size + 1) / p.size)
    cut = np.nonzero(passed)[0].max() + 1 if passed.any() else 0
    out = np.zeros(p.size, bool); out[order[:cut]] = True
    return out


def _enrich(labels, ids, subs_of, id_to_node, min_cluster):
    from scipy.stats import hypergeom
    ann = {pos: set(subs_of[id_to_node[int(nid)]]) for pos, nid in enumerate(ids)
           if id_to_node.get(int(nid)) in subs_of and subs_of[id_to_node[int(nid)]]}
    N = len(ann)
    if N == 0:
        return pd.DataFrame(), 0.0
    bg = collections.Counter(s for v in ann.values() for s in v)
    rows = []
    for c in sorted(set(labels)):
        members = [p for p in range(len(labels)) if labels[p] == c]
        ann_m = [p for p in members if p in ann]
        if len(ann_m) < min_cluster:
            continue
        cnt = collections.Counter(s for p in ann_m for s in ann[p])
        for sub, k_ in cnt.items():
            K, n = bg[sub], len(ann_m)
            rows.append(dict(cluster=int(c), subsystem=sub, in_cluster=int(k_), expected=n * K / N,
                             fold=k_ / max(n * K / N, 1e-9), p=float(hypergeom.sf(k_ - 1, N, K, n)),
                             cluster_size=len(members), annotated=len(ann_m)))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["significant"] = _bh(df["p"].values)
    return df, N / max(len(ids), 1)


def _motif(labels, ids, refnet, min_cluster, seed=0, n_null=20):
    idx = refnet.adj_sparse.coalesce().indices().cpu().numpy()
    E = set(zip(idx[0].tolist(), idx[1].tolist()))
    node_ids = [int(v) for v in ids]
    rng = np.random.default_rng(seed)
    def dens(group):
        k = sum(1 for u in group for v in group if u < v and ((u, v) in E or (v, u) in E))
        n = len(group); return k / max(n * (n - 1) / 2, 1)
    out = []
    for c in sorted(set(labels)):
        mem = [node_ids[p] for p in range(len(labels)) if labels[p] == c]
        if len(mem) < min_cluster:
            continue
        obs = dens(mem)
        nl = float(np.mean([dens(list(rng.choice(node_ids, size=len(mem), replace=False))) for _ in range(n_null)]))
        out.append(dict(cluster=int(c), size=len(mem), internal_density=obs, random_density=nl,
                        fold=obs / nl if nl > 0 else None))
    return out


GENERIC_SUBS = {"Isolated", "Miscellaneous", "Exchange/demand reactions", "Transport reactions", "nan", "", "None"}


def embedding_clusters(run, data: dict, suites: dict, device, n_clusters=24, min_cluster=8,
                       fdr_q=0.05, n_exemplars=6, seed=0, top_clusters=8) -> dict:
    """k-means on each model's L2-normalised entity table (cosine geometry),
    hypergeometric subsystem enrichment with BH correction, two controls
    (random embeddings, shuffled labels), reference-network motifs, and a
    2-D projection to draw the clusters on."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from graphpath.plotting.network_maps import _entity_table, embed_model_space

    def cluster(W):
        X = W / np.clip(np.linalg.norm(W, axis=1, keepdims=True), 1e-9, None)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(X)
        sil = (silhouette_score(X, km.labels_, sample_size=min(4000, len(X)), random_state=seed)
               if len(set(km.labels_)) > 1 else float("nan"))
        return km.labels_, float(sil), X

    out = {}
    for dname, b in data.items():
        subs_of = {k: {x for x in v if x not in GENERIC_SUBS} for k, v in b.subsystems_of().items()}
        for d, arm, v in run.all_weights():
            if d != dname:
                continue
            model, _ = A.load_arm(run, d, arm, v, device)
            try:
                W = _entity_table(model).cpu().float().numpy()
            except ValueError as e:
                out[key(d, arm, v)] = dict(skipped=str(e)); free(model); continue
            ids = np.arange(W.shape[0]); keep = ids != 0
            W, ids = W[keep], ids[keep]
            labels, sil, X = cluster(W)
            enr, cov = _enrich(labels, ids, subs_of, b.id_to_node, min_cluster)
            mot = _motif(labels, ids, suites[dname].ref, min_cluster, seed)
            rng = np.random.default_rng(seed)
            lab_r, sil_r, _ = cluster(rng.normal(size=W.shape).astype(np.float32))
            enr_r, _ = _enrich(lab_r, ids, subs_of, b.id_to_node, min_cluster)
            enr_s, _ = _enrich(labels[rng.permutation(len(labels))], ids, subs_of, b.id_to_node, min_cluster)
            n_sig = lambda df: int(df["significant"].sum()) if not df.empty else 0
            sig = enr[enr["significant"]].sort_values(["cluster", "p"]) if not enr.empty else enr
            table = []
            for c in list(dict.fromkeys(sig["cluster"]))[:top_clusters] if not sig.empty else []:
                sub = sig[sig["cluster"] == c].head(3)
                mem = np.flatnonzero(labels == c)
                types = collections.Counter(b.ntype(b.id_to_node[int(ids[p])]) for p in mem if int(ids[p]) in b.id_to_node)
                cen = X[mem].mean(0); cen /= max(np.linalg.norm(cen), 1e-9)
                order = mem[np.argsort(-(X[mem] @ cen))][:n_exemplars]
                mt = next((m for m in mot if m["cluster"] == int(c)), None)
                table.append(dict(cluster=int(c), size=int(len(mem)),
                                  top_subsystems=[dict(subsystem=r.subsystem, fold=r.fold, p=r.p) for r in sub.itertuples()],
                                  types=dict(types), motif_fold=(mt["fold"] if mt else None),
                                  exemplars=[str(b.common(b.id_to_node.get(int(ids[p]), ids[p]))) for p in order]))
            try:
                emb_ids, xy = embed_model_space(model, seed=seed, verbose=False)
                emb = dict(ids=emb_ids.tolist(), xy=np.asarray(xy).tolist())
            except Exception as e:  # noqa: BLE001
                emb = dict(failed=f"{type(e).__name__}: {e}")
            out[key(d, arm, v)] = dict(
                dataset=d, arm=arm, n_clusters=n_clusters, silhouette=sil, silhouette_random=sil_r,
                annotation_coverage=cov, n_sig=n_sig(enr), n_sig_random=n_sig(enr_r), n_sig_shuffled=n_sig(enr_s),
                n_clusters_sig=(int(enr.loc[enr["significant"], "cluster"].nunique()) if not enr.empty else 0),
                n_clusters_sig_random=(int(enr_r.loc[enr_r["significant"], "cluster"].nunique()) if not enr_r.empty else 0),
                n_clusters_sig_shuffled=(int(enr_s.loc[enr_s["significant"], "cluster"].nunique()) if not enr_s.empty else 0),
                significant_clusters=sorted(set(enr.loc[enr["significant"], "cluster"].astype(int))) if not enr.empty else [],
                labels=labels.tolist(), ids=ids.tolist(), motif=mot, table=table, embedding=emb,
                motif_median_fold=float(np.nanmedian([m["fold"] for m in mot if m["fold"] is not None])) if mot else None)
            print(f"  {key(d, arm, v):<22} silhouette {sil:.3f} (random {sil_r:.3f})  enriched pairs "
                  f"{n_sig(enr)} [random {n_sig(enr_r)}, shuffled {n_sig(enr_s)}]")
            free(model)
    dump_json(out, run.analysis_dir() / "clusters.json")
    return out


# ── 7. attention ──────────────────────────────────────────────────────────

def _probe_inputs(b: DatasetBundle, n: int, seed=0):
    """Partially-noised REAL graphs, so attention reflects a realistic state."""
    idx = torch.randperm(len(b.train), generator=torch.Generator().manual_seed(seed))[:n]
    nf, ad = [], []
    for i in idx.tolist():
        a_, n_, msk = b.train[i][:3]
        n_ = torch.as_tensor(n_).long().clone(); n_[~torch.as_tensor(msk).bool()] = 0
        nf.append(n_); ad.append(torch.as_tensor(np.asarray(a_)).long())
    return torch.stack(nf), torch.stack(ad)


def _reaction_partners(b: DatasetBundle, node_toks, adj):
    """(B,N,N) bool: do slots i and j both connect to the same reaction node?"""
    B, N = node_toks.shape
    out = torch.zeros(B, N, N, dtype=torch.bool)
    for g in range(B):
        names = [b.id_to_node.get(int(v)) for v in node_toks[g]]
        rxn = [k for k, nm in enumerate(names) if nm is not None and b.ntype(nm) == "reaction"]
        if rxn:
            inc = (adj[g][:, rxn] > 0).float()
            out[g] = (inc @ inc.T) > 0
    return out


def _shared_reaction_stats(attn, shares, adj, node_toks):
    """Per layer: attention on pairs sharing a reaction vs other unconnected pairs.
    Direct edges are excluded from both groups."""
    real = node_toks != 0
    valid = real[:, :, None] & real[:, None, :]
    eye = torch.eye(attn.shape[-1], dtype=torch.bool)[None]
    base = valid & ~eye & ~(adj > 0)
    pos, neg = base & shares, base & ~shares
    rows = []
    if not (pos.any() and neg.any()):
        return rows, int(pos.sum()), int(neg.sum())
    for l in range(attn.shape[0]):
        a = attn[l].mean(1)
        p, n = a[pos].numpy(), a[neg].numpy()
        rows.append(dict(layer=l, shared=float(p.mean()), other=float(n.mean()),
                         ratio=float(p.mean() / max(n.mean(), 1e-12)), auc=_auc(p, n)))
    return rows, int(pos.sum()), int(neg.sum())


def attention(run, data: dict, device, n_graphs=4, fractions=(0.9, 0.7, 0.5, 0.25, 0.1),
              main_fraction=0.25, keep_slots=48) -> dict:
    """`collect` on partially-noised real graphs at several timesteps. Keeps
    the head-mean attention map of graph 0 (main probe), the type-flow
    matrices, edge alignment and the shared-reaction statistics."""
    from graphpath.benchmarks import attention_entropy, collect, edge_alignment, type_flow
    out = {}
    for dname, b in data.items():
        node_clean, adj_clean = _probe_inputs(b, n_graphs)
        shares = _reaction_partners(b, node_clean, adj_clean)
        nc, ac = node_clean.to(device), adj_clean.to(device)
        for d, arm, v in run.all_weights():
            if d != dname:
                continue
            model, _ = A.load_arm(run, d, arm, v, device)
            if arm == "transition" and device.type == "mps":
                model = model.to("cpu"); nc_, ac_ = node_clean, adj_clean
            else:
                nc_, ac_ = nc, ac
            dev = next(model.parameters()).device
            blocks = list(getattr(getattr(model, "tower", None), "blocks", []) or [])
            if not blocks or arm == "continuous":
                out[key(d, arm, v)] = dict(not_probeable=("forward() consumes embeddings, not tokens"
                                                           if arm == "continuous" else "no tower.blocks to hook"))
                free(model); continue
            T = model.cfg.T
            n_mask, e_mask = getattr(model, "node_mask", None), getattr(model, "edge_mask", None)
            t2t = getattr(model, "token_to_type", None)
            n_ty = int(getattr(model, "n_types", model.cfg.n_node_types))
            per_t = []
            for frac in fractions:
                tp = max(1, int(frac * T))
                tt = torch.full((n_graphs,), tp, dtype=torch.long, device=dev)
                if n_mask is not None:
                    n_in = model.schedule.q_sample(nc_, tt, n_mask); e_in = model.schedule.q_sample(ac_, tt, e_mask)
                else:
                    n_in = model.schedule.q_sample(nc_, tt, model.m_node); e_in = model.schedule.q_sample(ac_, tt, model.m_edge)
                cap = collect(model, n_in, e_in, t=tp, device=dev)
                attn = cap["attn"]
                rows, n_pos, n_neg = _shared_reaction_stats(attn, shares, adj_clean, node_clean)
                align = edge_alignment(cap, e_in.cpu(), edge_mask_id=e_mask) if e_mask is not None else None
                type_ids = t2t[n_in] if t2t is not None else None
                flow = type_flow(attn, type_ids.cpu(), n_ty) if (type_ids is not None and n_ty > 0) else None
                rec = dict(t=tp, frac=frac, shared=rows, n_pos=n_pos, n_neg=n_neg,
                           align=align, entropy=attention_entropy(attn).tolist(),
                           flow=(flow.tolist() if flow is not None else None))
                if abs(frac - main_fraction) < 1e-9:
                    real = (n_in[0] != 0).cpu(); keep = torch.nonzero(real).flatten()[:keep_slots]
                    rec["map_head_mean"] = [attn[l, 0].mean(0)[keep][:, keep].tolist() for l in range(attn.shape[0])]
                    rec["map_head_sd"] = [float(attn[l, 0].std(0)[keep][:, keep].mean()) for l in range(attn.shape[0])]
                per_t.append(rec)
            out[key(d, arm, v)] = dict(T=T, probes=per_t, typed=t2t is not None,
                                       type_names=A.TYPE_ORDER[:n_ty] if n_ty else [])
            main = min(per_t, key=lambda r: abs(r["frac"] - main_fraction))
            best = max(main["shared"], key=lambda r: r["ratio"]) if main["shared"] else None
            print(f"  {key(d, arm, v):<22} t={main['t']}: best shared-reaction ratio "
                  f"{best['ratio'] if best else float('nan'):.2f}")
            free(model)
    dump_json(out, run.analysis_dir() / "attention.json")
    return out


# ── 8. ablation: the masked arm under different reveal rules ──────────────

ABLATION_ARMS = {
    "baseline": {},
    "random order": dict(use_confidence_unmask=False),
    "noisy order": dict(confidence_temperature=3.0),
    "remask to t=0": dict(remask_frac=0.15, remask_stop_frac=0.0),
    "force-reveal 0.9": dict(edge_confidence_reveal_thresh=0.9),
}
ABLATION_METRICS = [("known", "edge_precision", "up"), ("known", "edge_f1", "up"),
                    ("known", "novel_edge_frac", "down"), ("topo", "density", "target"),
                    ("topo", "square_clustering", "target"), ("topo", "degree", "target"),
                    ("topo", "n_components", "target"), ("flux", "participation_precision", "up"),
                    ("flux", "mass_balanced_frac", "up")]


@torch.no_grad()
def ablation(run, data: dict, suites: dict, device, scores: Optional[dict] = None, kind="masked",
             seeds=(0, 1), n=32, max_nodes=64, arms=ABLATION_ARMS, benches=("known", "topo", "flux")) -> dict:
    """Same weights, different reveal rule, re-scored. The second baseline
    seed is the ruler: a difference smaller than the gap between two draws of
    the same arm is not a result."""
    from graphpath.pipeline.common import result_to_dict
    iu = torch.triu_indices(max_nodes, max_nodes, 1)
    out = {}
    for d, arm, v in run.all_weights():
        if d not in data or arm != kind:
            continue
        model, _ = A.load_arm(run, d, arm, v, device)
        runs = {}
        for label, kw in arms.items():
            for sd in (seeds if label == "baseline" else seeds[:1]):
                name = label if sd == seeds[0] else f"{label} (seed {sd})"
                torch.manual_seed(sd)
                ckw, _ = A.clamp_kwargs(arm, v, data[d].sizes(), n, max_nodes, sd, A.pad_scale_for(run, d))
                n_, a_ = model.sample(N=max_nodes, B=n, device=device, **kw, **A._to(ckw, device))
                nodes, adj = n_.long().cpu(), (a_ > 0).long().cpu()
                grab = {}
                def cb(t, nn_, e, grab=grab):
                    if int(t) == model.cfg.T // 2:
                        up = e[0][iu[0], iu[1]].cpu(); dec = up != model.edge_mask
                        grab.update(decided=float(dec.float().mean()),
                                    present_given_decided=float((up[dec] == 1).float().mean()) if dec.any() else None,
                                    nodes_masked=float((nn_[0].cpu() == model.node_mask).float().mean()))
                torch.manual_seed(sd)
                ckw1, _ = A.clamp_kwargs(arm, v, data[d].sizes(), 1, max_nodes, sd, A.pad_scale_for(run, d))
                model.sample(N=max_nodes, B=1, device=device, callback=cb, **kw, **A._to(ckw1, device))
                res = {b_: result_to_dict(suites[d].benches[b_].evaluate(nodes, adj))
                       for b_ in benches if b_ in suites[d].benches}
                runs[name] = dict(results=res, mechanism=grab,
                                  density=float(adj[:, iu[0], iu[1]].float().mean()), seed=sd, kwargs=kw)
                print(f"  {d}/{name:<22} density {runs[name]['density']:.4f}  at t=T/2 "
                      f"{grab.get('decided', float('nan')):.0%} decided")
        out[key(d, arm, v)] = dict(runs=runs, metrics=ABLATION_METRICS, seeds=list(seeds), n=n,
                                   held_out={b_: scores["held-out"][d][b_] for b_ in benches
                                             if scores and b_ in scores["held-out"].get(d, {})})
        free(model)
    dump_json(out, run.analysis_dir() / "ablation.json")
    return out


# ── 9. the continuous arm's decision threshold ───────────────────────────

@torch.no_grad()
def threshold_sweep(run, data: dict, suites: dict, device, thresholds=(0.1, 0.3, 0.5, 0.75, 0.9, 0.952, 0.99),
                    n=24, max_nodes=64, seed=0, kind="continuous") -> dict:
    """Weighted BCE at pos_weight w is minimised at q* = w p / (1 + p (w - 1)),
    so the 0.5 cut keeps pairs with implied p > 0.5/(w - 0.5(w-1)); the sweep
    says where the density crosses the data's."""
    iu = torch.triu_indices(max_nodes, max_nodes, 1)
    out = {}
    for d, arm, v in run.all_weights():
        if d not in data or arm != kind:
            continue
        model, ck = A.load_arm(run, d, arm, v, device)
        w = float(ck.get("cfg", {}).get("adj_pos_weight", 20.0))
        rows = []
        for t in thresholds:
            torch.manual_seed(seed)
            n_, a_ = model.sample(N=max_nodes, B=n, device=device, edge_thresh=t)
            n_, a_ = n_.long().cpu(), (a_ > 0).long().cpu()
            comps, trans, iso = [], [], []
            for g in range(len(n_)):
                keep = np.flatnonzero(n_[g].numpy() != 0)
                if len(keep) < 3:
                    continue
                Am = (a_[g].numpy()[np.ix_(keep, keep)] > 0); Am = Am | Am.T; np.fill_diagonal(Am, False)
                G = nx.from_numpy_array(Am.astype(int))
                comps.append(nx.number_connected_components(G)); iso.append(float(np.mean([G.degree(x) == 0 for x in G])))
                H = G.subgraph([x for x in G if G.degree(x) > 0]); trans.append(nx.transitivity(H) if H.number_of_nodes() > 2 else 0.0)
            ep = None
            if "known" in suites[d].benches:
                ep = suites[d].benches["known"].evaluate(n_, a_).summary.get("edge_precision")
            den = w - t * (w - 1)
            rows.append(dict(thresh=t, implied_p=(t / den if den > 0 else None),
                             density=float(a_[:, iu[0], iu[1]].float().mean()),
                             edges=float(a_[:, iu[0], iu[1]].sum(1).float().mean()),
                             components=float(np.mean(comps)) if comps else None,
                             isolated=float(np.mean(iso)) if iso else None,
                             transitivity=float(np.mean(trans)) if trans else None, edge_precision=ep))
        out[key(d, arm, v)] = dict(adj_pos_weight=w, data_density=data[d].density, rows=rows,
                                   q_star=w * data[d].density / (1 + data[d].density * (w - 1)))
        print(f"  {key(d, arm, v):<22} " + "  ".join(f"{r['thresh']:g}->{r['density']:.4f}" for r in rows))
        free(model)
    dump_json(out, run.analysis_dir() / "threshold.json")
    return out


# ── 10. maps ──────────────────────────────────────────────────────────────

def maps(run, data: dict, samples: dict, suites: dict, device, n_show=6, seed=0) -> dict:
    """2-D layouts with sets marked on them: the tissue co-expression network
    (one layout per dataset, shared by every arm) and each model's own
    embedding space (one layout per model). Stores coordinates, the marked
    sets, and the per-set statistics the library figures print in titles
    (co-expressed pair share, compactness), so the maps can be redrawn."""
    from graphpath.plotting.network_maps import _entity_table, embed_model_space, embed_tissue_network
    out = {}
    g = torch.Generator().manual_seed(seed)

    def node_sets(toks, n=n_show):
        return [sorted({int(x) for x in row.tolist() if int(x) != 0}) for row in toks[:n]
                if len({int(x) for x in row.tolist() if int(x) != 0}) > 1]

    for dname, b in data.items():
        V = b.vocab_size
        train_toks = torch.stack([b.train[i][1].long().clone().masked_fill(~b.train[i][2].bool(), 0)
                                  for i in torch.randperm(len(b.train), generator=g)[:n_show * 3].tolist()])
        rand = lambda sets: [torch.randint(1, V, (len(s),), generator=g).tolist() for s in sets]
        entry = dict(tissue=None, model_space={})
        # the tissue network, once per dataset. Needs only a co-expression source: human has
        # one (used by coexpr_support) but no `coexpr` benchmark, so genes are then picked the
        # way coexpr_support picks them.
        if suites[dname].expr is not None:
            expr, coexp = suites[dname].expr, suites[dname].benches.get("coexpr")
            genes_of = (coexp.extract_gene_nodes if coexp is not None
                        else lambda r, b=b, ex=expr: [x for x in _genes_of(b, r) if x < ex.vocab_size])
            gene_sets = lambda toks, n=n_show: [s for s in (genes_of(r) for r in toks[:n]) if len(s) > 1]
            train_genes = gene_sets(train_toks, len(train_toks))[:n_show]
            tissue = suites[dname].tissue
            try:
                ids, xy = embed_tissue_network(expr, tissue, verbose=False)
                sets = {"training": train_genes, "random": rand(train_genes)}
                for k, (nodes, _) in _by_dataset(samples, dname):
                    sets[arm_label(k)] = gene_sets(nodes)
                def coexpressed(ss):
                    """(share of within-set pairs co-expressed, those pairs). The pairs are what
                    the map draws as lines: two genes the set put together that co-express."""
                    hits = tot = 0; pairs = []
                    for s in ss:
                        gg = torch.tensor(sorted(set(s)), device=expr.device)
                        if gg.numel() < 2:
                            continue
                        p = torch.combinations(gg, r=2)
                        with torch.no_grad():
                            vv = expr._lookup(tissue, p[:, 0], p[:, 1])
                        hits += int((vv != 0).sum()); tot += vv.numel()
                        pairs += p[vv != 0].cpu().tolist()
                    return (hits / tot if tot else None), pairs
                entry["tissue"] = dict(name=tissue, ids=ids.tolist(), xy=np.asarray(xy).tolist(), sets={})
                for k, s in sets.items():
                    hr, pairs = coexpressed(s)
                    entry["tissue"]["sets"][k] = dict(sets=s, pairs_coexpressed=hr, coexpressed_pairs=pairs)
            except Exception as e:  # noqa: BLE001
                entry["tissue"] = dict(failed=f"{type(e).__name__}: {e}")
        # each model's own space
        train_set = node_sets(train_toks, len(train_toks))[:n_show]
        for k, (nodes, _) in _by_dataset(samples, dname):
            arm = arm_label(k)
            gen_set = node_sets(nodes)
            if not gen_set:
                continue
            try:
                model, _ = A.load_arm(run, dname, split_key(k)[1], split_key(k)[2], device)
                ids, xy = embed_model_space(model, seed=seed, verbose=False)
                Wn = torch.nn.functional.normalize(_entity_table(model).cpu().float(), dim=-1)
                gg = torch.Generator().manual_seed(seed)
                rp = torch.randint(0, Wn.shape[0], (4000, 2), generator=gg); rp = rp[rp[:, 0] != rp[:, 1]]
                baseline = float((1 - (Wn[rp[:, 0]] * Wn[rp[:, 1]]).sum(-1)).median())
                def compactness(sets):
                    ds_ = []
                    for s in sets:
                        t = torch.tensor(sorted({int(x) for x in s if int(x) != 0})); t = t[t < Wn.shape[0]]
                        if t.numel() < 2:
                            continue
                        p = torch.combinations(t, r=2); ds_.append(1 - (Wn[p[:, 0]] * Wn[p[:, 1]]).sum(-1))
                    return float(torch.cat(ds_).median() / max(baseline, 1e-9)) if ds_ else None
                sets = {"training": train_set, "random": rand(gen_set), "generated": gen_set}
                entry["model_space"][arm] = dict(ids=ids.tolist(), xy=np.asarray(xy).tolist(),
                                                 types=[b.ntype(b.id_to_node.get(int(i), "")) for i in ids],
                                                 sets={kk: dict(sets=s, compactness=compactness(s)) for kk, s in sets.items()})
                free(model)
            except Exception as e:  # noqa: BLE001
                entry["model_space"][arm] = dict(failed=f"{type(e).__name__}: {e}")
        out[dname] = entry
    dump_json(out, run.analysis_dir() / "maps.json")
    return out
