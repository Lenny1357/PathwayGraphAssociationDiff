"""Which evidence source tracks pathway membership, and how much a tissue's
co-expression network says at all. Properties of the DATA: no model is loaded.

    signal_comparison   given a GENE PAIR, does a source say whether the two share a
                        metabolic subsystem? AUC, Cohen's d, enrichment by score
                        percentile, precision at coverage — for E. coli correlation,
                        E. coli iModulons, human co-expression, human Perturb-seq
                        (row = shared consequence, column = co-response)
    coexpr_information  does within-pathway co-expression in tissue T carry information
                        about the pathway's activity in T (MI vs a permutation null), and
                        how different are the tissue networks from each other

Both write one JSON file; figures are drawn from it.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graphpath import paths
from graphpath.pipeline.common import dump_json

COVS = np.array([0.005, 0.01, 0.02, 0.05, 0.10, 0.25])
N_BINS = 20


def score_source(score, y, name, n_bins=N_BINS, covs=COVS, keep=60_000) -> dict:
    """AUC, Cohen's d, per-decile enrichment, precision at coverage, and a
    bounded subsample of the raw scores so the distributions can be drawn."""
    score = np.asarray(score, dtype=float); ok = np.isfinite(score)
    s, lab = score[ok], np.asarray(y, bool)[ok]
    base = float(lab.mean())
    o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    n1 = int(lab.sum())
    auc = float((rk[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(lab) - n1)))
    d = float((s[lab].mean() - s[~lab].mean()) / s.std()) if s.std() > 0 else None
    q = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    enr = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (s >= lo) & (s < hi)
        enr.append(float(lab[m].mean() / base) if m.sum() > 50 else None)
    prec = [float(lab[np.argpartition(-s, max(1, int(len(s) * c)))[:max(1, int(len(s) * c))]].mean() / base)
            for c in covs]
    ties = float(np.bincount(np.unique(s, return_inverse=True)[1]).max() / len(s)) if len(s) else None
    pick = np.random.default_rng(0).choice(len(s), min(len(s), keep), replace=False)
    return dict(name=name, auc=auc, d=d, base=base, n=int(len(s)), tie_frac=ties,
                sample_score=s[pick].tolist(), sample_label=lab[pick].tolist(), enrichment=enr,
                percentile=np.linspace(100 / n_bins / 2, 100 - 100 / n_bins / 2, n_bins).tolist(),
                precision=prec, coverage=covs.tolist(), computed=True)


def _subsystems(edges_csv) -> dict:
    df = pd.read_csv(edges_csv)
    col = next(c for c in df.columns if c.lower() == "subsystem")
    sub = collections.defaultdict(set)
    for c in ("source", "target"):
        for n, s in zip(df[c].astype(str).str.strip(), df[col]):
            sub[n].add(str(s))
    return sub


def _pairs(genes, n_pairs, seed):
    rs = np.random.default_rng(seed)
    ia, ib = rs.choice(genes, n_pairs), rs.choice(genes, n_pairs)
    m = ia != ib
    return ia[m], ib[m]


def signal_comparison(out_dir: Path, n_pairs=300_000, seed=0) -> dict:
    """One pair sample per organism, scored by every source whose data is present."""
    sources, log = {}, []
    # ── E. coli ─────────────────────────────────────────────────────────
    ecoli = paths.DATA / "ecoli"
    if (paths.ECOLI_EXPR / "meta.json").exists():
        import pickle
        from graphpath.data.expression_ecoli import EcoliExpressionLoader
        from graphpath.data.node_types import ecoli_node_type
        nid = pickle.load(open(paths.ECOLI_REF / "ecoli_node_id_map.pkl", "rb")); id2n = {v: k for k, v in nid.items()}
        sub_of = _subsystems(paths.ECOLI_REF / "ecoli_gem_reaction_node_edges.csv")
        genes = [i for i, n in id2n.items() if ecoli_node_type(n) == "gene" and sub_of.get(n)]
        ia, ib = _pairs(genes, n_pairs, seed)
        y = np.array([bool(sub_of[id2n[int(u)]] & sub_of[id2n[int(v)]]) for u, v in zip(ia, ib)])
        log.append(f"E. coli: {len(genes):,} genes, {len(y):,} pairs, P(same subsystem) = {y.mean():.4f}")
        expr = EcoliExpressionLoader(str(paths.ECOLI_EXPR), device="cpu")
        tu, tv = torch.tensor(ia), torch.tensor(ib)
        vals = [expr._lookup(t, tu, tv).float().numpy() for t in expr.tissues]
        sources["E. coli correlation"] = score_source(np.nanmean(np.stack(vals), 0), y, "E. coli correlation")
        M_CSV = ecoli / "precise1k/data/precise1k/M.csv"
        if M_CSV.exists():
            Mdf = pd.read_csv(M_CSV, index_col=0)
            pos = {g: k for k, g in enumerate(Mdf.index)}
            Mn = Mdf.values / np.linalg.norm(Mdf.values, axis=1, keepdims=True).clip(1e-9)
            hi = np.array([pos.get(id2n[int(u)], -1) for u in ia]); hj = np.array([pos.get(id2n[int(v)], -1) for v in ib])
            both = (hi >= 0) & (hj >= 0)
            cos = np.full(len(y), np.nan); cos[both] = (Mn[hi[both]] * Mn[hj[both]]).sum(1)
            sources["E. coli iModulon"] = score_source(cos, y, "E. coli iModulon")
            log.append(f"iModulons: {Mdf.shape[1]} modules, {both.mean():.1%} of pairs have both genes")
        else:
            log.append("E. coli iModulon skipped (precise1k M.csv missing)")
    else:
        log.append("E. coli sources skipped (ecoli_expr_nodespace missing)")
    # ── human ───────────────────────────────────────────────────────────
    PT = paths.PT_FILES
    human_ready = (PT / "human_gem_sampled_graphs.pkl").exists()
    hy = hia = hib = id2nh = hsub = None
    if human_ready:
        import pickle
        from graphpath.data.expression import ExpressionLoader
        nidh = pickle.load(open(PT / "node_id_map.pkl", "rb")); id2nh = {v: k for k, v in nidh.items()}
        hsub = _subsystems(PT / "human_gem_reaction_node_edges2.csv")
        hgenes = [i for i, n in id2nh.items() if str(n).startswith("ENSG") and hsub.get(n)]
        hia, hib = _pairs(hgenes, n_pairs, seed)
        hy = np.array([bool(hsub[id2nh[int(u)]] & hsub[id2nh[int(v)]]) for u, v in zip(hia, hib)])
        log.append(f"human: {len(hgenes):,} genes, {len(hy):,} pairs, P(same subsystem) = {hy.mean():.4f}")
        hexpr = ExpressionLoader(path=str(PT), device="cpu")
        tu, tv = torch.tensor(hia), torch.tensor(hib)
        vals = [hexpr._lookup(t, tu, tv).float().numpy() for t in hexpr.tissues]
        sources["human co-expression"] = score_source(np.nanmean(np.stack(vals), 0), hy, "human co-expression")
    else:
        # the values measured in an earlier run, flagged so a stored number is never mistaken for a computed one
        sources["human co-expression"] = dict(name="human co-expression", auc=None, d=None, base=0.1027, n=0,
                                              percentile=None, enrichment=None, precision=[1.6, 1.4, 1.3, 1.2, 1.1, 1.1],
                                              coverage=COVS.tolist(), computed=False)
        log.append(f"human co-expression: {PT} not reachable — recorded values used")
    PERT = paths.PERTURB_REF
    if (PERT / "meta.json").exists():
        meta = json.loads((PERT / "meta.json").read_text())
        E = np.load(PERT / "effect_matrix.npy").astype(np.float64)
        Z = E - E.mean(1, keepdims=True); Z = Z / Z.std(1, keepdims=True).clip(1e-12)
        prow = {g: i for i, g in enumerate(meta["genes"])}
        if human_ready:
            pi = np.array([prow.get(id2nh[int(u)], -1) for u in hia]); pj = np.array([prow.get(id2nh[int(v)], -1) for v in hib])
            both = (pi >= 0) & (pj >= 0)
            sc = np.full(len(hy), np.nan); sc[both] = (Z[pi[both]] * Z[pj[both]]).sum(1) / E.shape[1]
            sources["human Perturb-seq"] = score_source(sc, hy, "human Perturb-seq")
            gsub = {n: hsub[n] for n in hsub if str(n).startswith("ENSG")}
        else:
            gm = json.loads((paths.GEM_REF / "meta.json").read_text())
            import scipy.sparse as sp
            z = np.load(paths.GEM_REF / "gem.npz")
            RG = sp.coo_matrix((np.ones(len(z["rgm_row"])), (z["rgm_row"], z["rgm_col"])),
                               shape=(gm["n_rxns"], gm["n_genes"])).tocsc()
            rxn_sub = dict(zip(gm["rxns"], gm["subsystems"]))
            gsub = collections.defaultdict(set)
            for j, gname in enumerate(gm["genes"]):
                for r in RG.indices[RG.indptr[j]:RG.indptr[j + 1]]:
                    s = rxn_sub.get(gm["rxns"][r])
                    if s:
                        gsub[gname].add(str(s))
            cand = [g for g in meta["genes"] if gsub.get(g)]
            pa, pb = _pairs(np.arange(len(cand)), n_pairs, seed)
            py = np.array([bool(gsub[cand[u]] & gsub[cand[v]]) for u, v in zip(pa, pb)])
            sc = (Z[[prow[cand[u]] for u in pa]] * Z[[prow[cand[v]] for v in pb]]).sum(1) / E.shape[1]
            sources["human Perturb-seq"] = score_source(sc, py, "human Perturb-seq")
        # the column view: do genes X and Y move together across the knockdowns?
        C = E.astype(np.float32); C = C - C.mean(0, keepdims=True); C = (C / C.std(0, keepdims=True).clip(1e-6)).astype(np.float32)
        mcol = {g: j for j, g in enumerate(meta["measured_genes"])}
        cand = [g for g in meta["measured_genes"] if gsub.get(g)]
        a, b = _pairs(np.arange(len(cand)), n_pairs, seed)
        yc = np.array([bool(gsub[cand[u]] & gsub[cand[v]]) for u, v in zip(a, b)])
        ca = np.array([mcol[cand[u]] for u in a]); cb = np.array([mcol[cand[v]] for v in b])
        sc = np.empty(len(ca), dtype=np.float32)
        for i in range(0, len(ca), 20_000):
            sl = slice(i, i + 20_000)
            sc[sl] = np.einsum("ij,ij->j", C[:, ca[sl]], C[:, cb[sl]]) / C.shape[0]
        sources["human Perturb-seq (co-response)"] = score_source(sc, yc, "human Perturb-seq (co-response)")
        log.append(f"Perturb-seq: {len(prow):,} perturbed, {C.shape[1]:,} measured genes")
    else:
        log.append("Perturb-seq skipped (perturb_ref missing)")
    out = dict(sources=sources, log=log, n_pairs=n_pairs, seed=seed, covs=COVS.tolist())
    dump_json(out, Path(out_dir) / "signal_comparison.json")
    for l in log:
        print("  " + l)
    return out


# ── co-expression information content (human tissue tables) ──────────────

def _mi_binned(x, y, bins=8):
    xb = np.quantile(x, np.linspace(0, 1, bins + 1)); xb[-1] += 1e-9
    yb = np.quantile(y, np.linspace(0, 1, bins + 1)); yb[-1] += 1e-9
    c, _, _ = np.histogram2d(x, y, bins=[np.unique(xb), np.unique(yb)])
    p = c / c.sum(); px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True); nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


def coexpr_information(out_dir: Path, min_genes=4, n_path=1500, n_perm=200, top_k=200_000,
                       n_pairs=200_000, seed=0, device="cpu") -> dict:
    from graphpath.data.expression import ExpressionLoader
    from graphpath.data.loaders import load_graphs
    from graphpath.data.dataset import PathwayGraphDataset
    from graphpath.data.node_types import human_node_type
    graphs, nid = load_graphs("human")
    dataset = PathwayGraphDataset(node_id_map=nid, graphs=graphs, max_nodes=64, device=device)
    expr = ExpressionLoader(path=str(paths.PT_FILES), device=device, verbose=False)
    tissues, T = expr.tissues, len(expr.tissues)
    rng = np.random.default_rng(seed)
    # 1. pathway x tissue matrices, z-scored across tissues per pathway
    rows, sizes, pairs = [], [], []
    for i in torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist():
        _, nf, mask = dataset[i][:3]
        ids = torch.as_tensor(nf)[torch.as_tensor(mask).bool()].long()
        genes = torch.tensor(sorted({int(v) for v in ids.tolist() if human_node_type(dataset.id_to_node[int(v)]) == "gene"}))
        if genes.numel() < min_genes:
            continue
        rows.append(expr.within_list_co_exp(genes.to(device)).cpu().numpy()); sizes.append(int(genes.numel()))
        pairs.append(genes.numel() * (genes.numel() - 1) // 2)
        if len(rows) >= n_path:
            break
    S = np.stack(rows)
    def z(M):
        sd = M.std(1, keepdims=True); return (M - M.mean(1, keepdims=True)) / np.where(sd > 1e-12, sd, np.nan)
    ACT, COEX = z(S[:, :, 1]), z(S[:, :, 3])
    keep = ~(np.isnan(ACT).any(1) | np.isnan(COEX).any(1)); ACT, COEX = ACT[keep], COEX[keep]
    x, y = COEX.ravel(), ACT.ravel()
    def permuted(M):
        o = M.copy()
        for i in range(o.shape[0]):
            o[i] = o[i, rng.permutation(o.shape[1])]
        return o
    obs = _mi_binned(x, y); null = np.array([_mi_binned(permuted(COEX).ravel(), y) for _ in range(n_perm)])
    from scipy.stats import pearsonr, spearmanr
    per_t = [_mi_binned(COEX[:, t], ACT[:, t]) for t in range(T)]
    per_t_null = [float(np.mean([_mi_binned(COEX[rng.permutation(len(COEX)), t], ACT[:, t]) for _ in range(30)])) for t in range(T)]
    # size-matched random gene sets
    gene_ids = np.array([i for i, n in dataset.id_to_node.items() if human_node_type(n) == "gene"])
    R = np.stack([expr.within_list_co_exp(torch.tensor(rng.choice(gene_ids, int(s), replace=False)).to(device)).cpu().numpy()
                  for s in np.array(sizes)[keep][:400]])
    rA, rC = z(R[:, :, 1]), z(R[:, :, 3]); ok = ~(np.isnan(rA).any(1) | np.isnan(rC).any(1))
    rmi = _mi_binned(rC[ok].ravel(), rA[ok].ravel())
    rnull = np.array([_mi_binned(permuted(rC[ok]).ravel(), rA[ok].ravel()) for _ in range(50)])
    # 2. how different are the tissue networks
    keys, vals = {}, {}
    for t in tissues:
        sp_ = expr.co_expression[t].coalesce(); u, v = sp_.indices()
        k = torch.minimum(u, v).long() * dataset.vocab_size + torch.maximum(u, v).long()
        k, inv = torch.unique(k, return_inverse=True)
        w = torch.zeros(k.numel(), device=sp_.device, dtype=sp_.values().dtype); w.scatter_reduce_(0, inv, sp_.values(), reduce="amax")
        keys[t], vals[t] = k.cpu().numpy(), w.cpu().numpy()
    JAC, WR, TOPK = np.eye(T), np.eye(T), np.eye(T)
    tops = {t: set(keys[t][np.argpartition(-np.abs(vals[t]), min(top_k, len(vals[t])) - 1)[:min(top_k, len(vals[t]))]].tolist()) for t in tissues}
    for i in range(T):
        for j in range(i + 1, T):
            a, b = tissues[i], tissues[j]
            inter, ia, ib = np.intersect1d(keys[a], keys[b], return_indices=True)
            union = len(keys[a]) + len(keys[b]) - len(inter)
            JAC[i, j] = JAC[j, i] = len(inter) / union if union else 0.0
            if len(inter) > 2:
                WR[i, j] = WR[j, i] = np.corrcoef(vals[a][ia], vals[b][ib])[0, 1]
            s, u_ = tops[a] & tops[b], tops[a] | tops[b]
            TOPK[i, j] = TOPK[j, i] = len(s) / len(u_) if u_ else 0.0
    all_keys = np.concatenate([keys[t] for t in tissues]); uniq, counts = np.unique(all_keys, return_counts=True)
    share = (np.bincount(counts, minlength=T + 1)[1:] / len(uniq)).tolist()
    # 3. the ceiling: MI(tissue ; value)
    sel = rng.choice(uniq, min(n_pairs, len(uniq)), replace=False)
    Vm = np.zeros((len(sel), T))
    for i, t in enumerate(tissues):
        idx = np.clip(np.searchsorted(keys[t], sel), 0, len(keys[t]) - 1); hit = keys[t][idx] == sel
        Vm[hit, i] = vals[t][idx[hit]]
    value, tissue_lab = Vm.ravel(), np.tile(np.arange(T), len(sel))
    edges = np.quantile(value, np.linspace(0, 1, 21)); edges[-1] += 1e-9
    vb = np.clip(np.digitize(value, np.unique(edges)) - 1, 0, None)
    def mi_tv(tl):
        c = np.zeros((T, vb.max() + 1)); np.add.at(c, (tl, vb), 1); p = c / c.sum()
        px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True); nz = p > 0
        return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())
    mi = mi_tv(tissue_lab); mi_null = np.array([mi_tv(rng.permutation(tissue_lab)) for _ in range(30)])
    between, within = Vm.mean(0).var(), Vm.var(1).mean()
    off = ~np.eye(T, dtype=bool)
    out = dict(
        tissues=tissues, n_pathways=int(len(ACT)), genes_per_pathway=np.array(sizes)[keep].tolist(),
        activity=dict(mi=obs, mi_null_mean=float(null.mean()), mi_null_sd=float(null.std()), null=null.tolist(),
                      spearman=float(spearmanr(x, y).statistic), pearson=float(pearsonr(x, y).statistic),
                      per_tissue_mi=per_t, per_tissue_null=per_t_null,
                      random_sets=dict(mi=rmi, null_mean=float(rnull.mean())),
                      scatter=dict(x=x[:20000].tolist(), y=y[:20000].tolist())),
        similarity=dict(jaccard=JAC.tolist(), weight_r=WR.tolist(), topk_jaccard=TOPK.tolist(), top_k=top_k,
                        pairs_in_k_tissues=share, means=dict(jaccard=float(JAC[off].mean()), weight_r=float(WR[off].mean()),
                                                              topk=float(TOPK[off].mean()))),
        ceiling=dict(mi_tissue_value=mi, null_mean=float(mi_null.mean()), null_sd=float(mi_null.std()),
                     H_tissue=math.log(T), variance_between=float(between), variance_within=float(within),
                     share_in_all_tissues=share[-1], n_pairs=int(len(sel))))
    dump_json(out, Path(out_dir) / "coexpr_information.json")
    print(f"  MI(coexpr; activity) {obs:.4f} vs null {null.mean():.4f}   MI(tissue; value) {mi:.5f} vs null {mi_null.mean():.5f}")
    return out
