"""Replogle genome-scale Perturb-seq -> a functional gene network for Human-GEM.

Why this dataset and not another. Perturb-seq gives what co-expression cannot:
a CAUSAL readout. Knock down gene A, measure the whole transcriptome, and the
result is "A affects B", not "A and B move together". The analyses in
data/ecoli/ecoli_coexpression_information.ipynb found pairwise co-expression
barely distinguishes pathway members from random pairs (Cohen's d = 0.13);
perturbation-profile similarity over Human-GEM genes reaches d = 0.29 on the
same kind of test.

Replogle et al. 2022 (figshare 20029387) publish PSEUDOBULK matrices as well
as the raw single-cell data. The pseudobulk is what this needs and it is two
orders of magnitude smaller: 375 MB against 66 GB for the genome-wide
single-cell file.

    python build_perturbseq.py --dataset gwps --out-dir perturb_ref

Coverage against Human-GEM, for the genome-wide K562 screen:

    2,848 Human-GEM genes
      1,626 perturbed (57.1%)     1,445 measured (50.7%)     1,409 both

so the usable object is a 1,626 x 1,445 causal effect matrix over metabolic
genes, from 9,852 perturbations x 8,248 measured genes overall.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

FILES = {                       # figshare file ids, pseudobulk only
    "gwps":      (35773217, "K562_gwps_bulk.h5ad", 374_587_922,
                  "genome-wide K562, 9,852 perturbations"),
    "essential": (35780870, "K562_essential_bulk.h5ad", 79_766_954,
                  "K562 essential genes, 2,057 perturbations"),
    "rpe1":      (35775512, "rpe1_bulk.h5ad", 95_400_000,
                  "RPE1 essential genes, a second cell line"),
}
URL = "https://ndownloader.figshare.com/files/{}"


def fetch(dataset: str, cache: Path, refresh=False) -> Path:
    fid, name, size, _ = FILES[dataset]
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / name
    if dst.exists() and not refresh:
        return dst
    print(f"  downloading {name} ({size/1e6:.0f} MB) ...", flush=True)
    urllib.request.urlretrieve(URL.format(fid), dst)
    return dst


def perturbed_gene(label: str):
    """obs names look like '10023_ZC3H18_P1P2_ENSG00000158545'."""
    for part in str(label).split("_"):
        if part.startswith("ENSG"):
            return part
    return None



def build_response_network(E, pert_names, measured_names, keep=None,
                           top_k=32, min_abs=0.1, chunk=512, verbose=True):
    """The COLUMN-direction network: an edge means two measured genes RESPOND
    alike across the knockdown panel.

    The row-direction network (below) correlates perturbation profiles — two
    genes whose knockdowns break the same thing. This one correlates response
    profiles: for each measured gene, its column of E is how it reacted to
    each of the ~847 knockdowns, and two genes are joined when those reaction
    vectors agree. Co-regulation, read out from interventions instead of from
    observation. They are different relations over different gene sets, and
    the column one covers far more genes because every measured gene has a
    column whereas only perturbed genes have a row.

    SELF-KNOCKDOWNS ARE MASKED, and it matters much more here than it does in
    the row direction. A gene that was both perturbed and measured carries its
    own knockdown at its own row — roughly 9x a typical entry and always
    negative. Against a row of 8,175 readouts that is one outlier among many;
    against a COLUMN of 847 it dominates the variance, and standardising the
    column would then be standardising by the intervention rather than by the
    biology.

    A masked cell is centred to zero so it contributes nothing to any dot
    product, and the denominator is left at n_perturbations for every pair —
    the same approximation PerturbationBenchmark makes, and with at most one
    masked cell per column it moves a correlation by ~0.1%.

    Returns (row, col, val, names, n_self_masked). row/col index into `names`.
    """
    import numpy as np

    idx = ([j for j, g in enumerate(measured_names) if g in keep]
           if keep is not None else list(range(len(measured_names))))
    names = [measured_names[j] for j in idx]
    if len(names) < 2:
        return (np.zeros(0, np.int32), np.zeros(0, np.int32),
                np.zeros(0, np.float32), names, 0)

    R = np.asarray(E, dtype=np.float64)[:, idx].copy()
    pert_at = {g: i for i, g in enumerate(pert_names)}
    n_self = 0
    for j, g in enumerate(names):
        i = pert_at.get(g)
        if i is not None:
            R[i, j] = np.nan
            n_self += 1

    with np.errstate(invalid="ignore"):
        mu = np.nanmean(R, axis=0, keepdims=True)
        sd = np.nanstd(R, axis=0, keepdims=True)
    Z = np.where(sd > 1e-9, (R - mu) / np.clip(sd, 1e-9, None), 0.0)
    Z = np.nan_to_num(Z, nan=0.0)

    n, denom = len(names), Z.shape[0]
    k = max(1, min(top_k, n - 1))
    # chunked: the full matrix is n^2, which at 8,175 measured genes is 534 MB
    # in float64 before anything is done with it
    rr, cc, vv = [], [], []
    for s0 in range(0, n, chunk):
        s1 = min(s0 + chunk, n)
        S = (Z[:, s0:s1].T @ Z) / denom
        for a_i in range(s1 - s0):
            row = S[a_i]
            row[s0 + a_i] = 0.0                     # no self-edge
            sel = np.argpartition(-np.abs(row), k)[:k]
            for j in sel:
                v = float(row[j])
                if abs(v) >= min_abs:
                    rr.append(s0 + a_i); cc.append(int(j)); vv.append(v)
    r = np.array(rr, np.int32); c = np.array(cc, np.int32)
    v = np.array(vv, np.float32)
    if verbose:
        deg = np.bincount(r, minlength=n)
        print(f"  response net    {len(v):,} edges over {n:,} measured genes, "
              f"|r| >= {min_abs}, top {top_k}")
        print(f"                  {n_self} self-knockdown cells masked; "
              f"degree median {int(np.median(deg))}, max {int(deg.max())}, "
              f"isolated {int((deg == 0).sum())}")
    return r, c, v, names, n_self


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="gwps", choices=sorted(FILES))
    ap.add_argument("--out-dir", default="perturb_ref")
    ap.add_argument("--cache-dir", default="pseudobulk")
    ap.add_argument("--gem-meta", default="../gem/gem_ref/meta.json")
    ap.add_argument("--top-k", type=int, default=32,
                    help="partners kept per gene in the similarity network")
    ap.add_argument("--min-abs", type=float, default=0.1)
    ap.add_argument("--response-all", action="store_true",
                    help="build the response-direction network over ALL "
                         "measured genes rather than only Human-GEM ones "
                         "(8,175 vs ~1,445 nodes; ~534 MB peak instead of "
                         "~17 MB)")
    ap.add_argument("--max-p", type=float, default=0.05,
                    help="keep only perturbations with a significant "
                         "transcriptional phenotype (energy_test_p_value). "
                         "1.0 keeps everything.")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args(argv)

    import anndata as ad

    here = Path(__file__).resolve().parent
    out = here / a.out_dir if not Path(a.out_dir).is_absolute() else Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = fetch(a.dataset, here / a.cache_dir, a.refresh)

    print(f"reading {path.name} ...", flush=True)
    A = ad.read_h5ad(path)
    print(f"  {A.shape[0]:,} perturbations x {A.shape[1]:,} measured genes")

    gm = Path(a.gem_meta)
    gm = gm if gm.is_absolute() else (here / gm)
    gem = json.loads(gm.read_text())
    G = list(dict.fromkeys(gem["genes"]))
    Gset = set(G)

    pert_row = {}
    for i, lbl in enumerate(A.obs_names):
        g = perturbed_gene(lbl)
        if g is not None:
            pert_row.setdefault(g, i)
    measured = {g: j for j, g in enumerate(A.var_names)}

    # Keep only knockdowns that DID something. Replogle score every
    # perturbation for whether its transcriptome differs from control
    # (`energy_test_p_value`); roughly half do not, and an unaffected
    # knockdown's profile is measurement noise. Left in, those genes correlate
    # with nothing and sit isolated in the network — 1,291 of 1,626 were
    # isolated before this filter, 847 metabolic genes survive it.
    pval = (A.obs["energy_test_p_value"].values.astype(float)
            if "energy_test_p_value" in A.obs else None)
    metab_pert = [g for g in G if g in pert_row]
    n_before = len(metab_pert)
    if pval is not None and a.max_p < 1.0:
        metab_pert = [g for g in metab_pert if pval[pert_row[g]] < a.max_p]
    print(f"\nHuman-GEM: {len(G):,} genes")
    print(f"  perturbed here  {n_before:>6,}  ({n_before/len(G):.1%})")
    print(f"  ...with a significant phenotype (p < {a.max_p})  "
          f"{len(metab_pert):,}  <- the usable set")
    print(f"  measured here   {len(Gset & set(measured)):>6,}  "
          f"({len(Gset & set(measured))/len(G):.1%})")

    # ── the effect matrix, metabolic genes only ────────────────────────────
    #
    # 6,881 entries are +inf — measured genes whose control expression is ~0,
    # so the ratio is undefined. They sit in only 73 of the 8,248 readout
    # columns, so the fix is to DROP those columns rather than patch the
    # values. Note np.nan_to_num is the wrong tool here: its default maps inf
    # to the dtype maximum (3.4e38 in float32), which then overflows the
    # moment anything squares it.
    Xall = np.asarray(A.X)
    bad_cols = ~np.isfinite(Xall).all(0)
    good = np.flatnonzero(~bad_cols)
    if bad_cols.any():
        print(f"  dropping {int(bad_cols.sum())} readout genes with non-finite "
              f"values ({int((~np.isfinite(Xall)).sum()):,} entries)")
    rows = np.array([pert_row[g] for g in metab_pert])
    E = np.asarray(Xall[np.ix_(rows, good)], dtype=np.float64)
    measured_names = [str(x) for x in np.asarray(A.var_names)[good]]
    assert np.isfinite(E).all(), "non-finite values survived the column filter"
    np.save(out / "effect_matrix.npy", E.astype(np.float32))

    # ── the functional network: correlate perturbation PROFILES ────────────
    #
    # Two genes whose knockdowns produce the same transcriptional consequence
    # are functionally related — this is what the Replogle paper is built on,
    # and it recovers complexes and pathways where correlation of expression
    # does not. The profile is taken over ALL measured genes, not just the
    # metabolic ones, because the readout is more informative the wider it is.
    # float64 throughout: the raw values span orders of magnitude and the
    # standardisation squares them
    sd = E.std(1, keepdims=True)
    Z = np.where(sd > 1e-9, (E - E.mean(1, keepdims=True)) / sd.clip(1e-9), 0.0)
    S = (Z @ Z.T) / Z.shape[1]
    np.fill_diagonal(S, 0.0)

    k = min(a.top_k, S.shape[1] - 1)
    idx = np.argpartition(-np.abs(S), k, axis=1)[:, :k]
    r, c, v = [], [], []
    for i in range(S.shape[0]):
        for j in idx[i]:
            if abs(S[i, j]) >= a.min_abs:
                r.append(i); c.append(int(j)); v.append(float(S[i, j]))
    r, c, v = np.array(r, np.int32), np.array(c, np.int32), np.array(v, np.float32)
    np.savez_compressed(out / "similarity.npz", row=r, col=c, val=v)

    # ── the OTHER direction: correlate RESPONSE profiles (columns) ─────────
    #
    # Restricted to Human-GEM genes by default, so the nodes are entities the
    # model actually has in its vocabulary — the column analogue of building
    # the row network over `metab_pert` rather than all 9,852 perturbations.
    # --response-all lifts that and uses every measured gene.
    rr, rc, rv, resp_names, n_self = build_response_network(
        E, metab_pert, measured_names,
        keep=None if a.response_all else Gset,
        top_k=a.top_k, min_abs=a.min_abs)
    np.savez_compressed(out / "similarity_response.npz",
                        row=rr, col=rc, val=rv,
                        names=np.array(resp_names, dtype=object),
                        allow_pickle=True)

    meta = {"dataset": a.dataset, "source": URL.format(FILES[a.dataset][0]),
            "genes": metab_pert, "n_measured": int(E.shape[1]),
            "measured_genes": measured_names,
            "top_k": a.top_k, "min_abs": a.min_abs,
            # the column-direction network. Its nodes are a DIFFERENT gene set
            # from `genes`, so row/col in similarity_response.npz index into
            # `response_genes`, never into `genes`.
            "response_genes": resp_names,
            "response_all_measured": bool(a.response_all),
            "response_n_self_masked": int(n_self),
            "note": "similarity.npz row/col index into `genes`; val is "
                    "perturbation-profile correlation (two genes whose "
                    "knockdowns break the same thing). "
                    "similarity_response.npz row/col index into "
                    "`response_genes`; val is response-profile correlation "
                    "(two genes that react alike across the knockdown panel), "
                    "with self-knockdown cells masked. "
                    "effect_matrix.npy is (len(genes), n_measured)."}
    (out / "meta.json").write_text(json.dumps(meta))

    deg = np.bincount(r, minlength=len(metab_pert))
    print(f"\n  effect matrix   {E.shape[0]:,} x {E.shape[1]:,}  -> effect_matrix.npy")
    print(f"  similarity net  {len(v):,} edges over {len(metab_pert):,} genes, "
          f"|r| >= {a.min_abs}, top {a.top_k}")
    print(f"  degree          median {int(np.median(deg))}, max {deg.max()}, "
          f"isolated {(deg == 0).sum()}")
    print(f"\n  wrote {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
