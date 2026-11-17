"""iML1515 -> tensors.

The E. coli counterpart of build_gem.py, writing the SAME gem.npz / meta.json
schema so `StoichModel`, `validity.py` and `flux_sampler.py` read an E. coli
reference with no code change. The id bridge needs no change either: this
project's E. coli node space already uses BiGG ids verbatim — reactions as
"rxn_PFK", metabolites as bare "10fthf_c", genes as bare "b0002" — which is
exactly what StoichModel.strip expects.

The source is BiGG's iML1515.json rather than a .mat: it is the canonical
distribution of the model, it carries formulas, charges, bounds, subsystems,
EC codes and GPR rules in one file, and it needs no cobra or libsbml to read.
The copy in data/ecoli/ is used if present (it is byte-identical to BiGG's).

    python build_gem_ecoli.py --out-dir gem_ref_ecoli --cache-dir gem_cache

Unlike the Human-GEM builder this one also VERIFIES the assembly by solving
the growth LP with scipy and comparing against the published iML1515 optimum
(0.8770 h^-1 on M9 glucose aerobic, the medium the file's exchange bounds
already encode). A stoichiometry or bounds bug shows up there immediately,
where the shape and nnz counts would not catch it.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from build_gem import GENERIC, element_matrix

URL = "http://bigg.ucsd.edu/static/models/iML1515.json"
LOCAL = Path(__file__).resolve().parents[2] / "data" / "ecoli" / "iML1515.json"

# published optimum for iML1515 on the medium its own exchange bounds encode
PUBLISHED_GROWTH = 0.8770


def fetch(cache: Path, refresh: bool = False) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / "iML1515.json"
    if dst.exists() and not refresh:
        return dst
    if LOCAL.exists() and not refresh:
        print(f"  using {LOCAL}")
        dst.write_bytes(LOCAL.read_bytes())
        return dst
    print(f"  downloading {URL} ...", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        dst.write_bytes(r.read())
    return dst


def gene_tokens(rule: str):
    """Gene ids out of a GPR string. iML1515 rules are `and`/`or`/parens over
    bare locus tags, so splitting on non-id characters is exact — verified by
    the unmatched-token count printed at the end of the build."""
    if not rule:
        return []
    out, cur = [], []
    for ch in rule:
        if ch.isalnum() or ch in "_.-":
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return [t for t in out if t.lower() not in ("and", "or")]


def growth_lp(S, lb, ub, c):
    """max c'v subject to Sv = 0, lb <= v <= ub. Returns (status, objective)."""
    try:
        from scipy.optimize import linprog
    except Exception as e:                                   # pragma: no cover
        return f"scipy unavailable ({type(e).__name__})", float("nan")
    r = linprog(-c, A_eq=S, b_eq=np.zeros(S.shape[0]),
                bounds=list(zip(lb, ub)), method="highs")
    return ("optimal" if r.success else r.message), (-r.fun if r.success else float("nan"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="gem_ref_ecoli")
    ap.add_argument("--cache-dir", default="gem_cache")
    ap.add_argument("--node-id-map", default=None,
                    help="pickle of this project's E. coli node ids; if given, "
                         "the build reports how much of the GEM it covers")
    ap.add_argument("--refresh", action="store_true", help="re-download")
    ap.add_argument("--no-verify", action="store_true", help="skip the growth LP")
    a = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    cache = here / a.cache_dir if not Path(a.cache_dir).is_absolute() else Path(a.cache_dir)
    out = here / a.out_dir if not Path(a.out_dir).is_absolute() else Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    src = fetch(cache, a.refresh)
    print(f"parsing {src.name} ...")
    m = json.loads(src.read_text())

    mets = [x["id"] for x in m["metabolites"]]
    rxns = [r["id"] for r in m["reactions"]]
    genes = [g["id"] for g in m["genes"]]
    met_of = {x: i for i, x in enumerate(mets)}
    gene_of = {g: i for i, g in enumerate(genes)}

    rows, cols, vals = [], [], []
    unmatched = set()
    rgm_r, rgm_c = [], []
    for j, r in enumerate(m["reactions"]):
        for met, coef in r["metabolites"].items():
            rows.append(met_of[met]); cols.append(j); vals.append(float(coef))
        for t in gene_tokens(r.get("gene_reaction_rule")):
            if t in gene_of:
                rgm_r.append(j); rgm_c.append(gene_of[t])
            else:
                unmatched.add(t)

    S = sp.coo_matrix((vals, (rows, cols)), shape=(len(mets), len(rxns)))
    lb = np.array([float(r["lower_bound"]) for r in m["reactions"]])
    ub = np.array([float(r["upper_bound"]) for r in m["reactions"]])
    c = np.array([float(r.get("objective_coefficient", 0.0)) for r in m["reactions"]])
    charges = np.array([float(x.get("charge") or 0.0) for x in m["metabolites"]])

    E, els, form_ok, generic = element_matrix([x.get("formula") for x in m["metabolites"]])

    # Same convention as the human builder: exchange reactions (one
    # participant) and the biomass pseudo-reaction are unbalanced BY DESIGN,
    # and R/X placeholder formulas (ACP-bound acyl chains, Fe-S clusters)
    # cannot be checked. Both are labelled, not counted as failures.
    Sd = sp.csc_matrix(S)
    n_part = np.diff(Sd.indptr)
    atom_imbalance = (np.asarray(np.abs(E @ Sd).max(axis=0)).ravel() if E.size
                      else np.zeros(len(rxns)))
    charge_imbalance = np.abs(charges @ Sd).ravel()
    checkable = np.ones(len(rxns), bool)
    for j in range(len(rxns)):
        idx = Sd.indices[Sd.indptr[j]:Sd.indptr[j + 1]]
        if n_part[j] <= 1 or not form_ok[idx].all() or generic[idx].any():
            checkable[j] = False
    # Both biomass pseudo-reactions have many participants and no R/X, so the
    # filter above lets them through — but they drain ~64 net carbon by
    # construction and are not balance failures. The WT one is not the
    # objective (c = 0), so testing c alone would still leave it in.
    # Human-GEM's builder never hits this: its biomass reaction is written
    # with placeholder formulas and is already excluded.
    is_biomass = np.array([r.startswith("BIOMASS") for r in rxns]) | (c != 0)
    checkable[is_biomass] = False
    balanced = checkable & (atom_imbalance < 1e-6) & (charge_imbalance < 1e-6)

    rgm = sp.coo_matrix((np.ones(len(rgm_r)), (rgm_r, rgm_c)),
                        shape=(len(rxns), len(genes)))
    rgm.sum_duplicates()

    comps = sorted(m["compartments"])
    comp_of = {ch: i for i, ch in enumerate(comps)}

    np.savez_compressed(
        out / "gem.npz",
        S_row=S.row.astype(np.int32), S_col=S.col.astype(np.int32),
        S_val=S.data.astype(np.float64), S_shape=np.array(S.shape),
        lb=lb, ub=ub, c=c,
        E=E, met_charges=charges,
        formula_ok=form_ok, formula_generic=generic,
        rxn_checkable=checkable, rxn_balanced=balanced,
        rxn_atom_imbalance=atom_imbalance, rxn_charge_imbalance=charge_imbalance,
        rgm_row=rgm.row.astype(np.int32), rgm_col=rgm.col.astype(np.int32),
        rxn_n_participants=n_part.astype(np.int32),
    )
    meta = {
        "source": URL,
        "organism": "Escherichia coli str. K-12 substr. MG1655",
        "version": str(m.get("version", "")),
        "rxns": rxns, "mets": mets, "genes": genes,
        "elements": els,
        "subsystems": [r.get("subsystem", "") for r in m["reactions"]],
        "met_names": [x.get("name", "") for x in m["metabolites"]],
        "rxn_names": [r.get("name", "") for r in m["reactions"]],
        "met_comps": [comp_of[x["compartment"]] for x in m["metabolites"]],
        "comps": comps,
        "comp_names": [m["compartments"][ch] for ch in comps],
        "gene_names": [g.get("name", "") for g in m["genes"]],
        "rxn_ec": [r.get("annotation", {}).get("ec-code", []) for r in m["reactions"]],
        "objective": [rxns[i] for i in np.flatnonzero(c)],
        "n_rxns": len(rxns), "n_mets": len(mets), "n_genes": len(genes),
    }
    (out / "meta.json").write_text(json.dumps(meta))

    print(f"\n  S            {S.shape[0]:,} metabolites x {S.shape[1]:,} reactions, "
          f"{S.nnz:,} nonzeros")
    print(f"  reversible   {(lb < 0).sum():,}   objective: {meta['objective']}")
    comp_str = ", ".join(f"{ch}={m['compartments'][ch]}" for ch in comps)
    print(f"  compartments {comp_str}")
    print(f"  elements     {len(els)}: {' '.join(els)}")
    print(f"  GPR          {rgm.nnz:,} reaction-gene links over "
          f"{len(set(rgm.col)):,} genes; {len(unmatched)} unmatched tokens")
    print(f"  mass+charge  {balanced.sum():,} of {checkable.sum():,} checkable "
          f"reactions balance ({100 * balanced.sum() / max(checkable.sum(), 1):.1f}%)")
    print(f"               {(~checkable).sum():,} not checkable "
          f"({(n_part <= 1).sum():,} exchange/sink, "
          f"{int(generic.sum()):,} metabolites with R/X placeholders)")

    if a.node_id_map:
        import pickle
        with open(a.node_id_map, "rb") as f:
            node_ids = pickle.load(f)
        want = set(node_ids)
        hit_r = sum(1 for r in rxns if f"rxn_{r}" in want)
        hit_m = sum(1 for x in mets if x in want)
        hit_g = sum(1 for g in genes if g in want)
        print(f"\n  node-space bridge ({Path(a.node_id_map).name}, {len(want):,} ids)")
        print(f"    reactions   {hit_r:,}/{len(rxns):,} ({hit_r/len(rxns):.0%})")
        print(f"    metabolites {hit_m:,}/{len(mets):,} ({hit_m/len(mets):.0%})")
        print(f"    genes       {hit_g:,}/{len(genes):,} ({hit_g/len(genes):.0%})")

    if not a.no_verify:
        print("\n  verifying by solving the growth LP ...", flush=True)
        status, mu = growth_lp(S.tocsc().toarray(), lb, ub, c)
        ok = np.isfinite(mu) and abs(mu - PUBLISHED_GROWTH) < 5e-3
        print(f"    {status}   mu = {mu:.4f} /h   "
              f"published {PUBLISHED_GROWTH:.4f}   {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            print("    the assembled S/lb/ub do not reproduce the published optimum")
            return 1

    print(f"\n  wrote {out/'gem.npz'} and {out/'meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
