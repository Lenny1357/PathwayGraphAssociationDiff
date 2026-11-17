"""Human-GEM -> tensors.

Downloads the model from SysBioChalmers/Human-GEM and writes everything the
torch FBA layer needs as plain arrays: the stoichiometric matrix in COO, the
flux bounds, the objective, the element-composition matrix for mass balance,
and the id lists that bridge onto this project's node-id space.

The .mat is the source rather than the SBML: 0.78 MB against 43 MB, it carries
S / lb / ub / c / rxnGeneMat / metFormulas / metCharges already assembled, and
scipy reads it without cobra or libsbml.

    python build_gem.py --out-dir gem_ref --cache-dir gem_cache
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

RAW = "https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/main/model"
FILES = ["Human-GEM.mat", "reactions.tsv", "metabolites.tsv", "genes.tsv"]

# Placeholder atoms in Human-GEM formulas: R/X mark generic side chains on
# lipid and glycan species, so those reactions cannot be mass-checked and are
# reported separately rather than counted as failures.
GENERIC = {"R", "X"}
_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def as_str_list(v) -> np.ndarray:
    """
    MATLAB cell arrays come back ragged: an empty cell is an empty ndarray,
    not "". Seven of Human-GEM's metFormulas are like that, and .astype(str)
    on the whole array raises rather than coercing them.
    """
    out = []
    for x in np.asarray(v, dtype=object).ravel():
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, np.ndarray):
            out.append("" if x.size == 0 else str(x.ravel()[0]))
        else:
            out.append("" if x is None else str(x))
    return np.array(out, dtype=object)


def fetch(cache: Path, refresh: bool = False) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dst = cache / f
        if dst.exists() and not refresh:
            continue
        print(f"  downloading {f} ...", flush=True)
        urllib.request.urlretrieve(f"{RAW}/{f}", dst)
    return cache


def parse_formula(f) -> dict:
    """'C10H16O' -> {'C': 10, 'H': 16, 'O': 1}. Empty/invalid -> None."""
    if f is None:
        return None
    f = str(f).strip()
    if not f or f in ("NaN", "nan", "[]"):
        return None
    out, pos = {}, 0
    for m in _TOKEN.finditer(f):
        if m.start() != pos:            # something unparseable in between
            return None
        pos = m.end()
        el, n = m.group(1), m.group(2)
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out if pos == len(f) and out else None


def element_matrix(formulas):
    """
    (n_elements, n_mets) counts, the element list, and a per-metabolite flag
    for whether its formula could be read at all.

    E @ S then gives, for every reaction, the net atoms of each element it
    creates or destroys. A balanced reaction is a zero column.
    """
    parsed = [parse_formula(f) for f in formulas]
    ok = np.array([p is not None for p in parsed])
    generic = np.array([p is not None and bool(set(p) & GENERIC) for p in parsed])
    els = sorted({e for p in parsed if p for e in p})
    idx = {e: i for i, e in enumerate(els)}
    E = np.zeros((len(els), len(parsed)))
    for j, p in enumerate(parsed):
        if p:
            for e, n in p.items():
                E[idx[e], j] = n
    return E, els, ok, generic


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="gem_ref")
    ap.add_argument("--cache-dir", default="gem_cache")
    ap.add_argument("--refresh", action="store_true", help="re-download")
    a = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    cache = fetch(here / a.cache_dir if not Path(a.cache_dir).is_absolute()
                  else Path(a.cache_dir), a.refresh)
    out = here / a.out_dir if not Path(a.out_dir).is_absolute() else Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("parsing Human-GEM.mat ...")
    g = sio.loadmat(cache / "Human-GEM.mat", struct_as_record=False,
                    squeeze_me=True)["humanGEM"]

    S = sp.coo_matrix(sp.csc_matrix(g.S))
    rxns = as_str_list(g.rxns)
    mets = as_str_list(g.mets)
    genes = as_str_list(g.genes)
    lb, ub, c = (np.asarray(getattr(g, k), dtype=np.float64) for k in ("lb", "ub", "c"))
    charges = np.nan_to_num(np.asarray(g.metCharges, dtype=np.float64))
    rgm = sp.coo_matrix(sp.csc_matrix(g.rxnGeneMat))

    E, els, form_ok, generic = element_matrix(as_str_list(g.metFormulas))

    # Reference mass/charge balance, so a generated graph has something to be
    # judged against. Exchange reactions (one participant) and the biomass
    # pseudo-reaction are unbalanced BY DESIGN — the model would be wrong if
    # they weren't — so they are labelled, not counted as failures.
    Sd = sp.csc_matrix(g.S)
    n_part = np.diff(Sd.indptr)
    atom_imbalance = np.abs(E @ Sd).max(axis=0) if E.size else np.zeros(len(rxns))
    atom_imbalance = np.asarray(atom_imbalance).ravel()
    charge_imbalance = np.abs(charges @ Sd).ravel()
    checkable = np.ones(len(rxns), bool)
    for j in range(len(rxns)):
        rows = Sd.indices[Sd.indptr[j]:Sd.indptr[j + 1]]
        if n_part[j] <= 1 or not form_ok[rows].all() or generic[rows].any():
            checkable[j] = False
    balanced = checkable & (atom_imbalance < 1e-6) & (charge_imbalance < 1e-6)

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
        "source": RAW,
        "version": str(getattr(g, "version", "")),
        "rxns": list(rxns), "mets": list(mets), "genes": list(genes),
        "elements": els,
        "subsystems": as_str_list(g.subSystems).tolist(),
        "met_names": as_str_list(g.metNames).tolist(),
        "rxn_names": as_str_list(g.rxnNames).tolist(),
        "met_comps": np.asarray(g.metComps, dtype=int).tolist(),
        "comps": as_str_list(g.comps).tolist(),
        "comp_names": as_str_list(g.compNames).tolist(),
        "objective": rxns[np.flatnonzero(c)].tolist(),
        "n_rxns": int(len(rxns)), "n_mets": int(len(mets)), "n_genes": int(len(genes)),
    }
    (out / "meta.json").write_text(json.dumps(meta))

    print(f"\n  S            {S.shape[0]:,} metabolites x {S.shape[1]:,} reactions, "
          f"{S.nnz:,} nonzeros")
    print(f"  reversible   {(lb < 0).sum():,}   objective: {meta['objective']}")
    print(f"  elements     {len(els)}: {' '.join(els)}")
    print(f"  mass+charge  {balanced.sum():,} of {checkable.sum():,} checkable "
          f"reactions balance ({100*balanced.sum()/max(checkable.sum(),1):.1f}%)")
    print(f"               {(~checkable).sum():,} not checkable "
          f"(exchange/biomass, or R/X placeholder formulas)")
    print(f"\n  wrote {out/'gem.npz'} and {out/'meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
