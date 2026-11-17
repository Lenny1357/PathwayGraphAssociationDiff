#!/usr/bin/env python3
"""iML1515 -> reference network edge list, in the shape GraphPathModel expects.

Emits the E. coli counterparts of human_gem_reaction_node_edges2.csv and
node_name_map.csv:

    ecoli_gem_reaction_node_edges.csv   source,target,edge_type,source_type,
                                        target_type,subsystem,reaction
    node_name_map.csv                   id,common_name
    node_types.csv                      id,type
    subsystems.csv                      subsystem,n_reactions,n_nodes

Node ids
    rxn_PGI                 reaction
    enz_b0002+b0003+b0004   enzyme complex, genes sorted, '+' joined
    b4025                   gene (Blattner number, matches PRECISE-1K exactly)
    g6p_c                   metabolite (BiGG id, compartment suffix kept)

Usage
    python build_reference.py --out-dir ./ecoli_ref
    python build_reference.py --model /path/to/iML1515.json --out-dir ./ecoli_ref
    python build_reference.py --out-dir ./ecoli_ref --hub-degree 60 --no-enzyme-layer

Requires cobra (pip install cobra) unless --model points at a JSON file, in
which case the JSON is parsed directly and cobra is optional.
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import os
import sys
import urllib.request

BIGG_JSON = "http://bigg.ucsd.edu/static/models/{model}.json"

# ---------------------------------------------------------------------------
# Currency metabolites.
#
# These are cofactors and small inorganics that participate in hundreds of
# reactions. Left in, they make the metabolite layer a hairball: every reaction
# is two hops from every other through h_c, and the co-expression propagation
# in the notebook spreads to everything by hops=2. Human-GEM pipelines strip
# the same class of node. Base ids, compartment-agnostic.
# ---------------------------------------------------------------------------
CURRENCY = {
    # protons, water, gases
    "h", "h2o", "o2", "co2", "co", "n2", "h2", "h2o2", "o2s", "oh1",
    # phosphate series
    "pi", "ppi", "pppi",
    # nucleotide energy carriers
    "atp", "adp", "amp", "gtp", "gdp", "gmp", "ctp", "cdp", "cmp",
    "utp", "udp", "ump", "itp", "idp", "imp", "datp", "dadp", "damp",
    "dgtp", "dgdp", "dgmp", "dctp", "dcdp", "dcmp", "dttp", "dtdp", "dtmp",
    # redox carriers
    "nad", "nadh", "nadp", "nadph", "fad", "fadh2", "fmn", "fmnh2",
    "q8", "q8h2", "mqn8", "mql8", "2dmmq8", "2dmmql8",
    "trdox", "trdrd", "grxox", "grxrd", "gthox", "gthrd",
    # group carriers
    "coa", "actp",
    # inorganic ions and salts
    "nh4", "no2", "no3", "so4", "so3", "s2o3", "h2s", "hco3", "cl",
    "na1", "k", "mg2", "ca2", "fe2", "fe3", "mn2", "zn2", "cu", "cu2",
    "cobalt2", "ni2", "mobd", "tungs", "sel", "cd2", "hg2", "ag", "pb",
}

# Reaction ids to drop outright: boundary pseudo-reactions and biomass, which
# is a ~100-degree hub that belongs to no pathway.
DROP_PREFIXES = ("EX_", "DM_", "SK_", "BIOMASS")

# cobra's placeholder gene for spontaneous / orphan reactions
SPONTANEOUS = {"s0001", "S0001"}


# ---------------------------------------------------------------------------
# GPR parsing
# ---------------------------------------------------------------------------
def _balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        depth += ch == "("
        depth -= ch == ")"
        if depth < 0:
            return False
    return depth == 0


def _top_level_split(s: str, op: str) -> list:
    """Split on `op` only at parenthesis depth 0."""
    out, depth, cur, i, n = [], 0, [], 0, len(s)
    pad = f" {op} "
    while i < n:
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and s[i:i + len(pad)].lower() == pad:
            out.append("".join(cur))
            cur = []
            i += len(pad)
            continue
        cur.append(ch)
        i += 1
    out.append("".join(cur))
    return [p.strip() for p in out if p.strip()]


def parse_gpr_dnf(rule: str) -> list:
    """Gene-protein-reaction rule -> list of complexes (frozensets of genes).

    'b0001 or (b0002 and b0003)' -> [{b0001}, {b0002, b0003}]

    Handles nesting by expanding to disjunctive normal form. iML1515's rules
    are shallow, so this is exact for that model; on a deeply nested rule the
    expansion is still correct but can produce many complexes.
    """
    rule = (rule or "").strip()
    if not rule:
        return []
    while rule.startswith("(") and rule.endswith(")") and _balanced(rule[1:-1]):
        rule = rule[1:-1].strip()
    if not rule:
        return []

    ors = _top_level_split(rule, "or")
    if len(ors) > 1:
        out = []
        for part in ors:
            out.extend(parse_gpr_dnf(part))
        return out

    ands = _top_level_split(rule, "and")
    if len(ands) > 1:
        combos = [frozenset()]
        for part in ands:
            sub = parse_gpr_dnf(part)
            if not sub:
                continue
            combos = [a | b for a, b in itertools.product(combos, sub)]
        return [c for c in combos if c]

    token = rule.strip("() ")
    return [frozenset({token})] if token else []


def complex_id(genes) -> str:
    return "enz_" + "+".join(sorted(genes))


# ---------------------------------------------------------------------------
# Model loading. cobra if available, raw BiGG JSON otherwise.
# ---------------------------------------------------------------------------
class Model:
    """Minimal view over a BiGG model: reactions, metabolites, genes."""

    def __init__(self, reactions, metabolites, genes):
        self.reactions = reactions      # list of dicts
        self.metabolites = metabolites  # {id: {name, compartment}}
        self.genes = genes              # {id: name}


def load_model(path_or_id: str) -> Model:
    if os.path.exists(path_or_id):
        path = path_or_id
    else:
        path = f"{path_or_id}.json"
        if not os.path.exists(path):
            url = BIGG_JSON.format(model=path_or_id)
            print(f"downloading {url}", file=sys.stderr)
            urllib.request.urlretrieve(url, path)

    if path.endswith((".xml", ".sbml", ".mat")):
        try:
            import cobra
        except ImportError:
            sys.exit("SBML/MAT input needs cobra: pip install cobra "
                     "(or pass the .json from BiGG instead)")
        m = cobra.io.read_sbml_model(path) if path.endswith((".xml", ".sbml")) \
            else cobra.io.load_matlab_model(path)
        return Model(
            reactions=[{
                "id": r.id, "name": r.name, "subsystem": r.subsystem or "",
                "gpr": r.gene_reaction_rule,
                "metabolites": {mt.id: co for mt, co in r.metabolites.items()},
            } for r in m.reactions],
            metabolites={mt.id: {"name": mt.name, "compartment": mt.compartment}
                         for mt in m.metabolites},
            genes={g.id: (g.name or g.id) for g in m.genes},
        )

    with open(path) as fh:
        raw = json.load(fh)
    return Model(
        reactions=[{
            "id": r["id"], "name": r.get("name", ""),
            "subsystem": r.get("subsystem", "") or "",
            "gpr": r.get("gene_reaction_rule", ""),
            "metabolites": r.get("metabolites", {}),
        } for r in raw["reactions"]],
        metabolites={mt["id"]: {"name": mt.get("name", ""),
                                "compartment": mt.get("compartment", "")}
                     for mt in raw["metabolites"]},
        genes={g["id"]: (g.get("name") or g["id"]) for g in raw.get("genes", [])},
    )


def base_id(met_id: str) -> str:
    """g6p_c -> g6p. BiGG metabolite ids are <base>_<compartment>."""
    return met_id.rsplit("_", 1)[0] if "_" in met_id else met_id


# ---------------------------------------------------------------------------
def build(model: Model, *, enzyme_layer=True, drop_currency=True,
          hub_degree=0, merge_compartments=False, keep_exchanges=False):
    """-> (edges, name_map, node_types, subsystem_of_node)"""

    reactions = [r for r in model.reactions
                 if keep_exchanges or not r["id"].startswith(DROP_PREFIXES)]

    # Metabolite degree, for the hub cut. Counted before any filtering so the
    # threshold means the same thing regardless of what else was dropped.
    degree = collections.Counter()
    for r in reactions:
        for m in r["metabolites"]:
            degree[m] += 1

    def met_node(m):
        return base_id(m) if merge_compartments else m

    def keep_met(m):
        if drop_currency and base_id(m) in CURRENCY:
            return False
        if hub_degree and degree[m] >= hub_degree:
            return False
        return True

    edges = []          # (source, target, edge_type, s_type, t_type, subsys, rxn)
    name_map = {}
    node_types = {}
    subsystem_of = collections.defaultdict(set)
    dropped_mets = set()

    def note(node, ntype, name, subsys=None):
        node_types[node] = ntype
        if name and node not in name_map:
            name_map[node] = name
        if subsys:
            subsystem_of[node].add(subsys)

    for r in reactions:
        rid, sub = f"rxn_{r['id']}", r["subsystem"]
        note(rid, "reaction", r["name"] or r["id"], sub)

        # reaction <-> metabolite
        for m, coeff in r["metabolites"].items():
            if not keep_met(m):
                dropped_mets.add(m)
                continue
            mid = met_node(m)
            meta = model.metabolites.get(m, {})
            note(mid, "metabolite", meta.get("name", ""), sub)
            role = "substrate" if coeff < 0 else "product"
            edges.append((mid, rid, role, "metabolite", "reaction", sub, r["id"]))

        # reaction <-> enzyme <-> gene   (or reaction <-> gene directly)
        complexes = parse_gpr_dnf(r["gpr"])
        for cx in complexes:
            cx = frozenset(g for g in cx if g not in SPONTANEOUS)
            if not cx:
                continue
            gene_names = [model.genes.get(g, g) for g in sorted(cx)]
            if enzyme_layer:
                eid = complex_id(cx)
                note(eid, "enzyme", "-".join(gene_names), sub)
                edges.append((eid, rid, "catalyses", "enzyme", "reaction", sub, r["id"]))
                for g in sorted(cx):
                    note(g, "gene", model.genes.get(g, g), sub)
                    edges.append((g, eid, "encodes", "gene", "enzyme", sub, r["id"]))
            else:
                for g in sorted(cx):
                    note(g, "gene", model.genes.get(g, g), sub)
                    edges.append((g, rid, "encodes", "gene", "reaction", sub, r["id"]))

    # de-duplicate, keeping first occurrence
    seen, uniq = set(), []
    for e in edges:
        key = (e[0], e[1], e[2])
        if key not in seen:
            seen.add(key)
            uniq.append(e)

    return uniq, name_map, node_types, subsystem_of, dropped_mets


def write_outputs(out_dir, edges, name_map, node_types, subsystem_of, model):
    os.makedirs(out_dir, exist_ok=True)
    p = lambda f: os.path.join(out_dir, f)

    with open(p("ecoli_gem_reaction_node_edges.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "target", "edge_type", "source_type",
                    "target_type", "subsystem", "reaction"])
        w.writerows(edges)

    with open(p("node_name_map.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "common_name"])
        for k in sorted(name_map):
            w.writerow([k, name_map[k]])

    with open(p("node_types.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "type"])
        for k in sorted(node_types):
            w.writerow([k, node_types[k]])

    # node -> subsystem, the association ReferenceNetwork.build_pathway_
    # associations() derives. Written out as well so it can be loaded directly.
    with open(p("node_subsystems.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "subsystem"])
        for k in sorted(subsystem_of):
            for s in sorted(subsystem_of[k]):
                if s:
                    w.writerow([k, s])

    counts = collections.Counter(node_types.values())
    subs = collections.Counter()
    for node, ss in subsystem_of.items():
        for s in ss:
            if s:
                subs[s] += 1
    with open(p("subsystems.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subsystem", "n_nodes"])
        for s, n in subs.most_common():
            w.writerow([s, n])
    return counts, subs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="iML1515",
                    help="BiGG model id (downloaded) or path to .json/.xml/.mat")
    ap.add_argument("--out-dir", default="ecoli_ref")
    ap.add_argument("--no-enzyme-layer", action="store_true",
                    help="link genes straight to reactions, no enzyme nodes "
                         "(collapses the 4-type grammar to 3 - retrain if so)")
    ap.add_argument("--keep-currency", action="store_true")
    ap.add_argument("--hub-degree", type=int, default=0,
                    help="also drop metabolites in >= this many reactions "
                         "(0 = off; ~60 is a reasonable second cut)")
    ap.add_argument("--merge-compartments", action="store_true",
                    help="treat g6p_c and g6p_p as one node")
    ap.add_argument("--keep-exchanges", action="store_true",
                    help="keep EX_/DM_/SK_/BIOMASS pseudo-reactions")
    a = ap.parse_args()

    model = load_model(a.model)
    print(f"loaded: {len(model.reactions)} reactions, "
          f"{len(model.metabolites)} metabolites, {len(model.genes)} genes")

    edges, name_map, node_types, subsystem_of, dropped = build(
        model,
        enzyme_layer=not a.no_enzyme_layer,
        drop_currency=not a.keep_currency,
        hub_degree=a.hub_degree,
        merge_compartments=a.merge_compartments,
        keep_exchanges=a.keep_exchanges,
    )
    counts, subs = write_outputs(a.out_dir, edges, name_map, node_types,
                                 subsystem_of, model)

    print(f"\ndropped {len(dropped)} currency/hub metabolite species")
    print(f"nodes  : {sum(counts.values())}  " +
          "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"edges  : {len(edges)}")
    print(f"subsystems: {len(subs)}")
    print(f"\nwritten to {a.out_dir}/")
    print("  ecoli_gem_reaction_node_edges.csv  <- EDGES_CSV in the notebook")
    print("  node_name_map.csv                  <- NAME_MAP in the notebook")
    print("  node_types.csv, node_subsystems.csv, subsystems.csv")

    if sum(counts.values()) < 3000:
        print("\nnote: this graph is ~4x smaller than Human-GEM. With "
              "max_nodes=64 a subgraph now covers a large share of a whole\n"
              "subsystem, so NearestTrainingBenchmark will be far more "
              "sensitive - expect higher near_duplicate_rate at the same\n"
              "model capacity, and consider dropping d_model/n_layers.")


if __name__ == "__main__":
    main()
