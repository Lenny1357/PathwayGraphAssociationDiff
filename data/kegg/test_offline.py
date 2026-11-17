#!/usr/bin/env python3
"""Offline checks for the KEGG pipeline — no network, no KEGG account.

Runs the parser and the graph builder over hand-written flat-file fragments
taken verbatim from real KEGG entries, so a failure here is a bug in the code
rather than a problem with the download. Run this first.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_kegg import (build_module_graph, module_class, parse_compound_block,
                        parse_entry, parse_equation, parse_orthology_block,
                        parse_reaction_block, write_outputs)
from graphpath.data.node_types import TYPE_MAP, kegg_node_type as get_node_type

FAILED = []


def check(label, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        FAILED.append(label)


def check_true(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(label)


# Verbatim from `rest.kegg.jp/get/M00001`, trimmed to a few lines per field.
M00001 = """ENTRY       M00001            Pathway   Module
NAME        Glycolysis (Embden-Meyerhof pathway), glucose => pyruvate
DEFINITION  (K00844,K12407) (K01810,K06859) K01803
CLASS       Pathway modules; Carbohydrate metabolism; Central carbohydrate metabolism
PATHWAY     map00010  Glycolysis / Gluconeogenesis
            map01200  Carbon metabolism
ORTHOLOGY   K00844,K12407  hexokinase/glucokinase [EC:2.7.1.1 2.7.1.2] [RN:R01786]
            K01810,K06859  glucose-6-phosphate isomerase [EC:5.3.1.9] [RN:R13199]
            K01803  triosephosphate isomerase [EC:5.3.1.1] [RN:R01015]
REACTION    R01786,R02189  C00267 -> C00668
            R13199  C00668 -> C00085
            R01068  C00354 -> C00111 + C00118
COMPOUND    C00267  alpha-D-Glucose
            C00668  alpha-D-Glucose 6-phosphate
            C00085  D-Fructose 6-phosphate
            C00354  D-Fructose 1,6-bisphosphate
            C00111  Glycerone phosphate
            C00118  D-Glyceraldehyde 3-phosphate
///"""

R01786 = """ENTRY       R01786                      Reaction
NAME        ATP:alpha-D-glucose 6-phosphotransferase
EQUATION    C00002 + C00267 <=> C00008 + C00668
ENZYME      2.7.1.1         2.7.1.2
///"""


def main():
    print("\n1. flat-file parsing (12-column fields + continuation lines)")
    f = parse_entry(M00001)
    check("ENTRY field", f["ENTRY"][0].split()[0], "M00001")
    check("NAME field", f["NAME"][0][:10], "Glycolysis")
    check("PATHWAY continuation line kept", len(f["PATHWAY"]), 2)
    check("COMPOUND lines", len(f["COMPOUND"]), 6)

    print("\n2. CLASS -> subsystem")
    check("module_class", module_class(f),
          ("Pathway modules", "Central carbohydrate metabolism"))
    check("missing CLASS falls back", module_class({}), ("Unclassified", "Unclassified"))

    print("\n3. REACTION block")
    steps = parse_reaction_block(f["REACTION"])
    check("step count", len(steps), 3)
    check("alternatives on one step", steps[0], (["R01786", "R02189"],
                                                 ["C00267"], ["C00668"]))
    check("'+' splits co-products", steps[2],
          (["R01068"], ["C00354"], ["C00111", "C00118"]))
    check("<-> arrow accepted", parse_reaction_block(["R00001  C00001 <-> C00002"]),
          [(["R00001"], ["C00001"], ["C00002"])])
    check("malformed ids rejected", parse_reaction_block(["R1  C1 -> CoA"]), [])
    check("real ids only", parse_reaction_block(["R00200  C00074 <-> C00022"]),
          [(["R00200"], ["C00074"], ["C00022"])])

    print("\n4. ORTHOLOGY block")
    rows = parse_orthology_block(f["ORTHOLOGY"])
    check("row count", len(rows), 3)
    check("KOs on one row", rows[0]["kos"], ["K00844", "K12407"])
    check("two ECs parsed", rows[0]["ecs"], ["2.7.1.1", "2.7.1.2"])
    check("RN parsed", rows[0]["rns"], ["R01786"])
    check("name stripped of brackets", rows[0]["name"], "hexokinase/glucokinase")

    print("\n5. COMPOUND block and reaction equations")
    check("compound name", parse_compound_block(f["COMPOUND"])["C00267"],
          "alpha-D-Glucose")
    check("equation split", parse_equation(R01786),
          (["C00002", "C00267"], ["C00008", "C00668"]))

    print("\n6. node typing")
    for nid, want in (("rxn_R01786", "reaction"), ("ec_2.7.1.1", "enzyme"),
                      ("ko_K00844", "gene"), ("cpd_C00267", "metabolite")):
        check(f"get_node_type({nid})", get_node_type(nid), want)
    check_true("TYPE_MAP order unchanged",
               TYPE_MAP == {"gene": 0, "metabolite": 1, "enzyme": 2, "reaction": 3})

    print("\n7. graph construction")
    edges, names = build_module_graph("M00001", f, enzyme_layer=True)
    et = {(u, v) for u, v, *_ in edges}
    check_true("substrate edge present", ("cpd_C00267", "rxn_R01786") in et)
    check_true("product edge present", ("rxn_R01786", "cpd_C00668") in et)
    check_true("gene -> enzyme edge", ("ko_K00844", "ec_2.7.1.1") in et)
    check_true("enzyme -> reaction edge", ("ec_2.7.1.1", "rxn_R01786") in et)
    check_true("no self-loops", all(u != v for u, v, *_ in edges))
    check_true("compound name carried", names.get("cpd_C00267") == "alpha-D-Glucose")
    types = {t for _, _, _, t1, t2 in edges for t in (t1, t2)}
    check_true("only the four known types", types <= set(TYPE_MAP), str(sorted(types)))

    e2, _ = build_module_graph("M00001", f, enzyme_layer=False)
    et2 = {(u, v) for u, v, *_ in e2}
    check_true("--no-enzyme-layer links KO straight to reaction",
               ("ko_K00844", "rxn_R01786") in et2
               and not any(a.startswith("ec_") or b.startswith("ec_") for a, b in et2))

    eq = {"R01786": (["C00002", "C00267"], ["C00008", "C00668"])}
    e3, _ = build_module_graph("M00001", f, True, equations=eq)
    et3 = {(u, v) for u, v, *_ in e3}
    check_true("--full-equations brings currency back",
               ("cpd_C00002", "rxn_R01786") in et3)
    check_true("module steps exclude it by default",
               ("cpd_C00002", "rxn_R01786") not in et)

    print("\n8. end-to-end write (into a temp dir)")
    import pickle
    with tempfile.TemporaryDirectory() as td:
        per = {"M00001": (edges, names)}
        rows, nodes, graphs, skipped = write_outputs(
            td, per, names, {"M00001": module_class(f)}, 64, 4,
            {"M00001": "Glycolysis"}, {"M00001": "module"})
        for fn in ("kegg_edges.csv", "node_name_map.csv", "node_types.csv",
                   "node_subsystems.csv", "units.csv", "meta.json",
                   "kegg_sampled_graphs.pkl", "node_id_map.pkl"):
            check_true(f"wrote {fn}", os.path.exists(os.path.join(td, fn)))
        check_true("one graph per module", len(graphs) == 1)
        g = graphs[0]
        check_true("adjacency is square and matches num_nodes",
                   g["adj_matrix"].shape == (g["num_nodes"], g["num_nodes"]))
        check_true("adjacency symmetric",
                   bool((g["adj_matrix"] == g["adj_matrix"].T).all()))
        check_true("no self-loops in adjacency",
                   float(g["adj_matrix"].diagonal().sum()) == 0.0)
        check_true("node_ids index into node_id_map",
                   len(g["node_ids"]) == g["num_nodes"])
        nid = pickle.load(open(os.path.join(td, "node_id_map.pkl"), "rb"))
        check_true("node_id_map covers every node id",
                   set(g["node_ids"]) <= set(nid.values()))
        check_true("dataset keys match PathwayGraphDataset",
                   {"node_ids", "adj_matrix", "num_nodes",
                    "original_group_id", "center_node"} <= set(g))

    print("\n9. pathway builder (bulk link tables)")
    from build_kegg import build_pathway_graphs, CURRENCY
    links = {
        "pathway_list":   "path:map00010\tGlycolysis / Gluconeogenesis",
        "rxn_of_pathway": "path:map00010\trn:R01786\npath:map00010\trn:R13199",
        # R01786 carries ATP and H2O alongside the real substrates
        "cpd_of_rxn": ("rn:R01786\tcpd:C00267\nrn:R01786\tcpd:C00668\n"
                       "rn:R01786\tcpd:C00002\nrn:R01786\tcpd:C00001\n"
                       "rn:R13199\tcpd:C00668\nrn:R13199\tcpd:C00085"),
        "ko_of_rxn":      "rn:R01786\tko:K00844",
        "ec_of_rxn":      "rn:R01786\tec:2.7.1.1",
        "compound_list":  "cpd:C00267\talpha-D-Glucose\ncpd:C00002\tATP; Adenosine 5'-triphosphate",
        "reaction_list":  "rn:R01786\thexokinase",
        "ko_list":        "ko:K00844\tHK; hexokinase",
    }
    per, cls, pn = build_pathway_graphs(links, True, drop_currency=True)
    et = {(u, v) for u, v, *_ in per["map00010"][0]}
    check_true("pathway graph built", "map00010" in per)
    check_true("real substrate kept", ("cpd_C00267", "rxn_R01786") in et)
    check_true("ATP stripped as currency", ("cpd_C00002", "rxn_R01786") not in et)
    check_true("H2O stripped as currency", ("cpd_C00001", "rxn_R01786") not in et)
    check_true("gene -> enzyme -> reaction", ("ko_K00844", "ec_2.7.1.1") in et
               and ("ec_2.7.1.1", "rxn_R01786") in et)
    check_true("pathway name recovered",
               cls["map00010"][1] == "Glycolysis / Gluconeogenesis")
    per2, _, _ = build_pathway_graphs(links, True, drop_currency=False)
    et2 = {(u, v) for u, v, *_ in per2["map00010"][0]}
    check_true("--keep-currency puts ATP back", ("cpd_C00002", "rxn_R01786") in et2)
    per3, _, _ = build_pathway_graphs(links, True, True, hub_degree=2)
    et3 = {(u, v) for u, v, *_ in per3["map00010"][0]}
    check_true("--hub-degree cuts a shared compound",
               ("cpd_C00668", "rxn_R01786") not in et3)

    print(f"\n{'ALL PASS' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
