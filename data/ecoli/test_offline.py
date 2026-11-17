#!/usr/bin/env python3
"""Offline smoke test: fake mini-GEM + fake PRECISE matrix through both builders.

No network and no cobra needed. Exercises GPR->DNF parsing, currency and hub
filtering, edge/name-map emission, metadata stratification, the top-k
correlation, and the node-id-space join. Run it before touching the real data
so a failure here is a bug in these scripts, not in the download.

    python test_offline.py
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_reference import parse_gpr_dnf, complex_id  # noqa: E402
from graphpath.data.node_types import ecoli_node_type as get_node_type  # noqa: E402

ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")


# ---------------------------------------------------------------------------
print("\n1. GPR -> DNF")
cases = [
    ("b0001", [{"b0001"}]),
    ("b0001 or b0002", [{"b0001"}, {"b0002"}]),
    ("b0001 and b0002", [{"b0001", "b0002"}]),
    ("(b0001 and b0002) or b0003", [{"b0001", "b0002"}, {"b0003"}]),
    ("b0001 and (b0002 or b0003)", [{"b0001", "b0002"}, {"b0001", "b0003"}]),
    ("((b1) or (b2)) and b3", [{"b1", "b3"}, {"b2", "b3"}]),
    ("", []),
]
for rule, want in cases:
    got = [set(c) for c in parse_gpr_dnf(rule)]
    check(f"{rule!r:38s} -> {got}", sorted(map(sorted, got)) == sorted(map(sorted, want)))

check("complex id is order-independent",
      complex_id({"b0003", "b0001"}) == complex_id({"b0001", "b0003"}) ==
      "enz_b0001+b0003")

# ---------------------------------------------------------------------------
print("\n2. node type detection")
for node, want in [("rxn_PGI", "reaction"), ("enz_b0002+b0003", "enzyme"),
                   ("b4025", "gene"), ("g6p_c", "metabolite"),
                   ("glc__D_e", "metabolite"), ("13dpg_c", "metabolite")]:
    check(f"{node:<20s} -> {want}", get_node_type(node) == want)

# ---------------------------------------------------------------------------
print("\n3. build_reference on a fake mini-GEM")
tmp = tempfile.mkdtemp(prefix="ecoli_test_")
mini = {
    "id": "mini",
    "genes": [{"id": f"b{i:04d}", "name": n} for i, n in
              enumerate(["thrA", "thrB", "pgi", "pfkA", "pfkB", "fbaA"], start=1)]
             + [{"id": "s0001", "name": "spontaneous"}],
    "metabolites": [
        {"id": "g6p_c", "name": "D-Glucose 6-phosphate", "compartment": "c"},
        {"id": "f6p_c", "name": "D-Fructose 6-phosphate", "compartment": "c"},
        {"id": "fdp_c", "name": "D-Fructose 1,6-bisphosphate", "compartment": "c"},
        {"id": "dhap_c", "name": "Dihydroxyacetone phosphate", "compartment": "c"},
        {"id": "atp_c", "name": "ATP", "compartment": "c"},
        {"id": "adp_c", "name": "ADP", "compartment": "c"},
        {"id": "h_c", "name": "H+", "compartment": "c"},
        {"id": "glc__D_e", "name": "D-Glucose", "compartment": "e"},
    ],
    "reactions": [
        {"id": "PGI", "name": "glucose-6-phosphate isomerase",
         "subsystem": "Glycolysis", "gene_reaction_rule": "b0003",
         "metabolites": {"g6p_c": -1, "f6p_c": 1}},
        {"id": "PFK", "name": "phosphofructokinase", "subsystem": "Glycolysis",
         "gene_reaction_rule": "b0004 or b0005",
         "metabolites": {"f6p_c": -1, "atp_c": -1, "fdp_c": 1, "adp_c": 1, "h_c": 1}},
        {"id": "FBA", "name": "fructose-bisphosphate aldolase",
         "subsystem": "Glycolysis", "gene_reaction_rule": "b0006",
         "metabolites": {"fdp_c": -1, "dhap_c": 1}},
        {"id": "ASP", "name": "aspartate kinase", "subsystem": "Threonine",
         "gene_reaction_rule": "b0001 and b0002",
         "metabolites": {"atp_c": -1, "adp_c": 1, "dhap_c": -1, "g6p_c": 1}},
        {"id": "SPON", "name": "spontaneous thing", "subsystem": "",
         "gene_reaction_rule": "s0001", "metabolites": {"h_c": -1, "f6p_c": 1}},
        {"id": "EX_glc__D_e", "name": "glucose exchange", "subsystem": "",
         "gene_reaction_rule": "", "metabolites": {"glc__D_e": -1}},
    ],
}
mini_path = os.path.join(tmp, "mini.json")
with open(mini_path, "w") as fh:
    json.dump(mini, fh)

ref_dir = os.path.join(tmp, "ref")
r = subprocess.run([sys.executable, os.path.join(HERE, "build_reference.py"),
                    "--model", mini_path, "--out-dir", ref_dir],
                   capture_output=True, text=True)
print("   " + r.stdout.strip().replace("\n", "\n   "))
if r.returncode:
    print(r.stderr)
check("build_reference exits 0", r.returncode == 0)

edges = pd.read_csv(os.path.join(ref_dir, "ecoli_gem_reaction_node_edges.csv"))
types = pd.read_csv(os.path.join(ref_dir, "node_types.csv"))
names = pd.read_csv(os.path.join(ref_dir, "node_name_map.csv"))
tset = dict(zip(types["id"], types["type"]))

check("currency metabolites stripped",
      not {"atp_c", "adp_c", "h_c"} & set(tset), f"nodes: {sorted(tset)[:6]}...")
check("exchange reaction dropped", "rxn_EX_glc__D_e" not in tset)
check("spontaneous gene s0001 not a node", "s0001" not in tset)
check("enzyme complex from AND rule present", "enz_b0001+b0002" in tset)
check("isozymes give two enzymes", {"enz_b0004", "enz_b0005"} <= set(tset))
check("gene->enzyme->reaction chain",
      len(edges[(edges.source == "b0004") & (edges.target == "enz_b0004")]) == 1 and
      len(edges[(edges.source == "enz_b0004") & (edges.target == "rxn_PFK")]) == 1)
check("substrate/product roles distinguished",
      set(edges[edges.source_type == "metabolite"].edge_type) == {"substrate", "product"})
check("subsystem carried on edges",
      set(edges[edges.reaction == "PGI"].subsystem) == {"Glycolysis"})
check("names resolve", dict(zip(names["id"], names["common_name"])).get("b0003") == "pgi")
check("all four entity types present",
      set(tset.values()) == {"gene", "metabolite", "enzyme", "reaction"},
      str(sorted(set(tset.values()))))
check("node_types agrees with get_node_type",
      all(get_node_type(n) == t for n, t in tset.items()),
      str([(n, t, get_node_type(n)) for n, t in tset.items()
           if get_node_type(n) != t][:3]))

# hub cut
ref2 = os.path.join(tmp, "ref_hub")
r2 = subprocess.run([sys.executable, os.path.join(HERE, "build_reference.py"),
                     "--model", mini_path, "--out-dir", ref2, "--hub-degree", "2"],
                    capture_output=True, text=True)
t2 = set(pd.read_csv(os.path.join(ref2, "node_types.csv"))["id"])
check("--hub-degree drops high-degree metabolites",
      "g6p_c" not in t2 and "f6p_c" not in t2, f"kept: {sorted(t2 & {'g6p_c','f6p_c','fdp_c'})}")

# ---------------------------------------------------------------------------
print("\n4. build_coexpression on a fake PRECISE-1K")
rng = np.random.default_rng(0)
# 10 gene nodes in two planted modules of 5, plus 2 genes absent from the graph
genes = [f"b{i:04d}" for i in range(1, 11)] + ["b9998", "b9999"]
samples = [f"S{i:03d}" for i in range(60)]
lat = rng.normal(size=(2, len(samples)))
X = np.vstack([lat[0] + 0.3 * rng.normal(size=len(samples)) for _ in range(5)] +
              [lat[1] + 0.3 * rng.normal(size=len(samples)) for _ in range(5)] +
              [rng.normal(size=len(samples)) for _ in range(2)])
expr = pd.DataFrame(X, index=genes, columns=samples)
meta = pd.DataFrame({
    "project": ["glucose"] * 30 + ["acetate"] * 30,
    "Carbon Source (g/L)": ["glc"] * 30 + ["ac"] * 30,
}, index=samples)
expr_csv, meta_csv = os.path.join(tmp, "log_tpm_qc.csv"), os.path.join(tmp, "metadata_qc.csv")
expr.to_csv(expr_csv)
meta.to_csv(meta_csv)

# node id map keyed on graph node names, as PathwayGraphDataset would have it
extra_gene_nodes = [f"b{i:04d}" for i in range(7, 11)]
node_id_map = {n: i + 1 for i, n in enumerate(sorted(set(tset) | set(extra_gene_nodes)))}
map_path = os.path.join(tmp, "node_id_map.pkl")
with open(map_path, "wb") as fh:
    pickle.dump(node_id_map, fh)

expr_dir = os.path.join(tmp, "expr")
r3 = subprocess.run([sys.executable, os.path.join(HERE, "build_coexpression.py"),
                     "--expr", expr_csv, "--meta", meta_csv,
                     "--stratify-by", "project", "--node-id-map", map_path,
                     "--out-dir", expr_dir, "--top-k", "3", "--min-samples", "10",
                     "--min-abs", "0.3"],
                    capture_output=True, text=True)
print("   " + (r3.stdout + r3.stderr).strip().replace("\n", "\n   "))
check("build_coexpression exits 0", r3.returncode == 0)

meta_out = json.load(open(os.path.join(expr_dir, "meta.json")))
check("two strata", len(meta_out["tissues"]) == 2, str(meta_out["tissues"]))
check("node-id space", meta_out["space"] == "node")
check("vocab matches the id map",
      meta_out["vocab_size"] == max(node_id_map.values()) + 1)
check("only graph genes kept (b9998/b9999 dropped)",
      set(meta_out["genes"]) <= set(node_id_map), str(meta_out["genes"]))

z = np.load(os.path.join(expr_dir, "coexp_000.npz"))
V = meta_out["vocab_size"]
check("indices within vocab", z["row"].max() < V and z["col"].max() < V)
check("symmetric", len(z["row"]) == len(z["col"]) and
      set(zip(z["row"].tolist(), z["col"].tolist())) ==
      set(zip(z["col"].tolist(), z["row"].tolist())))
check("no self loops", not (z["row"] == z["col"]).any())
check("|r| in range", np.abs(z["val"]).max() <= 1.0)

# the two latent modules should be recovered: b0001-b0004 vs b0005-b0006
id2node = {v: k for k, v in node_id_map.items()}
pairs = {(id2node[int(a)], id2node[int(b)]) for a, b in zip(z["row"], z["col"])}
mod1 = {f"b{i:04d}" for i in range(1, 6)}
within = sum(1 for a, b in pairs if (a in mod1) == (b in mod1))
check("top-k recovers the planted modules", within / max(len(pairs), 1) > 0.8,
      f"{within}/{len(pairs)} within-module")

E = np.load(os.path.join(expr_dir, "expression.npy"))
check("expression matrix shape", E.shape == (V, 2), str(E.shape))

print("\n5. loader (numpy inspection path; torch path needs torch installed)")
from graphpath.data import expression_ecoli  # noqa: E402
m = expression_ecoli.inspect(expr_dir)
check("inspect returns the strata", len(m["tissues"]) == 2)

print("\n6. build_regulondb on a fake RegulonDB dump")
net_path = os.path.join(tmp, "network_tf_gene.txt")
with open(net_path, "w") as fh:
    fh.write("# RegulonDB fake dump\n# columns: TF, gene, effect, evidence\n")
    for tf, tg, eff in [("thrA", "pgi", "+"), ("thrA", "pfkA", "-"),
                        ("pgi", "fbaA", "+-"), ("thrA", "notInGem", "+"),
                        ("thrA", "pgi", "+")]:
        fh.write(f"{tf}\t{tg}\t{eff}\tstrong\n")

r4 = subprocess.run([sys.executable, os.path.join(HERE, "build_regulondb.py"),
                     "--network", net_path,
                     "--name-map", os.path.join(ref_dir, "node_name_map.csv"),
                     "--out-dir", os.path.join(tmp, "reg")],
                    capture_output=True, text=True)
print("   " + (r4.stdout + r4.stderr).strip().replace("\n", "\n   ")[:1200])
check("build_regulondb exits 0", r4.returncode == 0)
if r4.returncode == 0:
    reg = pd.read_csv(os.path.join(tmp, "reg", "ecoli_regulondb_edges.csv"))
    check("symbols mapped to b-numbers",
          set(reg["source"]) <= {"b0001", "b0003"} and
          set(reg["target"]) <= {"b0003", "b0004", "b0006"}, str(reg.values.tolist()))
    check("unmapped target dropped", "notInGem" not in set(reg["target"]))
    check("duplicate edge collapsed", len(reg) == 3, f"{len(reg)} edges")
    check("effect decoded", set(reg["edge_type"]) == {"activates", "represses", "dual"},
          str(sorted(set(reg["edge_type"]))))
    check("same schema as the GEM edges",
          list(reg.columns) == list(edges.columns))

shutil.rmtree(tmp)
print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
