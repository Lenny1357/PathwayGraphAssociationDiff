#!/usr/bin/env python3
"""Download every KEGG module and turn it into this project's inputs.

Produces the same four reference files the Human-GEM and E. coli pipelines
produce, PLUS the pathway dataset itself — which the other two cannot, because
they have to sample subgraphs out of one genome-scale network. A KEGG module
IS a curated pathway, so the training graphs come straight from the source and
no sampler is involved.

Everything is derived from `get/M#####`. One module entry carries its
reactions, its main-chain transformations, its compounds and its KO -> EC ->
reaction links, so 573 modules cost ~58 requests and no compound or reaction
lookups at all.

The REACTION block is used for topology rather than full reaction equations,
and that is deliberate: KEGG's module steps already exclude currency
metabolites. `R01786  C00267 -> C00668` is hexokinase without the ATP/ADP, so
there is no hub-stripping heuristic to tune here — the curation has done it.
Pass --full-equations to fetch R entries and use the complete stoichiometry
instead, hubs and all.

Outputs (in --out-dir):
    kegg_module_edges.csv    <- EDGES_CSV in the notebook
    node_name_map.csv        <- NAME_MAP in the notebook
    node_types.csv, node_subsystems.csv, modules.csv
    kegg_sampled_graphs.pkl  <- the pathway dataset
    node_id_map.pkl          <- its vocabulary
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

REST = "https://rest.kegg.jp"
BATCH = 10                      # KEGG's documented limit for `get`


# ── fetching ───────────────────────────────────────────────────────────────

def _get(path: str, retries: int = 3, pause: float = 0.34) -> str:
    """One REST call, with a short pause so we stay a polite client."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{REST}/{path}", timeout=60) as fh:
                out = fh.read().decode("utf-8", "replace")
            time.sleep(pause)
            return out
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  {type(e).__name__} on {path}; retrying in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    return ""


def fetch_modules(cache_dir: str, refresh: bool = False) -> dict:
    """{module_id: raw flat-file text}, cached on disk."""
    os.makedirs(cache_dir, exist_ok=True)
    listing = os.path.join(cache_dir, "module_list.tsv")
    if refresh or not os.path.exists(listing):
        with open(listing, "w") as fh:
            fh.write(_get("list/module"))
    ids = [ln.split("\t")[0].strip() for ln in
           open(listing).read().strip().splitlines() if ln.strip()]
    ids = [i.split(":")[-1] for i in ids]
    print(f"{len(ids)} modules listed")

    raw = {}
    todo = []
    for m in ids:
        p = os.path.join(cache_dir, f"{m}.txt")
        if os.path.exists(p) and not refresh:
            raw[m] = open(p).read()
        else:
            todo.append(m)
    if todo:
        print(f"fetching {len(todo)} module entries "
              f"({(len(todo) + BATCH - 1)//BATCH} requests)")
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            text = _get("get/" + "+".join(chunk))
            for entry in text.split("///"):
                entry = entry.strip("\n")
                if not entry.strip():
                    continue
                mid = re.search(r"^ENTRY\s+(M\d+)", entry, re.M)
                if not mid:
                    continue
                mid = mid.group(1)
                raw[mid] = entry
                with open(os.path.join(cache_dir, f"{mid}.txt"), "w") as fh:
                    fh.write(entry)
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)}", end="\r",
                  file=sys.stderr)
        print(file=sys.stderr)
    return raw


def fetch_reactions(rids, cache_dir: str, refresh: bool = False) -> dict:
    """{reaction_id: raw text} — only needed for --full-equations."""
    os.makedirs(cache_dir, exist_ok=True)
    raw, todo = {}, []
    for r in sorted(rids):
        p = os.path.join(cache_dir, f"{r}.txt")
        if os.path.exists(p) and not refresh:
            raw[r] = open(p).read()
        else:
            todo.append(r)
    if todo:
        print(f"fetching {len(todo)} reaction entries "
              f"({(len(todo) + BATCH - 1)//BATCH} requests)")
        for i in range(0, len(todo), BATCH):
            text = _get("get/" + "+".join(todo[i:i + BATCH]))
            for entry in text.split("///"):
                rid = re.search(r"^ENTRY\s+(R\d+)", entry.strip("\n"), re.M)
                if not rid:
                    continue
                raw[rid.group(1)] = entry
                with open(os.path.join(cache_dir, f"{rid.group(1)}.txt"), "w") as fh:
                    fh.write(entry)
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)}", end="\r",
                  file=sys.stderr)
        print(file=sys.stderr)
    return raw


# ── parsing ────────────────────────────────────────────────────────────────

def parse_entry(text: str) -> dict:
    """
    KEGG flat file -> {FIELD: [line, ...]}.

    The format is fixed-width: the field name occupies the first 12 columns
    and a blank there means "continuation of the previous field".
    """
    out, field = defaultdict(list), None
    for line in text.splitlines():
        if not line.strip() or line.startswith("///"):
            continue
        head, body = line[:12].strip(), line[12:].rstrip()
        if head:
            field = head
        if field:
            out[field].append(body)
    return dict(out)


_STEP_RE = re.compile(r"^([\w,]+)\s+(.*?)\s*(?:->|<->|=>)\s*(.*)$")
# KEGG ids are fixed width. Matching a bare prefix would let "CoA" or "Cx"
# through as compounds, which is how junk enters the graph silently.
_CID_RE = re.compile(r"^C\d{5}$")
_RID_RE = re.compile(r"^R\d{5}$")


def parse_reaction_block(lines) -> list:
    """
    'R01786,R02189  C00267 -> C00668'  ->  [(rxns, [subs], [prods]), ...]

    Multiple reactions per step are alternatives for the same transformation;
    a '+' separates co-substrates within one side.
    """
    steps = []
    for ln in lines:
        m = _STEP_RE.match(ln.strip())
        if not m:
            continue
        rxns = [r for r in m.group(1).split(",") if _RID_RE.match(r.strip())]
        subs = [c.strip() for c in m.group(2).split("+") if _CID_RE.match(c.strip())]
        prods = [c.strip() for c in m.group(3).split("+") if _CID_RE.match(c.strip())]
        if rxns and (subs or prods):
            steps.append((rxns, subs, prods))
    return steps


_EC_RE = re.compile(r"\[EC:([^\]]+)\]")
_RN_RE = re.compile(r"\[RN:([^\]]+)\]")


def parse_orthology_block(lines) -> list:
    """'K00844,K12407  hexokinase [EC:2.7.1.1 2.7.1.2] [RN:R01786]' -> rows."""
    rows = []
    for ln in lines:
        ln = ln.strip()
        if not ln or not ln.startswith("K"):
            continue
        kos_part, _, rest = ln.partition("  ")
        kos = [k.strip() for k in kos_part.split(",")
               if re.match(r"^K\d{5}$", k.strip())]
        if not kos:
            continue
        ecs = []
        m = _EC_RE.search(rest)
        if m:
            ecs = [e.strip() for e in m.group(1).split() if e.strip() and e[0].isdigit()]
        rns = []
        m = _RN_RE.search(rest)
        if m:
            rns = [r.strip() for r in m.group(1).split() if _RID_RE.match(r.strip())]
        name = _RN_RE.sub("", _EC_RE.sub("", rest)).strip()
        rows.append({"kos": kos, "ecs": ecs, "rns": rns, "name": name})
    return rows


def parse_compound_block(lines) -> dict:
    out = {}
    for ln in lines:
        parts = ln.strip().split("  ", 1)
        if parts and _CID_RE.match(parts[0].strip()):
            out[parts[0].strip()] = parts[1].strip() if len(parts) > 1 else parts[0]
    return out


def module_class(fields) -> tuple:
    """CLASS -> (top level, subsystem). Falls back to the module id."""
    cls = " ".join(fields.get("CLASS", [])).strip()
    parts = [p.strip() for p in cls.split(";") if p.strip()]
    if not parts:
        return ("Unclassified", "Unclassified")
    return (parts[0], parts[-1])


# ── graph construction ─────────────────────────────────────────────────────

def rxn_id(r):  return f"rxn_{r}"
def ko_id(k):   return f"ko_{k}"
def ec_id(e):   return f"ec_{e}"
def cpd_id(c):  return f"cpd_{c}"


def build_module_graph(mid, fields, enzyme_layer=True, equations=None):
    """
    One module -> (edges, names). Edges are (u, v, edge_type, u_type, v_type).

    Topology comes from the module's own REACTION steps, which are the
    main-chain transformations. With `equations` (from --full-equations) the
    complete stoichiometry is used instead, currency metabolites included.
    """
    edges, names = [], {}
    steps = parse_reaction_block(fields.get("REACTION", []))
    compounds = parse_compound_block(fields.get("COMPOUND", []))
    for c, n in compounds.items():
        names[cpd_id(c)] = n

    def add(u, v, et, ut, vt):
        if u != v:
            edges.append((u, v, et, ut, vt))

    seen_rxn = set()
    for rxns, subs, prods in steps:
        for r in rxns:
            seen_rxn.add(r)
            names.setdefault(rxn_id(r), r)
            if equations is not None and r in equations:
                s2, p2 = equations[r]
            else:
                s2, p2 = subs, prods
            for c in s2:
                add(cpd_id(c), rxn_id(r), "substrate", "metabolite", "reaction")
            for c in p2:
                add(rxn_id(r), cpd_id(c), "product", "reaction", "metabolite")

    for row in parse_orthology_block(fields.get("ORTHOLOGY", [])):
        targets = [r for r in row["rns"] if r in seen_rxn] or sorted(seen_rxn)
        for k in row["kos"]:
            names.setdefault(ko_id(k), row["name"] or k)
            if enzyme_layer and row["ecs"]:
                for e in row["ecs"]:
                    names.setdefault(ec_id(e), row["name"] or e)
                    add(ko_id(k), ec_id(e), "encodes", "gene", "enzyme")
                    for r in targets:
                        add(ec_id(e), rxn_id(r), "catalyses", "enzyme", "reaction")
            else:
                for r in targets:
                    add(ko_id(k), rxn_id(r), "catalyses", "gene", "reaction")
    return edges, names


def parse_equation(text: str):
    """R entry -> ([substrate C ids], [product C ids])."""
    f = parse_entry(text)
    eq = " ".join(f.get("EQUATION", []))
    if "<=>" not in eq:
        return [], []
    lhs, rhs = eq.split("<=>", 1)
    grab = lambda s: re.findall(r"\bC\d{5}\b", s)
    return grab(lhs), grab(rhs)


# ── outputs ────────────────────────────────────────────────────────────────

def _bfs_from(seed, adj, pool, max_nodes):
    """Breadth-first neighbourhood of `seed`, restricted to `pool`."""
    seen, order, queue = {seed}, [seed], [seed]
    while queue and len(order) < max_nodes:
        nxt = []
        for n in queue:
            for m in sorted(adj[n]):
                if m in pool and m not in seen:
                    seen.add(m); order.append(m); nxt.append(m)
                    if len(order) >= max_nodes:
                        break
            if len(order) >= max_nodes:
                break
        queue = nxt
    return sorted(order)


def _chunk(ids, edges, max_nodes, min_nodes, n_chunks=1):
    """
    Cut a unit into up to `n_chunks` connected pieces of at most `max_nodes`.

    Chunks are DISJOINT — each node lands in exactly one — so a memorisation
    benchmark is not confounded by two training graphs sharing most of their
    nodes. Each chunk grows breadth-first from the highest-degree node not yet
    used, which keeps a piece coherent instead of arbitrary.

    With n_chunks=1 a large pathway keeps only its hub neighbourhood: 166
    pathway maps lose 22,431 of 45,115 nodes that way, so raising this is how
    the rest of the curation gets used.
    """
    keep = set(ids)
    adj = defaultdict(set)
    for u, v, *_ in edges:
        if u in keep and v in keep:
            adj[u].add(v); adj[v].add(u)
    if not adj:
        return [ids[:max_nodes]] if len(ids) >= min_nodes else []

    out, remaining = [], set(adj)
    while remaining and len(out) < n_chunks:
        seed = max(remaining, key=lambda n: (len(adj[n] & remaining), n))
        piece = _bfs_from(seed, adj, remaining, max_nodes)
        remaining -= set(piece)
        if len(piece) >= min_nodes:
            out.append(piece)
        elif not out:                       # never return nothing at all
            out.append(piece)
            break
    return out


def write_outputs(out_dir, per_module, names, classes, max_nodes, min_nodes,
                  module_names=None, unit_kind=None, chunks_per_unit=1):
    os.makedirs(out_dir, exist_ok=True)
    p = lambda f: os.path.join(out_dir, f)

    from graphpath.data.node_types import kegg_node_type as get_node_type

    # ── edge table ────────────────────────────────────────────────────────
    seen, rows = set(), []
    for mid, (edges, _) in per_module.items():
        sub = classes[mid][1]
        for u, v, et, ut, vt in edges:
            key = (min(u, v), max(u, v), mid)
            if key in seen:
                continue
            seen.add(key)
            rows.append([u, v, et, ut, vt, sub, "", mid,
                         (unit_kind or {}).get(mid, "module")])
    with open(p("kegg_edges.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "target", "edge_type", "source_type",
                    "target_type", "subsystem", "reaction", "unit", "unit_kind"])
        w.writerows(rows)

    nodes = sorted({r[0] for r in rows} | {r[1] for r in rows})
    with open(p("node_name_map.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "common_name"])
        for n in nodes:
            w.writerow([n, names.get(n, n)])
    with open(p("node_types.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "type"])
        for n in nodes:
            w.writerow([n, get_node_type(n)])

    node_subs = defaultdict(set)
    for u, v, sub in ((r[0], r[1], r[5]) for r in rows):
        node_subs[u].add(sub); node_subs[v].add(sub)
    with open(p("node_subsystems.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "subsystem"])
        for n in nodes:
            for s in sorted(node_subs[n]):
                w.writerow([n, s])

    # ── the pathway dataset: one graph per module ─────────────────────────
    node_id_map = {n: i for i, n in enumerate(nodes)}
    graphs, skipped = [], {"too_small": 0, "no_edges": 0}
    for gi, (mid, (edges, _)) in enumerate(sorted(per_module.items())):
        ids = sorted({u for u, *_ in edges} | {v for _, v, *_ in edges})
        if not edges:
            skipped["no_edges"] += 1; continue
        if len(ids) < min_nodes:
            skipped["too_small"] += 1; continue
        # Connected chunks, not the alphabetically-first max_nodes ids —
        # truncating a 612-node map that way left 52 components and 77%
        # isolated nodes.
        pieces = _chunk(ids, edges, max_nodes, min_nodes, chunks_per_unit)
        if len(pieces) > 1:
            skipped["multi_chunk"] = skipped.get("multi_chunk", 0) + 1
        import numpy as np
        for ci, piece in enumerate(pieces):
            pos = {n: i for i, n in enumerate(piece)}
            n = len(piece)
            adj = np.zeros((n, n), dtype=np.float32)
            for u, v, *_ in edges:
                if u in pos and v in pos:
                    adj[pos[u], pos[v]] = adj[pos[v], pos[u]] = 1.0
            graphs.append({
                "node_ids": [node_id_map[x] for x in piece],
                "adj_matrix": adj,
                "num_nodes": n,
                "original_group_id": classes[mid][1],
                "center_node": node_id_map[piece[0]],
                "unit": mid,
                "unit_kind": (unit_kind or {}).get(mid, "module"),
                "chunk": ci,
                "unit_n_nodes": len(ids),
            })
    with open(p("kegg_sampled_graphs.pkl"), "wb") as fh:
        pickle.dump(graphs, fh)
    with open(p("node_id_map.pkl"), "wb") as fh:
        pickle.dump(node_id_map, fh)

    with open(p("units.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "kind", "name", "top_class", "subsystem",
                    "n_nodes", "n_edges"])
        for mid, (edges, _) in sorted(per_module.items()):
            ids = {u for u, *_ in edges} | {v for _, v, *_ in edges}
            w.writerow([mid, (unit_kind or {}).get(mid, "module"),
                        (module_names or {}).get(mid, mid),
                        classes[mid][0], classes[mid][1], len(ids), len(edges)])

    with open(p("meta.json"), "w") as fh:
        json.dump({"source": "KEGG via rest.kegg.jp",
                   "n_units": len(per_module),
                   "n_modules": sum(1 for m in per_module
                                    if (unit_kind or {}).get(m) == "module"),
                   "n_pathways": sum(1 for m in per_module
                                     if (unit_kind or {}).get(m) == "pathway"), "n_graphs": len(graphs),
                   "n_nodes": len(nodes), "n_edges": len(rows),
                   "max_nodes": max_nodes, "min_nodes": min_nodes,
                   "chunks_per_unit": chunks_per_unit,
                   "skipped": skipped}, fh, indent=1)
    return rows, nodes, graphs, skipped



# ── pathways ───────────────────────────────────────────────────────────────
#
# A module entry carries its own topology; a pathway map does not, and KGML is
# not served over the REST API. But the bulk `link/` endpoints give complete
# membership in one request each, and an undirected graph does not need to
# know substrate from product — so seven calls reconstruct every pathway.
#
# The cost is that reaction->compound links are unfiltered, so the currency
# metabolites the module steps had already excluded come back: H2O appears in
# 4,389 reactions, H+ in 2,625, O2 in 2,032. Left in, every reaction sits two
# hops from every other. They are stripped by default.

CURRENCY = {
    "C00001": "H2O", "C00002": "ATP", "C00008": "ADP", "C00020": "AMP",
    "C00003": "NAD+", "C00004": "NADH", "C00006": "NADP+", "C00005": "NADPH",
    "C00080": "H+", "C00009": "phosphate", "C00013": "diphosphate",
    "C00011": "CO2", "C00007": "O2", "C00014": "NH3", "C00027": "H2O2",
    "C00016": "FAD", "C01352": "FADH2", "C00044": "GTP", "C00035": "GDP",
    "C00015": "UDP", "C00075": "UTP", "C00063": "CTP", "C00019": "SAM",
    "C00021": "SAH", "C00010": "CoA", "C00034": "Mn2+", "C00038": "Zn2+",
    "C00070": "Cu2+", "C14818": "Fe2+", "C14819": "Fe3+", "C00087": "S",
    "C00282": "H2", "C00237": "CO", "C00697": "N2", "C00533": "NO",
}


def fetch_links(cache_dir, refresh=False):
    """The bulk link/list tables, one request each, cached."""
    os.makedirs(cache_dir, exist_ok=True)
    want = {
        "pathway_list":    "list/pathway",
        "rxn_of_pathway":  "link/reaction/pathway",
        "cpd_of_rxn":      "link/compound/reaction",
        "ko_of_rxn":       "link/ko/reaction",
        "ec_of_rxn":       "link/enzyme/reaction",
        "compound_list":   "list/compound",
        "reaction_list":   "list/reaction",
        "ko_list":         "list/ko",
    }
    out = {}
    for key, path in want.items():
        f = os.path.join(cache_dir, f"{key}.tsv")
        if refresh or not os.path.exists(f):
            print(f"  fetching {path}")
            with open(f, "w") as fh:
                fh.write(_get(path))
        out[key] = open(f).read()
    return out


def _pairs(text):
    for ln in text.strip().splitlines():
        if "\t" in ln:
            a, b = ln.split("\t")[:2]
            yield a.split(":")[-1].strip(), b.split(":")[-1].strip()


def _names(text):
    out = {}
    for ln in text.strip().splitlines():
        if "\t" not in ln:
            continue
        k, v = ln.split("\t")[:2]
        out[k.split(":")[-1].strip()] = v.split(";")[0].strip()
    return out


def build_pathway_graphs(links, enzyme_layer=True, drop_currency=True,
                         hub_degree=0, drop_global=True):
    """{pathway_id: (edges, names)} plus {pathway_id: (class, name)}."""
    rxn_of_path = defaultdict(set)
    for pid, rid in _pairs(links["rxn_of_pathway"]):
        if pid.startswith("map") and _RID_RE.match(rid):
            rxn_of_path[pid].add(rid)

    cpd_of_rxn = defaultdict(set)
    for rid, cid in _pairs(links["cpd_of_rxn"]):
        if _CID_RE.match(cid):
            cpd_of_rxn[rid].add(cid)

    if drop_currency:
        for r in cpd_of_rxn:
            cpd_of_rxn[r] -= set(CURRENCY)
    if hub_degree:
        # a second, data-driven cut for whatever the fixed list misses
        freq = Counter(c for cs in cpd_of_rxn.values() for c in cs)
        hubs = {c for c, n in freq.items() if n >= hub_degree}
        for r in cpd_of_rxn:
            cpd_of_rxn[r] -= hubs
        print(f"  hub cut at degree {hub_degree} removed {len(hubs)} compounds")

    ko_of_rxn, ec_of_rxn = defaultdict(set), defaultdict(set)
    for rid, k in _pairs(links["ko_of_rxn"]):
        ko_of_rxn[rid].add(k)
    for rid, e in _pairs(links["ec_of_rxn"]):
        ec_of_rxn[rid].add(e)

    cpd_names = _names(links["compound_list"])
    rxn_names = _names(links["reaction_list"])
    ko_names = _names(links["ko_list"])
    path_names = _names(links["pathway_list"])

    per_path, classes = {}, {}
    for pid, rxns in rxn_of_path.items():
        if drop_global and (pid.startswith("map011") or pid.startswith("map012")):
            continue          # map01100 "Metabolic pathways" etc. — supersets
        edges, names = [], {}
        add = lambda u, v, et, ut, vt: edges.append((u, v, et, ut, vt)) if u != v else None
        for r in sorted(rxns):
            names[rxn_id(r)] = rxn_names.get(r, r)
            for c in sorted(cpd_of_rxn.get(r, ())):
                names[cpd_id(c)] = cpd_names.get(c, c)
                add(cpd_id(c), rxn_id(r), "participates", "metabolite", "reaction")
            for k in sorted(ko_of_rxn.get(r, ())):
                names[ko_id(k)] = ko_names.get(k, k)
                if enzyme_layer and ec_of_rxn.get(r):
                    for e in sorted(ec_of_rxn[r]):
                        names[ec_id(e)] = e
                        add(ko_id(k), ec_id(e), "encodes", "gene", "enzyme")
                        add(ec_id(e), rxn_id(r), "catalyses", "enzyme", "reaction")
                else:
                    add(ko_id(k), rxn_id(r), "catalyses", "gene", "reaction")
        if edges:
            per_path[pid] = (edges, names)
            classes[pid] = ("Pathway maps", path_names.get(pid, pid))
    return per_path, classes, path_names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="kegg_ref")
    ap.add_argument("--source", choices=["modules", "pathways", "both"],
                    default="both",
                    help="which curated units become pathway graphs. Modules "
                         "are smaller and cleaner (their steps already exclude "
                         "currency metabolites); pathway maps are larger and "
                         "need the currency cut below. 'both' keeps each unit "
                         "as its own graph, which means modules are also "
                         "contained inside the pathways they belong to.")
    ap.add_argument("--keep-currency", action="store_true",
                    help="pathway mode only: keep H2O/ATP/NAD/H+ etc. Left in, "
                         "every reaction sits two hops from every other.")
    ap.add_argument("--keep-global", action="store_true",
                    help="keep KEGG's global overview maps (map011xx/map012xx, "
                         "e.g. map01100 'Metabolic pathways' at ~16,000 nodes). "
                         "They are unions of other pathways, not pathways.")
    ap.add_argument("--hub-degree", type=int, default=0,
                    help="pathway mode only: also drop any compound appearing "
                         "in at least this many reactions")
    ap.add_argument("--cache-dir", default="kegg_cache")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download even if cached")
    ap.add_argument("--no-enzyme-layer", action="store_true",
                    help="link KO straight to reaction, no EC node in between")
    ap.add_argument("--full-equations", action="store_true",
                    help="fetch R entries and use complete stoichiometry "
                         "(brings currency metabolites back in)")
    ap.add_argument("--max-nodes", type=int, default=64,
                    help="cap per pathway graph, matching the dataset")
    ap.add_argument("--min-nodes", type=int, default=4)
    ap.add_argument("--chunks-per-unit", type=int, default=1,
                    help="cut a unit larger than --max-nodes into up to this "
                         "many DISJOINT connected chunks instead of keeping "
                         "only its hub neighbourhood. 1 discards half of all "
                         "KEGG nodes; 12 recovers nearly all of them.")
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    per_unit, classes, unit_names, unit_kind = {}, {}, {}, {}
    names = {}

    if a.source in ("modules", "both"):
        raw = fetch_modules(a.cache_dir, a.refresh)
        parsed = {m: parse_entry(t) for m, t in raw.items()}
        classes = {m: module_class(f) for m, f in parsed.items()}

    equations = None
    if a.full_equations and a.source in ("modules", "both"):
        rids = set()
        for f in parsed.values():
            for rxns, _, _ in parse_reaction_block(f.get("REACTION", [])):
                rids.update(rxns)
        eq_raw = fetch_reactions(rids, os.path.join(a.cache_dir, "reactions"),
                                 a.refresh)
        equations = {r: parse_equation(t) for r, t in eq_raw.items()}
        print(f"parsed {len(equations)} reaction equations")

    if a.source in ("modules", "both"):
        for m, f in parsed.items():
            e, n = build_module_graph(m, f, not a.no_enzyme_layer, equations)
            per_unit[m] = (e, n); names.update(n)
            unit_names[m] = " ".join(f.get("NAME", [])).strip() or m
            unit_kind[m] = "module"
        print(f"modules: {len(per_unit)} units")

    if a.source in ("pathways", "both"):
        print("fetching bulk link tables (7 requests)")
        links = fetch_links(os.path.join(a.cache_dir, "links"), a.refresh)
        per_path, pclasses, pnames = build_pathway_graphs(
            links, not a.no_enzyme_layer, not a.keep_currency, a.hub_degree,
            not a.keep_global)
        for pid, (e, n) in per_path.items():
            per_unit[pid] = (e, n); names.update(n)
            unit_names[pid] = pnames.get(pid, pid)
            unit_kind[pid] = "pathway"
        classes.update(pclasses)
        print(f"pathways: {len(per_path)} units")

    rows, nodes, graphs, skipped = write_outputs(
        a.out_dir, per_unit, names, classes, a.max_nodes, a.min_nodes,
        unit_names, unit_kind, a.chunks_per_unit)

    from graphpath.data.node_types import kegg_node_type as get_node_type
    from collections import Counter
    tc = Counter(get_node_type(n) for n in nodes)
    nm = sum(1 for m in unit_kind.values() if m == "module")
    npw = sum(1 for m in unit_kind.values() if m == "pathway")
    print(f"\n{nm} modules + {npw} pathways -> {len(nodes):,} nodes, "
          f"{len(rows):,} edges")
    for t, c in tc.most_common():
        print(f"  {t:<12}{c:>7,}")
    print(f"pathway graphs: {len(graphs)}  "
          f"(skipped {skipped['no_edges']} with no edges, "
          f"{skipped['too_small']} under --min-nodes)")
    print(f"\nwrote to {a.out_dir}/")
    print("  kegg_edges.csv           <- EDGES_CSV in the notebook")
    print("  node_name_map.csv        <- NAME_MAP in the notebook")
    print("  kegg_sampled_graphs.pkl + node_id_map.pkl  <- the dataset")
    print("  node_types.csv, node_subsystems.csv, units.csv, meta.json")


if __name__ == "__main__":
    main()
