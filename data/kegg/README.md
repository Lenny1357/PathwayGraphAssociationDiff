# Swapping GraphPathModel from Human-GEM to KEGG modules

One script that downloads every KEGG module and produces the three inputs the
notebook consumes — reference network, node name map, node types — **plus the
pathway dataset itself**.

That last part is the difference from the other two setups. Human-GEM and
iML1515 are single genome-scale networks that have to be cut into pathways by
a sampler; `data/ecoli/README.md` says so explicitly and does not port one.
A KEGG module is *already* a curated pathway, so the training graphs come
straight from the source and every graph is a unit a biologist named.

## Files

| file | what it does |
|---|---|
| `build_kegg.py` | fetch all modules → edges CSV, name map, types, subsystems, **and the dataset pickle** |
| `node_types.py` | KEGG `get_node_type` / `get_common_name`, unchanged `TYPE_MAP` order |
| `test_offline.py` | parser + builder checks against real flat-file fragments, no network |
| `data/kegg.ipynb` | runs the build, then stats and figures |

Run `python test_offline.py` first. It passes here, so a failure on your side
is an environment problem rather than a bug.

## Build

```bash
python build_kegg.py --out-dir kegg_ref --cache-dir kegg_cache
```

No downloads to arrange by hand and no API key. 573 modules at 10 per request
is ~58 calls, cached on disk, so a re-run costs nothing. Roughly a minute.

Useful flags:

- `--no-enzyme-layer` — link KO straight to reaction, no EC node between
- `--full-equations` — fetch R entries and use complete stoichiometry
- `--max-nodes 64` — cap per graph, matching `PathwayGraphDataset`
- `--refresh` — re-download instead of using the cache

## What it produces

```
8,825 nodes   11,578 edges   573 modules   531 pathway graphs
gene 2,838 · reaction 2,343 · metabolite 2,025 · enzyme 1,712
nodes per graph: median 21, range 4-64      edge density ~0.10
```

| file | role |
|---|---|
| `kegg_module_edges.csv` | `EDGES_CSV` in the notebook |
| `node_name_map.csv` | `NAME_MAP` in the notebook |
| `kegg_sampled_graphs.pkl` + `node_id_map.pkl` | the dataset — feed straight to `PathwayGraphDataset` |
| `node_types.csv`, `node_subsystems.csv`, `modules.csv`, `meta.json` | metadata |

## Node ids

```
rxn_R01786    reaction
ec_2.7.1.1    enzyme
ko_K00844     gene          (KEGG Orthology group)
cpd_C00267    metabolite
```

The gene entity is a **KO group, not an organism's gene**. That is the point:
a module is defined across species, so a generated pathway is a claim about
function rather than about one genome. If you need organism-specific loci,
`link/<org>/ko` maps a KO onto them.

`TYPE_MAP` keeps the order `{gene: 0, metabolite: 1, enzyme: 2, reaction: 3}`,
so `model.set_node_types()` means what it did and a type-pair bias trained on
Human-GEM is at least index-compatible.

## Notebook changes

**Cell 0** — the type functions:

```python
from node_types import get_node_type, load_name_map, make_get_common_name
name_map = load_name_map('.../kegg_ref/node_name_map.csv')
get_common_name = make_get_common_name(name_map)
```

**Cell 0** — the dataset, which no longer needs a sampler:

```python
import pickle
pathway_dataset = pickle.load(open('.../kegg_ref/kegg_sampled_graphs.pkl', 'rb'))
node_id_map     = pickle.load(open('.../kegg_ref/node_id_map.pkl', 'rb'))
dataset = PathwayGraphDataset(node_id_map=node_id_map, graphs=pathway_dataset,
                              max_nodes=64, device=device)
```

**Cell 7** — the constants:

```python
EDGES_CSV = '.../kegg_ref/kegg_module_edges.csv'
NAME_MAP  = '.../kegg_ref/node_name_map.csv'
```

There is no co-expression source here, so `CoExpressionBenchmark`, the
conditioning cells and the propagation cells have nothing to read. Either
skip them or bring an expression compendium keyed on KO.

## Three things that behave differently

**Currency metabolites are already gone, and this is the big one.** The
topology comes from each module's `REACTION` block, which lists main-chain
transformations: `R01786  C00267 -> C00668` is hexokinase *without* the
ATP/ADP. So there is no hub-stripping heuristic to tune — the curation did it.
It shows up in the numbers against the E. coli build:

| | KEGG | E. coli (iML1515) |
|---|---|---|
| max degree | **80** | 286 |
| degree CV | **1.05** | 2.37 |
| top-1% share of edges | **8.0%** | 11.6% |

`--full-equations` puts the hubs back if you want to see what the other
pipelines are fighting.

**The graph is deliberately disconnected.** 48 components, because modules are
separate curated units rather than one network. In the other two setups that
would be a defect; here it is the structure being exploited.

**There is no expression axis at all.** KEGG modules are species-agnostic, so
there is no tissue and no condition to condition on. Everything in the project
that reads `ExpressionLoader` is inert. What KEGG gives instead is a clean,
curated, sampler-free pathway dataset — useful for asking whether the model
generates plausible *pathway structure*, separately from whether it can be
steered by expression.

## Not covered

A KO→organism mapping, if you want to project generated pathways onto a
specific genome, and an expression compendium keyed on KO. Both are one REST
call and one join away (`link/hsa/ko`, `link/eco/ko`), but neither is built
here because neither is needed to train on this dataset.
