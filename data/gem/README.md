# Steady-state flux checks on Human-GEM and iML1515, in torch

Stoichiometric validity for generated pathways, and a differentiable version
of the same score for steering the diffusion model.

## Files

| file | what it does |
|---|---|
| `build_gem.py` | downloads Human-GEM → `gem_ref/gem.npz` + `meta.json` |
| `build_gem_ecoli.py` | iML1515 → `gem_ref_ecoli/`, same schema |
| `torch_fba.py` | `StoichModel`, `null_dim`, `flux_residual`, `dead_end_metabolites`, `mass_imbalance` |
| `validity.py` | `FluxValidityBenchmark` (plugs into `benchmarks/`), `guidance_residual` |
| `test_offline.py` | 32 checks on networks worked out by hand, plus Human-GEM itself |
| `flux_sampler.py` | `FluxSampler` — subgraphs that can actually carry flux |
| `test_flux_sampler.py` | 19 checks on the sampler |
| `data/gem.ipynb` | build, inspect, and calibrate against real pathways |

```bash
python build_gem.py          # ~10 s, downloads 3.5 MB
python build_gem_ecoli.py    # ~30 s, reuses data/ecoli/iML1515.json
python test_offline.py       # 32 passed, 0 failed
```

## The E. coli reference

`gem_ref_ecoli/` holds iML1515 in the **same schema** as `gem_ref/`, so
`StoichModel`, `FluxValidityBenchmark`, `guidance_residual` and `FluxSampler`
all read it with no code change:

```python
sm = StoichModel("data/gem/gem_ref_ecoli")
# StoichModel(1,877 mets x 2,712 rxns, 10,575 nnz, cpu)
```

The id bridge needs no change either. This project's E. coli node space is
already BiGG ids verbatim — reactions `rxn_PFK`, metabolites `10fthf_c`,
genes `b0002` — which is what `StoichModel.strip` expects. Coverage against
`ecoli_node_id_map.pkl`: 2,242/2,712 reactions (83%), 1,651/1,877 metabolites
(88%), 1,413/1,516 genes (93%). The GEM is the larger object, so the missing
fraction is entities the pathway sampler never placed in a chunk, not
entities the bridge fails to resolve.

**Nothing needed downloading.** `data/ecoli/iML1515.json` is byte-identical
to BiGG's canonical copy (sha256 `b0f9199f…`, 3,062,491 bytes), so the
builder uses it and only falls back to the network if it is absent.

### It verifies itself

Unlike the Human-GEM builder, this one solves the growth LP with scipy and
compares against the published iML1515 optimum:

```
optimal   mu = 0.8770 /h   published 0.8770   MATCH
```

That is worth more than it looks. Shape and nnz counts pass even when signs
are flipped or bounds are transposed; reproducing the published growth rate
to four figures means S, `lb`, `ub` and `c` are all assembled correctly, on
the M9-glucose-aerobic medium the file's own exchange bounds encode. Pass
`--no-verify` to skip it.

### How it compares with Human-GEM

| | Human-GEM | iML1515 |
|---|---|---|
| metabolites × reactions | 8,461 × 12,931 | 1,877 × 2,712 |
| genes | 2,848 | 1,516 |
| checkable reactions that balance | 98.1% (9,656/9,842) | **100.0%** (2,200/2,200) |
| formulas that parse | 8,449/8,461 | **1,877/1,877** |

iML1515 is small enough that the dense S is 41 MB rather than 875 MB, so the
`dense` property is usable interactively here where it is not on the human
model. It is also far better curated — every formula parses and every
checkable reaction balances to 1e-6.

Three exclusions from the balance check, all by design: 337 exchange/sink
reactions have one participant, 97 metabolites carry `R`/`X` placeholders
(ACP-bound acyl chains, Fe-S clusters), and both `BIOMASS_*` pseudo-reactions
drain ~64 net carbon by construction. Note the second biomass reaction is
*not* the objective (`c = 0`), so excluding on `c` alone leaves it in — the
builder excludes on the `BIOMASS` id prefix as well.

### Real vs rewired, on E. coli

48 real iML1515 chunks against the same node sets with edges shuffled:

| metric | real | rewired |
|---|---|---|
| `participation_precision` | **1.000** | 0.136 |
| `participation_recall` | **1.000** | 0.047 |
| `isolated_frac` | 0.000 | 0.891 |
| `mass_balanced_frac` | 0.314 | 0.000 |
| `flux_residual` | 0.059 | **0.000** |

Participation precision/recall are the metrics that discriminate. Read
`flux_residual` with the caution the section below sets out — it is *lower*
for the rewired graphs, because shuffling isolates 89% of the nodes and an
isolated node has no constraint to violate. The same inversion measured on
Human-GEM holds here.

## How this differs from COBRApy

**It is not a replacement, and for anything COBRApy can do you should use
COBRApy and believe it over this.** COBRApy calls a real LP solver (GLPK,
Gurobi) through optlang and returns a certified optimum from exact simplex
pivoting. Nothing here does that. `flux_residual` is a projected-gradient
minimum of a Rayleigh quotient — it answers "how close to a steady state can
this subnetwork get" rather than "is it feasible, yes or no", and it carries
float64 rounding rather than a solver's careful conditioning.

What it buys, and why those two things are the whole reason this folder exists:

**Batching.** COBRApy solves one model per call, on the CPU, through a Python
object graph. Scoring 32 generated graphs at each of 200 reverse steps is
6,400 solver calls per sample. Here the batch is one dense `(B, m, n)` tensor
and one batched SVD.

**Gradients.** Guidance needs `d(score)/d(edge probability)`. An LP returns a
number, and the vertex it came from is a discontinuous function of the input —
there is no useful derivative. `flux_residual` is a smooth function of the
stoichiometry handed to it, so autograd reaches the model's edge logits.

Scale matters here. The guidance case works on ≤64-node subnetworks, so the
extracted matrices are perhaps 40×25 and the dense batched treatment is right.
Do not point this at the full 8,461 × 12,931 model — use COBRApy for that.

If you install cobra later, the useful move is to validate this layer against
it on a few hundred subnetworks rather than to replace it.

## What it computes

| | exact? | differentiable? | cost |
|---|---|---|---|
| `mass_imbalance` | yes | yes | one matmul |
| `dead_end_metabolites` | yes | no (boolean) | one reduction |
| `null_dim` | yes | no (rank) | batched SVD |
| `flux_residual` | no — local minimum | **yes** | SVD + ≤200 matmuls |

`null_dim` is `n_reactions − rank(S)`: zero means the subnetwork is completely
blocked, since no flux assignment holds every internal metabolite at steady
state. It **ignores irreversibility**, so it is a necessary condition, not a
sufficient one — `test_offline.py` has the worked example where `null_dim` is
1 but the only null direction needs a reaction to run backwards.

`flux_residual` minimises `‖Sv‖²/‖v‖²`. The Rayleigh quotient rather than plain
least squares because `‖Sv‖²` is minimised at `v = 0`, which is a steady state
and tells you nothing; dividing by `‖v‖²` excludes that without an extra
constraint, and makes the score scale-free — so flux *magnitude* bounds drop
out and only the *sign* constraints survive, which is the part of `lb/ub` that
carries biology.

## Read it against real pathways, never against zero

A real chunk is cut out of a bigger network, so its boundary metabolites have
no source or sink inside the chunk and it scores badly in absolute terms too.
`FluxValidityBenchmark` inherits `evaluate_dataset`, so the reference is one
call away. The only question worth asking is whether generated graphs score
like real ones.

## The trap: an empty graph is perfectly balanced

From `test_offline.py`, on the same 14 glycolysis reactions — once wired with
their true participations, once with random edges at the same density:

| | participation precision | **flux residual** | unusable metabolites | mass-balanced |
|---|---|---|---|---|
| correctly wired | 1.000 | 1.338 | 0.62 (0 isolated) | 1.00 |
| randomly wired | 0.122 | **0** | 0.98 (0.76 isolated) | 0.00 |

The random graph gets a *perfect* residual, and the correct one does not. Not
a bug: draw few enough edges and `S` is mostly zeros, everything is null space,
and the residual is trivially zero. Every other column ranks them correctly.

Two consequences, and they are the main thing to take from this folder:

1. **Never report `flux_residual` alone.** Pair it with
   `participation_precision` and `unusable_frac`, which is what the benchmark
   does.
2. **Never guide on `flux_residual` alone.** Gradient descent on it will find
   the empty graph, because the empty graph is the global optimum. Guidance
   needs a term that keeps edges — the edge-count prior the model already has,
   or an explicit penalty. Start with a small `scale` and check the benchmark
   after, because pushing the residual down is not the same as making the
   graph biologically better.

`unusable_frac` exists because of this too. `dead_end_frac` counts only
metabolites that are wired in at all, so a sparse graph flatters itself by
leaving most of its metabolites isolated; `unusable_frac` is the union of dead
and isolated, and it is the column that ranks the two rows correctly.

## Guidance

```python
from torch_fba import StoichModel
from validity import guidance_residual

st = StoichModel('data/gem/gem_ref', device=device)
tables = st.token_tables(dataset.id_to_node)          # once

p = edge_logits.softmax(-1)[..., 1].detach().requires_grad_()
g = torch.autograd.grad(guidance_residual(st, tables, node_t, p).sum(), p)[0]
edge_logits[..., 1] = edge_logits[..., 1] - scale * g
```

Insert it in `GraphDiffBERTUniform.sample` between the forward pass and
`posterior_logits`. Note the anchoring: the posterior only moves 2–5% of edges
per step in the middle of the chain, so guidance applied there has many steps
to act through and `scale` should be small.

## The id bridge

graphpath node ids map onto Human-GEM directly — reactions are `rxn_MAR03905`
(strip the prefix), metabolites are bare `MAM01965c`, genes are bare
`ENSG00000130203`. `enz_*` nodes have no counterpart and are reported as
unmatched rather than silently dropped. `StoichModel.index` returns
slot-aligned arrays so a generated adjacency can be cut to the same slots.

## Notes on the source

The `.mat` is used rather than the SBML: 0.78 MB against 43 MB, and it carries
`S`, `lb`, `ub`, `c`, `rxnGeneMat`, `metFormulas` and `metCharges` already
assembled, so neither cobra nor libsbml is needed.

```
S            8,461 metabolites x 12,931 reactions, 55,198 nonzeros
reversible   5,725      objective: MAR13082 (biomass)
elements     25         mass+charge: 9,656 of 9,842 checkable reactions balance (98.1%)
```

3,089 reactions are not mass-checkable: exchange and biomass pseudo-reactions
are unbalanced by design, and lipid/glycan species carry `R`/`X` placeholder
atoms in their formulas. They are labelled, not counted as failures.

MATLAB cell arrays come back ragged — seven of Human-GEM's `metFormulas` are
empty ndarrays rather than `""`, and `.astype(str)` raises on them rather than
coercing. `as_str_list` handles it.

## Sampling subgraphs that can carry flux

```python
from torch_fba import StoichModel
from flux_sampler import FluxSampler

st = StoichModel('data/gem/gem_ref')
s  = FluxSampler(st, max_nodes=64)        # or 128
graphs = s.sample(n=2000, seed=0)
ds, node_id_map = s.to_dataset(graphs)    # PathwayGraphDataset format
```

Prebuilt: `gem_ref/humangem_flux_graphs_{64,128}.pkl` + matching node id maps.

**The samplers elsewhere in this project grow by NODE** — BFS outward until the
budget is spent — which cuts reactions in half. A reaction with three of its
five metabolites is not chemistry: mass balance is undefined for it, and the
steady-state residual measures the truncation rather than the biology.

**This one grows by REACTION.** Adding a reaction adds all of its metabolites,
so every reaction in the result is complete. Same seeds, same 64-node budget:

| | completeness | mass-balanced | null_dim | residual |
|---|---|---|---|---|
| **flux-aware** | **1.00** | **1.00** | 9.7 | 1.121 |
| naive node-BFS | 0.41 | 0.38 | 47.0 | 0.074 |

Note the naive sampler scores BETTER on residual and null_dim. That is the
empty-graph trap from the section above, in a new guise: 53 reactions against
11 metabolites is barely constrained, so almost anything satisfies it. Only
completeness and mass balance can compare two samplers; the flux scores cannot.

### What drives the growth

- **completeness** — a reaction is admitted only if all its metabolites fit.
- **flux closure** — candidates are scored by how many *dangling* metabolites
  they resolve. A metabolite only produced inside the subgraph, never consumed,
  cannot sit at steady state; a reaction that consumes it closes the loop. This
  is what turns a connected blob into something with a nonzero flux space.
- **budget** — a reaction dragging in six new metabolites costs six nodes, so
  candidates are scored per node spent.

`sample()` then rejects graphs whose flux space is empty (`null_dim == 0`) —
roughly a quarter of attempts at 64 nodes.

### Three settings that matter

**`expand_max_degree=30`** — hubs are kept as NODES (water and protons are real
participants; dropping them makes every reaction fail mass balance) but are not
traversed THROUGH. 2% of metabolites carry 49% of all incidences, so expanding
from water puts 2,722 reactions in the candidate list on every step. This is
both a quality fix and a 76x speedup: 0.28 s to 3.7 ms per graph.

**`gene_frac=0.20`** — reserves a fifth of the budget for gene nodes. Without
it reactions and metabolites spend everything and genes almost never appear
(184 across 400 graphs), leaving the model a node type it never sees. Genes
cannot affect flux or mass balance, so the reservation costs chemistry per
graph, not correctness.

**`max_nodes`** — the requested parameter. Output at each:

```
 64 nodes: 2,000 graphs in 21s   22 reactions + 29 metabolites + 12 genes
           mass-balanced 0.997   null_dim 4.3   density 0.072   121 subsystems
128 nodes: 2,000 graphs in 42s   48 reactions + 54 metabolites + 24 genes
           mass-balanced 0.991   null_dim 9.5   density 0.041   129 subsystems
```

Edges are metabolite-reaction and gene-reaction only. Two metabolites of one
reaction are connected *through* it, which is what the reaction node is for;
adding the clique would triple the edge count without adding information.

## Not covered

FVA, pFBA, gap-filling, gene knockouts, and anything needing a certified
optimum. Those are COBRApy's, and the `rxnGeneMat` is already parsed into
`gem.npz` if you want to build knockout support on top.
