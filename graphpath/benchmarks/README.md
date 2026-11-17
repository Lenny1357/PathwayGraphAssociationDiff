# Benchmarks

Metrics for judging generated pathways. Each one scores a batch of graphs —
the `(node_toks, adjacency)` pair `model.sample()` returns — and reports
per-graph values plus a summary.

| File | Contents |
| --- | --- |
| [base.py](base.py) | `Benchmark` base class, `BenchmarkResult` |
| [coexpression.py](coexpression.py) | `CoExpressionBenchmark` — co-expression of the genes placed in one pathway |
| [discovery.py](discovery.py) | `DiscoveryFinder` — novel links supported by co-expression propagated through the metabolic graph |
| [nearest_training.py](nearest_training.py) | `NearestTrainingBenchmark` — closest training pathway by node overlap (memorisation) |
| [novel_edges.py](novel_edges.py) | `NovelEdgeBenchmark` — for edges absent from training, how many hops apart their endpoints are in the reference |
| [perturbation.py](perturbation.py) | `PerturbationBenchmark` — do the grouped genes respond alike when knocked down |
| [../plotting/](../plotting/) | publication figures, incl. three network maps (co-expression, metabolic, model embedding) — `graphpath.plotting` |
| [baselines.py](baselines.py) | `random_like`, `shuffled_like`, `compare` |
| [known_structure.py](known_structure.py) | `KnownStructureBenchmark` — how much of a generated pathway exists in the curated network |
| [reference_network.py](reference_network.py) | `ReferenceNetwork` — the curated edge table, in node-id space |
| [../data/node_types.py](../data/node_types.py) | shared node id → name → type helpers — `graphpath.data.node_types` |
| [flux.py](flux.py) | `FluxValidityBenchmark` — stoichiometric validity via `graphpath.fba`; `guidance_residual` for steering |
| [topology.py](topology.py) | `TopologyBenchmark` — graph structure (degree, clustering, path lengths, motifs) against the training distribution |

## Usage

```python
from graphpath.benchmarks import CoExpressionBenchmark
from graphpath.data.expression import ExpressionLoader

bench = CoExpressionBenchmark(
    dataset,
    expression_loader=ExpressionLoader(),
    device=device,
    name_map_path='pt_files/node_name_map.csv',   # optional
)

print(bench.evaluate_model(model, n_graphs=32, N=64))   # generated
print(bench.evaluate_dataset(dataset, n_graphs=32))     # real pathways, for reference
```

A generated score means little on its own — read it against the dataset
reference, and against a shuffled control (sample, then permute the gene
tokens between graphs) if you want a floor.

## Interpreting a result

```
gene co-expression within generated pathways  (29/32 graphs scored, coverage 91%)
  genes_per_graph_mean         8.58
  unique_genes_per_graph       8.08
  duplicate_gene_rate          0.06
  co_expression_mean           0.31
  co_expression_std            0.05
  ! 3 graph(s) skipped for having < 2 gene nodes
```

`coverage` matters as much as the score: a strong number over 3 of 32 graphs
is not a strong result. Graphs with fewer than `min_genes` gene nodes are
skipped, since co-expression is undefined for a single gene.

`duplicate_gene_rate` matters too. Generated graphs repeat entities, and a
repeated gene contributes a self-pair with co-expression 1.0. The benchmark
de-duplicates by default (`unique_genes=True`); turning that off inflated the
score from −0.01 to +0.18 on a test batch.

## `KnownStructureBenchmark`

Checks every generated node and edge against the curated network.

```python
from graphpath.benchmarks import ReferenceNetwork, KnownStructureBenchmark

ref = ReferenceNetwork(dataset.node_to_id, device=device,
                       num_nodes=dataset.vocab_size)   # pass the real vocab size
ref.load_edges('pt_files/edges.csv')
ref.build_adjacency()
subs = ref.build_pathway_associations()                # optional

bench = KnownStructureBenchmark(ref, node_to_subsystems=subs, device=device)
node_toks, adj = model.sample(N=64, B=32, device=device)
print(bench.evaluate(node_toks, adj))
```

| metric | meaning |
| --- | --- |
| `node_known_frac` | fraction of generated nodes that exist in the reference network |
| `unique_node_frac` | 1.0 means no repeated entities within a graph |
| `edge_precision` | generated edges that are real edges |
| `edge_recall` | reference edges among the generated nodes that the model found |
| `novel_edge_frac` | `1 - precision`; invented links |
| `n_subsystems` / `subsystem_purity` | how concentrated the nodes are; a real pathway sits in few subsystems, a random node set sprays across many |

Fidelity and novelty pull against each other. 100% precision with 0% novel
edges means the model memorised a reference pathway; 0% precision means it
invented everything. Neither alone is a good result — read them together,
and against `evaluate_dataset` for what real pathways score.

### Highlighting recovered edges in plots

`create_graph_objs(..., subgraphs=...)` draws reference edges thick and
green (the `weight == 3` branch in `plotting.py`). The benchmark produces
that input directly:

```python
subgraphs = bench.known_edge_indices(node_toks)        # list of (2, E) slot indices
graphs, names, labels, types, ids = create_graph_objs(
    node_toks, adj, dataset, get_common_name, get_node_type, subgraphs=subgraphs)
```

## `NearestTrainingBenchmark`

For each generated graph, the training graph sharing the most nodes.

```python
nn = NearestTrainingBenchmark(dataset, device=device)
print(nn.evaluate(node_toks, adj))
idx, jac = nn.nearest(node_toks, k=3)      # inspect what it matched
```

| metric | meaning |
| --- | --- |
| `nn_node_jaccard_mean` | overall set similarity to the closest training pathway |
| `nn_containment_mean` | how much of the generated graph fits inside one training example; high with low Jaccard = a reproduced fragment |
| `nn_edge_jaccard_mean` | edge agreement with that same match — identical nodes wired differently is recombination, wired identically is a copy |
| `near_duplicate_rate` | fraction at or above `duplicate_threshold` |
| `distinct_matches` | how many different training pathways were matched; far below the graph count means mode collapse |

Score the training set against itself with `exclude_identical=True` for the
reference — otherwise every graph matches itself at 1.0. That column answers
"how similar are real pathways to each other", which is the level a model
with genuine novelty should sit at.

## Adding a benchmark

Subclass `Benchmark`, set `name`, implement `evaluate`:

```python
from graphpath.benchmarks.base import Benchmark, BenchmarkResult

class EdgeTypeViolationBenchmark(Benchmark):
    name = "type-adjacency violations"

    def evaluate(self, node_toks, adjacency=None, **kw):
        ...
        return BenchmarkResult(self.name, values, summary,
                               n_graphs=len(node_toks), n_scored=len(values))
```

`evaluate_model` and `evaluate_dataset` come from the base class. Export it
from `__init__.py`.

## Expression source

`ExpressionLoader` lives in [graphpath/data/expression.py](../graphpath/data/expression.py),
since the model's co-expression *conditioning* needs the same tables the
benchmark scores against.

```python
from graphpath.data.expression import ExpressionLoader

loader = ExpressionLoader(path='pt_files', device=device)
```

It reads `co_expression_sparse_*_<tissue>.pt` and
`tissue_expression_dict.pkl` from `path` — without changing the working
directory, so relative paths the caller holds keep working.

The same object feeds conditioning:

```python
co_expr = loader.as_condition(tissue_ids)      # {"coexpr_sparse", "tissue_ids"}
model.compute_loss(node_feats, adjcs, mask, co_expr=co_expr)
model.sample(N=64, B=8, co_expr=co_expr, guidance_scale=2.0)
```

`within_list_co_exp(gene_ids)` returns `(n_tissues, 4)` with columns
`expr_count, expr_mean, coexpr_count, coexpr_mean`.

### Propagated co-expression

Raw tables only speak to genes. `PropagatedExpression` spreads the signal
along the metabolic network so metabolites and reactions inherit a value,
and is a drop-in replacement:

```python
from graphpath.data.propagation import PropagatedExpression

source = PropagatedExpression(expr_loader, ref.adj_sparse, hops=3, decay=0.5)
bench  = CoExpressionBenchmark(dataset, expression_loader=source,
                               device=device, node_filter="all")
```

`hops=0` reduces exactly to the raw loader — use it as the control, so any
change in a number is attributable to the propagation alone. `node_filter`
stays `"gene"` unless propagating; `"all"` scores every entity type. Any other object with
that method works; a non-matching shape falls back to a flat mean and says
so in the notes.

### Reading the tissue columns

A pathway is expressed in a *tissue*, so averaging across all of them is the
wrong summary — genes strongly co-expressed in liver and silent elsewhere
get dragged toward zero. The benchmark reports both:

| metric | meaning |
| --- | --- |
| `coexpr_best_tissue_mean` | co-expression in each graph's best tissue, averaged |
| `coexpr_mean_over_tissues` | flat mean; much lower, and only comparable across runs |
| `coexpr_pairs_found_mean` | gene pairs with any co-expression entry |
| `top_tissue` / `top_tissue_share` | which tissue wins most often, and how dominantly |

`last_per_tissue` on the benchmark holds the `(n_scored, n_tissues)` matrix
if you want to plot the distribution.

## Topology

`TopologyBenchmark` ignores which entities were emitted and asks only whether
the graph has the shape of a pathway.

```python
from graphpath.benchmarks import (TopologyBenchmark, topology_distances, topology_table,
                        composition_table, topology_figure, random_like)

topo  = TopologyBenchmark(dataset.id_to_node)          # heavy=False to skip
gen   = topo.evaluate(nodes, adj)                      # betweenness/modularity
train = topo.evaluate_dataset(dataset, n_graphs=96)
rand  = topo.evaluate(*random_like(nodes, adj, dataset.vocab_size))

results = {"generated": gen, "random": rand, "training": train}
print(topology_table(results))                 # mean (median ± sd) per metric
print(composition_table(results, "edge_types"))
print(topology_distances(gen, train))          # W1 + KS per distribution
topology_figure(results, save="topology.png")
```

Three things about it are deliberate and change what the numbers mean:

**Padding is dropped by token, not by degree.** Dropping every isolated node
first is the obvious move, and it silently deletes a real defect: a node the
model emitted and never connected is not padding. `isolated_node_frac`
reports it instead.

**Path length and diameter come from the largest connected component**, not
from connected graphs only. Skipping disconnected graphs means a model that
fragments its output is never charged for it — the bad graphs just leave the
average. `n_components` and `lcc_frac` report the fragmentation separately.

**`shuffled_like` is the wrong control here.** It permutes identities and
keeps the adjacency, and topology does not depend on identity — so the
shuffled arm *is* the generated arm, except that displaced padding strands
real nodes as isolates and makes it look worse for a non-structural reason.
Use `random_like` as the floor.

`topology_distances` is the summary worth quoting: comparing means hides
shape, since a generated set can match the training mean degree exactly while
having no hubs and no leaves. It reports 1-Wasserstein (normalised by the
reference mean, so counts and fractions share a scale) and the KS statistic
over the pooled degree, clustering, path-length and shared-neighbour
distributions. Zero is a perfect match.

## `NovelEdgeBenchmark`

`KnownStructureBenchmark` asks whether a generated edge exists in the curated
network. This asks the follow-up about the ones that were never in TRAINING:
how far apart are their endpoints in the reference, and what type pair are they?

```python
from graphpath.benchmarks import (ReferenceNetwork, NovelEdgeBenchmark, novel_edge_table)

ref = ReferenceNetwork(train_dataset.node_to_id, device=device,
                       num_nodes=train_dataset.vocab_size)
ref.load_edges(EDGES_CSV); ref.build_adjacency()

novel = NovelEdgeBenchmark(train_dataset, ref, node_type_fn=get_node_type)
res = novel.evaluate(node_toks, adjacency)
print(novel_edge_table(res))
```

The hop count separates three things that "novel edge" otherwise lumps together:

| hops | what it means |
|---|---|
| **1** | the edge IS in the reference but was never in a training chunk — the model put back something real it had not been shown |
| **2** | endpoints share a reference neighbour: a plausible shortcut, not a discovery |
| **3+** | a long-range claim, and mostly a mistake |
| unreachable | different components entirely |

**Always read it against the control**, which `evaluate` computes automatically:
random endpoint pairs with the same node-type mix as the novel edges. A dense
reference has a small diameter, so a share of random pairs also land 2-3 hops
apart; only the excess is informative. `excess_le2_over_control` is that number.

Sanity check on E. coli, 48 held-out REAL pathways scored against a model that
was trained on the other 1,400 graphs:

```
hops apart in the reference      novel   control   ratio
  1 (in reference)              100.0%      1.1%    94.0
  unreachable                     0.0%     95.2%     0.0

by node-type pair                  n  <=2 hops  median
  metabolite-reaction             27    100.0%     1.0
  enzyme-gene                     15    100.0%     1.0
```

Real pathways contribute few novel edges (1.3% of their edges) and every one
of them is a genuine reference edge. Random tokens at the same density score
`novel_in_reference` 0.000 and 97% unreachable. That 94x gap is the range the
metric has to work in.

Run `python benchmarks/test_novel_edges.py` — 18 checks on a path graph whose
hop distances are worked out by hand.

## `PerturbationBenchmark`

The causal counterpart of `CoExpressionBenchmark`. Co-expression asks whether
two genes move together; Perturb-seq asks what happens when you knock one out.

```python
from graphpath.benchmarks import PerturbationBenchmark, perturbation_table

pert = PerturbationBenchmark('data/perturbseq/perturb_ref', dataset, min_genes=3)
print(perturbation_table({"generated": pert.evaluate(nodes, adj),
                          "shuffled":  pert.evaluate(sn, sa),
                          "random":    pert.evaluate(rn, ra)}))
```

Two scores, measuring different things:

- **profile similarity** — `corr(profile_i, profile_j)` for two *perturbed*
  genes: knocking these down has the same downstream consequence.
- **cross effect** — `|effect[i, j]|`: how much knocking down *i* moves *j*.
  Directed and causal, and the thing a correlation cannot express. Needs *i*
  perturbed and *j* measured, so it covers a smaller set.

Both come with a **size-matched random control** drawn from the same metabolic
gene pool, and `*_excess` is the number to read. Raw values are close to
meaningless on their own — the co-expression work found random gene sets
reaching 70–100% of the pathway score.

### Read `pooled_d`, not the per-graph mean

Only 847 of Human-GEM's 2,848 genes have a profile that survived Replogle's
phenotype filter, so a 64-node graph contributes about **2.4** scorable genes
and its per-graph mean is computed over one or two pairs. Averaging within a
graph before comparing throws the power away. `pooled_d` pools every
within-graph pair and compares distributions, which is the correct estimator
at this coverage and is directly comparable to effect sizes measured elsewhere.

### Validation

Synthetic, with a planted pathway (`python benchmarks/test_perturbation.py`,
13 checks): planted set scores 0.802 against a 0.057 null, z = 14.6; a random
set of the same size scores 0.002.

Real Human-GEM gene sets:

| | pooled d | pooled AUC | genes/graph |
|---|---|---|---|
| Human-GEM subsystem sets | **0.155** | 0.532 | 9.8 |
| flux-sampler chunks | 0.072 | 0.510 | 2.4 |
| shuffled | −0.136 | 0.465 | 2.4 |

It orders them correctly, but note the second row. **BFS-grown chunks are not
functionally coherent gene sets** — they span whatever the network reaches
within the node budget, crossing subsystems — so a functional signal is weak on
them by construction. That is a finding about the chunks, not the metric, and
it is worth knowing before reading a generated graph's score against them.
