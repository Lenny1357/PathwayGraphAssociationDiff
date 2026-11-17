# Swapping GraphPathModel from Human-GEM to E. coli

Four scripts that produce the E. coli counterparts of the three inputs your
notebook consumes: the reference network, the node name map, and the
per-context co-expression tables. Plus a curated regulatory layer Human-GEM
has no equivalent of.

Everything keys on **b-numbers** (`b4025`). iML1515's genes, PRECISE-1K's
expression matrix, RegulonDB and EcoliNet all use them natively, so there is
no ID mapping step anywhere in this pipeline — the main source of join loss on
the human side disappears.

## Files

| file | what it does |
|---|---|
| `build_reference.py` | iML1515 → `ecoli_gem_reaction_node_edges.csv`, `node_name_map.csv`, `node_types.csv`, `node_subsystems.csv` |
| `build_coexpression.py` | PRECISE-1K → per-condition sparse co-expression in node-id space |
| `expression_ecoli.py` | `EcoliExpressionLoader`, drop-in for `ExpressionLoader` |
| `build_regulondb.py` | RegulonDB TF–gene → a second reference network, same schema |
| `build_pathways.py` | iML1515 → `ecoli_sampled_graphs.pkl` + `ecoli_node_id_map.pkl`, the training dataset |
| `node_types.py` | E. coli `get_node_type` / `get_common_name` |
| `test_offline.py` | synthetic end-to-end check, no network or cobra needed |
| `ecoli_setup.ipynb` | runs the build, then stats and figures for the network **and the co-expression tables** |
| `ecoli_coexpression_information.ipynb` | the E. coli counterpart of `coexpression_information.ipynb` — MI between co-expression and pathway activity, and how different the 24 condition networks are |

Run `python test_offline.py` first. It passes here, so a failure on your side
is an environment problem rather than a bug in these scripts.

## Downloads

```bash
# 1. the GEM  (json needs no cobra; the xml does)
wget http://bigg.ucsd.edu/static/models/iML1515.json

# 2. the expression compendium
git clone https://github.com/SBRG/precise1k
#   precise1k/data/precise1k/log_tpm_qc.csv    1035 samples
#   precise1k/data/precise1k/metadata_qc.csv
#   precise1k/data/k12_modulome/               2710 samples, if you want more

# 3. the regulatory network  (optional second reference)
#    regulondb.ccg.unam.mx -> Downloads -> "TF - gene interactions"
```

## Build

```bash
python build_reference.py --model iML1515.json --out-dir ecoli_ref

# look at the metadata before choosing the condition axis
python build_coexpression.py --expr precise1k/data/precise1k/log_tpm_qc.csv \
    --meta precise1k/data/precise1k/metadata_qc.csv --list-columns

python build_coexpression.py \
    --expr precise1k/data/precise1k/log_tpm_qc.csv \
    --meta precise1k/data/precise1k/metadata_qc.csv \
    --stratify-by "carbon source" \
    --node-id-map pt_files/ecoli_node_id_map.pkl \
    --out-dir ecoli_expr --top-k 32 --min-samples 12

python build_regulondb.py --network network_tf_gene.txt \
    --name-map ecoli_ref/node_name_map.csv --out-dir ecoli_ref
```

`--node-id-map` matters: `CoExpressionPropagator(ref.adj_sparse,
expr_loader.co_expression[tissue], ...)` multiplies the co-expression matrix
against the metabolic adjacency, so both have to be in dataset node-id space.
Build the pathway dataset first, then point the co-expression build at the
`node_id_map.pkl` it produced. (If your `ExpressionLoader` in fact keyed on
gene symbols and re-indexed internally, drop the flag — the output is then in
gene space and `meta.json` records `space: "gene"`.)

## Notebook changes

**Cell 0** — paths and the type functions:

```python
from node_types import get_node_type, load_name_map, make_get_common_name
name_map = load_name_map('.../ecoli_ref/node_name_map.csv')
get_common_name = make_get_common_name(name_map)
```

The old `get_node_type` keyed on `rxn_` / `MAM` / `ENS` prefixes. The E. coli
one keys on `rxn_` / `enz_` / `b####`, everything else a metabolite. The
`TYPE_MAP` order is unchanged, so `model.set_node_types()` still means the
same thing.

**Cell 7** — the three constants:

```python
EXPR_PATH = '.../ecoli_expr'
EDGES_CSV = '.../ecoli_ref/ecoli_gem_reaction_node_edges.csv'
NAME_MAP  = '.../ecoli_ref/node_name_map.csv'

from expression_ecoli import EcoliExpressionLoader
expr_loader = EcoliExpressionLoader(path=EXPR_PATH, device=device)
```

Everything downstream — `per_tissue_coexpr`, best-tissue selection in the
discovery cell, `tissue_heatmap` — works unchanged; "tissue" now reads
"condition".

## Three things that will behave differently

**The graph is ~4× smaller.** iML1515 is 1515 genes / 1877 metabolites / 2712
reactions against Human-GEM's ~3600 / 8400 / 13100. With `max_nodes=64`, a
sampled subgraph now covers a large share of an entire subsystem, so
`NearestTrainingBenchmark` gets much sharper teeth: expect `near_duplicate_rate`
to rise at the capacity you trained at on human. Drop `d_model` / `n_layers`
before concluding the model memorises.

**Currency metabolites are worse here.** `h_c`, `atp_c`, `h2o_c` and friends
appear in hundreds of a 2712-reaction model's reactions. Left in, every
reaction is two hops from every other and the propagation cell saturates at
`hops=2` instead of 3. `build_reference.py` strips a standard list by default;
`--hub-degree 60` adds a degree-based second cut if the propagation ratio still
collapses toward 1.

**There is no tissue axis.** Condition strata are a weaker signal than tissues:
two tissues differ far more than two carbon sources. Expect a lower ceiling on
`coexpr_hit_rate_*` across the board — read the `gen/random` ratio and
`% to train` columns, which are scale-free, not the raw level. Stratifying by
`project` gives more, larger groups than `carbon source`; try both.

## The perturbation slot

No genome-scale single-cell Perturb-seq exists for E. coli. In descending order
of fit to your discovery cell:

- **RegulonDB v12** — curated TF→gene edges. `build_regulondb.py` puts it in
  the same schema, so a generated edge can be scored against curated
  *regulation* as well as curated *chemistry*. An edge with precision 0 against
  `ref` but > 0 against `ref_reg` is exactly the population your discovery
  benchmark is trying to surface.
- **CRISPRi knockdown → bulk RNA-seq of essential genes** (Silvis et al., mBio
  2022) — a real perturbation × transcriptome matrix; GEO accession in the paper.
- **CRISPRi of regulators × promoter-reporter library** (Nat Commun 2023,
  `s41467-023-41572-4`).
- **TF knockout / overexpression samples already inside PRECISE-1K** — filter
  `metadata_qc.csv`; free, since you are loading it anyway.
- **DREAM5 E. coli** gold standard, if you want a published baseline for the
  novel-link ranking.

## Looking at the co-expression tables

Cells 8 and 9 of the notebook. They need `ecoli_expr/` to exist, so run cell 7
with `BUILD_COEXPRESSION = True` first; both no-op with a message otherwise.

**Cell 8** draws the network — one stratum, strongest 15k edges, largest
component, force layout seeded spectrally so it is deterministic. Two panels
over the same layout: Louvain communities on the left, iML1515 subsystems on
the right. The comparison is the point. The graph is strongly modular
(Q = 0.79 on `ros_tale`) but the subsystems scatter across those modules
rather than lining up with them.

Only the positive edges lay the graph out. `spring_layout` reads edge weight
as attraction, so feeding it signed correlations pulls anti-correlated genes
together and you get a hairball; negative edges are drawn on top in red
afterwards.

**Cell 9** asks whether the tables are worth conditioning on. Two numbers
matter:

| | |
|---|---|
| same-subsystem gene pairs, co-expression rate | **2.4x** background (range 1.7–3.2 across strata) |
| stratum-to-stratum edge overlap | mean Jaccard **0.027**, against 0.0061 for independent graphs |

So there is real signal in both directions — co-expression does track
metabolic organisation, and the strata are not redrawings of one distribution
— but neither margin is large.

**The caveat is panel f, and it is the reason cell 9 exists.** Stratifying
1,035 samples by `project` gives 24 strata of 14–101 samples. At `--min-abs
0.3` a correlation only reaches p < 0.05 from n = 44, so **16 of the 24 strata
sit below their own significance threshold**, and 78.8% of the union edge set
appears in exactly one stratum. Some of that 4.5x-over-random separation is
the condition axis and some is small-sample noise, and as built these are not
separable.

If you intend to condition on this, either raise `--min-samples` well above 44
and accept fewer strata, or stratify on something coarser than `project` —
`carbon source` gives fewer, larger groups. `--top-k 32` also forces a minimum
degree of 32 on every gene whether or not it has 32 real partners, which is
what flattens the left tail in panel b.

## Is the condition axis informative?

`ecoli_coexpression_information.ipynb`, mirroring the human notebook. Headline
numbers from a local run:

| | |
|---|---|
| MI excess over permutation null, curated subsystems | +0.026 nats (z = +4.0) |
| ...size-matched RANDOM gene sets | **+0.037 nats — higher** |
| MI excess, sampled chunks | +0.008 nats (z = +50.6) |
| ...size-matched random | +0.006 (random reaches 70%) |
| Spearman, co-expression vs activity | −0.03 to −0.07 |

**The coupling is not specific to pathway membership.** For the curated
subsystems random gene sets of the same size score *higher* than the real
ones; for chunks they reach 70%. That is the same conclusion the human
notebook reached (random reached 83% there), slightly worse here.

The ceiling on conditioning, from the variance decomposition: pair identity
explains **60.7%** of the variance in co-expression values, and the condition
label explains **11.9%** of what remains (shuffle null 0.0%). Real but modest,
and an upper bound — a model must infer the pair rather than being told it.

One methodological note carried in the notebook: the pooled
`MI(condition ; value)` the human version uses returns ~0 here whatever the
data says, because every stratum was built with the same top-k and threshold
so their marginal value distributions are identical by construction, and
densified 94% of the cells are zero. The variance decomposition replaces it.

### What the 24 strata actually are

`project` is an **experimental study**, not a biological condition. PRECISE-1K
has 45; the build kept the 24 largest (855 of 1,035 samples). Each bundles a
strain, a medium, a perturbation, a lab and a sequencing run — so differences
between the networks confound biology with **strain and batch**, and this table
cannot separate them. Five strains appear (MG1655, BW25113, GMOS, DGF-298,
W3110) and several strata are ALE/TALE experiments, i.e. *evolved* lines whose
expression differs by mutation, not only by condition.

Most are M9 + glucose and differ by what was done to the cells — paraquat,
H2O2, ethanol, trimethoprim. `--stratify-by "carbon source"` gives a cleaner
biological axis and is worth comparing against.

### Pathway members vs random gene pairs

Computed from the sample-level log-TPM matrix, not the stored top-k networks,
so no threshold artefact:

| | mean |r| | signed mean r |
|---|---|---|
| same-subsystem pairs | 0.379 | 0.136 |
| random metabolic pairs | 0.356 | 0.081 |
| ratio | **1.06x** | **1.68x** |

Consistent in sign (23 of 24 strata) but small: Cohen's d = **0.13** on the
pooled correlations, and the two distributions almost completely overlap.

Two things the figure makes plain. Both arms sit far above the uncorrelated
noise floor `sqrt(2/pi)/sqrt(n-1)`, so most of the correlation in this
compendium is **global covariation** — growth rate, batch — shared by pathway
and random pairs alike. And mean |r| tracks sample size almost perfectly
(0.44 at n=14, 0.27 at n=101) in **both** arms, so absolute |r| is largely a
small-sample artefact and only the pathway-minus-random gap is interpretable.

### iModulons vs raw correlation

`M.csv` in PRECISE-1K is an ICA decomposition into 201 iModulons (117 with a
named regulator), fitted over all 1,035 samples rather than one stratum's
14–101. Cell 9 asks whether it tracks pathway co-membership better than
pairwise correlation, at **matched coverage** — a comparison that has to be
made carefully, since "shares an iModulon" fires on only 0.76% of pairs and
its precision cannot be set against a mean over all pairs.

| % of pairs kept | raw correlation | iModulon cosine | both |
|---|---|---|---|
| 0.5% | 3.1x | **4.1x** | 4.0x |
| 1% | 2.5x | **2.9x** | 3.0x |
| 2% | 2.2x | 2.2x | **2.5x** |
| 10% | **1.7x** | 1.4x | 1.6x |
| AUC | **0.579** | 0.531 | 0.570 |

**They are good at different things.** iModulons win the sparse, high-precision
tail; raw correlation is the better broad ranker. Shared-module count is
cleanly monotone — 1 shared iModulon gives 2.7x, two gives 4.6x, three gives
5.9x — but only 0.06% of pairs reach two.

The other hypothesis is dead: stripping global principal components per
stratum makes things **worse**, d = 0.131 -> 0.074 at 5 PCs, because both arms
shrink together. Global covariation is not masking a pathway signal.

Practical read: neither is a usable per-edge predictor of pathway membership
(AUC 0.58 at best). Use them as a **sparse high-precision prior** — top ~1% of
pairs, where both reach 3x — rather than as a dense edge weight.

## The pathway dataset

`build_pathways.py` fills the gap the rest of this README used to end on.

```bash
python build_pathways.py                       # 1,778 graphs
python build_coexpression.py ... --node-id-map ecoli_ref/ecoli_node_id_map.pkl \
    --out-dir ecoli_expr_nodespace             # co-expression in the same id space
```

Chunks are grown per subsystem from a seed reaction outward, overlapping by
default. Overlap is a leak in the human dataset — held-out chunks sit at ~0.85
Jaccard to a training chunk — but not here: **mean pairwise Jaccard 0.011**,
because the network is sparse enough that different seeds inside a subsystem
land in different places. `--disjoint` gives 458 graphs instead.

Two things it does that the human pickle does not:

- **Strips the 30 highest-degree entities.** In iML1515 those are promiscuous
  *enzymes* (ompF at degree 286), not currency metabolites — either way they
  locate nothing, and left in, every chunk is the same hub neighbourhood.
- **Deduplicates by node set.** A node belongs to several subsystems, so the
  same neighbourhood gets grown more than once: 464 of 2,242 graphs were exact
  duplicates and 168 node sets spanned several subsystems, which would defeat
  a subsystem-held-out split outright. `also_in` records every subsystem a
  surviving chunk belongs to.

```
1,778 graphs   vocabulary 6,472   nodes/graph median 64   edge density 0.037
reaction 2,242 · metabolite 1,651 · gene 1,413 · enzyme 1,166 · 37 subsystems
```

`compare_models_ecoli.ipynb` in the project root trains the four generative
variants on it, splitting by **subsystem** rather than at random — train-vs-test
Jaccard mean 0.009, max 0.73, against ~0.85 for the human random split.

## Not covered

The *human* sampler's exact behaviour. `build_pathways.py` is a
reimplementation against the format `PathwayGraphDataset` consumes, not a port
— it produces the same fields but makes its own choices about hub stripping,
overlap and deduplication, all documented above and all switchable.
