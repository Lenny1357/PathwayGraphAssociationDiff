# Splits that don't leak

Two held-out sets for this project, both at the level of **edges** rather than
graphs, plus the evaluation that reads them.

| file | |
|---|---|
| `edge_split.py` | `EdgeSplit`, `MaskedEdgeDataset`, `link_prediction` |
| `version_split.py` | `Release`, `VersionSplit` — train on an older Human-GEM |
| `test_offline.py` | 20 checks, no model and no network |
| `splits.ipynb` | build both, inspect, evaluate |

```bash
python test_offline.py     # 20 passed, 0 failed
```

## Why not the 90/10 graph split

Chunks are cut from one connected network, so a held-out chunk sits at ~0.85
Jaccard to a training chunk — the model has already seen almost every node and
most of the edges. Whatever that measures, it is not generalisation.

Both splits here hold out **edges globally**: a held-out association is removed
from every graph it appears in, so it is genuinely unseen, while its endpoints
keep their other edges. That makes the task link prediction rather than cold
start, which is the question worth asking.

## 1. `EdgeSplit` — hold out associations at random

```python
from edge_split import EdgeSplit, MaskedEdgeDataset, link_prediction, link_table
from benchmarks.node_naming import default_node_type

split = EdgeSplit(dataset, default_node_type, frac=0.10, seed=0,
                  edge_types=[("enzyme", "reaction"), ("gene", "enzyme"),
                              ("metabolite", "reaction")])
print(split.summary())

train_ds = MaskedEdgeDataset(dataset, split)     # train on this
...
print(link_table(link_prediction(model, dataset, split, t=5)))
```

Two properties, both asserted in the tests:

- **Global.** An edge is a canonical `(token, token)` pair, not a position in
  one graph. Removing it from graph 7 and leaving it in graph 12 would leak it.
- **Typed.** `edge_types` selects which node-type pairs to hold out, so
  "does it know enzyme→reaction associations" and "does it know which
  metabolites share a reaction" are separable questions.

`min_degree` refuses to hold out an edge that would leave either endpoint with
nothing else. Holding out a node's only edge makes it unpredictable rather than
held out — there is no context left to predict from.

### The evaluation

`link_prediction` feeds each graph exactly as the model saw it in training,
lightly corrupted at timestep `t`, and reads P(edge) at held-out positions
against **type-matched negatives**.

Matching the negatives is not optional. Without it the learned type-pair bias
separates positives from negatives on its own, and the AUC measures nothing but
"does it know a gene does not bind a gene". The harness is verified against an
oracle scorer (AUC 1.000) and a random one (0.482).

## 2. `VersionSplit` — train on an older release

```python
from version_split import Release, VersionSplit
old, new = Release.fetch("v1.13.0"), Release.fetch("main")
vs = VersionSplit(old, new)
print(vs.summary())
split = vs.to_edge_split(dataset, default_node_type, which="new_chemistry")
```

Same evaluation, different hold-out rule: hide what the curators had not added
yet. Anything the model proposes that later turned out to be real is evidence
it generalises to biology rather than to one release's quirks.

### New nodes cannot be predicted, and it is worth being clear why

v1.13.0 → main adds 178 reactions, 177 metabolites and 46 genes. **None of
them can be emitted by a model trained on the old release** — the tokens do not
exist in its vocabulary. That is not a failure to generalise; there is no
probability to measure. Even the 61 new reactions whose participants all
already existed carry a new `MAR…` id of their own.

What a new reaction *does* create, that an old-vocabulary model can express, is
new **co-membership** among entities that already existed:

```
v1.13.0 -> main
  GAINED associations   19,744     <- the prediction target
    new chemistry          154     <- from reactions that did not exist
    re-annotation       19,590     <- gene rules edited on existing reactions
  LOST associations      3,757     <- negative control
```

**Read the provenance split before using the headline number.** 99.2% of the
gained associations come from curators editing the gene rule of a reaction
that already existed, not from new biochemistry. "Can it predict a
re-annotation" and "can it predict a reaction nobody had written down" are
different claims, and pooling them lets the easy one hide the hard one. Both
are available: `which="re_annotation"` and `which="new_chemistry"`.

`which="lost"` gives the negative control — associations curators **removed**.
A good model should rank those below the gained ones. Treat it as a mild
signal: a removal can be a re-annotation rather than a refutation.

### Choosing a base release

| tag | new reactions | new metabolites | new genes | dropped reactions |
|---|---|---|---|---|
| v1.13.0 | 178 | 177 | 46 | 273 |
| v1.15.0 | 118 | 93 | 42 | 260 |
| v1.17.0 | 77 | 29 | 21 | 115 |
| v1.19.0 | 5 | 9 | 15 | 45 |

v1.13.0 is the default: enough change to be powered, on a backbone that is
still ~98% shared. **v1.10.0 and earlier are not usable** — only 6,120 of its
8,370 metabolites survive into the current release, which is a
re-identification event rather than curation, and the diff would be dominated
by renames.

Note reaction counts *fall* over time (13,078 at v1.10.0 → 12,931 now). Human-
GEM is pruned as well as grown, which is why `lost` is a real category and not
an empty one.

## If you want new nodes to be predictable

The vocabulary has to carry information the model can generalise over —
sequence embeddings for genes, structure for metabolites — so an unseen token
still has a meaningful embedding rather than its random initialisation. That
is the point at which "predict the new nodes back in" becomes a well-posed
question, and it is why the sequence work is a prerequisite for it rather than
a nice-to-have.

## Not covered

Rebuilding pathway chunks from the *old* network. `MaskedEdgeDataset` gives
the same guarantee more cheaply — the model never sees a post-v1.13
association — while keeping the current vocabulary and node id map, so results
stay comparable with every other experiment in the project. Sampling fresh
chunks from v1.13 would need a chunker this project does not have
(`PathwayGraphDataset` only wraps pre-built graphs) and would renumber every
token.
