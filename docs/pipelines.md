# From notebooks to pipelines

The experiment notebooks were one long chain of cells each: load data, build
arms, train, sample, score, plot — with the scoring and the plotting welded
together, so redrawing a figure meant re-running the benchmark and re-running
an experiment meant re-running everything. They are now stages that talk to
each other through files:

```
scripts/train_compare.py   ->  outputs/runs/<run>/{<variant>/}<dataset>/{weights,samples}/<arm>.pt
                               outputs/runs/<run>/history.json, samples.json, manifest.json
scripts/score_run.py       ->  outputs/analysis/<run>/scores.json          every benchmark, four arms
                               outputs/analysis/<run>/*.json               one file per analysis
                               outputs/analysis/<run>/tables/*.csv
scripts/make_figures.py    ->  outputs/analysis/<run>/figures/<dataset>/*.png   from the JSON only
scripts/conditioning.py    ->  completion.json, validation.json
scripts/data_signals.py    ->  signal_comparison.json, coexpr_information.json
```

Every stage is resumable and independent: a run can be re-scored without
retraining, a figure redrawn without re-scoring, and `--analyses` picks the
subset to compute. Everything numeric is JSON (`graphpath.pipeline.common.dump_json`
handles numpy / torch / pandas); only weights and samples are `.pt`.

## Which notebook became what

| notebook | module | script | JSON written |
| --- | --- | --- | --- |
| `compare_models_sept.ipynb`, `_typed`, `_oct`, `_oct2` | `graphpath/pipeline/compare.py` (+ `arms.py` presets `sept`, `typed`, `nopad`, `pad`) | `train_compare.py --preset …` | `history.json`, `samples.json` |
| `benchmark_sept.ipynb` / `benchmark_typed.ipynb` cells 1–3, 6–8, 16 | `graphpath/pipeline/score.py`, `tables.py` | `score_run.py --analyses scores architecture` | `scores.json`, `architecture.json`, `tables/` |
| … cell 4 (reverse trajectory), cell 5 (remasking) | `graphpath/pipeline/trajectories.py` | `--analyses trajectories remask` | `trajectories.json`, `remask.json` |
| … cell 9 (pathways drawn), 17 (type-free clustering), 18 (graph statistics) | `graphpath/pipeline/analysis.py` | `--analyses pathway_stats clustering graph_stats` | `pathway_stats.json`, `clustering.json`, `graph_stats.json` |
| … cell 10 (co-expression support), 15 (discovery) | `analysis.coexpr_support`, `analysis.discovery` | `--analyses coexpr_support discovery` | `coexpr_support.json`, `discovery.json` |
| … cell 12 (maps), 13 (embedding clusters), 14 (attention) | `analysis.maps`, `.embedding_clusters`, `.attention` | `--analyses maps clusters attention` | `maps.json`, `clusters.json`, `attention.json` |
| … cell 20 (reveal-order ablation), 23 (edge threshold) | `analysis.ablation`, `.threshold_sweep` | `--analyses ablation threshold` | `ablation.json`, `threshold.json` |
| `benchmark_typed.ipynb` cell 22 (typed vs untyped) | `tables.compare_runs(scores_new, scores_old)` | — | |
| `conditioning_test.ipynb`, `conditioning_neighbourhood.ipynb` | `graphpath/pipeline/conditioning.py` | `conditioning.py [--validate]` | `completion.json`, `validation.json` |
| `signal_comparison.ipynb`, `coexpression_information.ipynb` | `graphpath/pipeline/signals.py` | `data_signals.py --what signal\|coexpr-info` | `signal_comparison.json`, `coexpr_information.json` |
| every figure cell | `graphpath/pipeline/figures.py` | `make_figures.py` | reads the above |

The converted notebooks are kept under `notebooks/archive/converted/` as the
record of the exploration; they are no longer the way to run anything.

Not converted (exploratory, one-off): `marg_model_test.ipynb`, `vae_debug.ipynb`,
`structural_encoding.ipynb`, `compare_models_embed.ipynb`,
`compare_models_ecoli.ipynb` (still in `notebooks/training/`), the data-setup
notebooks (`data/*/`, builder runners with exploratory figures) and the split
notebooks (`experiments/splits/`). They import from `graphpath` and run as before.

## Reproducing the September result

```bash
python scripts/train_compare.py --preset sept --datasets ecoli human     # ~12-20 GPU-hours
python scripts/score_run.py --run latest:compare_sept                   # every benchmark + analysis
python scripts/make_figures.py --run <run> --graphs
```

To score the run that already exists: `python scripts/score_run.py --run compare_sept_20260905_2327`.

## JSON schemas, briefly

- `scores.json` — `{"held-out": {dataset: {bench: result}}, "arms": {"dataset|arm": {bench: {generated|shuffled|random: result}}}}`;
  `result` has `summary`, `per_graph`, `distributions`, `n_graphs`, `n_scored`, `notes`.
  Runs with variants use keys like `"pad/ecoli|masked"`.
- `history.json` — `{"dataset|arm": [{"step", "loss", ...}, ...]}`.
- `trajectories.json` — per arm: the eight traces over `t`, plus `panels` (full node/edge states at eight points).
- `completion.json` — per dataset: `arms: {name: {mean: {...}, per_graph: {...}}}` where names are
  `freq`, `masked-plain`, `transition-plain`, `masked-cond <channel> g<scale>`.

Read them with `graphpath.pipeline.common.load_json`; `tables.py` turns `scores.json` into
DataFrames (`metric_table`, `coverage_table`, `rank_table`, `compare_runs`).
