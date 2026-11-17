# How this folder relates to the development repository

This folder was created by copying files from the development repository. Code was not
restructured or refactored, and module paths are unchanged, so `graphpath.*` imports and every
script's usage line work as they did for the paper.

## Changes made in the copy (documentation and defaults only)

| File | Change |
| --- | --- |
| `graphpath/paths.py`, `graphpath/data/expression.py`, `data/sequences/build_embeddings.py` | default human-input folder changed from a personal cluster path to `pt_files` (relative; override with `$PT_FILES`, as before) |
| `graphpath/benchmarks/README.md`, `data/ecoli/README.md`, `docs/data_sources.md` | the same path in examples |
| `data/gem/gem_gaps.ipynb` | cell outputs cleared; one personal path in a list of candidate folders replaced by `data/gem` |
| `README.md`, `requirements.txt`, `.gitignore`, `CITATION.cff`, this file | new |

## Included

- The whole `graphpath` library used for the paper, without the four extension modules listed
  below.
- The scripts that train, sample, score and draw the results of the paper, plus the
  dead-end analysis and the VAE diagnostics cited in Methods.
- The Slurm launchers for those scripts.
- Every dataset builder (`data/<source>/*.py` and its README).
- The split utilities that the offline tests import.
- The unit tests.

## Left out

- **Data, weights and results:** everything under `data/*/` except code and READMEs,
  `pt_files/`, `outputs/` and `logs/`. Datasets must be downloaded from their sources.
- **Notebooks**, apart from the data notebook `data/gem/gem_gaps.ipynb`. The others were
  exploratory and contain cluster paths and embedded outputs.
- **Exploratory and diagnostic scripts** not behind any number in the paper: underscore-prefixed
  scratch scripts, earlier probe versions, PAD size-clamp checks, VAE gradient debugging and
  palette validation. Also left out is `experiments/arch_bench`, a process sweep that the paper
  does not report.
- **Follow-up work not reported in the paper.** This covers pinned/anchored generation,
  classifier-free guidance on an anchor, ego-network training, the feature-based link predictor,
  retrieve-then-generate and the hub-gene robustness check. The modules left out for it are
  `graphpath/models/masked_anchored.py`, `masked_ego.py`, `masked_ego_feat.py` and
  `entity_features.py`; nothing included imports them.

## Known rough edges, left as in the paper's code

- `run_debug_vae.sh` defaults to `scripts/debug_vae.py`, which is not included. Always pass the
  script to run: `sbatch run_debug_vae.sh scripts/<script>.py [args]`.
- `scripts/calibrate_pad_schedule.py` has its run name set in `main()` rather than on the command
  line.
- `data/gem/flux_sampler.py` `FluxSampler.score` imports `torch_fba`, the earlier name of
  `graphpath/fba.py`. The paper's flux checks use `graphpath.fba` through the benchmarks and do
  not call this method.
- `docs/pipelines.md` and `docs/data_sources.md` mention notebook folders and
  `experiments/arch_bench`, which are not part of this release.
