# Graph diffusion for biological pathway generation — code for the paper

Code accompanying

> L. B. Otte, C. Hogstrand, A. Mardinoglu and M. Guo.
> *Graph Diffusion Generates Biological Pathways Supported by Independent Transcriptomic Evidence.*

The models generate typed pathway subgraphs (genes, enzymes, reactions and metabolites, at most
64 nodes) cut from the genome-scale metabolic reconstructions Human-GEM and *E. coli* iML1515.
They are then assessed against co-expression and Perturb-seq evidence that the models never see
in training.

This repository contains code only. **No datasets, trained weights or generated samples are
included**: every external dataset has to be downloaded from its original source (see
[Data](#data)).

---

## Contents

```
graphpath/            the library (pip install -e .)
  config.py           every hyper-parameter, with its rationale
  layers/             the interleaved node/edge graph tower, embeddings, type-pair prior, conditioning slot
  processes/          forward corruption: absorbing mask, marginal transition, Gaussian (and uniform)
  models/             one network + loss per arm: masked.py, joint.py/marginal.py, continuous.py, vae.py
  samplers/           reverse-time generation per process (confidence reveal, remasking, PAD damping)
  data/               datasets and loaders, node types, cofactor list, expression and propagation
  benchmarks/         reference-network, topology, memorisation, novel-link, co-expression, Perturb-seq, flux
  pipeline/           the experiment as stages: compare (train+sample), score, analysis, tables, figures
  plotting/           every figure-drawing function
scripts/              command-line entry points (listed below)
data/<source>/        one builder per external dataset; each downloads its own inputs
experiments/splits/   edge- and version-based split utilities (used by the offline tests)
tests/                unit tests (CPU, seconds) and the data builders' offline self-checks
docs/                 data_sources.md (every dataset and its origin), pipelines.md (stage JSON schemas)
run_*.sh              Slurm launchers used for the paper (one GPU each)
```

The paper's four generators share the tower in `graphpath/layers/` and differ only in the
corruption process:

| Paper name | `--arms` key | Model class | Process |
| --- | --- | --- | --- |
| absorbing mask | `masked` | `GraphDiffBERT` | `processes/absorbing.py`, `samplers/masked.py` |
| marginal transition | `transition` | `GraphDiffBERTJoint` | `processes/marginal.py`, `samplers/marginal.py` |
| continuous embedding | `continuous` | `ContinuousGraphDiffusion` | `processes/gaussian.py`, `samplers/continuous.py` |
| VAE reference | `vae` | `GraphVAE` | `samplers/vae.py` |

The settings used in the paper (typed nodes, padding supervised at its data frequency,
`d_model = 128`, `L = 4`, `T = 200`, cosine schedule, 20,000 steps, batch 32, AdamW 3e-4 / 1e-2,
seed 0) are the `pad` preset in `graphpath/pipeline/arms.py` (`PRESETS["pad"]`).

---

## Installation

Python ≥ 3.10 and a CUDA GPU. Every training, sampling and scoring step was run as a
single-GPU job.

```bash
python -m venv .venv && source .venv/bin/activate        # or a conda environment
pip install -e ".[viz,chem,data,dev]"
make test                                                # unit tests, CPU, no data needed
```

Versions used for the paper are pinned in [requirements.txt](requirements.txt): Python 3.11,
PyTorch 2.14 with CUDA 12.6.

The `run_*.sh` launchers call `$HOME/miniforge3/envs/graph_gen/bin/python` and use
partition `workq`. Edit the `PY=` line and the `#SBATCH` header for your cluster, or run the
`python scripts/...` commands below directly.

---

## Data

Nothing is redistributed. See [docs/data_sources.md](docs/data_sources.md) for every source, its
version and the script that converts it. In short:

| Dataset | Source | Builder |
| --- | --- | --- |
| Human-GEM **v1.19.0** (and v2.0.0 for the retrospective dead-end check) | <https://github.com/SysBioChalmers/Human-GEM> | `data/gem/build_gem.py`, `data/gem/gem_gaps.ipynb` |
| *E. coli* iML1515 | BiGG, <http://bigg.ucsd.edu/models/iML1515> | `data/gem/build_gem_ecoli.py`, `data/ecoli/build_reference.py`, `data/ecoli/build_pathways.py` |
| PRECISE-1K co-expression | <https://github.com/SBRG/precise1k> | `data/ecoli/build_coexpression.py` |
| Perturb-seq (Replogle et al. 2022; K562 genome-wide, K562 essential, RPE1) | figshare 20029387 | `data/perturbseq/build_perturbseq.py` |
| Human tissue co-expression (GeneFriends, SRA-derived) and Human Protein Atlas expression | <https://www.genefriends.org>, <https://www.proteinatlas.org> | see below |
| EnzymeMap, sequences, KEGG (optional annotations) | see `docs/data_sources.md` | `data/enzymemap/`, `data/sequences/`, `data/kegg/` |

```bash
make data-gem      # Human-GEM + iML1515 stoichiometric models   -> data/gem/
make data-ecoli    # E. coli reference network and 64-node pathways -> data/ecoli/ecoli_ref/
cd data/ecoli && python build_coexpression.py          # PRECISE-1K
cd data/perturbseq && python build_perturbseq.py       # Perturb-seq effect matrices
```

**Human-GEM version.** `data/gem/build_gem.py` downloads from the repository's `main` branch.
To reproduce the paper, check out tag `v1.19.0` of Human-GEM, or point the download at
`.../Human-GEM/v1.19.0/model`. The dead-end analysis reads
`data/gem/gem_cache/Human-GEM-v1.19.0.mat`. `data/gem/gem_gaps.ipynb` writes the dead-end lists
(`gem_gaps_metabolites.csv`, `gem_gaps_reactions.csv`) and runs the comparison with v2.0.0.

**Human training graphs and tissue co-expression (`$PT_FILES`).** The human pathway subgraphs,
the shared vocabulary and the per-tissue co-expression tables are read from a folder given by the
environment variable `PT_FILES` (default `./pt_files`):

| File | Content |
| --- | --- |
| `human_gem_sampled_graphs.pkl` | subsystem-restricted ≤ 64-node Human-GEM subgraphs (BFS + DFS sampler, Methods) |
| `node_id_map.pkl`, `node_name_map.csv`, `id_name_map.csv` | vocabulary and common names |
| `human_gem_reaction_node_edges2.csv` | the Human-GEM reference network (participation edges) |
| `co_expression_sparse_homo_sapiens_sra_<tissue>.pt`, `tissue_expression_dict.pkl` | GeneFriends tissue co-expression and HPA expression for 11 tissues |
| `ensembl_go_annotations.csv`, `gene_go_edges.csv`, `hpa_expression_data.csv` | gene annotations |

These files were produced by preprocessing code that is not part of this repository (see
[Limitations](#known-limitations)). The *E. coli* equivalents are fully rebuilt by the scripts
above.

---

## Reproducing the paper

All commands run from the repository root. Each writes under `outputs/`, which is created on
first use. `<run>` is the run folder that `train_compare.py` prints, e.g. `compare_pad_<stamp>`.

**1. Train and sample** the four arms on both organisms (96 graphs per arm, sampling seed 0):

```bash
python scripts/train_compare.py --preset pad --datasets ecoli human \
       --arms masked transition continuous vae            # -> outputs/runs/<run>/
```

**2. Calibrate padding damping** for the absorbing-mask arm, then redraw its samples with it
(Methods, *Padding damping during sampling*; chosen scales in the paper: *E. coli* 0.03,
human 0.005):

```bash
python scripts/calibrate_pad_scale.py --run <run>          # -> outputs/runs/<run>/pad_scale.json
python scripts/train_compare.py --preset pad --run <run> --sample-only
python scripts/calibrate_pad_schedule.py                   # constant vs time-relaxed damping (Methods);
                                                           # takes no arguments: set run_name in main()
```

**3. Score** every benchmark against the shuffled, random and held-out controls, and run the
analyses (clusters, trajectories, remasking, co-expression support, novel-link discovery,
square clustering, maps):

```bash
python scripts/score_run.py --run <run>                    # -> outputs/analysis/<run>/*.json, tables/*.csv
```

**4. Figures and tables**, drawn from the JSON alone (no GPU):

```bash
python scripts/make_figures.py --run <run>                 # benchmark, topology, embedding, trajectory panels
python scripts/paper_figures.py --run <run>                # named pathway panels, evidence chains, type violations
python scripts/discovery_routes.py --run <run>             # novel-link support routes (Fig. 6a, Supp.)
python scripts/draw_vae_samples.py --run <run>             # VAE samples next to held-out and diffusion graphs
python scripts/data_signals.py --what both                 # which evidence source tracks pathway membership
                                                           # (the question behind Fig. 2f,g; needs the human tables)
python scripts/compare_runs.py --new <run> --old <untyped run> --label-new typed --label-old untyped
```

The typed-versus-untyped comparison (Supplementary) needs a second run trained with
`--preset sept` (untyped, padding unsupervised).

**5. In-context link prediction for dead-end metabolites** (Results, Fig. 6b–d, Supplementary
Table of hypotheses):

```bash
python scripts/probe_controls.py --run <run>               # calibration with and without the degree cue; bootstrap CIs
python scripts/dead_end_hypotheses.py --run <run>          # score every dead end in every subgraph, assign tiers
python scripts/dead_end_figures.py --run <run>
```

**6. VAE baseline diagnostics** (Methods, *VAE reference*): `scripts/check_vae_learning.py`,
`check_vae_sampling.py` and `check_vae_seeds.py`.

`run_train.sh`, `run_calibrate_pad_scale.sh`, `run_slurm.sh` (score + figures),
`run_paper_figures.sh` and `run_cofactor_rescore.sh` are the Slurm wrappers for steps 1–4.
`run_debug_vae.sh <script> [args]` is the generic one-GPU wrapper that the other scripts' usage
lines refer to.

`scripts/conditioning.py` exercises the tissue-conditioning interface described in Methods and
in the Supplementary. Conditioned models were **not** evaluated in the paper.

### Where each result comes from

| Paper element | Computed by | Stored in `outputs/analysis/<run>/` |
| --- | --- | --- |
| Structure, edge precision, purity, memorisation (Supp. Tables) | `score_run.py` (`scores`) | `scores.json`, `tables/table_*` |
| Mean ranks against controls | `score_run.py` | `tables/rank_*.csv`, `tables/mean_rank_*.csv` |
| Embedding clusters and subsystem enrichment (Fig. 4e–i) | `score_run.py` (`clusters`) | `clusters.json` |
| Generation dynamics, remasking (Supp. dynamics figure) | `score_run.py` (`trajectories`, `remask`) | `trajectories.json`, `remask.json` |
| Type-incompatible edges | `paper_figures.py` | `type_violations.json` |
| Graph statistics, square clustering | `score_run.py` (`graph_stats`, `clustering`) | `graph_stats.json`, `clustering.json` |
| Edge-level co-expression support | `score_run.py` (`coexpr_support`) | `coexpr_support.json` |
| Novel-link enrichment vs random non-edges (Fig. 5e–f) | `score_run.py` (`discovery`) | `discovery.json` |
| Co-expression maps (Fig. 5h–j) | `score_run.py` (`maps`) | `maps.json` |
| Dead-end calibration and hypotheses (Fig. 6b–d) | `probe_controls.py`, `dead_end_hypotheses.py` | `dead_ends/` |

Currency metabolites and cofactors excluded from the evidence analyses are listed in
`graphpath/data/cofactors.py`.

---

## Tests

```bash
make test           # unit tests: processes, models (smoke), novel edges, perturbation, pipeline
make test-offline   # the data builders' self-checks (needs the built data folders)
```

---

## Known limitations

- **Human preprocessing is not included.** The Human-GEM subgraph sampler and the human
  tissue co-expression tables (`$PT_FILES`) were built outside this repository. All modelling,
  scoring and analysis code that reads them is here, and the *E. coli* half is rebuildable end to
  end.
- **Human-GEM version.** The builder fetches `main`; pin `v1.19.0` as described above.
- **Data-overview panels.** The descriptive data panels of Figs. 1–3 (network overview,
  dead-end summaries, evidence overviews, corruption illustration) were drawn interactively and
  are not regenerated by these scripts. The numbers they show come from the builders in `data/`.
- **One seed.** Every reported model is trained with seed 0 and sampled with seed 0 (96 graphs
  per model), as stated in the paper.
- **Assembled run.** In the reported run, `compare_oct2pad_20261112_1238`, the three diffusion
  arms come from one training run and the VAE baseline from a later run. That later run fixed
  the VAE's collapse with canonical order, BN on the posterior mean and edges from decoded types.
  Running steps 1–2 above trains all four arms with the current code in a single run.

See [RELEASE_NOTES.md](RELEASE_NOTES.md) for how this folder differs from the development
repository.

## Citation

See [CITATION.cff](CITATION.cff).
