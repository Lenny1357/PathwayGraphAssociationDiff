#!/bin/bash
#SBATCH --job-name=gp-bench-oct2
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/bench_oct2_%j.out
#SBATCH --error=logs/bench_oct2_%j.err
#
# Score the newest compare_oct2 run, draw every benchmark figure for it, and compare it
# against an earlier scored run.
#
#   mkdir -p logs && sbatch run_slurm.sh                        # newest compare_oct2_* with weights
#   sbatch run_slurm.sh compare_oct2_20260929_1542              # a specific run
#   sbatch run_slurm.sh latest:compare_oct2 "--resample"        # extra score_run.py flags
#   COMPARE= sbatch run_slurm.sh                                # skip the cross-run comparison
#
# Stages (see docs/pipelines.md):
#   scripts/score_run.py    -> outputs/analysis/<run>/*.json, tables/*.csv   (all 14 analyses)
#   scripts/make_figures.py -> outputs/analysis/<run>/figures/<dataset>/*.png
#   scripts/compare_runs.py -> outputs/analysis/<run>/compare_<old>.json, tables/compare_<old>.csv
#
# Every number a figure shows is in those JSON files, so figures are redrawn without the GPU,
# the models or re-scoring:
#   python scripts/make_figures.py --run <run> [--only scores maps ...] [--graphs]
#
# Human-GEM is read from $PT_FILES, by default the pt_files/ folder next to this script.
#   PT_FILES=/path/to/pt_files sbatch run_slurm.sh

RUN=${1:-latest:compare_oct2}
EXTRA=${2:-}
COMPARE=${COMPARE-compare_sept_20260905_2327}

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

PY=$HOME/miniforge3/envs/graph_gen/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export MPLBACKEND=Agg
[ -z "${PT_FILES:-}" ] && [ -d pt_files ] && export PT_FILES=$PWD/pt_files

# Pin "latest:" to a concrete name so both stages see the same run, even if a newer one appears.
RUN=$($PY -c "
import sys
from graphpath.pipeline.common import RunLayout
n = sys.argv[1]
print(RunLayout.latest(n.split(':', 1)[1]).name if n.startswith('latest:') else n)" "$RUN")

echo "host     $(hostname)"
echo "run      $RUN"
echo "PT_FILES ${PT_FILES:-<default>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# every analysis; each one failing is logged but does not stop the rest
$PY scripts/score_run.py --run "$RUN" ${EXTRA}

# every figure family, including the trajectory graph panels
$PY scripts/make_figures.py --run "$RUN" --graphs

# sampled GraphVAE graphs next to held-out and diffusion graphs (vae_samples_*.png, vae_vs_models.png)
$PY scripts/draw_vae_samples.py --run "$RUN" || echo "draw_vae_samples failed"

# distance to held-out, metric by metric, against the earlier run
if [ -n "$COMPARE" ] && [ -f "outputs/analysis/$COMPARE/scores.json" ]; then
    $PY scripts/compare_runs.py --new "$RUN" --old "$COMPARE" --label-new "${RUN%%_2*}" --label-old "${COMPARE%%_2*}"
fi

echo "done -> outputs/analysis/$RUN/figures"
