#!/bin/bash
#SBATCH --job-name=gp-paper-figs
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/paper_figs_%j.out
#SBATCH --error=logs/paper_figs_%j.err
#
# Paper-only figures for a scored run: named pathway panels, novel-link evidence chains
# (scripts/paper_figures.py), and the typed-vs-untyped comparison against the September run.
#
#   sbatch run_paper_figures.sh [run] [untyped_run]

RUN=${1:-compare_oct2pad_20261112_1238}
UNTYPED=${2:-compare_sept_20260905_2327}
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
PY=$HOME/miniforge3/envs/graph_gen/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export MPLBACKEND=Agg
[ -z "${PT_FILES:-}" ] && [ -d pt_files ] && export PT_FILES=$PWD/pt_files

$PY -u scripts/paper_figures.py --run "$RUN"
$PY -u scripts/compare_runs.py --new "$RUN" --old "$UNTYPED" --label-new typed --label-old untyped
echo done
