#!/bin/bash
#SBATCH --job-name=gp-pad-scale
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/pad_scale_%j.out
#SBATCH --error=logs/pad_scale_%j.err
#
# Calibrate the PAD damping of the pad masked arm (scripts/calibrate_pad_scale.py) and write
# <run>/pad_scale.json. Exits non-zero on failure, so a job submitted with
# --dependency=afterok on this one does not start.
#
#   sbatch run_calibrate_pad_scale.sh [run]

RUN=${1:-compare_oct2_20260929_1542}

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

PY=$HOME/miniforge3/envs/graph_gen/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
[ -z "${PT_FILES:-}" ] && [ -d pt_files ] && export PT_FILES=$PWD/pt_files

$PY -u scripts/calibrate_pad_scale.py --run "$RUN"
