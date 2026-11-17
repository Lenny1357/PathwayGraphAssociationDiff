#!/bin/bash
#SBATCH --job-name=gp-train
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#
# Train (resumably) and sample arms of a comparison run: arguments go straight to
# scripts/train_compare.py. One variant x dataset per job lets several run in parallel on one run:
#
#   sbatch run_train.sh --run compare_oct2fix_20261111_1637 --preset oct2 --variants pad --datasets human --arms vae

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

PY=$HOME/miniforge3/envs/graph_gen/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
[ -z "${PT_FILES:-}" ] && [ -d pt_files ] && export PT_FILES=$PWD/pt_files

$PY -u scripts/train_compare.py "$@"
