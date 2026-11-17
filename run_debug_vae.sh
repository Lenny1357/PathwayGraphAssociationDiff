#!/bin/bash
#SBATCH --job-name=gp-debug-vae
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=logs/debug_vae_%j.out
#SBATCH --error=logs/debug_vae_%j.err
#
# GraphVAE diagnostics: gradients, output spread and latent use under one-change-at-a-time
# variants (scripts/debug_vae.py).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

PY=$HOME/miniforge3/envs/graph_gen/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
[ -z "${PT_FILES:-}" ] && [ -d pt_files ] && export PT_FILES=$PWD/pt_files

$PY -u "${1:-scripts/debug_vae.py}" "${@:2}"
