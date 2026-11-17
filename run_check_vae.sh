#!/bin/bash
#SBATCH --job-name=gp-check-vae
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/check_vae_%j.out
#SBATCH --error=logs/check_vae_%j.err
#
# Short VAE training runs testing the canonical-order and bag-of-words fixes
# (scripts/check_vae_learning.py). Extra arguments go to the script, e.g. --steps 5000.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

PY=$HOME/miniforge3/envs/graph_gen/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
[ -z "${PT_FILES:-}" ] && [ -d pt_files ] && export PT_FILES=$PWD/pt_files

$PY -u scripts/check_vae_learning.py "$@"
