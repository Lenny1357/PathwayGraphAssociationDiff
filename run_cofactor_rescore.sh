#!/bin/bash
#SBATCH --job-name=gp-nocofactor
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=logs/nocofactor_%j.out
#SBATCH --error=logs/nocofactor_%j.err
#
# Re-run the co-expression evidence analyses with currency metabolites and cofactors
# excluded (graphpath/data/cofactors.py), then redraw every figure that reads them.
#   sbatch run_cofactor_rescore.sh [run]
RUN=${1:-compare_oct2pad_20261112_1238}
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
PY=$HOME/miniforge3/envs/graph_gen/bin/python
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export MPLBACKEND=Agg
[ -z "${PT_FILES:-}" ] && [ -d pt_files ] && export PT_FILES=$PWD/pt_files
$PY -c "
from graphpath.pipeline.common import load_bundle
from graphpath.data.cofactors import cofactor_ids
for d in ('ecoli','human'):
    b=load_bundle(d); c=sorted(cofactor_ids(b))
    print(d, len(c), 'cofactor tokens:', sorted({b.common(b.id_to_node[i]) for i in c}))
"
$PY -u scripts/score_run.py --run "$RUN" --analyses coexpr_support discovery
$PY -u scripts/make_figures.py --run "$RUN" --only coexpr_support discovery
$PY -u scripts/discovery_routes.py --run "$RUN"
$PY -u scripts/paper_figures.py --run "$RUN"
echo done
