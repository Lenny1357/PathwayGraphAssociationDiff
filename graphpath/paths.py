"""Where things live. Import these instead of re-deriving the project root.

    from graphpath.paths import ROOT, RUNS, WEIGHTS, FIGURES, DATA, PT_FILES

`PT_FILES` points at the Human-GEM inputs that are not built by this
repository (see docs/data_sources.md); override it with the environment
variable of the same name.
"""

import os
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
DATA     = ROOT / "data"
OUTPUTS  = ROOT / "outputs"
RUNS     = OUTPUTS / "runs"
WEIGHTS  = OUTPUTS / "weights"
FIGURES  = OUTPUTS / "figures"
PT_FILES = Path(os.environ.get("PT_FILES", "pt_files"))

ECOLI_REF     = DATA / "ecoli" / "ecoli_ref"
ECOLI_EXPR    = DATA / "ecoli" / "ecoli_expr_nodespace"
GEM_REF       = DATA / "gem" / "gem_ref"
GEM_REF_ECOLI = DATA / "gem" / "gem_ref_ecoli"
PERTURB_REF   = DATA / "perturbseq" / "perturb_ref"
SEQ_REF       = DATA / "sequences" / "seq_ref"
KEGG_REF      = DATA / "kegg" / "kegg_ref"
