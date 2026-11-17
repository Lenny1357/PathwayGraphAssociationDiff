"""Which evidence source tracks pathway membership, and what a tissue's
co-expression network can tell you. The two analysis notebooks; no model.

    python scripts/data_signals.py --what signal         # E. coli + human sources
    python scripts/data_signals.py --what coexpr-info    # needs the human tissue tables
    python scripts/data_signals.py --what both
"""

import argparse
from datetime import datetime

from graphpath.pipeline import figures as F
from graphpath.pipeline.common import ANALYSIS
from graphpath.pipeline.signals import coexpr_information, signal_comparison


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", default="signal", choices=["signal", "coexpr-info", "both"])
    ap.add_argument("--n-pairs", type=int, default=300_000)
    ap.add_argument("--name", default=None, help="analysis folder name (default signals_<stamp>)")
    a = ap.parse_args()
    name = a.name or f"signals_{datetime.now():%Y%m%d_%H%M}"
    out = ANALYSIS / name; out.mkdir(parents=True, exist_ok=True)
    if a.what in ("signal", "both"):
        signal_comparison(out, n_pairs=a.n_pairs)
    if a.what in ("coexpr-info", "both"):
        coexpr_information(out)
    paths = F.draw_all(name)
    print(f"{len(paths)} figures -> {out / 'figures'}")


if __name__ == "__main__":
    main()
