"""Redraw every figure of a run from its JSON. No model or data is loaded
unless --graphs is given (the trajectory graph panels need node names).

    python scripts/make_figures.py --run compare_sept_20260905_2327
    python scripts/make_figures.py --run compare_sept_20260905_2327 --only scores trajectories
    python scripts/make_figures.py --run signal_cmp_20261021_1200       # data-signal runs too
"""

import argparse

from graphpath.pipeline import figures as F
from graphpath.pipeline.common import ANALYSIS, RunLayout, load_bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--only", nargs="+", default=None,
                    help="figure families: loss samples scores trajectories remask pathway_stats clustering "
                         "coexpr_support discovery clusters attention ablation threshold maps completion "
                         "validation signal_comparison coexpr_information")
    ap.add_argument("--graphs", action="store_true", help="also draw the trajectory graph panels (loads the datasets)")
    a = ap.parse_args()
    bundles = None
    if a.graphs:
        run = RunLayout(a.run)
        bundles = {}
        for d in sorted({d for d, _, _ in run.all_weights()}):
            try:
                bundles[d] = load_bundle(d)
            except FileNotFoundError as e:
                print(f"  {d}: data missing, graph panels skipped ({e})")
    paths = F.draw_all(a.run, only=a.only, bundles=bundles)
    print(f"{len(paths)} figures -> {ANALYSIS / a.run / 'figures'}")
    for p in paths:
        print("  ", p.relative_to(ANALYSIS))


if __name__ == "__main__":
    main()
