"""Conditioned completion from revealed nodes. The conditioning notebooks.

    python scripts/conditioning.py --datasets human --arms masked-plain masked-cond transition-plain
    python scripts/conditioning.py --datasets human --arms masked-plain masked-cond masked-nbr --validate
    python scripts/conditioning.py --run conditioning_20260922_1845 --no-train     # re-evaluate

Weights and channels go to outputs/runs/<run>/; completion.json and
validation.json to outputs/analysis/<run>/. Figures: scripts/make_figures.py.
"""

import argparse

from graphpath.pipeline.common import RunLayout
from graphpath.pipeline.conditioning import ConditioningConfig, ConditioningExperiment
from graphpath.training.device import pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["human"])
    ap.add_argument("--arms", nargs="+", default=["masked-plain", "masked-cond", "transition-plain"])
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--k-revealed", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--run", default=None)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--validate", action="store_true", help="wiring / ablation / attribution checks for masked-nbr")
    a = ap.parse_args()
    cfg = ConditioningConfig(datasets=a.datasets, arms=a.arms, steps=a.steps, k_revealed=a.k_revealed, n_eval=a.n_eval)
    prefix = "nbrcond" if "masked-nbr" in a.arms else "conditioning"
    run = RunLayout(a.run) if a.run else RunLayout.new(prefix)
    exp = ConditioningExperiment(cfg, run, pick_device())
    print(f"run {run.path}")
    exp.load_data(); exp.channels()
    if not a.no_train:
        exp.train()
    exp.complete()
    if a.validate:
        exp.validate()
    print(f"\nnext: python scripts/make_figures.py --run {run.name}")


if __name__ == "__main__":
    main()
