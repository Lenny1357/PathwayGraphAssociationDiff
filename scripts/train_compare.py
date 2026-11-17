"""Train every arm on every dataset, then sample. The compare_models notebooks.

    python scripts/train_compare.py --preset sept --datasets ecoli human
    python scripts/train_compare.py --preset typed --steps 20000
    python scripts/train_compare.py --preset oct2                 # nopad + pad variants
    python scripts/train_compare.py --run compare_sept_20260905_2327 --sample-only
    python scripts/train_compare.py --run <run> --preset oct2 --variants pad --datasets human --arms vae

Resumable: an arm whose weights exist is skipped. Outputs go to
outputs/runs/<prefix>_<stamp>/; score them with scripts/score_run.py.
"""

import argparse

from graphpath.pipeline import arms as A
from graphpath.pipeline.common import RunLayout
from graphpath.pipeline.compare import CompareConfig, CompareExperiment
from graphpath.training.device import pick_device

VARIANTS = {"sept": {"": "sept"}, "typed": {"": "typed"}, "oct2": {"nopad": "nopad", "pad": "pad"},
            "nopad": {"": "nopad"}, "pad": {"": "pad"}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="sept", choices=sorted(VARIANTS))
    ap.add_argument("--datasets", nargs="+", default=["ecoli", "human"])
    ap.add_argument("--arms", nargs="+", default=list(A.DEFAULT_ARMS))
    ap.add_argument("--variants", nargs="+", default=None,
                    help="only these variants of the preset, e.g. pad (one job per variant runs in parallel)")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-samples", type=int, default=96)
    ap.add_argument("--run", default=None, help="existing run to resume (default: new compare_<preset>_<stamp>)")
    ap.add_argument("--sample-only", action="store_true", help="skip training; only (re)draw samples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    variants = VARIANTS[a.preset]
    if a.variants:
        variants = {v: p for v, p in variants.items() if v in a.variants}
        if not variants:
            ap.error(f"--variants {a.variants} not in preset {a.preset}: {sorted(VARIANTS[a.preset])}")
    cfg = CompareConfig(datasets=a.datasets, arms=a.arms, variants=variants, steps=a.steps,
                        batch_size=a.batch_size, n_samples=a.n_samples, seed=a.seed)
    run = RunLayout(a.run) if a.run else RunLayout.new(f"compare_{a.preset}")
    device = pick_device()
    print(f"run {run.path}\ndevice {device}\npreset {a.preset}: {cfg.variants}")
    exp = CompareExperiment(cfg, run, device)
    exp.load_data()
    if not a.sample_only:
        exp.train_all()
    exp.sample_all()
    print(f"\ndone -> {run.path}\nnext: python scripts/score_run.py --run {run.name}")


if __name__ == "__main__":
    main()
