"""What did a change buy? Two scored runs, metric by metric, as distance to
held-out. The typed-vs-untyped cell of benchmark_typed.ipynb.

    python scripts/compare_runs.py --new latest:compare_typed --old compare_sept_20260905_2327 \
        --label-new typed --label-old untyped

Reads outputs/analysis/<run>/scores.json for both, writes
outputs/analysis/<new>/compare_<old>.{json,csv} and a figure per dataset.
"""

import argparse

import numpy as np

from graphpath.pipeline import tables as T
from graphpath.pipeline.common import ANALYSIS, dump_json, load_json
from graphpath.pipeline.figures import INK, RC, SOFT, _save, plt
from scripts.score_run import resolve


def benefit_figure(df, out, label_new, label_old):
    paths = []
    with plt.rc_context(RC):
        for d in df.dataset.unique():
            sub = df[(df.dataset == d) & np.isfinite(df.closer)]
            if sub.empty:
                continue
            agg = sub.groupby("metric").agg(d_old=("d_old", "mean"), d_new=("d_new", "mean"),
                                            closer=("closer", "mean")).sort_values("closer")
            y = np.arange(len(agg))
            fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(agg) + 2.4))
            for yi, (_, r) in zip(y, agg.iterrows()):
                ax.plot([r.d_old, r.d_new], [yi, yi], lw=2.4, color="#1baf7a" if r.closer > 0 else "#c2402a", zorder=2)
                ax.scatter([r.d_old], [yi], s=46, color=SOFT, zorder=3, label=label_old if yi == 0 else None)
                ax.scatter([r.d_new], [yi], s=46, color="#2a78d6", zorder=4, marker="D", label=label_new if yi == 0 else None)
            ax.set_yticks(y); ax.set_yticklabels(agg.index, fontsize=8)
            ax.set_xlabel("distance to held-out   (0 = matches real pathways)"); ax.set_xlim(-0.02, 1.02)
            ax.legend(fontsize=8, frameon=False, loc="lower right"); ax.grid(alpha=.35, axis="x"); ax.set_axisbelow(True)
            ax.set_title(f"{d} — {label_new} vs {label_old}, averaged over arms: green moved toward real pathways "
                         f"({int((agg.closer > 0).sum())} of {len(agg)} metrics)", loc="left", fontsize=9, color=INK)
            paths.append(_save(fig, out / d / f"benefit_{label_new}_vs_{label_old}.png"))
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True); ap.add_argument("--old", required=True)
    ap.add_argument("--label-new", default="new"); ap.add_argument("--label-old", default="old")
    a = ap.parse_args()
    new, old = resolve(a.new), resolve(a.old)
    s_new, s_old = load_json(ANALYSIS / new.name / "scores.json"), load_json(ANALYSIS / old.name / "scores.json")
    df = T.compare_runs(s_new, s_old, a.label_new, a.label_old)
    out = ANALYSIS / new.name
    dump_json(df.to_dict(orient="records"), out / f"compare_{old.name}.json")
    df.to_csv(out / "tables" / f"compare_{old.name}.csv", index=False)
    print(df[["dataset", "arm", "metric", a.label_old, a.label_new, "held_out", "closer"]]
          .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    for p in benefit_figure(df, out / "figures", a.label_new, a.label_old):
        print("  ->", p)


if __name__ == "__main__":
    main()
