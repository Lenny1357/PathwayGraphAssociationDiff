"""Score a finished run and run the analyses. The benchmark notebooks.

    python scripts/score_run.py --run compare_sept_20260905_2327
    python scripts/score_run.py --run latest:compare_typed --analyses scores trajectories remask
    python scripts/score_run.py --run compare_sept_20260905_2327 --resample

Every analysis writes one JSON under outputs/analysis/<run>/ and nothing
else; draw the figures with scripts/make_figures.py.

analyses: scores architecture pathway_stats graph_stats clustering trajectories
          remask coexpr_support discovery clusters attention ablation threshold maps
"""

import argparse
import time

from graphpath.pipeline import analysis as AN
from graphpath.pipeline import tables as T
from graphpath.pipeline import trajectories as TR
from graphpath.pipeline.common import RunLayout, load_json
from graphpath.pipeline.score import ScoreRun
from graphpath.training.device import pick_device

ALL = ["scores", "architecture", "pathway_stats", "graph_stats", "clustering", "trajectories", "remask",
       "coexpr_support", "discovery", "clusters", "attention", "ablation", "threshold", "maps"]
CORE = ["scores", "architecture", "pathway_stats", "graph_stats", "clustering"]


def resolve(name: str) -> RunLayout:
    if name.startswith("latest:"):
        return RunLayout.latest(name.split(":", 1)[1])
    return RunLayout(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name, or latest:<prefix>")
    ap.add_argument("--analyses", nargs="+", default=ALL, help=f"subset of {ALL}; 'core' = {CORE}")
    ap.add_argument("--n-graphs", type=int, default=96)
    ap.add_argument("--resample", nargs="*", default=None, metavar="KEY",
                    help="draw fresh samples instead of scoring the saved ones: alone for every arm, "
                         "or the keys to redraw, e.g. pad/human|masked")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    analyses = CORE if a.analyses == ["core"] else a.analyses

    run = resolve(a.run)
    device = __import__("torch").device("cpu") if a.cpu else pick_device()
    print(f"run {run.path}\ndevice {device}\nanalysis -> {run.analysis_dir()}")
    sc = ScoreRun(run, device, n_graphs=a.n_graphs)
    sc.load_data()
    sc.load_samples(resample=(a.resample is not None and (set(a.resample) or True)))
    out = run.analysis_dir()
    scores = load_json(out / "scores.json") if (out / "scores.json").exists() else None

    def stage(name, fn):
        if name not in analyses:
            return
        print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
        t0 = time.time()
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — one failed analysis must not lose the others
            print(f"  {name} FAILED ({type(e).__name__}: {e})")
        print(f"  ({time.time() - t0:.0f}s)")

    def do_scores():
        nonlocal scores
        scores = sc.score()
        T.write_tables(out / "scores.json", out / "tables")
    stage("scores", do_scores)
    stage("architecture", lambda: sc.architecture())
    stage("pathway_stats", lambda: AN.pathway_stats(run, sc.data, sc.samples))
    stage("graph_stats", lambda: AN.graph_stats(run, sc.data, sc.samples))
    stage("clustering", lambda: AN.clustering(run, sc.data, sc.samples, scores))
    stage("trajectories", lambda: TR.trajectories(run, sc.data, device))
    stage("remask", lambda: TR.remask(run, sc.data, device))
    suites = {d: sc.suite(d) for d in sc.data} if any(x in analyses for x in
                                                      ("coexpr_support", "discovery", "clusters", "ablation", "threshold", "maps")) else {}
    support = None
    def do_support():
        nonlocal support
        support = AN.coexpr_support(run, sc.data, sc.samples, suites, device)
    stage("coexpr_support", do_support)
    stage("discovery", lambda: AN.discovery(run, sc.data, sc.samples, suites, device, support_json=support))
    stage("clusters", lambda: AN.embedding_clusters(run, sc.data, suites, device))
    stage("attention", lambda: AN.attention(run, sc.data, device))
    stage("ablation", lambda: AN.ablation(run, sc.data, suites, device, scores))
    stage("threshold", lambda: AN.threshold_sweep(run, sc.data, suites, device))
    stage("maps", lambda: AN.maps(run, sc.data, sc.samples, suites, device))
    print(f"\nnext: python scripts/make_figures.py --run {run.name}")


if __name__ == "__main__":
    main()
