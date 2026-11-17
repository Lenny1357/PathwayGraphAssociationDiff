"""Which PAD damping schedule reproduces the training graph sizes? GPU job only.

    sbatch run_debug_vae.sh scripts/calibrate_pad_schedule.py

Extends scripts/calibrate_pad_scale.py (one constant scale, written to
pad_scale.json and used by the pipeline) with two schedules for the pad
masked arm, sampled unclamped:

    constant     the same damping at every step (the current pipeline rule)
    time         damped early, easing to 1 below `relax` of the trajectory
    until_first  damped only until the graph holds its first PAD

Each setting is scored against the training size distribution: mean size,
share of graphs below 64 and at or below 32 nodes, PAD share, and the
1-Wasserstein distance between the size distributions. Results go to
<run>/pad_schedule_calibration.json; pad_scale.json is NOT changed, so the
pipeline keeps its current rule until a schedule is chosen.
"""

import sys
import time

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from graphpath.pipeline import arms as A
from graphpath.pipeline.common import RunLayout, dump_json, load_bundle

GRID = (
    [dict(pad_schedule="constant", pad_scale=s) for s in (0.3, 0.1, 0.03, 0.02, 0.01)]
    + [dict(pad_schedule="time", pad_scale=s, pad_relax=r) for s in (0.01, 0.003, 0.001) for r in (0.5, 0.25)]
    + [dict(pad_schedule="until_first", pad_scale=s) for s in (0.1, 0.03, 0.01, 0.005, 0.002, 0.001, 0.0005)]
)


def stats(sizes, N):
    return dict(mean=float(sizes.mean()), below_max=float((sizes < N).mean()),
                small=float((sizes <= N // 2).mean()), pad_share=float(1 - sizes.mean() / N))


def main(run_name="compare_oct2_20260929_1542", B=96, N=64, seed=0):
    device = torch.device("cuda")
    run = RunLayout(run_name)
    out = {}
    for d in sorted({d for d, a, v in run.all_weights() if a == "masked" and A.size_clamped(v)}):
        b = load_bundle(d)
        data = b.sizes().clip(max=N)
        ds = stats(data, N)
        print(f"\n{d}: training graphs  mean {ds['mean']:.1f}  <{N}: {ds['below_max']:.1%}  "
              f"<={N // 2}: {ds['small']:.1%}  PAD {ds['pad_share']:.3f}")
        model, _ = A.load_arm(run, d, "masked", "pad", device)
        rows = []
        for kw in GRID:
            torch.manual_seed(seed); t0 = time.time()
            with torch.no_grad():
                n_, a_ = model.sample(N=N, B=B, device=device, **kw)
            sizes = (n_ > 0).sum(1).cpu().numpy()
            edges = ((a_ > 0).sum((1, 2)) / 2).float().cpu().numpy()
            r = dict(**kw, **stats(sizes, N), w1=float(wasserstein_distance(sizes, data)),
                     edges=float(edges.mean()), zero_edge=int((edges == 0).sum()),
                     size_hist=np.bincount(sizes, minlength=N + 1).tolist())
            rows.append(r)
            name = kw["pad_schedule"] + (f" relax {kw['pad_relax']}" if "pad_relax" in kw else "")
            print(f"  {name:20} scale {kw['pad_scale']:<7} mean {r['mean']:5.1f}  <{N}: {r['below_max']:6.1%}  "
                  f"<={N // 2}: {r['small']:6.1%}  PAD {r['pad_share']:.3f}  W1 {r['w1']:5.2f}  "
                  f"edges/graph {r['edges']:5.1f}  zero-edge {r['zero_edge']:2d}  ({time.time() - t0:.0f}s)", flush=True)
        best = {s: min((r for r in rows if r["pad_schedule"] == s), key=lambda r: r["w1"])
                for s in ("constant", "time", "until_first")}
        for s, r in best.items():
            print(f"  best {s:12} scale {r['pad_scale']}" + (f" relax {r['pad_relax']}" if "pad_relax" in r else "")
                  + f"  W1 {r['w1']:.2f}  <{N} {r['below_max']:.1%}  <={N // 2} {r['small']:.1%}")
        out[d] = dict(data=ds, data_size_hist=np.bincount(data, minlength=N + 1).tolist(), grid=rows,
                      best={s: {k: v for k, v in r.items() if k != "size_hist"} for s, r in best.items()},
                      n=B, seed=seed)
        del model; torch.cuda.empty_cache()
    dump_json(out, run.path / "pad_schedule_calibration.json")
    print(f"\n-> {run.path / 'pad_schedule_calibration.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
