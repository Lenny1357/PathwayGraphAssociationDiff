"""Choose how far to damp PAD for the pad masked arm, per dataset. GPU job only.

    sbatch run_calibrate_pad_scale.sh

The pad masked arm sizes its own graphs by emitting PAD, but undamped it
collapses on human: an early PAD tips a graph into "small graph" mode and
the rest follow. `pad_scale` multiplies P(PAD) on every slot. This samples
the arm unclamped at each scale on a grid, compares the graph sizes it
emits with the training size distribution, and writes the scale with the
smallest 1-Wasserstein distance to <run>/pad_scale.json, where every
sampling site reads it (arms.pad_scale_for).

Fails (exit 1) if the chosen scale collapses graphs, or emits no small graph
for a dataset whose training graphs include small ones.
"""

import argparse
import sys
import time

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from graphpath.pipeline import arms as A
from graphpath.pipeline.common import RunLayout, dump_json, load_bundle

GRID = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]


def size_stats(sizes: np.ndarray, N: int) -> dict:
    return dict(mean=float(sizes.mean()), below_max=float((sizes < N).mean()),
                small=float((sizes <= N // 2).mean()), pad_share=float(1 - sizes.mean() / N))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="compare_oct2_20260929_1542")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device, N = torch.device("cuda"), 64
    run = RunLayout(a.run)
    out, ok = {}, True
    for d in sorted({d for d, arm, v in run.all_weights() if arm == "masked" and A.size_clamped(v)}):
        b = load_bundle(d)
        data = b.sizes().clip(max=N)
        ds = size_stats(data, N)
        print(f"\n{d}: training graphs  mean {ds['mean']:.1f} nodes  <{N}: {ds['below_max']:.1%}  "
              f"<={N // 2}: {ds['small']:.1%}  PAD share {ds['pad_share']:.3f}")
        model, _ = A.load_arm(run, d, "masked", "pad", device)
        rows = []
        for scale in GRID:
            torch.manual_seed(a.seed); t0 = time.time()
            with torch.no_grad():
                n_, a_ = model.sample(N=N, B=a.n, device=device, pad_scale=scale)
            sizes = (n_ > 0).sum(1).cpu().numpy()
            edges = ((a_ > 0).sum((1, 2)) / 2).float().cpu().numpy()
            r = dict(scale=scale, **size_stats(sizes, N), w1=float(wasserstein_distance(sizes, data)),
                     edges=float(edges.mean()), zero_edge=int((edges == 0).sum()),
                     size_hist=np.bincount(sizes, minlength=N + 1).tolist())
            rows.append(r)
            print(f"  scale {scale:<6}  mean {r['mean']:5.1f}  <{N}: {r['below_max']:6.1%}  <={N // 2}: {r['small']:6.1%}  "
                  f"PAD {r['pad_share']:.3f}  W1 {r['w1']:5.2f}  edges/graph {r['edges']:5.1f}  "
                  f"zero-edge {r['zero_edge']:2d}  ({time.time() - t0:.0f}s)")
        best = min(rows, key=lambda r: r["w1"])
        print(f"  -> chosen scale {best['scale']} (W1 {best['w1']:.2f})")
        if best["mean"] < 0.5 * ds["mean"] or (ds["below_max"] > 0.05 and best["below_max"] == 0):
            ok = False
        out[d] = dict(scale=best["scale"], w1=best["w1"], data=ds,
                      data_size_hist=np.bincount(data, minlength=N + 1).tolist(), grid=rows,
                      n=a.n, seed=a.seed)
    if ok:
        dump_json(out, run.path / "pad_scale.json")
        print(f"\nPASS -> {run.path / 'pad_scale.json'}")
    else:
        print("\nFAIL: chosen scale collapses graphs or emits no small graphs; pad_scale.json not written")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
