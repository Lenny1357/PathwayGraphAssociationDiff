"""Is the fixed nopad VAE's held-out loss a bad seed or real overfitting? GPU job only.

    sbatch run_debug_vae.sh scripts/check_vae_seeds.py

scripts/check_vae_sampling.py trained the fixed VAE (arms.build_arm defaults)
once per preset: pad generalised (held-out CE 7.65 vs marginal 8.31), nopad
did not (8.89), at nearly the same training loss. This trains nopad on E. coli
with three seeds and follows the held-out set every 500 steps:

    mean CE      what the check scored; unbounded, so a few confident misses dominate it
    median NLL   per real slot; robust to those misses
    acc / top10  how often the true entity is the argmax / in the top 10
    latent       held-out CE with z = 0 minus with z = mu

Diverging mean CE with a steady median means overconfidence on a few slots;
both rising together means overfitting.
"""

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from graphpath.pipeline import arms as A
from graphpath.pipeline.common import load_bundle
from graphpath.training.loop import train_step


@torch.no_grad()
def held_out(model, ids, adj, msk, batch_bn: bool = False):
    """batch_bn: normalise mu with the held-out batch's own statistics (as in
    training) instead of BN's running averages; running stats are restored."""
    model.eval()
    saved = None
    if batch_bn and model.mu_bn is not None:
        saved = (model.mu_bn.running_mean.clone(), model.mu_bn.running_var.clone())
        model.mu_bn.train()
    ids, adj, msk = model.canonical_order(ids, adj, msk)
    (nl, _), mu, _ = model.forward(ids, adj, msk)
    nll = F.cross_entropy(nl[msk], ids[msk], reduction="none")
    nl0, _ = model.decode(torch.zeros_like(mu), ids.shape[1])
    ce0 = F.cross_entropy(nl0[msk], ids[msk]).item()
    top10 = (nl[msk].topk(10, -1).indices == ids[msk][:, None]).any(-1).float().mean().item()
    if saved is not None:
        model.mu_bn.running_mean.copy_(saved[0]); model.mu_bn.running_var.copy_(saved[1])
    model.train()
    return dict(ce=nll.mean().item(), median=nll.median().item(), p90=nll.quantile(0.9).item(),
                acc=(nl.argmax(-1)[msk] == ids[msk]).float().mean().item(), top10=top10,
                latent=ce0 - nll.mean().item())


def main(steps=3000, seeds=(0, 1, 2), N=64, bs=32, every=500, compare_bn=False):
    device = torch.device("cuda")
    b = load_bundle("ecoli")
    counts = torch.as_tensor(b.token_counts(), dtype=torch.float)
    p = counts[counts > 0] / counts.sum(); marginal = float(-(p * p.log()).sum())
    hid, hadj = b.heldout_tensors(None, N)
    hmsk = hid > 0
    hid, hadj, hmsk = hid.to(device), hadj.long().to(device), hmsk.to(device)
    print(f"ecoli: token-marginal CE {marginal:.3f}  held-out graphs {len(hid)}")
    finals = []
    for seed in seeds:
        torch.manual_seed(seed)
        model, _ = A.build_arm("vae", b, A.PRESETS["nopad"], device, N, bs)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        g = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(b.train, batch_size=bs, shuffle=True, generator=g)
        it, hist, t0 = iter(loader), [], time.time()
        print(f"\n=== nopad seed {seed}")
        for n in range(1, steps + 1):
            try:
                x = next(it)
            except StopIteration:
                it = iter(loader); x = next(it)
            r = train_step(model, opt, x[1].to(device), x[0].long().to(device),
                           b.vocab_size, b.vocab_size + 1, mask=x[2].bool().to(device))
            hist.append(r["loss_node"])
            if n % every == 0:
                for tag, bb in ((("running",), False),) + (((("batch",), True),) if compare_bn else ()):
                    h = held_out(model, hid, hadj, hmsk, batch_bn=bb)
                    print(f"  step {n:5d} [{tag[0]:7} BN]  train {np.mean(hist[-every:]):.3f}  held-out CE {h['ce']:.3f}  "
                          f"median {h['median']:.3f}  p90 {h['p90']:.2f}  acc {h['acc']:.1%}  top10 {h['top10']:.1%}  "
                          f"latent {h['latent']:+.3f}  ({time.time() - t0:.0f}s)", flush=True)
                h = held_out(model, hid, hadj, hmsk)
        model.recalibrate_bn(torch.utils.data.DataLoader(b.train, batch_size=bs, shuffle=True), device)
        h = held_out(model, hid, hadj, hmsk)
        print(f"  after BN recalibration [running BN]  held-out CE {h['ce']:.3f}  median {h['median']:.3f}  "
              f"acc {h['acc']:.1%}  top10 {h['top10']:.1%}  latent {h['latent']:+.3f}", flush=True)
        finals.append(h)
        del model, opt; torch.cuda.empty_cache()
    ce = np.array([f["ce"] for f in finals]); med = np.array([f["median"] for f in finals])
    print(f"\nfinal held-out CE {ce.mean():.3f} +- {ce.std():.3f}  (per seed {np.round(ce, 3).tolist()})  "
          f"vs marginal {marginal:.3f};  median NLL {med.mean():.3f} +- {med.std():.3f}")
    return 0


if __name__ == "__main__":
    if "--compare-bn" in sys.argv:          # one seed, every 250 steps, running vs batch BN stats
        sys.exit(main(seeds=(0,), every=250, compare_bn=True))
    sys.exit(main())
