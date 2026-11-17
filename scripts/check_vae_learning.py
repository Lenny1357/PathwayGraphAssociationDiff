"""Does the GraphVAE learn with the canonical-order and bag-of-words fixes? GPU job only.

    sbatch run_check_vae.sh

Trains short VAEs on E. coli (same build and train_step as train_compare.py)
in four configurations and reports, on held-out graphs:

    node CE            unweighted, with z = mu, against the token-marginal baseline
    CE with z = 0      how much the decoder relies on the latent (gap = latent in use)
    token accuracy     argmax reconstruction, real slots only
    KL                 above the free-bits floor means the latent carries information

Passes when a fixed configuration beats the marginal baseline by >= 0.5 nats,
loses >= 0.5 nats when z is zeroed, and its training loss falls.
"""

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from graphpath.pipeline import arms as A
from graphpath.pipeline.common import load_bundle
from graphpath.training.loop import train_step

VARIANTS = {
    "original":           {},
    "canonical":          dict(vae_canonical_order=True),
    "bow":                dict(vae_bow_weight=1.0),
    "canonical+bow":      dict(vae_canonical_order=True, vae_bow_weight=1.0),
    "bn":                 dict(vae_mu_batchnorm=True),
    "bn+canonical":       dict(vae_mu_batchnorm=True, vae_canonical_order=True),
    "bn+bow":             dict(vae_mu_batchnorm=True, vae_bow_weight=1.0),
}


@torch.no_grad()
def evaluate(model, ids, adj, msk, canonical: bool) -> dict:
    model.eval()
    if canonical:
        ids, adj, msk = model.canonical_order(ids, adj, msk)
    (nl, _), mu, logvar = model.forward(ids, adj, msk)
    ce = F.cross_entropy(nl[msk], ids[msk]).item()
    nl0, _ = model.decode(torch.zeros_like(mu), ids.shape[1])
    ce0 = F.cross_entropy(nl0[msk], ids[msk]).item()
    acc = (nl.argmax(-1)[msk] == ids[msk]).float().mean().item()
    kl = (0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)).sum(1).mean().item()
    return dict(ce=ce, ce_zero_z=ce0, acc=acc, kl=kl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ecoli")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    device, N, bs = torch.device("cuda"), 64, 32
    b = load_bundle(a.dataset)
    counts = torch.as_tensor(b.token_counts(), dtype=torch.float)
    p = counts[counts > 0] / counts.sum()
    marginal = float(-(p * p.log()).sum())                     # CE of predicting the token marginal
    hid, hadj = b.heldout_tensors(None, N)
    hmsk = hid > 0
    hid, hadj, hmsk = hid.to(device), hadj.long().to(device), hmsk.to(device)
    print(f"{a.dataset}: token-marginal CE {marginal:.3f} nats  (chance ln V = {np.log(b.vocab_size):.3f})"
          f"  held-out graphs {len(hid)}")
    ok = False
    for name in a.variants:
        torch.manual_seed(0)
        model, _ = A.build_arm("vae", b, A.PRESETS["nopad"], device, N, bs, overrides=VARIANTS[name])
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        loader = torch.utils.data.DataLoader(b.train, batch_size=bs, shuffle=True)
        it, hist, t0 = iter(loader), [], time.time()
        for n in range(1, a.steps + 1):
            try:
                x = next(it)
            except StopIteration:
                it = iter(loader); x = next(it)
            adj, ids, msk = x[0].long().to(device), x[1].to(device), x[2].bool().to(device)
            r = train_step(model, opt, ids, adj, b.vocab_size, b.vocab_size + 1, mask=msk)
            hist.append(r["loss_node"])
            if n % 500 == 0:
                print(f"  {name:14} step {n:5d}  node {np.mean(hist[-500:]):.3f}  kl {r['loss_kl']:.2f}  "
                      f"bow {r.get('loss_bow', 0):.3f}  ({time.time() - t0:.0f}s)", flush=True)
        ev = evaluate(model, hid, hadj, hmsk, VARIANTS[name].get("vae_canonical_order", False))
        falling = np.mean(hist[-500:]) < np.mean(hist[:500]) - 0.05
        gain, used = marginal - ev["ce"], ev["ce_zero_z"] - ev["ce"]
        print(f"  {name:14} HELD-OUT  node CE {ev['ce']:.3f} ({gain:+.3f} vs marginal)  "
              f"z=0 CE {ev['ce_zero_z']:.3f} (latent worth {used:+.3f})  acc {ev['acc']:.1%}  "
              f"KL {ev['kl']:.1f}  train loss falling: {falling}")
        if name != "original" and gain >= 0.5 and used >= 0.5 and falling:
            ok = True
        del model, opt; torch.cuda.empty_cache()
    print("\nPASS" if ok else "\nFAIL: no fixed configuration learned past the marginal")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
