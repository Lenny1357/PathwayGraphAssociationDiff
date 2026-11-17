"""Does the fixed GraphVAE learn AND generate sensible graphs? GPU job only.

    sbatch run_debug_vae.sh scripts/check_vae_sampling.py

Trains the VAE as arms.build_arm now builds it (BN on mu, canonical order,
symmetric edges, edges from real node types) on E. coli, nopad and pad, then:

    learning     held-out node CE vs the token marginal, and the cost of zeroing z
    generation   96 graphs from the prior: edge density, type-violation rate
                 (edges whose type pair never occurs in a training graph), PAD
                 share and distinct entities per graph, each against the data

Passes when both presets learn (>= 0.5 nats better than the marginal, latent
worth >= 0.5 nats) and generate within a factor 1.5 of the data's density.
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


def legal_type_pairs(b, N):
    pairs = set()
    for g in b.train_graphs:
        k = g["num_nodes"]; ids = g["node_ids"][:k]
        adj = np.asarray(g["adj_matrix"])[:k, :k]
        ts = [b.ntype(b.id_to_node.get(int(i), "")) for i in ids]
        for i, j in zip(*np.nonzero(np.triu(adj, 1))):
            pairs.add(tuple(sorted((ts[i], ts[j]))))
    return pairs


def describe(b, nodes, adj, legal, N):
    iu = np.triu_indices(N, 1)
    dens, viol, n_e, pad, distinct = [], 0, 0, [], []
    for n, a in zip(nodes.tolist(), adj.cpu().numpy()):
        real = [i for i, t in enumerate(n) if t != 0]
        pad.append(1 - len(real) / N); distinct.append(len({n[i] for i in real}))
        if len(real) > 1:
            sub = a[np.ix_(real, real)]; m = len(real)
            dens.append(sub[np.triu_indices(m, 1)].mean())
        ts = [b.ntype(b.id_to_node.get(int(t), "")) for t in n]
        for i, j in zip(*iu):
            if a[i, j]:
                n_e += 1
                viol += tuple(sorted((ts[i], ts[j]))) not in legal
    return dict(density=float(np.mean(dens)), violation=viol / max(n_e, 1),
                pad=float(np.mean(pad)), distinct=float(np.mean(distinct)), edges=n_e / len(nodes))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    device, N, bs = torch.device("cuda"), 64, 32
    b = load_bundle("ecoli")
    legal = legal_type_pairs(b, N)
    counts = torch.as_tensor(b.token_counts(), dtype=torch.float)
    p = counts[counts > 0] / counts.sum(); marginal = float(-(p * p.log()).sum())
    hid, hadj = b.heldout_tensors(None, N)
    hmsk = hid > 0
    hid, hadj, hmsk = hid.to(device), hadj.long().to(device), hmsk.to(device)
    data = describe(b, hid.cpu(), hadj, legal, N)
    print(f"legal type pairs {sorted(legal)}")
    print(f"held-out data: density {data['density']:.4f}  violations {data['violation']:.1%}  "
          f"PAD {data['pad']:.3f}  distinct/graph {data['distinct']:.1f}  edges/graph {data['edges']:.1f}")
    ok = True
    for preset in ("nopad", "pad"):
        torch.manual_seed(0)
        model, cfg = A.build_arm("vae", b, A.PRESETS[preset], device, N, bs)
        flags = {k: getattr(cfg, k) for k in ("vae_mu_batchnorm", "vae_canonical_order", "vae_symmetric_edges",
                                              "vae_teacher_types", "supervise_pad")}
        print(f"\n=== {preset}  {flags}")
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        loader = torch.utils.data.DataLoader(b.train, batch_size=bs, shuffle=True)
        it, hist, t0 = iter(loader), [], time.time()
        for n in range(1, a.steps + 1):
            try:
                x = next(it)
            except StopIteration:
                it = iter(loader); x = next(it)
            r = train_step(model, opt, x[1].to(device), x[0].long().to(device),
                           b.vocab_size, b.vocab_size + 1, mask=x[2].bool().to(device))
            hist.append(r["loss_node"])
            if n % 1000 == 0:
                print(f"  step {n:5d}  node {np.mean(hist[-500:]):.3f}  edge {r['loss_edge']:.3f}  "
                      f"kl {r['loss_kl']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        model.eval()
        with torch.no_grad():
            ids, adj, msk = model.canonical_order(hid, hadj, hmsk)
            (nl, _), mu, _ = model.forward(ids, adj, msk)
            ce = F.cross_entropy(nl[msk], ids[msk]).item()
            nl0, _ = model.decode(torch.zeros_like(mu), N)
            worth = F.cross_entropy(nl0[msk], ids[msk]).item() - ce
            torch.manual_seed(1)
            sn, sa = model.sample(N=N, B=96, device=device)
        gen = describe(b, sn.cpu(), sa, legal, N)
        print(f"  held-out node CE {ce:.3f} ({marginal - ce:+.3f} vs marginal {marginal:.3f})  latent worth {worth:+.3f}")
        print(f"  generated: density {gen['density']:.4f} (data {data['density']:.4f})  violations {gen['violation']:.1%}  "
              f"PAD {gen['pad']:.3f}  distinct/graph {gen['distinct']:.1f}  edges/graph {gen['edges']:.1f}")
        ratio = gen["density"] / data["density"]
        if marginal - ce < 0.5 or worth < 0.5 or not (1 / 1.5 <= ratio <= 1.5):
            ok = False
        del model, opt; torch.cuda.empty_cache()
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
