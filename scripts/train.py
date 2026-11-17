"""Train one arm on one dataset and save a checkpoint.

    python scripts/train.py --dataset ecoli --model masked --steps 20000
    python scripts/train.py --dataset ecoli --model transition --typed
    python scripts/train.py --list

Every arm shares the training step; the transition arms need the data
marginals installed first, which this script does. Output lands in
outputs/runs/train_<stamp>/<dataset>/<model>/{weights,samples}.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from graphpath.config import Config, JointConfig, MarginalConfig
from graphpath.data.loaders import available_datasets, load_dataset, make_loader, token_to_type
from graphpath.models import ARMS, build_model
from graphpath.processes.marginal import estimate_marginals
from graphpath.training.device import pick_device
from graphpath.training.loop import batched_randperm, reorder_graph, train_step
from graphpath.training.runs import RunDir, save_checkpoint

SHARED = dict(T=200, n_layers=4, d_model=128, n_heads=4, edge_n_heads=4, d_ff=256, dropout=0.1)


def make_config(model: str, typed: bool):
    types = dict(n_node_types=4, use_type_bias=True) if typed else {}
    if model == "transition":
        return JointConfig(**SHARED, **types, supervise_pad=typed, pad_loss_weight=4.0 if typed else 1.0)
    if model == "marginal":
        return MarginalConfig(**SHARED, **types)
    return Config(**SHARED, **types, supervise_pad=typed, pad_loss_weight=4.0 if typed else 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ecoli", choices=["ecoli", "human", "kegg"])
    ap.add_argument("--model", default="masked", choices=sorted(ARMS))
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-nodes", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--typed", action="store_true", help="n_node_types=4 + supervise padding")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default=None, help="existing run directory to add to")
    ap.add_argument("--list", action="store_true", help="show available datasets and exit")
    a = ap.parse_args()
    if a.list:
        print("datasets:", available_datasets(), "\nmodels:", sorted(ARMS))
        return

    torch.manual_seed(a.seed)
    device = pick_device()
    if a.model in ("transition", "marginal") and device.type == "mps":
        # the marginal posterior is computed in float64, which MPS lacks
        print("MPS has no float64; the transition arms run on CPU")
        device = torch.device("cpu")
    train_ds, test_ds, info = load_dataset(a.dataset, a.max_nodes, seed=a.seed, device=device)
    loader = make_loader(train_ds, a.batch_size)
    cfg = make_config(a.model, a.typed)
    model = build_model(a.model, cfg, vocab_size=train_ds.vocab_size + 1).to(device)
    if a.model in ("transition", "marginal"):
        model.set_marginals(*estimate_marginals(loader, train_ds.vocab_size, cfg.edge_vocab,
                                                floor=cfg.marginal_floor))
    if a.typed:
        model.set_node_types(torch.tensor(token_to_type(train_ds, info["node_type_fn"])))
    permute = a.model != "vae"       # the VAE's loss is slot-wise; see GraphVAE docstring
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    print(f"{a.model} on {a.dataset}: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{len(train_ds)} train graphs, device {device}")

    run = RunDir(a.run) if a.run else RunDir.new("train")
    out = run.arm_dir(a.dataset, a.model)
    history, step, t0 = [], 0, time.time()
    while step < a.steps:
        for adjcs, node_feats, mask, *_ in loader:
            adjcs, node_feats, mask = adjcs.to(device).long(), node_feats.to(device), mask.to(device).bool()
            if permute:
                perm = batched_randperm(*node_feats.shape, node_feats.device)
                node_feats, adjcs = reorder_graph(node_feats, adjcs, perm)
                mask = mask.gather(1, perm)
            rec = train_step(model, opt, node_feats, adjcs, train_ds.vocab_size,
                             train_ds.vocab_size + 1, mask=mask)
            history.append(rec); step += 1
            if step % 100 == 0:
                print(f"  step {step:6d}  loss {rec['loss']:.4f}  ({time.time() - t0:.0f}s)")
            if step >= a.steps:
                break

    save_checkpoint(model, cfg, out / "weights" / f"{a.model}.pt", history=history,
                    arm=a.model, dataset=a.dataset, typed=a.typed)
    model.eval()
    nodes, adj = model.sample(N=a.max_nodes, B=16, device=device)
    torch.save({"nodes": nodes.cpu(), "adj": adj.cpu()}, out / "samples" / f"{a.model}.pt")
    (out / "history.json").write_text(json.dumps(history))
    print("saved to", out)


if __name__ == "__main__":
    main()
