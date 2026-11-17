"""Training step, isomorphism augmentation, and the epoch loop."""

from typing import TYPE_CHECKING, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:   # type hints only; the models import this module, so no runtime cycle
    from graphpath.models.continuous import ContinuousGraphDiffusion
    from graphpath.models.marginal import GraphDiffBERTMarginal
    from graphpath.models.masked import GraphDiffBERT
    from graphpath.models.uniform import GraphDiffBERTUniform
    from graphpath.models.vae import GraphVAE

    AnyGraphDiff = Union[GraphDiffBERT, GraphDiffBERTUniform, GraphDiffBERTMarginal,
                         ContinuousGraphDiffusion, GraphVAE]
else:
    AnyGraphDiff = nn.Module


# ── permutation augmentation ────────────────────────────────────────────────
# Node slots are not canonically ordered, so every batch is re-indexed by a
# random permutation. This is exactly why the model uses random per-node
# identity features (FIX 5) rather than a learned positional-embedding
# table — a positional table would fight this augmentation directly.

def reorder_graph(node_feats: Tensor, adjcs: Tensor, perm: Tensor):
    B, N = node_feats.shape
    batch_idx = torch.arange(B, device=node_feats.device).unsqueeze(1)
    new_node_feats = node_feats[batch_idx, perm]
    idx1 = perm.unsqueeze(-1).expand(-1, -1, N)
    idx2 = perm.unsqueeze(-2).expand(-1, N, -1)
    new_adjcs = adjcs.gather(1, idx1).gather(2, idx2)
    return new_node_feats, new_adjcs


def batched_randperm(B: int, N: int, device=None) -> Tensor:
    if device is None:
        device = torch.device("cpu")
    return torch.rand(B, N, device=device).argsort(dim=1)


# ── single optimisation step (drop-in) ──────────────────────────────────────

def train_step(
    model:            AnyGraphDiff,
    opt:              torch.optim.Optimizer,
    node_feats:       Tensor,
    adjcs:            Tensor,
    mask_token_id:    int,
    vocab_size:       int,
    mask:             Optional[Tensor] = None,
    loss_edge_weight: float = 1.0,
    co_expr:          Optional[dict] = None,
) -> dict:
    model.train()
    B, N   = node_feats.shape
    device = node_feats.device
    if mask is None:
        mask = torch.ones(B, N, dtype=torch.bool, device=device)

    result = model.compute_loss(
        node_feats, adjcs, mask,
        loss_edge_weight=loss_edge_weight,
        co_expr=co_expr,
    )
    opt.zero_grad()
    result["loss"].backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {k: (v.item() if isinstance(v, Tensor) else v)
            for k, v in result.items()}


# ── epoch loop ──────────────────────────────────────────────────────────────

def train(
    model:            AnyGraphDiff,
    opt:              torch.optim.Optimizer,
    loader,
    dataset,
    device:           torch.device,
    epochs:           int   = 100,
    loss_edge_weight: float = 1.0,
    sample_every:     int   = 2,
    on_sample=None,
    coexpr_sparse=None,
    tissue_ids_from:  Optional[str] = None,
    permute:          bool = True,
):
    """
    Run `epochs` passes over `loader`.

    `on_sample(model, epoch)` is called every `sample_every` epochs (pass
    e.g. a closure around `draw_sampled_graphs`); set it to None to skip
    generation during training.

    `permute` re-labels each batch with a random node permutation. Leave it
    on for the diffusion models — they are trained not to depend on node
    order. Turn it OFF for GraphVAE: its reconstruction loss compares slot i
    to slot i, so a fresh permutation every batch makes the target
    unlearnable beyond the per-slot marginal.

    Co-expression conditioning: pass the sparse (n_tissues, V, V) graph as
    `coexpr_sparse` and name which slot of the loader's 5-tuple carries the
    per-sample tissue id via `tissue_ids_from` ("groups" or "centers").
    Both must be given, and the model must have been built with
    cfg.use_coexpr=True; otherwise training runs unconditionally.
    """
    slot = {"groups": 3, "centers": 4}
    if tissue_ids_from is not None and tissue_ids_from not in slot:
        raise ValueError(f"tissue_ids_from must be one of {sorted(slot)}")
    condition = coexpr_sparse is not None and tissue_ids_from is not None
    losses, total_loss = [], []

    for epoch in range(epochs):
        run_loss = 0
        loss = None
        for step, batch in enumerate(loader):
            adjcs, node_feats, mask = batch[0], batch[1], batch[2]
            adjcs      = adjcs.to(device).long()
            node_feats = node_feats.to(device)
            mask       = mask.to(device).bool()

            co_expr = None
            if condition:
                co_expr = {
                    "coexpr_sparse": coexpr_sparse,
                    "tissue_ids":    batch[slot[tissue_ids_from]].to(device).long(),
                }

            if permute:
                iso_perm = batched_randperm(
                    node_feats.shape[0], node_feats.shape[1], node_feats.device
                )
                node_feats, adjcs = reorder_graph(node_feats, adjcs, iso_perm)
                # The mask has to move with the data it describes. Without
                # this line the mask still refers to the ORIGINAL slots, and
                # at this dataset's fill rate that misaligns ~3% of positions
                # — which makes the padding token ~3% of all node targets
                # against 0.004% for a uniform one, over-representing it
                # ~800x, so the loss-minimising prediction at every slot
                # becomes PAD.
                mask = mask.gather(1, iso_perm)

            loss = train_step(
                model, opt, node_feats, adjcs,
                mask_token_id=dataset.vocab_size,
                vocab_size=dataset.vocab_size + 1,
                mask=mask,
                loss_edge_weight=loss_edge_weight,
                co_expr=co_expr,
            )
            losses.append(loss)
            run_loss += loss["loss"]

        total_loss.append(run_loss / len(loader))
        # Each variant reports its own breakdown (the continuous model splits
        # the node term into an embedding MSE and a token CE), so render
        # whatever keys came back rather than a fixed pair.
        parts = " | ".join(f"{k[len('loss_'):]} {v:.3f}"
                           for k, v in loss.items() if k != "loss")
        print(f"Epoch: {epoch:3d} , Loss: {total_loss[-1]:.4f}  ({parts})")

        if on_sample is not None and epoch % sample_every == 0:
            print("\nSampling…")
            on_sample(model, epoch)

    return losses, total_loss
