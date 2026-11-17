"""Shared scaffolding for model-quality benchmarks.

Every benchmark scores a batch of generated graphs — the `(node_toks, adj)`
pair that `model.sample()` returns — and reports per-graph values plus a
summary. Scoring the training graphs through the same benchmark gives the
reference the generated numbers should be read against.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


@dataclass
class BenchmarkResult:
    name:     str
    values:   Optional[Tensor]        # (n_scored, ...) per-graph scores
    summary:  dict                    # scalar aggregates
    n_graphs: int = 0                 # graphs handed in
    n_scored: int = 0                 # graphs the metric could actually score
    notes:    list = field(default_factory=list)
    per_graph: dict = field(default_factory=dict)   # {metric: (n_scored,) tensor}
    # pooled samples that are not one-per-graph — a degree distribution has
    # one entry per NODE — plus any compositional breakdown
    distributions: dict = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Fraction of graphs the metric applied to. A high score over a
        handful of graphs is not the same result as one over all of them."""
        return self.n_scored / self.n_graphs if self.n_graphs else 0.0

    def __str__(self) -> str:
        lines = [f"{self.name}  ({self.n_scored}/{self.n_graphs} graphs scored, "
                 f"coverage {self.coverage:.0%})"]
        for k, v in self.summary.items():
            lines.append(f"  {k:<28} {v:.4f}" if isinstance(v, float)
                         else f"  {k:<28} {v}")
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines)


class Benchmark(ABC):
    """Subclass and implement `evaluate`."""

    name: str = "benchmark"

    @abstractmethod
    def evaluate(self, node_toks: Tensor, adjacency: Optional[Tensor] = None,
                 **kwargs) -> BenchmarkResult:
        """
        node_toks : (B, N) long — decoded node tokens
        adjacency : (B, N, N) optional — 0/1 adjacency, for edge-aware metrics
        """

    @torch.no_grad()
    def evaluate_model(self, model, n_graphs: int = 32, N: int = 64,
                       device=None, **sample_kwargs) -> BenchmarkResult:
        """Sample from a model and score the result."""
        device = device or next(model.parameters()).device
        was_training = model.training
        model.eval()
        node_toks, adj = model.sample(N=N, B=n_graphs, device=device, **sample_kwargs)
        if was_training:
            model.train()
        return self.evaluate(node_toks, adj)

    def evaluate_dataset(self, dataset, n_graphs: int = 32,
                         generator: Optional[torch.Generator] = None,
                         pad_id: int = 0) -> BenchmarkResult:
        """
        Score real pathways — the reference the generated numbers should be
        read against. Expects `dataset[i]` to yield
        `(adjacency, node_feats, mask, ...)` like the training loader.

        Padding is zeroed rather than compacted, so nodes and adjacency stay
        index-aligned and edge-aware benchmarks can use both.
        """
        n = min(n_graphs, len(dataset))
        idx = torch.randperm(len(dataset), generator=generator)[:n]
        nodes, adjs = [], []
        for i in idx.tolist():
            adj, node_feats, mask = dataset[i][:3]
            node_feats = torch.as_tensor(node_feats).long().clone()
            mask = torch.as_tensor(mask).bool()
            adj = torch.as_tensor(adj).long().clone()

            node_feats[~mask] = pad_id
            adj[~mask, :] = 0
            adj[:, ~mask] = 0
            nodes.append(node_feats)
            adjs.append(adj)

        res = self.evaluate(torch.stack(nodes), torch.stack(adjs))
        res.name += "  [training data]"
        return res
