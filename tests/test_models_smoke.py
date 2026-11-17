"""Every arm builds, trains one step and samples, on CPU with a toy config.

These are the checks the restructure was verified against: the model and
sampler interfaces agree, losses are finite, samples are symmetric and
self-loop free, and node-type registration works where it is supported.
"""

import pytest
import torch

from graphpath.config import Config, JointConfig, MarginalConfig
from graphpath.models import ARMS, build_model
from graphpath.processes.marginal import _mirror_upper

V, N, B = 12, 9, 3
SMALL = dict(T=8, n_layers=2, d_model=32, n_heads=2, edge_n_heads=2, d_ff=64, dropout=0.0)


def _cfg(arm, **kw):
    if arm == "transition":
        return JointConfig(**SMALL, **kw)
    if arm == "marginal":
        return MarginalConfig(**SMALL, **kw)
    return Config(**SMALL, rand_feat_dim=8, **kw)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    nf = torch.randint(1, V, (B, N), generator=g)
    aj = _mirror_upper((torch.rand(B, N, N, generator=g) < 0.2).long())
    aj = aj.masked_fill(torch.eye(N, dtype=torch.bool)[None], 0)
    pm = torch.ones(B, N, dtype=torch.bool); pm[2, 7:] = False
    nf[~pm] = 0
    return nf, aj, pm


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_loss_backward_and_sample(arm):
    torch.manual_seed(0)
    model = build_model(arm, _cfg(arm), vocab_size=V + 1)
    if arm in ("transition", "marginal"):
        model.set_marginals(torch.full((V,), 1.0 / V), torch.tensor([0.9, 0.1]))
    nf, aj, pm = _batch()
    out = model.compute_loss(nf, aj, pm)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    model.eval()
    nodes, adj = model.sample(N=N, B=2)
    assert nodes.shape == (2, N) and adj.shape == (2, N, N)
    assert torch.equal(adj, adj.transpose(1, 2)), "adjacency not symmetric"
    assert adj.diagonal(dim1=1, dim2=2).sum() == 0, "self-loops present"
    assert set(adj.unique().tolist()) <= {0, 1}


@pytest.mark.parametrize("arm", ["masked", "uniform", "continuous", "vae", "transition"])
def test_node_types(arm):
    model = build_model(arm, _cfg(arm, n_node_types=4), vocab_size=V + 1)
    if arm == "transition":
        model.set_marginals(torch.full((V,), 1.0 / V), torch.tensor([0.9, 0.1]))
    model.set_node_types(torch.arange(V) % 4)
    nf, aj, pm = _batch()
    assert torch.isfinite(model.compute_loss(nf, aj, pm)["loss"])


def test_state_dict_roundtrip():
    """Weights saved from one instance load into a fresh one of every arm."""
    for arm in sorted(ARMS):
        a = build_model(arm, _cfg(arm), vocab_size=V + 1)
        b = build_model(arm, _cfg(arm), vocab_size=V + 1)
        b.load_state_dict(a.state_dict())
        for (k, x), (_, y) in zip(a.state_dict().items(), b.state_dict().items()):
            assert torch.equal(x, y), (arm, k)
