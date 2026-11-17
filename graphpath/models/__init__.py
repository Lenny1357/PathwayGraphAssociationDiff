"""The generative models. All share one interface:

    model = <Class>(cfg, vocab_size=dataset.vocab_size + 1)
    model.compute_loss(node_feats, adjcs, pad_mask, loss_edge_weight=1.0) -> dict
    model.sample(N, B, ...) -> (node_tokens, adjacency)

`build_model(name, cfg, vocab_size)` maps the arm names used throughout the
experiments onto the classes.
"""

from graphpath.models.continuous import ContinuousGraphDiffusion
from graphpath.models.joint import GraphDiffBERTJoint
from graphpath.models.marginal import GraphDiffBERTMarginal
from graphpath.models.masked import GraphDiffBERT
from graphpath.models.uniform import GraphDiffBERTUniform
from graphpath.models.vae import GraphVAE

ARMS = {
    "masked":     GraphDiffBERT,            # absorbing-mask discrete diffusion
    "uniform":    GraphDiffBERTUniform,     # uniform transition matrix
    "marginal":   GraphDiffBERTMarginal,    # marginal transition, serial towers
    "transition": GraphDiffBERTJoint,       # marginal transition, joint tower (the main runs)
    "continuous": ContinuousGraphDiffusion, # Gaussian diffusion in embedding space
    "vae":        GraphVAE,                 # one-shot baseline
}


def build_model(name: str, cfg, vocab_size: int, **kwargs):
    """Instantiate an arm by name. `cfg` must be the matching config class
    (`MarginalConfig` / `JointConfig` for the transition arms, `Config` otherwise)."""
    try:
        cls = ARMS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(ARMS)}") from None
    return cls(cfg, vocab_size, **kwargs)


__all__ = ["GraphDiffBERT", "GraphDiffBERTUniform", "GraphDiffBERTMarginal", "GraphDiffBERTJoint",
           "ContinuousGraphDiffusion", "GraphVAE", "ARMS", "build_model"]
