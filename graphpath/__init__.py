"""graphpath — generative graph diffusion for biological pathways.

Layout (one concern per subpackage):

    config      Config / MarginalConfig / JointConfig — every hyper-parameter
    layers      attention blocks, towers, embeddings, typed connectivity,
                conditioning modules
    processes   forward corruption and reverse posterior for each noise
                process (absorbing, uniform, marginal, gaussian)
    models      network + training objective for each variant
    samplers    reverse-time generation, one function per process
    data        datasets, loaders, expression / perturbation signals
    benchmarks  model-quality metrics, each scored against controls
    plotting    sample drawings, benchmark figures, network maps
    training    train step / loop, class weights, run directories, devices
    fba         batched, differentiable stoichiometry checks
    paths       where the data, runs, weights and figures live

The names below resolve lazily, so `import graphpath.data.node_types` from a
data-building script does not pull in torch.
"""

import importlib

__version__ = "0.2.0"

_LAZY = {
    # config
    "Config": "graphpath.config", "MarginalConfig": "graphpath.config",
    "JointConfig": "graphpath.config",
    # models
    "GraphDiffBERT": "graphpath.models.masked",
    "GraphDiffBERTUniform": "graphpath.models.uniform",
    "ContinuousGraphDiffusion": "graphpath.models.continuous",
    "GraphVAE": "graphpath.models.vae",
    "GraphDiffBERTMarginal": "graphpath.models.marginal",
    "GraphDiffBERTJoint": "graphpath.models.joint",
    "build_model": "graphpath.models",
    # processes
    "AbsorbingMaskSchedule": "graphpath.processes.absorbing",
    "UniformTransitionSchedule": "graphpath.processes.uniform",
    "MarginalTransitionSchedule": "graphpath.processes.marginal",
    "ContinuousCosineSchedule": "graphpath.processes.gaussian",
    "estimate_marginals": "graphpath.processes.marginal",
    # layers
    "GraphBlock": "graphpath.layers.graph_block", "GraphTower": "graphpath.layers.graph_block",
    "ContinuousGraphTower": "graphpath.layers.graph_block",
    "GraphTransformerBlock": "graphpath.layers.graph_transformer",
    "SeparateTower": "graphpath.layers.separate_tower",
    "NodeTransformer": "graphpath.layers.separate_tower",
    "AdjacencyTransformer": "graphpath.layers.separate_tower",
    "UnifiedEmbeddingModule": "graphpath.layers.embedding",
    "CoExpressionEncoder": "graphpath.layers.conditioning",
    "GraphConditioner": "graphpath.layers.conditioning",
    "CoexprLookup": "graphpath.layers.pairwise_conditioning",
    "PairwiseEdgeBias": "graphpath.layers.pairwise_conditioning",
    "NeighbourhoodCrossAttention": "graphpath.layers.pairwise_conditioning",
    "NeighbourhoodConditioned": "graphpath.layers.neighbourhood",
    # data
    "PathwayGraphDataset": "graphpath.data.dataset",
    "ExpressionLoader": "graphpath.data.expression",
    "EcoliExpressionLoader": "graphpath.data.expression_ecoli",
    "CoExpressionPropagator": "graphpath.data.propagation",
    "PropagatedExpression": "graphpath.data.propagation",
    "propagate_all_tissues": "graphpath.data.propagation",
    # training
    "train": "graphpath.training.loop", "train_step": "graphpath.training.loop",
    "reorder_graph": "graphpath.training.loop", "batched_randperm": "graphpath.training.loop",
    "pick_device": "graphpath.training.device",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'graphpath' has no attribute {name!r}")


def __dir__():
    return __all__
