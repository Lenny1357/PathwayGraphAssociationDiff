"""Datasets, loaders and the evidence signals (expression, propagation).

Lazy on purpose: data-building scripts import `graphpath.data.node_types`
without needing torch.
"""

import importlib

_LAZY = {
    "PathwayGraphDataset": "graphpath.data.dataset",
    "ExpressionLoader": "graphpath.data.expression",
    "STAT_COLUMNS": "graphpath.data.expression",
    "DEFAULT_TISSUES": "graphpath.data.expression",
    "EcoliExpressionLoader": "graphpath.data.expression_ecoli",
    "CoExpressionPropagator": "graphpath.data.propagation",
    "PropagatedExpression": "graphpath.data.propagation",
    "propagate_all_tissues": "graphpath.data.propagation",
    "load_dataset": "graphpath.data.loaders", "load_graphs": "graphpath.data.loaders",
    "split_graphs": "graphpath.data.loaders", "make_loader": "graphpath.data.loaders",
    "available_datasets": "graphpath.data.loaders", "token_to_type": "graphpath.data.loaders",
    "REGISTRY": "graphpath.data.loaders",
    "TYPE_MAP": "graphpath.data.node_types", "TYPE_NAMES": "graphpath.data.node_types",
    "default_node_type": "graphpath.data.node_types", "human_node_type": "graphpath.data.node_types",
    "ecoli_node_type": "graphpath.data.node_types", "kegg_node_type": "graphpath.data.node_types",
    "node_type_fn": "graphpath.data.node_types", "load_name_map": "graphpath.data.node_types",
    "make_common_name": "graphpath.data.node_types",
}
__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'graphpath.data' has no attribute {name!r}")
