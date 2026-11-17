"""Load a pathway dataset by name and split it the way every experiment did.

    from graphpath.data.loaders import load_dataset, make_loader
    train_ds, test_ds, info = load_dataset("ecoli", max_nodes=64)
    loader = make_loader(train_ds, batch_size=32)

The registry names the pickles each dataset needs. E. coli is built in-tree
by data/ecoli/build_pathways.py; Human-GEM is read from `PT_FILES` (not
reproducible from this repository — see docs/data_sources.md).

The split is over graphs, 90/10 at `seed=0` by default, so the held-out set
is the one every benchmark table was read against. It leaks: chunks come
from one connected network. experiments/splits/ holds the edge-level
alternative.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from graphpath import paths
from graphpath.data.node_types import NODE_TYPE_FN, TYPE_MAP, load_name_map, make_common_name


@dataclass
class DatasetSpec:
    graphs:    Path            # list of {node_ids, adj_matrix, num_nodes, ...}
    node_map:  Path            # {node_id_str: int}
    name_map:  Optional[Path]  # id -> common name CSV
    edges:     Optional[Path]  # reference edge table (ReferenceNetwork)
    expr_dir:  Optional[Path]  # co-expression tables
    node_type: str             # key into NODE_TYPE_FN

    def available(self) -> bool:
        return self.graphs.exists() and self.node_map.exists()


REGISTRY = {
    "ecoli": DatasetSpec(
        graphs=paths.ECOLI_REF / "ecoli_sampled_graphs.pkl",
        node_map=paths.ECOLI_REF / "ecoli_node_id_map.pkl",
        name_map=paths.ECOLI_REF / "node_name_map.csv",
        edges=paths.ECOLI_REF / "ecoli_gem_reaction_node_edges.csv",
        expr_dir=paths.ECOLI_EXPR,
        node_type="ecoli",
    ),
    "human": DatasetSpec(
        graphs=paths.PT_FILES / "human_gem_sampled_graphs.pkl",
        node_map=paths.PT_FILES / "node_id_map.pkl",
        name_map=paths.PT_FILES / "node_name_map.csv",
        edges=paths.PT_FILES / "human_gem_reaction_node_edges2.csv",
        expr_dir=paths.PT_FILES,
        node_type="human",
    ),
    "kegg": DatasetSpec(
        graphs=paths.KEGG_REF / "kegg_sampled_graphs.pkl",
        node_map=paths.KEGG_REF / "node_id_map.pkl",
        name_map=paths.KEGG_REF / "node_name_map.csv",
        edges=paths.KEGG_REF / "kegg_module_edges.csv",
        expr_dir=None,
        node_type="kegg",
    ),
}


def available_datasets() -> list[str]:
    return [k for k, v in REGISTRY.items() if v.available()]


def load_graphs(name: str):
    """-> (graphs, node_id_map) for a registered dataset."""
    spec = REGISTRY[name]
    if not spec.available():
        raise FileNotFoundError(
            f"dataset {name!r}: missing {spec.graphs} or {spec.node_map}. "
            f"E. coli: run data/ecoli/build_pathways.py. Human: set PT_FILES.")
    with open(spec.graphs, "rb") as f:
        graphs = pickle.load(f)
    with open(spec.node_map, "rb") as f:
        node_map = pickle.load(f)
    return graphs, node_map


def split_graphs(graphs, split: float = 0.9, seed: int = 0,
                 n_graphs: Optional[int] = None):
    """Random graph-level split; the project default is 90/10 at seed 0."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(graphs))
    if n_graphs:
        perm = perm[:n_graphs]
    cut = int(len(perm) * split)
    return [graphs[i] for i in perm[:cut]], [graphs[i] for i in perm[cut:]]


def load_dataset(name: str, max_nodes: int = 64, split: float = 0.9, seed: int = 0,
                 n_graphs: Optional[int] = None, device="cpu") -> Tuple[object, object, dict]:
    """
    -> (train_dataset, test_dataset, info). `info` carries what the benchmarks
    ask for: node_type_fn, get_common_name, edges, expr_dir, raw graph lists.
    """
    from graphpath.data.dataset import PathwayGraphDataset   # torch, imported late

    spec = REGISTRY[name]
    graphs, node_map = load_graphs(name)
    train, test = split_graphs(graphs, split, seed, n_graphs)

    def mk(g):
        return PathwayGraphDataset(node_id_map=node_map, graphs=g,
                                   max_nodes=max_nodes, device=device)

    name_map = (load_name_map(str(spec.name_map))
                if spec.name_map and spec.name_map.exists() else None)
    info = dict(
        name=name, spec=spec, node_map=node_map, train_graphs=train, test_graphs=test,
        node_type_fn=NODE_TYPE_FN[spec.node_type],
        get_common_name=make_common_name(name_map),
        edges=str(spec.edges) if spec.edges and spec.edges.exists() else None,
        expr_dir=str(spec.expr_dir) if spec.expr_dir and spec.expr_dir.exists() else None,
    )
    return mk(train), mk(test), info


def make_loader(dataset, batch_size: int = 32, shuffle: bool = True):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def token_to_type(dataset, node_type_fn) -> list:
    """Per-token type ids for `model.set_node_types`, in `TYPE_MAP` order."""
    return [TYPE_MAP[node_type_fn(dataset.id_to_node[i])] for i in range(dataset.vocab_size)]
