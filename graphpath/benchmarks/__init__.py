"""Model-quality benchmarks.

    from graphpath.benchmarks import CoExpressionBenchmark

    bench = CoExpressionBenchmark(dataset, expression_loader=loader, device=device)
    print(bench.evaluate_model(model, n_graphs=32, N=64))
    print(bench.evaluate_dataset(dataset, n_graphs=32))   # the reference

Figures live in `graphpath.plotting`; node-type helpers in
`graphpath.data.node_types`.
"""

from graphpath.benchmarks.attention import (attention_entropy, capture_attention, collect,
                                            edge_alignment, type_flow)
from graphpath.benchmarks.base import Benchmark, BenchmarkResult
from graphpath.benchmarks.baselines import compare, random_like, shuffled_like
from graphpath.benchmarks.coexpression import CoExpressionBenchmark
from graphpath.benchmarks.conditioning_eval import assign_tissues, recovery_table, tissue_recovery
from graphpath.benchmarks.discovery import DiscoveryFinder
from graphpath.benchmarks.flux import FluxValidityBenchmark, guidance_residual
from graphpath.benchmarks.known_structure import KnownStructureBenchmark
from graphpath.benchmarks.nearest_training import NearestTrainingBenchmark
from graphpath.benchmarks.novel_edges import NovelEdgeBenchmark, novel_edge_table
from graphpath.benchmarks.perturbation import PerturbationBenchmark, perturbation_table
from graphpath.benchmarks.reference_network import ReferenceNetwork
from graphpath.benchmarks.topology import (TopologyBenchmark, composition_table,
                                           topology_distances, topology_table)

__all__ = [
    "Benchmark", "BenchmarkResult", "attention_entropy", "capture_attention", "collect",
    "edge_alignment", "type_flow", "compare", "random_like", "shuffled_like",
    "CoExpressionBenchmark", "assign_tissues", "recovery_table", "tissue_recovery",
    "DiscoveryFinder", "FluxValidityBenchmark", "guidance_residual", "KnownStructureBenchmark",
    "NearestTrainingBenchmark", "NovelEdgeBenchmark", "novel_edge_table",
    "PerturbationBenchmark", "perturbation_table", "ReferenceNetwork", "TopologyBenchmark",
    "composition_table", "topology_distances", "topology_table",
]
