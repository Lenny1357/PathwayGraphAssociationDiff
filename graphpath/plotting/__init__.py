"""Figures: sampled graphs, benchmark panels, network maps, attention."""

from graphpath.plotting.attention import (attention_edges_figure, attention_maps_figure,
                                          type_attention_figure)
from graphpath.plotting.benchmark_figures import (coexpression_figure, conditioning_figure,
                                                  guidance_sweep_figure, known_structure_figure,
                                                  memorisation_figure, tissue_heatmap,
                                                  topology_figure)
from graphpath.plotting.discovery import discovery_context_figure, discovery_figure
from graphpath.plotting.graphs import create_graph_objs, draw_sampled_graphs, plot_graph_samples
from graphpath.plotting.network_maps import (_entity_table, coexpression_network_figure,
                                             embed_graph, embed_metabolic_network,
                                             embed_model_space, embed_tissue_network,
                                             metabolic_network_map, model_embedding_map,
                                             propagated_overlay_figure, tissue_network_map)
from graphpath.plotting.style import (EDGE_STYLE, PALETTE, TYPE_COLORS, TYPE_HATCHES,
                                      TYPE_SHAPES, _color, _rc)

__all__ = [
    "attention_edges_figure", "attention_maps_figure", "type_attention_figure",
    "coexpression_figure", "conditioning_figure", "guidance_sweep_figure",
    "known_structure_figure", "memorisation_figure", "tissue_heatmap", "topology_figure",
    "discovery_context_figure", "discovery_figure", "create_graph_objs", "draw_sampled_graphs",
    "plot_graph_samples", "coexpression_network_figure", "embed_graph",
    "embed_metabolic_network", "embed_model_space", "embed_tissue_network",
    "metabolic_network_map", "model_embedding_map", "propagated_overlay_figure",
    "tissue_network_map", "EDGE_STYLE", "PALETTE", "TYPE_COLORS", "TYPE_HATCHES", "TYPE_SHAPES",
    "_color", "_rc", "_entity_table",
]
