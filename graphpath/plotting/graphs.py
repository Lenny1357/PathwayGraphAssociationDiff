# Plot graph samples
from time import time
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers
import matplotlib.transforms as mtransforms
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.legend_handler import HandlerPatch
import networkx as nx

def create_graph_objs(samples, connectivities, pathway_dataset, get_common_name, get_node_type, subgraphs=None):
    graphs = []
    node_names_list = []
    labels_list = []
    node_type_list = []
    id_maps = []

    for index in range(len(samples)):

        # --- original names and adjacency ---
        names = [
            pathway_dataset.id_to_node[w.item()] if w.item() != 0 else "<PAD>"
            for w in samples[index]
        ]
        adj = connectivities[index].detach().cpu().numpy()

        threshold = 0.99
        N = len(names)

        # --- collapse duplicate nodes ---
        name_to_new_idx = {}
        old_to_new = {}
        new_names = []

        for old_idx, name in enumerate(names):
            if name not in name_to_new_idx:
                new_idx = len(name_to_new_idx)
                name_to_new_idx[name] = new_idx
                new_names.append(name)
            old_to_new[old_idx] = name_to_new_idx[name]

        if subgraphs is not None:
            subgraph = [(x[0].item(), x[1].item()) for x in subgraphs[index].T]

        # --- build graph on unique nodes ---
        G = nx.Graph()
        G.add_nodes_from(range(len(new_names)))

        # --- build edges (skip duplicates automatically) ---
        for i in range(N):
            for j in range(i + 1, N):
                ni = old_to_new[i]
                nj = old_to_new[j]

                if subgraphs is not None and ((i,j) in subgraph or (j,i) in subgraph):
                    G.add_edge(ni, nj, weight=3)

                val = float(max(adj[i, j], adj[j, i]))
                if val <= threshold:
                    continue

                # skip self-loops caused by duplicates
                if ni == nj:
                    continue

                # keep strongest edge if duplicates created multiple
                if G.has_edge(ni, nj):
                    if val > G[ni][nj]["weight"]:
                        G[ni][nj]["weight"] = val
                else:
                    G.add_edge(ni, nj, weight=val)

        # --- remove isolated nodes ---
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)

        # --- re-center node indices ---
        old_nodes = list(G.nodes())
        relabel_map = {old_i: new_i for new_i, old_i in enumerate(old_nodes)}
        G = nx.relabel_nodes(G, relabel_map, copy=True)

        # --- rebuild node metadata ---
        label_map = {
            new_i: get_common_name(new_names[old_i])
            for old_i, new_i in relabel_map.items()
        }

        node_type_map = {
            new_i: get_node_type(new_names[old_i])
            for old_i, new_i in relabel_map.items()
        }

        id_map = {
            new_i: new_names[old_i]
            for old_i, new_i in relabel_map.items()
        }

        node_names = [label_map[i] for i in range(len(label_map))]

        # --- store ---
        graphs.append(G)
        node_names_list.append(node_names)
        labels_list.append(label_map)
        node_type_list.append(node_type_map)
        id_maps.append(id_map)

        # --- sanity checks ---
        assert set(G.nodes()) == set(range(G.number_of_nodes()))
        assert len(node_names) == G.number_of_nodes()

    return graphs, node_names_list, labels_list, node_type_list, id_maps

def _marker_legend_patch(marker, hatch, facecolor='white', edgecolor='black'):
    """Build a legend Patch that carries the given marker's shape path with an optional hatch pattern applied."""
    ms = mmarkers.MarkerStyle(marker)
    path = ms.get_path().transformed(ms.get_transform())
    patch = mpatches.PathPatch(
        path,
        facecolor=facecolor,
        edgecolor=edgecolor,
        hatch=hatch if hatch else None,
        linewidth=0.8,
    )
    return patch


class HandlerMarkerShape(HandlerPatch):
    """Legend handler that draws the wrapped PathPatch's real shape,
    scaled/centered to fit the legend key box, instead of matplotlib's
    default behavior of substituting a plain rectangle."""

    def create_artists(self, legend, orig_handle,
                        xdescent, ydescent, width, height, fontsize, trans):
        path = orig_handle.get_path()
        bbox = path.get_extents()

        w = bbox.width if bbox.width > 0 else 1.0
        h = bbox.height if bbox.height > 0 else 1.0

        scale = min(width / w, height / h) * 0.9
        tx = xdescent + width / 2 - (bbox.x0 + bbox.width / 2) * scale
        ty = ydescent + height / 2 - (bbox.y0 + bbox.height / 2) * scale

        p = mpatches.PathPatch(
            path,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            hatch=orig_handle.get_hatch(),
            linewidth=orig_handle.get_linewidth(),
        )
        t = mtransforms.Affine2D().scale(scale, scale).translate(tx, ty) + trans
        p.set_transform(t)
        return [p]


def _confidence_colormap(base_hex, low_color=(1.0, 1.0, 1.0)):
    """Build a LinearSegmentedColormap that goes from `low_color` (white by
    default, at confidence=0) to the full base color (at confidence=1).
    A straight RGB gradient reads much more clearly than a saturation/value
    tweak, since the eye picks up "how pale vs. vivid" very easily."""
    base_rgb = mcolors.to_rgb(base_hex)
    return mcolors.LinearSegmentedColormap.from_list(
        f"conf_{base_hex}", [low_color, base_rgb], N=256
    )


def _apply_confidence_to_color(base_hex, confidence, cmap_cache):
    """Map a confidence value in [0, 1] to an RGB color along the
    white -> base_hex gradient for this base color."""
    confidence = float(max(0.0, min(1.0, confidence)))
    if base_hex not in cmap_cache:
        cmap_cache[base_hex] = _confidence_colormap(base_hex)
    return cmap_cache[base_hex](confidence)


def plot_graph_samples(graphs, node_type_list, labels_list,
                       titles=None, suptitle=None, pos_list=None, verbose=True):
    """
    titles: optional list of per-panel titles (e.g. denoising timesteps);
        defaults to "Graph {i}".
    suptitle: optional figure-level title.
    pos_list: optional list of {node: (x, y)} layouts, one per panel. Use it
        to pin the same entity to the same place across panels (e.g. when the
        panels are stages of one denoising trajectory); defaults to an
        independent spring layout per panel.
    verbose: print per-panel edge counts.

    node_confidences: optional tensor/array-like of shape (num_graphs, num_nodes),
        values in [0, 1] indicating how confident we are about each node.
        Confidence is reflected as a color gradient (white = low confidence,
        fully saturated change/no_change color = high confidence), in the
        same node order as node_change_masks.
    """
    num_plots = min(12, len(graphs))
    nrows, ncols = 3, 4
    figsize = (16, 10)

    # -----------------------------
    # Define shape + hatch pattern per node type
    # -----------------------------
    type_shapes = {
        'gene': 'o',          # circle
        'metabolite': 's',    # square
        'enzyme': '^',        # triangle up
        'reaction': 'D',      # diamond
        'mask': 'X',          # X marker
    }
    type_hatches = {
        'gene': '',           # solid / no hatch
        'metabolite': '//',   # diagonal lines
        'enzyme': 'xx',       # crosshatch
        'reaction': '..',     # dots
        'mask': '++',         # plus pattern
    }
    default_shape = 'o'
    default_hatch = 'oo'

    node_change_colors = {
        'change': "#46aa27",    # green
        'no_change': "#f06711", # orange
    }

    type_colors = {
    'gene': "#b48c1f",
    'metabolite': "#0effdb",
    'enzyme': "#2c39a0",
    'reaction': "#d627b9"
    }

    cmap_cache = {}  # base_hex -> LinearSegmentedColormap, built once and reused

    # -----------------------------
    # Create figure
    # -----------------------------
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    # -----------------------------
    # Plot each graph
    # -----------------------------
    for i in range(num_plots):
        G = graphs[i]
        ax = axes[i]

        panel_title = titles[i] if titles is not None else f"Graph {i}"

        if G.number_of_nodes() == 0:
            ax.set_title(f"{panel_title} (empty)")
            ax.axis("off")
            continue

        # Layout
        pos = pos_list[i] if pos_list is not None else nx.spring_layout(G, seed=42)

        # Node types for this graph
        node_type_map = node_type_list[i]  # dict {node_idx: node_type}

        colors = [type_colors.get(n, '#7f7f7f')
        for n in node_type_list[i]]

        # -----------------------------
        # Draw nodes grouped by type: shape + hatch pattern per group
        # -----------------------------
        nodes_by_type = {}
        for n in G.nodes():
            ntype = node_type_map[n]
            nodes_by_type.setdefault(ntype, []).append(n)

        for ntype, node_list in nodes_by_type.items():
            shape = type_shapes.get(ntype, default_shape)
            hatch = type_hatches.get(ntype, default_hatch)
            color = type_colors.get(ntype, '#7f7f7f')

            collection = nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=node_list,
                node_shape=shape,
                node_color=color,
                node_size=160,
                edgecolors='black',
                linewidths=0.8,
                ax=ax
            )
            if collection is not None and hatch:
                collection.set_hatch(hatch)

        # -----------------------------
        # Separate edges by weight
        # -----------------------------
        normal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 1) == 1]
        masked_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 1) == 2]
        known_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 1) == 3]
        if verbose:
            print(f"{panel_title}: {len(normal_edges)} normal, "
                  f"{len(masked_edges)} masked, {len(known_edges)} known edges")

        # Draw normal edges (weight = 1)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=normal_edges,
            width=0.8,
            alpha=0.7,
            ax=ax
        )

        # Draw masked edges (weight = 2) as thin dotted lines
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=masked_edges,
            width=0.1,
            style="dotted",
            alpha=0.7,
            ax=ax
        )

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=known_edges,
            edge_color='green',
            width=3,
            alpha=1,
            ax=ax
        )

        # Draw labels if provided
        labels = labels_list[i]
        if labels is not None and False:
            nx.draw_networkx_labels(
                G,
                pos,
                labels=labels,
                font_size=6,
                ax=ax
            )

        ax.set_title(panel_title, fontsize=10)
        ax.axis("off")

    # -----------------------------
    # Hide unused subplots
    # -----------------------------
    for j in range(num_plots, nrows * ncols):
        axes[j].axis("off")

    # -----------------------------
    # Legends: node type (true shape + hatch) + change status (color)
    # -----------------------------
    shape_legend_handles = [
        _marker_legend_patch(shape, type_hatches.get(ntype, default_hatch), facecolor=type_colors.get(ntype, '#7f7f7f'), edgecolor='black')
        for ntype, shape in type_shapes.items()
    ]
    shape_legend_labels = list(type_shapes.keys())

    color_legend_handles = [
        mlines.Line2D([], [], color=c, marker='o', linestyle='None',
                      markersize=8, label=label)
        for label, c in node_change_colors.items()
    ]

    fig.legend(
        handles=shape_legend_handles + color_legend_handles,
        labels=shape_legend_labels + list(node_change_colors.keys()),
        handler_map={mpatches.PathPatch: HandlerMarkerShape()},
        loc='upper right',
        fontsize=14,
        title="Node type (shape+pattern) / change (color)"
    )


    if suptitle is not None:
# centred over the panel area only — the legend occupies the right 28%
        fig.suptitle(suptitle, fontsize=13, x=0.36, y=0.995)

    plt.tight_layout(rect=[0, 0, 0.72, 0.97 if suptitle else 1])
    plt.show()
    return fig


def draw_sampled_graphs(
    model,
    n_samples=6, N=10, temperature=1.0,
    save_path="sampled_graphs.png", seed=42, label_fn=None,
    **sample_kwargs,
) -> str:
    """Sample `n_samples` graphs from any model with `.sample()` and draw them
    in a grid. `label_fn(token) -> str` names the nodes; default is the id."""
    import math
    namer = label_fn or (lambda tok: str(int(tok)))

    torch.manual_seed(seed)
    device = next(model.parameters()).device
    model.eval()

    node_toks, adj_mats = model.sample(
        N=N, B=n_samples, temperature=temperature, device=device, **sample_kwargs
    )
    node_toks = node_toks.cpu().numpy()
    # collapse any K>2 edge typing down to presence for the drawing
    adj_mats  = (adj_mats.cpu().numpy() > 0).astype(int)

    # the density the model should be reproducing, where it knows it
    m_edge = getattr(model, "m_edge", None)
    m_edge_pos = 1.0 - m_edge[0].item() if m_edge is not None else float("nan")

    ncols = 3
    nrows = math.ceil(n_samples / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             facecolor="#DED9C7")
    axes = np.array(axes).flatten()

    for i in range(n_samples):
        ax = axes[i]
        ax.set_facecolor("#F1EFE8")
        G   = nx.from_numpy_array(adj_mats[i])
        pos = nx.spring_layout(G, seed=seed + i, k=1.8 / math.sqrt(max(N, 2)))
        labels = {n: namer(node_toks[i][n]) for n in G.nodes()}
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#73726c",
                               width=1.4, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=540,
                               linewidths=0.8, edgecolors="#ffffff")
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                                font_size=10, font_color="#ffffff",
                                font_weight="bold")
        ne = G.number_of_edges()
        ax.set_title(
            f"Sample {i+1}  ·  {N}N {ne}E  "
            f"deg {2*ne/N:.1f}  ρ={2*ne/(N*(N-1)):.3f}  (m={m_edge_pos:.3f})",
            fontsize=9, color="#3d3d3a", pad=5)
        ax.axis("off")

    for j in range(n_samples, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{type(model).__name__} samples",
                 fontsize=12, color="#3d3d3a", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {save_path}")
    return save_path
