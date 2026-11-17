"""Whole-network maps: co-expression, the curated metabolic network, the
model's own embedding space, and propagated co-expression overlays."""

from typing import Optional, Sequence

import numpy as np

from graphpath.plotting.style import (EDGE_STYLE, FALLBACK, GRID, INK, INK_MUTED,
                                      INK_SOFT, PALETTE, SURFACE, TYPE_COLORS,
                                      TYPE_HATCHES, TYPE_SHAPES, _as_np, _bars,
                                      _color, _rc)

from graphpath.data.node_types import default_node_type


# ── co-expression network context ──────────────────────────────────────────

def _incident_edges(sp, genes, device):
    """Co-expression edges touching `genes`, as (u, v, w) tensors."""
    import torch
    sp = sp.coalesce()
    rows, cols = sp.indices()
    vals = sp.values()
    g = genes.to(rows.device)
    m = torch.isin(rows, g) | torch.isin(cols, g)
    return rows[m], cols[m], vals[m]


def coexpression_network_figure(
    expr_loader, gene_sets: dict, tissue: Optional[str] = None,
    max_context: int = 260, max_edges: int = 1400, min_weight: float = 0.0,
    save: Optional[str] = None, seed: int = 0, layout_iter: int = 60,
):
    """
    The co-expression network as a backdrop, with each condition's genes lit
    up on it.

    `gene_sets` maps a panel label -> list of gene-id lists (one per graph).
    Every panel shares ONE layout computed on the union, so position means
    the same thing in each: if the model groups co-expressed genes, its
    genes land together and the shuffled control scatters. That comparison
    is only readable because the coordinates are shared — laying each panel
    out separately would show two unrelated pictures.

    The backdrop is a neighbourhood, not the whole network: every gene shown
    is either highlighted or a strong co-expression partner of one. Drawing
    all V genes would be a grey disc.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import torch

    labels = list(gene_sets)
    if tissue is None:                       # the tissue the first set fits best
        first = torch.tensor(sorted({g for gs in gene_sets[labels[0]] for g in gs}))
        stats = expr_loader.within_list_co_exp(first)
        tissue = expr_loader.tissues[int(stats[:, 3].argmax())]

    sp = expr_loader.co_expression[tissue].coalesce()
    device = sp.device

    highlight = {lbl: sorted({int(g) for gs in gene_sets[lbl] for g in gs})
                 for lbl in labels}
    all_hl = torch.tensor(sorted(set().union(*highlight.values())), device=device)

    # context ring: strongest partners of the highlighted genes
    u, v, w = _incident_edges(sp, all_hl, device)
    if min_weight > 0:
        keep = w.abs() >= min_weight
        u, v, w = u[keep], v[keep], w[keep]
    order = w.abs().argsort(descending=True)
    u, v, w = u[order], v[order], w[order]

    hl_set = set(all_hl.tolist())
    context, kept = [], []
    for i in range(u.numel()):
        a, b = int(u[i]), int(v[i])
        other = b if a in hl_set else a
        if other not in hl_set and len(context) < max_context:
            context.append(other)
        kept.append(i)
        if len(kept) >= max_edges and len(context) >= max_context:
            break
    shown = hl_set | set(context)

    G = nx.Graph()
    G.add_nodes_from(shown)
    for i in kept[:max_edges]:
        a, b = int(u[i]), int(v[i])
        if a in shown and b in shown and a != b:
            G.add_edge(a, b, weight=float(abs(w[i])))

    pos = nx.spring_layout(G, seed=seed, iterations=layout_iter,
                           weight="weight", k=1.6 / max(len(G) ** 0.5, 1))

    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, len(labels),
                                 figsize=(4.3 * len(labels), 4.4), squeeze=False)
        for ax, lbl, i in zip(axes[0], labels, range(len(labels))):
            col = _color(lbl, i)
            hl = [n for n in G.nodes() if n in set(highlight[lbl])]
            ctx = [n for n in G.nodes() if n not in set(highlight[lbl])]

            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GRID,
                                   width=0.4, alpha=0.7)
            # edges between two genes the model put in the same pathway
            within = [(a, b) for a, b in G.edges()
                      if any(a in set(gs) and b in set(gs) for gs in gene_sets[lbl])]
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=within,
                                   edge_color=col, width=1.6, alpha=0.85)
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=ctx, node_size=7,
                                   node_color=INK_MUTED, alpha=0.30, linewidths=0)
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=hl, node_size=42,
                                   node_color=col, edgecolors=SURFACE, linewidths=0.9)

            ax.set_title(f"{lbl}   ({len(hl)} genes, {len(within)} co-expressed pairs)",
                         loc="left", pad=6, fontsize=9.5)
            ax.set_axis_off()

        fig.suptitle(f"Generated genes on the {tissue} co-expression network",
                     x=0.02, ha="left", fontsize=11, color=INK)
        fig.text(0.02, 0.015,
                 f"Grey: {len(context)} neighbouring genes giving context. "
                 f"Coloured: genes the model placed in a pathway; thick edges join "
                 f"two of them. Shared layout — position is comparable across panels.",
                 fontsize=8, color=INK_MUTED, ha="left")
        fig.tight_layout(rect=[0, 0.05, 1, 0.94])
        if save:
            fig.savefig(save, dpi=300)
        return fig, tissue


# ── whole-network map ──────────────────────────────────────────────────────

def embed_tissue_network(expr_loader, tissue: str, n_components: int = 32,
                         method: str = "umap", seed: int = 0, verbose: bool = True):
    """
    A 2-D position for EVERY gene in a tissue's co-expression network.

    Truncated SVD over the full sparse adjacency, then a non-linear
    projection of those components to the plane. Both stages consume every
    edge, so unlike a drawn subgraph this really is the whole network — the
    layout just cannot show all of it as lines, so the map shows genes as a
    point cloud and lets position carry the structure.

    Returns (gene_ids, xy) — feed both to `tissue_network_map` so the
    embedding is computed once and reused across figures.
    """
    import numpy as np
    import scipy.sparse as ssp
    from scipy.sparse.linalg import svds

    sp = expr_loader.co_expression[tissue].coalesce()
    return _embed_sparse(sp, expr_loader.vocab_size, n_components, method,
                         seed, verbose, label=tissue)


def _embed_sparse(sp, V: int, n_components: int = 32, method: str = "umap",
                  seed: int = 0, verbose: bool = True, label: str = "network"):
    """Positions for every non-isolated node of a sparse adjacency."""
    import numpy as np
    import scipy.sparse as ssp
    from scipy.sparse.linalg import svds

    sp = sp.coalesce()
    idx = sp.indices().cpu().numpy()
    val = np.abs(sp.values().cpu().numpy().astype(np.float64))

    A = ssp.coo_matrix((val, (idx[0], idx[1])), shape=(V, V)).tocsr()
    A = A.maximum(A.T)                                   # symmetric
    deg = np.asarray(A.sum(1)).ravel()
    genes = np.flatnonzero(deg > 0)                      # isolated nodes have no position
    A = A[genes][:, genes]
    if verbose:
        print(f"{label}: {len(genes)} connected nodes, {A.nnz // 2} edges")

    k = min(n_components, min(A.shape) - 1)
    # a fixed starting vector rather than random_state: this scipy build
    # rejects a RandomState there, and v0 gives determinism either way
    U, S, _ = svds(A.astype(np.float64), k=k,
                   v0=np.full(A.shape[0], 1.0 / np.sqrt(A.shape[0])))
    X = U[:, ::-1] * S[::-1]                             # strongest components first

    if method == "umap":
        try:
            import umap
            xy = umap.UMAP(n_neighbors=15, min_dist=0.12, random_state=seed,
                           verbose=False).fit_transform(X)
        except Exception:
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        xy = TSNE(n_components=2, init="pca", random_state=seed,
                  perplexity=min(30, max(5, len(genes) // 4))).fit_transform(X)
    elif method == "svd":
        xy = X[:, :2]
    return genes, np.asarray(xy)


def tissue_network_map(expr_loader, gene_sets: dict, tissue: Optional[str] = None,
                       embedding=None, save: Optional[str] = None, seed: int = 0,
                       link_pairs: bool = True, method: str = "umap"):
    """
    The whole tissue co-expression network as a map, with each condition's
    genes marked on it.

    Every gene in the network is a grey point, positioned by its
    co-expression with every other gene; genes that are co-expressed sit
    near each other, so the cloud's structure IS the network's module
    structure. Each panel then marks where one condition's genes fall.

    The question the figure answers: do the model's genes land inside
    modules the way real pathway genes do, or spread across the map the way
    randomly drawn genes do? Panels share the map, so that is a direct
    visual comparison.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    labels = list(gene_sets)
    if tissue is None:
        first = torch.tensor(sorted({g for gs in gene_sets[labels[0]] for g in gs}))
        tissue = expr_loader.tissues[int(expr_loader.within_list_co_exp(first)[:, 3].argmax())]

    if embedding is None:
        embedding = embed_tissue_network(expr_loader, tissue, method=method, seed=seed)
    genes, xy = embedding
    pos_of = {int(g): i for i, g in enumerate(genes)}

    def hit_rate(sets):
        """Share of within-pathway gene pairs that are co-expressed at all —
        the same quantity the benchmark reports, printed on the panel so the
        picture and the number cannot drift apart."""
        hits = tot = 0
        for gs in sets:
            g = torch.tensor(sorted(set(gs)), device=expr_loader.device)
            if g.numel() < 2:
                continue
            p = torch.combinations(g, r=2)
            v = expr_loader._lookup(tissue, p[:, 0], p[:, 1])
            hits += int((v != 0).sum()); tot += v.numel()
        return hits / tot if tot else float("nan")

    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, len(labels), figsize=(4.1 * len(labels), 4.3),
                                 squeeze=False, sharex=True, sharey=True)
        for ax, lbl, i in zip(axes[0], labels, range(len(labels))):
            col = _color(lbl, i)
            # the backdrop has to be legible enough to read module structure,
            # while staying clearly subordinate to the marked genes
            ax.scatter(xy[:, 0], xy[:, 1], s=4.0, c="#b9b8b1", alpha=0.45,
                       linewidths=0, zorder=1)

            pts, n_hl = [], 0
            for gs in gene_sets[lbl]:
                ids = [pos_of[int(g)] for g in set(gs) if int(g) in pos_of]
                n_hl += len(ids)
                pts.extend(ids)
                if link_pairs and len(ids) > 1:
                    g_t = torch.tensor(sorted({int(g) for g in gs if int(g) in pos_of}),
                                       device=expr_loader.device)
                    p = torch.combinations(g_t, r=2)
                    v = expr_loader._lookup(tissue, p[:, 0], p[:, 1]).cpu()
                    for (a, b), vv in zip(p.cpu().tolist(), v.tolist()):
                        if vv != 0:
                            ax.plot(*zip(xy[pos_of[a]], xy[pos_of[b]]), color=col,
                                    lw=1.0, alpha=0.55, zorder=2)
            if pts:
                ax.scatter(xy[pts, 0], xy[pts, 1], s=34, c=col, zorder=3,
                           edgecolors=SURFACE, linewidths=1.0)

            hr = hit_rate(gene_sets[lbl])
            ax.set_title(f"{lbl}\n{n_hl} genes · {hr:.0%} of pairs co-expressed",
                         loc="left", pad=6, fontsize=9.5)
            ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_color(GRID)

        fig.suptitle(f"Genes on the complete {tissue} co-expression network",
                     x=0.02, ha="left", fontsize=11, color=INK)
        fig.text(0.02, 0.015,
                 f"Grey: all {len(genes)} connected genes in the tissue, positioned by "
                 f"co-expression with every other gene — nearby genes are co-expressed. "
                 f"Lines join co-expressed genes placed in the same pathway.",
                 fontsize=8, color=INK_MUTED, ha="left")
        fig.tight_layout(rect=[0, 0.05, 1, 0.93])
        if save:
            fig.savefig(save, dpi=300)
        return fig, embedding, tissue




def embed_graph(sp, V: int, layout: str = "drl", n_components: int = 32,
                method: str = "umap", seed: int = 0, verbose: bool = True,
                label: str = "graph"):
    """
    A position for every non-isolated node of a sparse adjacency.

    igraph's DrL when available — built for graphs this size, and it places
    high-degree nodes centrally, which is what makes a hub-and-spoke network
    readable. Falls back to the truncated-SVD projection otherwise.

    Returns (node_ids, xy, degree).
    """
    import numpy as np

    sp = sp.coalesce()
    if layout == "drl":
        try:
            import igraph as ig
        except ImportError:
            if verbose:
                print("igraph not installed — falling back to the SVD projection")
            layout = "svd"

    if layout == "drl":
        import igraph as ig
        idx = sp.indices().cpu().numpy()
        g = ig.Graph(n=V, edges=list(zip(idx[0].tolist(), idx[1].tolist())),
                     directed=False)
        g.simplify(multiple=True, loops=True)
        isolated = [v.index for v in g.vs if v.degree() == 0]
        kept = np.array([v for v in range(V) if v not in set(isolated)])
        if isolated:
            g.delete_vertices(isolated)
        if verbose:
            print(f"{label}: {g.vcount()} connected nodes, {g.ecount()} edges")
        ig.set_random_number_generator(__import__("random").Random(seed))
        return kept, np.asarray(g.layout_drl().coords), np.asarray(g.degree())

    ids, xy = _embed_sparse(sp, V, n_components, method, seed, verbose, label=label)
    deg = np.zeros(V)
    u = sp.indices()[0].cpu().numpy()
    np.add.at(deg, u, 1)
    return ids, xy, deg[ids]


def embed_metabolic_network(reference, layout: str = "drl", n_components: int = 32,
                            method: str = "umap", seed: int = 0, verbose: bool = True):
    """Positions for the curated metabolic network. See `embed_graph`."""
    return embed_graph(reference.adj_sparse, reference.num_nodes, layout,
                       n_components, method, seed, verbose,
                       label="metabolic network")


def metabolic_network_map(reference, dataset, node_sets: dict,
                          node_type_fn=None, embedding=None, draw_edges=None,
                          max_edges: int = 40000, save: Optional[str] = None,
                          seed: int = 0, pad_id: int = 0, verbose: bool = True):
    """
    The curated metabolic network as a backdrop, with each condition's nodes
    marked on it and coloured by entity type.

    One categorical system in this figure: the colour is the NODE TYPE, and
    the condition is the panel. (The bar charts colour by condition, because
    there the condition is the categorical dimension — within a figure only
    one thing may own the hue.)

    Each panel reports the share of within-pathway node pairs that are
    genuinely adjacent in the reference network, so the picture carries its
    own statistic.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.collections import LineCollection

    node_type_fn = node_type_fn or default_node_type

    if embedding is None:
        embedding = embed_metabolic_network(reference, seed=seed, verbose=verbose)
    ids, xy, deg = embedding
    pos_of = {int(n): i for i, n in enumerate(ids)}

    def type_of(node_id):
        name = dataset.id_to_node.get(int(node_id))
        return node_type_fn(name) if name is not None else "unknown"

    types = np.array([type_of(n) for n in ids])
    sizes = 2.0 + 6.0 * (deg / max(deg.max(), 1)) ** 0.5

    # edges only when there are few enough to stay a texture rather than mud
    edge_xy = None
    if draw_edges is not False:
        u, v = reference.adj_sparse.coalesce().indices().cpu().numpy()
        keep = u < v
        u, v = u[keep], v[keep]
        if draw_edges or len(u) <= max_edges:
            sel = [(pos_of[a], pos_of[b]) for a, b in zip(u.tolist(), v.tolist())
                   if a in pos_of and b in pos_of][:max_edges]
            edge_xy = np.array([[xy[a], xy[b]] for a, b in sel]) if sel else None
        elif verbose:
            print(f"{len(u)} edges > max_edges={max_edges}; drawing nodes only")

    def adjacency_rate(sets):
        """Within-pathway node pairs that are actually adjacent in the
        reference — the spatial claim of the figure, as a number."""
        hits = tot = 0
        for s in sets:
            t = torch.tensor(sorted({int(x) for x in s if int(x) != pad_id}))
            if t.numel() < 2:
                continue
            p = torch.combinations(t, r=2)
            hits += int(reference.has_edge(p[:, 0], p[:, 1]).sum())
            tot += p.shape[0]
        return hits / tot if tot else float("nan")

    labels = list(node_sets)
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, len(labels), figsize=(4.3 * len(labels), 4.6),
                                 squeeze=False, sharex=True, sharey=True)
        for ax, lbl in zip(axes[0], labels):
            if edge_xy is not None:
                ax.add_collection(LineCollection(edge_xy, colors="#00000012",
                                                 linewidths=0.15, zorder=1))
            ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c="#b9b8b1", alpha=0.40,
                       linewidths=0, zorder=2)

            n_hl = n_off = 0
            for s in node_sets[lbl]:
                for n in {int(x) for x in s if int(x) != pad_id}:
                    if n not in pos_of:
                        n_off += 1                       # absent from the network
                        continue
                    i = pos_of[n]
                    ax.scatter(xy[i, 0], xy[i, 1], s=44,
                               c=TYPE_COLORS.get(type_of(n), "#7f7f7f"),
                               edgecolors=SURFACE, linewidths=1.0, zorder=4)
                    n_hl += 1

            rate = adjacency_rate(node_sets[lbl])
            extra = f" · {n_off} not in network" if n_off else ""
            ax.set_title(f"{lbl}\n{n_hl} nodes · {rate:.0%} of pairs adjacent{extra}",
                         loc="left", pad=6, fontsize=9.5)
            ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_color(GRID)

        present = [t for t in TYPE_COLORS if (types == t).any()]
        fig.legend(handles=[plt.Line2D([0], [0], marker="o", linestyle="",
                                       color=TYPE_COLORS[t], label=t, markersize=7)
                            for t in present],
                   loc="lower center", ncol=len(present), bbox_to_anchor=(0.5, -0.01))
        fig.suptitle("Generated nodes on the curated metabolic network",
                     x=0.02, ha="left", fontsize=11, color=INK)
        fig.text(0.02, 0.055,
                 f"Grey: all {len(ids)} connected nodes of the reference network, "
                 f"sized by degree. Coloured: nodes placed in a pathway, by entity type.",
                 fontsize=8, color=INK_MUTED, ha="left")
        fig.tight_layout(rect=[0, 0.10, 1, 0.93])
        if save:
            fig.savefig(save, dpi=300)
        return fig, embedding


# ── the model's own embedding space ────────────────────────────────────────

def _entity_table(model):
    """The model's learned per-entity vectors, whichever variant it is."""
    tower = getattr(model, "tower", None)
    if tower is not None and hasattr(tower, "gene_embeddings"):
        try:
            return tower.gene_embeddings().detach()
        except RuntimeError:
            pass
    emb = getattr(model, "embedding", None)          # continuous variant
    if emb is not None and hasattr(emb, "embed"):
        return emb.embed.weight[: model.node_vocab].detach()
    if tower is not None and hasattr(tower, "tok_emb"):
        return tower.tok_emb.weight[: model.node_vocab].detach()
    encoder = getattr(model, "encoder", None)        # VAE: entities enter through the encoder
    if encoder is not None and hasattr(encoder, "tok_emb"):
        return encoder.tok_emb.weight[: model.node_vocab].detach()
    raise ValueError("could not locate an entity embedding table on this model")


def embed_model_space(model, method: str = "umap", seed: int = 0,
                      metric: str = "cosine", pad_id: int = 0,
                      verbose: bool = True):
    """
    Project the model's learned entity embeddings to 2-D.

    This is the model's *own* view of the vocabulary — not the
    co-expression network, not the curated network, but what it internally
    believes about which entities resemble which. Comparing it against the
    other two maps is the point: structure that appears here and in the
    reference networks is structure the model actually learned.

    Cosine geometry by default; an embedding table's magnitudes carry
    frequency effects that direction does not.
    """
    import numpy as np

    W = _entity_table(model).cpu().float().numpy()
    ids = np.array([i for i in range(W.shape[0]) if i != pad_id])
    X = W[ids]
    if verbose:
        print(f"model embedding table: {X.shape[0]} entities x {X.shape[1]} dims")

    if method == "umap":
        try:
            import umap
            xy = umap.UMAP(n_neighbors=15, min_dist=0.12, metric=metric,
                           random_state=seed, verbose=False).fit_transform(X)
            if verbose:
                print("  projected with UMAP")
            return ids, np.asarray(xy)
        except Exception:
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        xy = TSNE(n_components=2, init="pca", metric=metric, random_state=seed,
                  perplexity=min(30, max(5, len(ids) // 4))).fit_transform(X)
        if verbose:
            print("  projected with t-SNE")
    else:
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=seed).fit_transform(X)
        if verbose:
            print("  projected with PCA")
    return ids, np.asarray(xy)


def model_embedding_map(model, dataset, node_sets: Optional[dict] = None,
                        node_type_fn=None, embedding=None, save: Optional[str] = None,
                        seed: int = 0, pad_id: int = 0, verbose: bool = True):
    """
    The model's embedding space, with each condition's entities marked.

    With `node_sets=None` it draws one panel coloured by entity type — the
    diagnostic for whether the model organised its vocabulary by type at
    all, or left that entirely to the separate type embedding.

    Each panel reports a COMPACTNESS ratio: the median cosine distance
    between entities placed in the same pathway, over the median between
    random entities. Below 1 means the model already regarded those
    entities as related; ~1 means the pathway is an arbitrary set as far as
    its embedding is concerned. The ratio is computed in the full embedding
    space, not on the 2-D projection, so it does not inherit UMAP's
    distortions.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    node_type_fn = node_type_fn or default_node_type

    if embedding is None:
        embedding = embed_model_space(model, seed=seed, pad_id=pad_id, verbose=verbose)
    ids, xy = embedding
    pos_of = {int(n): i for i, n in enumerate(ids)}

    W = _entity_table(model).cpu().float()
    Wn = torch.nn.functional.normalize(W, dim=-1)

    g = torch.Generator().manual_seed(seed)
    ref_pairs = torch.randint(0, Wn.shape[0], (4000, 2), generator=g)
    ref_pairs = ref_pairs[ref_pairs[:, 0] != ref_pairs[:, 1]]
    baseline = float((1 - (Wn[ref_pairs[:, 0]] * Wn[ref_pairs[:, 1]]).sum(-1)).median())

    def compactness(sets):
        d = []
        for s in sets:
            t = torch.tensor(sorted({int(x) for x in s if int(x) != pad_id}))
            t = t[t < Wn.shape[0]]
            if t.numel() < 2:
                continue
            p = torch.combinations(t, r=2)
            d.append(1 - (Wn[p[:, 0]] * Wn[p[:, 1]]).sum(-1))
        if not d:
            return float("nan")
        return float(torch.cat(d).median() / max(baseline, 1e-9))

    def type_of(n):
        name = dataset.id_to_node.get(int(n))
        return node_type_fn(name) if name is not None else "unknown"

    labels = list(node_sets) if node_sets else ["embedding space"]
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, len(labels), figsize=(4.3 * len(labels), 4.6),
                                 squeeze=False, sharex=True, sharey=True)
        types = np.array([type_of(n) for n in ids])

        for ax, lbl in zip(axes[0], labels):
            if node_sets is None:
                for t in [t for t in TYPE_COLORS if (types == t).any()]:
                    m = types == t
                    ax.scatter(xy[m, 0], xy[m, 1], s=6, c=TYPE_COLORS[t],
                               alpha=0.55, linewidths=0, label=t, zorder=2)
                ax.set_title("entities coloured by type", loc="left", pad=6, fontsize=9.5)
            else:
                ax.scatter(xy[:, 0], xy[:, 1], s=4.0, c="#b9b8b1", alpha=0.40,
                           linewidths=0, zorder=1)
                n_hl, drawn = 0, []
                for s in node_sets[lbl]:
                    for n in {int(x) for x in s if int(x) != pad_id}:
                        if n in pos_of:
                            i = pos_of[n]
                            ax.scatter(xy[i, 0], xy[i, 1], s=42,
                                       c=TYPE_COLORS.get(type_of(n), "#7f7f7f"),
                                       edgecolors=SURFACE, linewidths=1.0, zorder=3)
                            drawn.append(xy[i])
                            n_hl += 1
                # entities the model embeds almost identically land on the
                # same pixel; without this the panel shows five dots and
                # claims thirty-two
                span = (xy.max(0) - xy.min(0)).max()
                distinct = len({tuple(np.round(p / (span * 0.004))) for p in drawn}) \
                    if drawn else 0
                overlap = (f" · {n_hl - distinct} overlapping"
                           if distinct and n_hl - distinct > 0.15 * n_hl else "")
                ratio = compactness(node_sets[lbl])
                ax.set_title(f"{lbl}\n{n_hl} entities · compactness {ratio:.2f}{overlap}",
                             loc="left", pad=6, fontsize=9.5)
            ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_color(GRID)

        present = [t for t in TYPE_COLORS if (types == t).any()]
        fig.legend(handles=[plt.Line2D([0], [0], marker="o", linestyle="",
                                       color=TYPE_COLORS[t], label=t, markersize=7)
                            for t in present],
                   loc="lower center", ncol=len(present), bbox_to_anchor=(0.5, -0.01))
        fig.suptitle("The model's own entity embedding space",
                     x=0.02, ha="left", fontsize=11, color=INK)
        note = ("Compactness: median cosine distance within a pathway / between random "
                "entities, measured in the full embedding space. Below 1 = the model "
                "already treats those entities as related.")
        fig.text(0.02, 0.055, note if node_sets else
                 "Every entity in the vocabulary, positioned by the model's learned "
                 "embedding. Clustering by type means the model organised its "
                 "vocabulary that way on its own.",
                 fontsize=8, color=INK_MUTED, ha="left")
        fig.tight_layout(rect=[0, 0.10, 1, 0.93])
        if save:
            fig.savefig(save, dpi=300)
        return fig, embedding


# ── propagated co-expression laid over the metabolic network ───────────────

def propagated_overlay_figure(reference, propagator, dataset,
                              embedding=None, top_k: int = 4,
                              max_edges: int = 3000, max_nodes: int = 2500,
                              node_type_fn=None, save: Optional[str] = None,
                              seed: int = 0, verbose: bool = True):
    """
    Propagated co-expression over the metabolic network, as two panels.

        thickness  propagated co-expression strength
        panel a    links that are ALSO metabolic edges — transcriptional
                   evidence agreeing with known chemistry
        panel b    links the reference does not have — the population the
                   discovery benchmark ranks, seen whole

    Three things make this readable where a single overlay is not:

    * The layout is computed on the UNION of the metabolic edges and the
      propagated links actually drawn. Laying out on the metabolic graph
      alone leaves every co-expression link a long chord across the figure,
      because the layout knows nothing about them; on the union, related
      nodes sit together and the links become short.
    * Nodes are kept by DEGREE in that union — the `max_nodes` best
      connected. A metabolic network is mostly low-degree periphery, which
      contributes nothing but long spokes.
    * The two classes get their own panels. Agreement is rare (often 25:1
      against), so overlaying them means the rare, interesting class
      disappears under the common one.

    Faint grey in both panels is the metabolic network itself, so the
    reference's shape stays visible under whichever class is being shown.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    node_type_fn = node_type_fn or default_node_type
    V = reference.num_nodes

    # ── the propagated links, strongest first ─────────────────────────────
    net = propagator.top_k_network(k=top_k).coalesce()
    pu, pv = net.indices().cpu().numpy()
    pw = net.values().cpu().numpy()
    m = pu < pv
    pu, pv, pw = pu[m], pv[m], pw[m]
    n_all = len(pu)
    if n_all > max_edges:
        sel = np.argsort(-pw)[:max_edges]
        pu, pv, pw = pu[sel], pv[sel], pw[sel]
        if verbose:
            print(f"drawing the {max_edges} strongest of {n_all} propagated links "
                  f"({max_edges / n_all:.0%})")

    mu, mv = reference.adj_sparse.coalesce().indices().cpu().numpy()
    mm = mu < mv
    mu, mv = mu[mm], mv[mm]

    # ── layout on the union, so the drawn links are short ─────────────────
    if embedding is None:
        uu = np.concatenate([pu, pv, mu, mv])
        vv = np.concatenate([pv, pu, mv, mu])
        union = torch.sparse_coo_tensor(
            torch.as_tensor(np.stack([uu, vv]), dtype=torch.long),
            torch.ones(len(uu)), (V, V)).coalesce()

        # keep the best-connected nodes; the periphery is all spokes
        deg_all = np.zeros(V)
        np.add.at(deg_all, uu, 1)
        keep_ids = np.flatnonzero(deg_all > 0)
        if len(keep_ids) > max_nodes:
            keep_ids = keep_ids[np.argsort(-deg_all[keep_ids])[:max_nodes]]
            if verbose:
                print(f"keeping the {max_nodes} best-connected of "
                      f"{int((deg_all > 0).sum())} nodes")
        keep_set = set(keep_ids.tolist())
        sub = torch.as_tensor([[a, b] for a, b in zip(uu.tolist(), vv.tolist())
                               if a in keep_set and b in keep_set], dtype=torch.long).T
        union = torch.sparse_coo_tensor(sub, torch.ones(sub.shape[1]), (V, V)).coalesce()
        embedding = embed_graph(union, V, seed=seed, verbose=verbose,
                                label="metabolic + propagated union")
    ids, xy, deg = embedding
    pos_of = {int(n): i for i, n in enumerate(ids)}

    def segs_for(a, b):
        ok = np.array([x in pos_of and y in pos_of for x, y in zip(a, b)])
        if not ok.any():
            return np.empty((0, 2, 2)), ok
        return np.array([[xy[pos_of[x]], xy[pos_of[y]]]
                         for x, y in zip(a[ok], b[ok])]), ok

    p_segs, p_ok = segs_for(pu, pv)
    pw_ok = pw[p_ok]
    is_met = reference.has_edge(torch.as_tensor(pu[p_ok]),
                                torch.as_tensor(pv[p_ok])).cpu().numpy()
    m_segs, m_ok = segs_for(mu, mv)
    lw = 0.3 + 2.6 * (pw_ok / max(pw_ok.max(), 1e-9)) ** 0.7

    n_hit, n_tot = int(is_met.sum()), len(is_met)
    n_nodes = len(ids)
    # density of the SUBGRAPH actually drawn: dividing the whole network's
    # edge count by the kept node count overstates it, and understates the
    # enrichment by the same factor
    n_met_kept = int(m_ok.sum())
    density = n_met_kept / max(n_nodes * (n_nodes - 1) / 2, 1)
    enrich = (n_hit / n_tot) / density if n_tot and density > 0 else float("nan")

    panels = [("a  Agreeing with the metabolic network", is_met, "#1baf7a", 0.8),
              ("b  Co-expression only", ~is_met, "#eb6834", 0.35)]

    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.8), sharex=True, sharey=True)
        for ax, (title, mask, col, alpha) in zip(axes, panels):
            if len(m_segs):
                ax.add_collection(LineCollection(m_segs, colors="#00000012",
                                                 linewidths=0.25, zorder=1))
            ax.scatter(xy[:, 0], xy[:, 1],
                       s=1.5 + 6.0 * (deg / max(deg.max(), 1)) ** 0.5,
                       c="#b9b8b1", alpha=0.5, linewidths=0, zorder=2)
            if mask.any():
                ax.add_collection(LineCollection(p_segs[mask], colors=col,
                                                 linewidths=lw[mask], alpha=alpha,
                                                 zorder=3))
            ax.set_title(f"{title}\n{int(mask.sum())} links", loc="left",
                         pad=8, fontsize=9.5)
            ax.set_axis_off()
        axes[0].autoscale()

        fig.suptitle(
            f"Propagated co-expression over the metabolic network — "
            f"{n_hit} of {n_tot} strongest links ({n_hit / max(n_tot,1):.0%}) are real "
            f"edges, {enrich:.0f}x the drawn subgraph's density",
            x=0.02, ha="left", fontsize=11, color=INK)
        fig.legend(handles=[
            Line2D([0], [0], color="#8a8880", lw=0.5, label="line thickness = propagated strength"),
            Line2D([0], [0], color="#c8c7c0", lw=1.0, label="metabolic network (context)"),
            Line2D([0], [0], marker="o", lw=0, markerfacecolor="#b9b8b1",
                   markeredgecolor="none", markersize=6,
                   label="node size = degree in the union"),
        ], loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=[0, 0.05, 1, 0.93])
        if save:
            fig.savefig(save, dpi=300)
        return fig, embedding
