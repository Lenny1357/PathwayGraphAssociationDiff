"""Does a generated pathway have the SHAPE of a pathway?

Every other benchmark here scores *which* entities and links were emitted.
This one ignores identity entirely and asks only about topology: degree
distribution, clustering, path lengths, community structure, and the
metabolic motifs (a metabolite feeding several reactions, a reaction
producing several metabolites) that give these networks their characteristic
form.

A model can score well on edge precision and still produce graphs that are
structurally wrong — a hairball, or a spray of disconnected pairs. Read
against `evaluate_dataset`, this says how close the generated topology is to
the real one, and `topology_distances` turns that into a single number per
metric rather than leaving it to eyeball two columns.

A note on controls. `shuffled_like` is the wrong one here: it permutes node
identities and keeps the adjacency, and topology does not depend on identity,
so the shuffled arm is the generated arm — except that displaced padding
strands real nodes as isolates, which makes it look worse for a reason that
has nothing to do with structure. Use `random_like` as the floor and the
training data as the target.

Two departures from the obvious implementation, both of which change what
the numbers mean:

  * Padding slots are dropped by TOKEN, not by degree. Removing every
    isolated node first, as is tempting, silently deletes a real defect —
    a node the model emitted and then failed to connect — by treating it as
    if it were padding. Isolation is reported as its own metric.
  * Path length and diameter are computed on the largest connected
    component, not skipped when the graph is disconnected. Skipping means a
    model that fragments its output is never charged for it, because the
    fragmented graphs quietly leave the average.
"""

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .base import Benchmark, BenchmarkResult
from graphpath.data.node_types import default_node_type

# Metrics whose pooled distribution — not just its mean — is worth keeping,
# because "captures the structure" is a statement about the whole shape.
POOLED = ("degree", "clustering_node", "square_clustering_node",
          "path_length", "shared_neighbour")


class TopologyBenchmark(Benchmark):
    """
    id_to_node : {token id -> node name}, e.g. `dataset.id_to_node`. Needed
        for the type-aware metrics (edge types, fan-in/fan-out, bipartite
        clustering); without it those are skipped and a note says so.
    heavy : include betweenness, edge betweenness and modularity. They are
        the expensive third of the benchmark and are dominated by the cheap
        metrics for most comparisons, so turn them off for quick passes.
    """

    name = "graph topology"

    def __init__(self, id_to_node: Optional[dict] = None,
                 node_type_fn=default_node_type, pad_id: int = 0,
                 heavy: bool = True):
        self.id_to_node = id_to_node
        self.node_type_fn = node_type_fn
        self.pad_id = pad_id
        self.heavy = heavy

    # ── graph construction ─────────────────────────────────────────────────

    def to_networkx(self, node_toks: Tensor, adjacency: Optional[Tensor] = None):
        """
        Decode a batch into NetworkX graphs plus per-graph metadata.

        Returns (graphs, names, types, pad_counts, isolated_counts), where
        `names[i]` and `types[i]` are lists index-aligned with `graphs[i]`'s
        relabelled nodes.
        """
        import networkx as nx

        node_toks = node_toks if node_toks.dim() == 2 else node_toks.unsqueeze(0)
        toks = node_toks.detach().cpu().numpy()
        adjs = (adjacency.detach().cpu().numpy()
                if adjacency is not None else None)

        graphs, names_out, types_out, pads, isolates = [], [], [], [], []
        for b in range(len(toks)):
            row = toks[b]
            real = np.flatnonzero(row != self.pad_id)   # padding is not a node
            pads.append(int(len(row) - len(real)))

            names = [self._name(int(row[i])) for i in real]
            types = [self.node_type_fn(n) if n is not None else "unknown"
                     for n in names]

            G = nx.Graph()
            G.add_nodes_from(range(len(real)))
            if adjs is not None:
                a = adjs[b][np.ix_(real, real)]
                a = np.maximum(a, a.T)                  # the model may be asymmetric
                iu = np.triu_indices(len(real), k=1)
                hit = np.flatnonzero(a[iu] > 0)
                G.add_edges_from(zip(iu[0][hit], iu[1][hit]))

            # An emitted-but-unconnected node is a defect, so count it before
            # dropping it; the connectivity metrics need it gone, the
            # isolation rate needs it seen.
            iso = list(nx.isolates(G))
            isolates.append(len(iso))
            G.remove_nodes_from(iso)
            keep = [i for i in range(len(real)) if i not in set(iso)]
            G = nx.relabel_nodes(G, {old: new for new, old in enumerate(keep)},
                                 copy=True)

            graphs.append(G)
            names_out.append([names[i] for i in keep])
            types_out.append([types[i] for i in keep])
        return graphs, names_out, types_out, pads, isolates

    def _name(self, tok: int):
        if self.id_to_node is None:
            return None
        return self.id_to_node.get(tok) if hasattr(self.id_to_node, "get") \
            else self.id_to_node[tok]

    # ── per-graph metrics ──────────────────────────────────────────────────

    def _score_graph(self, G, names, types, n_pad, n_iso) -> tuple:
        """(scalar metrics, pooled distributions) for one graph."""
        import networkx as nx
        from networkx.algorithms import bipartite, community

        n, m = G.number_of_nodes(), G.number_of_edges()
        n_emitted = n + n_iso
        out = {
            "n_nodes": float(n),
            "n_edges": float(m),
            "n_pad_slots": float(n_pad),
            "isolated_node_frac": n_iso / n_emitted if n_emitted else np.nan,
        }
        pooled = {k: [] for k in POOLED}
        if n == 0:
            return out, pooled

        out["density"] = nx.density(G)
        # kept for continuity with the original script; it is 1/density up to
        # a constant, so read one or the other, not both
        out["sparsity"] = n ** 2 / m if m else np.nan

        deg = np.array([d for _, d in G.degree()], dtype=float)
        pooled["degree"] = deg.tolist()
        out["degree"] = float(deg.mean())
        out["degree_max"] = float(deg.max())
        # heavy-tailed vs flat: a real network has hubs, a uniform sample
        # does not, and the mean alone cannot tell them apart
        out["degree_cv"] = float(deg.std() / deg.mean()) if deg.mean() else np.nan

        # Two clusterings, because one of them is structurally zero here and
        # a reader who sees only that concludes the metric is broken.
        #
        #   clustering         triangles. IRRESPECTIVE OF TYPE, over every
        #                      edge. A metabolite-reaction network is
        #                      bipartite, so it has no odd cycles and this is
        #                      exactly 0 by construction, for real pathways
        #                      and for any generated graph that respects the
        #                      same grammar. Non-zero means type violations.
        #   square_clustering  4-cycles. Also irrespective of type, but it is
        #                      the closure measure that SURVIVES bipartite
        #                      structure: two reactions sharing two
        #                      metabolites close a square, not a triangle.
        cl = nx.clustering(G)
        pooled["clustering_node"] = list(cl.values())
        out["clustering"] = float(np.mean(list(cl.values())))

        sq = nx.square_clustering(G)
        if sq:
            pooled["square_clustering_node"] = list(sq.values())
            out["square_clustering"] = float(np.mean(list(sq.values())))

        if m > 0:
            # undefined for a regular graph (zero degree variance); networkx
            # divides by it anyway and lets numpy warn
            with np.errstate(invalid="ignore", divide="ignore"):
                a = nx.degree_assortativity_coefficient(G)
            if not np.isnan(a):
                out["assortativity"] = float(a)

        # ── connectivity, from ONE all-pairs pass reused three ways ───────
        comps = list(nx.connected_components(G))
        out["n_components"] = float(len(comps))
        out["lcc_frac"] = max(len(c) for c in comps) / n
        out["connected"] = float(len(comps) == 1)

        sp = dict(nx.all_pairs_shortest_path_length(G))
        lcc = max(comps, key=len)
        lengths, shared = [], []
        for u in lcc:
            for v, d in sp[u].items():
                if v in lcc and u < v:
                    lengths.append(d)
        # Path length on the largest component: a disconnected graph still
        # gets a number, so fragmenting the output cannot buy a better score
        # by removing the graph from the average.
        if lengths:
            out["avg_path_length"] = float(np.mean(lengths))
            out["diameter"] = float(np.max(lengths))
            pooled["path_length"] = lengths

        # Two-step neighbours sharing a neighbourhood — the substrate-sharing
        # signature. One BFS pass instead of a path search per pair.
        for u in G.nodes():
            nu = set(G.neighbors(u))
            for v, d in sp[u].items():
                if d == 2 and u < v:
                    nv = set(G.neighbors(v))
                    total = min(len(nu), len(nv))
                    if total:
                        shared.append(len(nu & nv) / total)
        if shared:
            out["shared_neighbour_ratio"] = float(np.mean(shared))
            pooled["shared_neighbour"] = shared

        # ── type-aware structure ──────────────────────────────────────────
        if self.id_to_node is not None:
            rxn = {i for i, t in enumerate(types) if t == "reaction"}
            met = {i for i, t in enumerate(types) if t == "metabolite"}

            # Substrate sharing is a statement about reaction-metabolite
            # links, so build that bipartite part explicitly rather than
            # testing whether the whole graph happens to be bipartite: it
            # never is (gene-enzyme edges), and reaction-reaction edges make
            # even the reaction/metabolite subgraph non-bipartite, at which
            # point networkx raises instead of returning False.
            sub = nx.Graph()
            sub.add_nodes_from(rxn | met)
            sub.add_edges_from((u, v) for u, v in G.edges()
                               if {types[u], types[v]} == {"reaction", "metabolite"})
            sub.remove_nodes_from(list(nx.isolates(sub)))
            # The type-AWARE counterpart: co-membership measured only across
            # the reaction/metabolite bipartition, so it answers "do two
            # metabolites share reactions" rather than "is there any closure".
            # `nodes` must be intersected with the subgraph or networkx raises
            # a KeyError on a type that ended up with no cross-type edge.
            sub_met = met & set(sub.nodes())
            if sub_met and sub.number_of_edges():
                cd = bipartite.clustering(sub, nodes=sub_met)
                if cd:
                    out["bipartite_clustering"] = float(np.mean(list(cd.values())))
                out["rxn_met_edge_frac"] = (
                    sub.number_of_edges() / m if m else np.nan)

            fi, fo = [], []
            for i in G.nodes():
                nbrs = list(G.neighbors(i))
                if types[i] == "metabolite":
                    r = sum(1 for j in nbrs if types[j] == "reaction")
                    if r > 1:
                        fi.append(r)
                elif types[i] == "reaction":
                    mm = sum(1 for j in nbrs if types[j] == "metabolite")
                    if mm > 1:
                        fo.append(mm)
            if fi:
                out["fan_in_motifs"] = float(np.mean(fi))
            if fo:
                out["fan_out_motifs"] = float(np.mean(fo))

        if self.heavy and m > 0:
            b = nx.betweenness_centrality(G)
            out["avg_betweenness"] = float(np.mean(list(b.values())))
            eb = nx.edge_betweenness_centrality(G)
            out["avg_edge_betweenness"] = float(np.mean(list(eb.values())))
            comms = community.greedy_modularity_communities(G)
            out["modularity"] = float(community.modularity(G, comms))
            out["n_communities"] = float(len(comms))

        return out, pooled

    # ── the benchmark ──────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, node_toks: Tensor, adjacency: Optional[Tensor] = None,
                 **kwargs) -> BenchmarkResult:
        node_toks = node_toks if node_toks.dim() == 2 else node_toks.unsqueeze(0)
        notes = []
        if adjacency is None:
            return BenchmarkResult(
                self.name, None, {}, len(node_toks), 0,
                ["no adjacency passed — topology cannot be measured"])
        if self.id_to_node is None:
            notes.append("no id_to_node — edge-type, motif and bipartite "
                         "metrics skipped")

        graphs, names, types, pads, isos = self.to_networkx(node_toks, adjacency)

        per_graph, pooled = [], {k: [] for k in POOLED}
        node_types, edge_types = defaultdict(int), defaultdict(int)
        for G, nm, ty, np_, ni in zip(graphs, names, types, pads, isos):
            s, p = self._score_graph(G, nm, ty, np_, ni)
            if G.number_of_nodes() == 0:
                continue
            per_graph.append(s)
            for k in POOLED:
                pooled[k].extend(p[k])
            if self.id_to_node is not None:
                for t in ty:
                    node_types[t] += 1
                for u, v in G.edges():
                    edge_types[tuple(sorted((ty[u], ty[v])))] += 1

        if not per_graph:
            notes.append("every graph was empty after dropping padding")
            return BenchmarkResult(self.name, None, {}, len(node_toks), 0, notes)

        keys, seen = [], set()
        for g in per_graph:
            for k in g:
                if k not in seen:
                    seen.add(k); keys.append(k)

        summary, by_metric = {}, {}
        for k in keys:
            v = torch.tensor([g.get(k, float("nan")) for g in per_graph],
                             dtype=torch.float)
            by_metric[k] = v
            fin = v[~v.isnan()]
            if fin.numel():
                summary[k] = float(fin.mean())
            # a metric only some graphs support is a coverage fact, not a mean
            if int(v.isnan().sum()):
                notes.append(f"{k}: {int(v.isnan().sum())}/{len(per_graph)} "
                             f"graphs could not be scored")

        values = torch.stack([by_metric[k] for k in keys], dim=1)
        res = BenchmarkResult(self.name, values, summary,
                              n_graphs=len(node_toks), n_scored=len(per_graph),
                              notes=notes, per_graph=by_metric)
        res.distributions = {k: np.asarray(v, dtype=float)
                             for k, v in pooled.items()}
        tot_n, tot_e = sum(node_types.values()), sum(edge_types.values())
        res.distributions["node_types"] = {
            k: 100 * v / tot_n for k, v in sorted(node_types.items())} if tot_n else {}
        res.distributions["edge_types"] = {
            k: 100 * v / tot_e for k, v in sorted(edge_types.items())} if tot_e else {}
        return res


# ── comparison against the reference distribution ──────────────────────────

def _wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    """1-Wasserstein between two empirical samples, via sorted quantiles."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    q = np.linspace(0, 1, 512)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic, no SciPy dependency."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.union1d(a, b)
    ca = np.searchsorted(np.sort(a), grid, side="right") / a.size
    cb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.abs(ca - cb).max())


def topology_distances(result: BenchmarkResult, reference: BenchmarkResult,
                       normalise: bool = True) -> dict:
    """
    How far each pooled distribution sits from the reference one.

    Comparing means hides shape: a generated set can match the training mean
    degree exactly while having no hubs and no leaves. Wasserstein and KS
    both see that; both are reported because they disagree in a useful way —
    KS is sensitive to where the bulk sits, Wasserstein to how far the tail
    reaches.

    `normalise` divides the Wasserstein distance by the reference's mean, so
    degree (counts) and clustering (a fraction) can be read on one scale.
    Zero is a perfect match in every case.
    """
    out = {}
    for k in POOLED:
        a = np.asarray(result.distributions.get(k, []), dtype=float)
        b = np.asarray(reference.distributions.get(k, []), dtype=float)
        w = _wasserstein(a, b)
        if normalise and b.size and abs(b.mean()) > 1e-12:
            w = w / abs(b.mean())
        out[f"{k}_w1"] = w
        out[f"{k}_ks"] = _ks(a, b)
    return out


def topology_table(results: dict, keys: Optional[Sequence[str]] = None) -> str:
    """
    Metric table across conditions, with median and spread — the summary that
    `compare()` prints holds means only, and these metrics are skewed enough
    that a mean alone misleads.
    """
    names = list(results)
    if keys is None:
        keys, seen = [], set()
        for r in results.values():
            for k in r.per_graph:
                if k not in seen:
                    seen.add(k); keys.append(k)
    w = max(len(k) for k in keys) + 2
    head = f"{'metric':<{w}}" + "".join(f"{n:>26}" for n in names)
    lines = [head, f"{'':<{w}}" + "".join(f"{'mean (median +- sd)':>26}" for _ in names),
             "-" * len(head)]
    for k in keys:
        row = f"{k:<{w}}"
        for n in names:
            v = results[n].per_graph.get(k)
            if v is None:
                row += f"{'-':>26}"
                continue
            v = v[~v.isnan()]
            row += (f"{'-':>26}" if v.numel() == 0 else
                    f"{float(v.mean()):>12.3f} ({float(v.median()):.3f} "
                    f"+-{float(v.std()):.2f})".rjust(26))
        lines.append(row)
    return "\n".join(lines)


def composition_table(results: dict, kind: str = "node_types") -> str:
    """Node- or edge-type composition, in percent, across conditions."""
    names = list(results)
    cats, seen = [], set()
    for r in results.values():
        for k in r.distributions.get(kind, {}):
            if k not in seen:
                seen.add(k); cats.append(k)
    if not cats:
        return f"(no {kind} recorded — was id_to_node passed?)"
    label = (lambda c: c if isinstance(c, str) else " - ".join(c))
    w = max(len(label(c)) for c in cats) + 2
    head = f"{kind:<{w}}" + "".join(f"{n:>18}" for n in names)
    lines = [head, "-" * len(head)]
    for c in cats:
        row = f"{label(c):<{w}}"
        for n in names:
            v = results[n].distributions.get(kind, {}).get(c)
            row += f"{v:>17.1f}%" if v is not None else f"{'-':>18}"
        lines.append(row)
    return "\n".join(lines)
