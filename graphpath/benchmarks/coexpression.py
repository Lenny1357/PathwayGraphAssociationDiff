"""Co-expression of the gene nodes a model puts in the same pathway.

A generated pathway should not just look structurally plausible — the genes
in it should actually be co-expressed. This benchmark pulls the gene nodes
out of each generated graph and scores them with an expression source.

The expression source is `graphpath.data.expression.ExpressionLoader`, or
anything exposing `within_list_co_exp(gene_ids) -> (n_tissues, 4)` with the
columns of `STAT_COLUMNS`.

A pathway is expressed in a TISSUE, so a single number averaged over all
tissues is the wrong summary: a set of genes strongly co-expressed in liver
and dead everywhere else is a good pathway, but averaging drags it toward
zero. The per-tissue maximum is reported alongside the mean, and the tissue
achieving it is recorded — a coherent pathway should also agree on which
tissue that is across graphs.
"""

from typing import Callable, Optional

import torch
from torch import Tensor

from graphpath.data.expression import STAT_COLUMNS

from .base import Benchmark, BenchmarkResult
from graphpath.data.node_types import default_node_type, load_name_map, make_common_name


class CoExpressionBenchmark(Benchmark):

    name = "gene co-expression within generated pathways"

    def __init__(
        self,
        dataset,                                   # needs .id_to_node
        expression_loader=None,                    # ExpressionLoader, or .within_list_co_exp
        device='cuda',
        name_map_path: Optional[str] = None,
        node_type_fn: Callable[[str], str] = default_node_type,
        unique_genes: bool = True,
        min_genes: int = 2,
        pad_id: int = 0,
        node_filter: str = "gene",
    ):
        """
        unique_genes : de-duplicate gene ids before scoring. Generated graphs
            repeat entities fairly often, and a repeated gene contributes a
            self-pair with co-expression 1.0, which inflates the score.
        min_genes : graphs with fewer gene nodes than this are skipped (the
            metric is undefined for a single gene); the skipped count is
            reported so a good score over 3 graphs is not mistaken for a good
            score over 50.
        pad_id : token id used for padding, excluded from the gene set.
        node_filter : "gene" scores only gene nodes, which is all the raw
            co-expression tables can speak to. "all" scores every entity —
            only meaningful with a `PropagatedExpression` source, where
            metabolites and reactions have inherited a value through the
            metabolic network.
        """
        if node_filter not in ("gene", "all"):
            raise ValueError("node_filter must be 'gene' or 'all'")
        self.dataset = dataset
        self.device = device
        self.unique_genes = unique_genes
        self.min_genes = min_genes
        self.pad_id = pad_id
        self.node_type_fn = node_type_fn
        self.node_filter = node_filter

        self.expression_loader = expression_loader
        self.name_map = load_name_map(name_map_path) if name_map_path else None
        self.get_common_name = make_common_name(self.name_map)

    # ── utilities ──────────────────────────────────────────────────────────

    def get_node_type(self, name: str) -> str:
        return self.node_type_fn(name)

    def node_name(self, node_id: int):
        return self.dataset.id_to_node.get(int(node_id))

    def extract_nodes(self, nodes: Tensor):
        """Scoreable node ids from one graph's token vector.

        Unknown ids (padding, the MASK slot, anything outside the vocabulary)
        are dropped rather than raising, so a partially-decoded graph can
        still be scored.
        """
        out, seen = [], set()
        for n in nodes.flatten().tolist():
            if n == self.pad_id:
                continue
            name = self.node_name(n)
            if name is None:
                continue
            if self.node_filter == "gene" and self.get_node_type(name) != 'gene':
                continue
            if self.unique_genes:
                if n in seen:
                    continue
                seen.add(n)
            out.append(n)
        return out

    # kept under the old name: the notebook cells call it
    extract_gene_nodes = extract_nodes

    def gene_counts(self, node_toks: Tensor):
        """(total, unique) gene nodes per graph — the composition the metric
        depends on, worth reporting alongside it."""
        totals, uniques = [], []
        for row in node_toks:
            ids = [n for n in row.flatten().tolist()
                   if n != self.pad_id and self.node_name(n) is not None
                   and (self.node_filter == "all"
                        or self.get_node_type(self.node_name(n)) == 'gene')]
            totals.append(len(ids))
            uniques.append(len(set(ids)))
        return totals, uniques

    # ── the benchmark ──────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, node_toks: Tensor, adjacency: Optional[Tensor] = None,
                 **kwargs) -> BenchmarkResult:
        if self.expression_loader is None:
            raise RuntimeError(
                "CoExpressionBenchmark needs an expression source; pass "
                "expression_loader=ExpressionLoader(path=...) from "
                "graphpath.data.expression, or anything exposing "
                "within_list_co_exp(gene_ids) -> (n_tissues, 4)."
            )

        node_toks = node_toks if node_toks.dim() == 2 else node_toks.unsqueeze(0)
        scores, n_used, scored_idx = [], [], []

        for gi, nodes in enumerate(node_toks):
            gene_nodes = self.extract_nodes(nodes.to(self.device))
            if len(gene_nodes) < self.min_genes:
                continue
            scored_idx.append(gi)
            res = self.expression_loader.within_list_co_exp(
                torch.tensor(gene_nodes, device=self.device)
            )
            scores.append(torch.as_tensor(res).detach().float().cpu())
            n_used.append(len(gene_nodes))

        totals, uniques = self.gene_counts(node_toks)
        notes_prefix = []
        if self.node_filter == "all":
            notes_prefix.append("scoring ALL entity types, not only genes — the "
                                "'genes_*' keys count every scored entity")
        if getattr(self.expression_loader, "hops", None) is not None:
            notes_prefix.append(f"propagated co-expression "
                                f"({self.expression_loader.hops} hops, "
                                f"decay {self.expression_loader.decay})")
        summary = {
            "genes_per_graph_mean":  float(sum(totals) / max(len(totals), 1)),
            "unique_genes_per_graph": float(sum(uniques) / max(len(uniques), 1)),
            "duplicate_gene_rate":   1 - (sum(uniques) / max(sum(totals), 1)),
        }
        notes = list(notes_prefix)

        if not scores:
            notes.append(f"no graph had >= {self.min_genes} gene nodes — nothing scored")
            return BenchmarkResult(self.name, None, summary,
                                   n_graphs=len(node_toks), n_scored=0, notes=notes)

        values = torch.stack(scores)          # (n_scored, n_tissues, 4)
        summary["genes_scored_mean"] = float(sum(n_used) / len(n_used))

        if values.dim() == 3 and values.shape[-1] == len(STAT_COLUMNS):
            # named columns, per tissue
            coexpr = values[..., STAT_COLUMNS.index("coexpr_mean")]   # (G, T)
            count  = values[..., STAT_COLUMNS.index("coexpr_count")]
            expr   = values[..., STAT_COLUMNS.index("expr_mean")]
            best_t = coexpr.argmax(dim=1)                             # (G,)
            best_v = coexpr.gather(1, best_t[:, None]).squeeze(1)

            # coexpr_mean is an average over ALL pairs with absent counted as
            # zero, so it factorises as (hit rate) x (mean value when
            # present). Both halves matter and they can move independently:
            # a set can score well by touching few pairs very strongly, or
            # many pairs weakly. Neither term depends on set size.
            n_pairs = torch.tensor([n * (n - 1) / 2 for n in n_used],
                                   dtype=torch.float).clamp(min=1)
            hit = count / n_pairs[:, None]                             # (G, T)
            hit_best = hit.gather(1, best_t[:, None]).squeeze(1)
            with torch.no_grad():
                val_present = torch.where(hit_best > 0, best_v / hit_best.clamp(min=1e-9),
                                          torch.zeros_like(best_v))

            # Three hit-rate estimators, because they answer different
            # questions and can differ by ~10 points:
            #   best        in the tissue with the highest coexpr_mean. The
            #               one that keeps hit x strength = score, so the
            #               figure's three panels multiply.
            #   pooled      total hits / total pairs in that same tissue. The
            #               natural estimator for a RATE — every pair counts
            #               once, instead of every graph counting once, so a
            #               2-gene graph (one pair, 0% or 100%) stops
            #               swinging the average.
            #   max_tissue  the best tissue for hit rate specifically. An
            #               upper bound: a maximum over 11 tissues is
            #               optimistic even under the null.
            counts_best = count.gather(1, best_t[:, None]).squeeze(1)
            summary.update({
                "coexpr_mean_over_tissues": float(coexpr.mean()),
                "coexpr_best_tissue_mean":  float(best_v.mean()),
                "coexpr_best_tissue_std":   float(best_v.std()) if len(best_v) > 1 else 0.0,
                "coexpr_hit_rate_best":     float(hit_best.mean()),
                "coexpr_hit_rate_pooled":   float(counts_best.sum() / n_pairs.sum()),
                "coexpr_hit_rate_max_tissue": float(hit.max(dim=1).values.mean()),
                "coexpr_hit_rate_mean":     float(hit.mean()),
                "coexpr_value_when_present": float(val_present[hit_best > 0].mean())
                                             if bool((hit_best > 0).any()) else 0.0,
                "coexpr_pairs_found_mean":  float(count.max(dim=1).values.mean()),
                "coexpr_pairs_total_mean":  float(n_pairs.mean()),
                "expr_mean_over_tissues":   float(expr.mean()),
            })
            per_graph = {
                "coexpr_best":        best_v,
                "hit_rate_pooled_num": counts_best,
                "hit_rate_pooled_den": n_pairs,
                "hit_rate_best":      hit_best,
                "value_when_present": val_present,
                "n_genes":            torch.tensor(n_used, dtype=torch.float),
                "per_tissue_coexpr":  coexpr,
                # which input graphs these rows correspond to — graphs with
                # too few genes are skipped, so row i is not graph i
                "graph_index":        torch.tensor(scored_idx, dtype=torch.long),
            }
            tissues = getattr(self.expression_loader, "tissues", None)
            if tissues is not None:
                counts = torch.bincount(best_t, minlength=len(tissues))
                top = int(counts.argmax())
                summary["top_tissue"] = tissues[top]
                # a coherent set of pathways should keep landing on the same
                # tissue; spread across all of them suggests noise
                summary["top_tissue_share"] = float(counts[top] / counts.sum())
            self.last_per_tissue = coexpr        # (n_scored, n_tissues), for plots
        else:
            per_graph = {}
            flat = values.flatten(1) if values.dim() > 1 else values[:, None]
            summary.update({
                "co_expression_mean": float(flat.mean()),
                "co_expression_std":  float(flat.std()) if len(flat) > 1 else 0.0,
            })
            notes.append("expression source returned an unrecognised shape "
                         f"{tuple(values.shape[1:])}; reporting a flat mean")
        if len(scores) < len(node_toks):
            notes.append(f"{len(node_toks) - len(scores)} graph(s) skipped for "
                         f"having < {self.min_genes} gene nodes")
        if not self.unique_genes and summary["duplicate_gene_rate"] > 0:
            notes.append("duplicate genes are being scored (unique_genes=False), "
                         "which inflates co-expression via self-pairs")

        return BenchmarkResult(self.name, values, summary,
                               n_graphs=len(node_toks), n_scored=len(scores),
                               notes=notes, per_graph=per_graph)
