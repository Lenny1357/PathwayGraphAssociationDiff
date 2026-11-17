"""Spread gene co-expression across the whole metabolic network.

Co-expression is measured between genes, so three quarters of the entities
in a pathway have none. But a gene sits beside the enzyme it encodes, which
sits beside a reaction, which sits beside its metabolites — so a gene's
transcriptional neighbourhood implies something about the entities around
it. This propagates that signal outward and gives every pair of nodes, of
any type, a co-expression-derived score.

Method — spread each entity over its neighbourhood, then read the gene
co-expression between the two neighbourhoods:

      P = sum_k  decay^k * A_norm^k          k = 0 .. hops
      S = P C P^T

  Row u of P is entity u smeared across the graph; S[u,v] is then the
  co-expression between u's neighbourhood and v's. A gene keeps its own
  value at hop 0; a metabolite three hops from a gene inherits a
  decay^3-weighted share of it.

  (An earlier draft factorised C ~ Z0 Z0^T by SVD and diffused the factor.
  That is only valid for a positive semi-definite C; a co-expression matrix
  with a zero diagonal has negative eigenvalues, and the SVD silently flips
  their sign — planted pairs scored BELOW unrelated controls. Everything
  here is computed exactly on the sparse matrices instead.)

Hyperparameters, in the order they matter:

  hops    how far the signal travels. 1 reaches the enzymes beside a gene,
          2 the reactions, 3 the metabolites. Beyond ~3 hub metabolites
          connect everything to everything and the score stops separating.
  decay   how much survives each hop. 0 disables propagation entirely
          (scores collapse to the plain gene-gene matrix); 1 lets a distant
          gene count as much as an adjacent one.
  normalize  'sym' (D^-1/2 A D^-1/2) keeps hubs from dominating and is
          symmetric; 'row' (D^-1 A) makes each node average its neighbours.
"""

from typing import Optional

import numpy as np
import torch
from torch import Tensor


class CoExpressionPropagator:

    def __init__(
        self,
        adjacency: Tensor,          # sparse (V, V) metabolic network
        coexpr: Tensor,             # sparse (V, V) gene-gene co-expression
        vocab_size: Optional[int] = None,
        hops: int = 2,
        decay: float = 0.5,
        normalize: str = "sym",
        normalized: bool = True,
        prune: float = 1e-6,        # drop tiny entries of P to keep it sparse
        device=None,
        verbose: bool = True,
    ):
        self.V = int(vocab_size or adjacency.shape[0])
        self.hops = hops
        self.decay = decay
        self.normalize = normalize
        self.normalized = normalized
        self.prune = prune
        self.device = torch.device(device) if device is not None else adjacency.device
        self.verbose = verbose
        self._adj = adjacency.coalesce()
        self._coexpr = coexpr.coalesce()
        self.P = None               # (V, V) sparse spread operator
        self.M = None               # P @ C
        self._mass = None           # row sums of P, the normaliser

    # ── fitting ────────────────────────────────────────────────────────────

    def _to_csr(self, sp, absolute=False):
        import scipy.sparse as ssp
        idx = sp.indices().cpu().numpy()
        val = sp.values().cpu().numpy().astype(np.float64)
        if absolute:
            val = np.abs(val)
        return ssp.coo_matrix((val, (idx[0], idx[1])), shape=(self.V, self.V)).tocsr()

    def _normalised_adjacency(self):
        import scipy.sparse as ssp
        A = self._to_csr(self._adj, absolute=True)
        A = A.maximum(A.T)
        deg = np.asarray(A.sum(1)).ravel()
        deg[deg == 0] = 1.0
        if self.normalize == "row":
            return ssp.diags(1.0 / deg) @ A
        d = ssp.diags(1.0 / np.sqrt(deg))
        return d @ A @ d

    def build_spread_operator(self):
        """
        P = sum_k decay^k * A_norm^k. Depends only on the metabolic graph,
        so it can be built once and reused for every tissue.
        """
        import scipy.sparse as ssp
        A = self._normalised_adjacency()
        P = ssp.identity(self.V, format="csr")
        cur = P
        for h in range(1, self.hops + 1):
            cur = (cur @ A).tocsr()
            if self.prune:
                cur.data[np.abs(cur.data) < self.prune] = 0.0
                cur.eliminate_zeros()
            P = P + (self.decay ** h) * cur
        return P.tocsr()

    def fit(self, spread=None):
        """Build the spread operator P and the propagated scores."""
        C = self._to_csr(self._coexpr)
        C = C.maximum(C.T)                      # the score is symmetric either way
        P = self.build_spread_operator() if spread is None else spread

        self.P = P
        self.M = (P @ C).tocsr()                # rows: each entity's spread profile
        # Normalise by the mass that reaches GENES, which are the only
        # entities carrying co-expression. Two alternatives are worse:
        # dividing by S[u,u] explodes (the co-expression diagonal is zero),
        # and dividing by the total spread mass buries every non-gene score,
        # since most of a metabolite's neighbourhood has no co-expression at
        # all. With this denominator the score is the weighted mean
        # co-expression between the genes the two entities can reach, on the
        # same scale as C: hops=0 returns C[u,v] exactly for a gene pair, and
        # a metabolite whose only reachable gene is A scores, against a
        # metabolite reaching only B, exactly the A-B co-expression.
        has_coexpr = (np.asarray(np.abs(C).sum(1)).ravel() > 0)
        self.gene_idx = np.flatnonzero(has_coexpr)
        self._mass = np.clip(
            np.asarray(P[:, self.gene_idx].sum(1)).ravel(), 1e-12, None)
        self._reaches_gene = np.asarray(P[:, self.gene_idx].sum(1)).ravel() > 0

        if self.verbose:
            seeded = int((np.asarray(np.abs(C).sum(1)).ravel() > 0).sum())
            reached = int((np.asarray(np.abs(self.M).sum(1)).ravel() > 0).sum())
            print(f"propagated co-expression: {self.hops} hops, decay {self.decay}, "
                  f"{self.normalize}-normalised")
            print(f"  entities with a signal: {seeded} seeded -> {reached} after "
                  f"propagation ({reached / max(self.V, 1):.0%} of the vocabulary)")
            print(f"  spread operator P: {P.nnz} nonzeros "
                  f"({P.nnz / max(self.V, 1):.1f} per entity)")
            print(f"  entities that can reach a gene: "
                  f"{int(self._reaches_gene.sum())} / {self.V}")
        return self

    # ── scoring ────────────────────────────────────────────────────────────

    def _require_fit(self):
        if self.M is None:
            self.fit()

    def score(self, u: Tensor, v: Tensor) -> Tensor:
        """Propagated co-expression for pairs (u_i, v_i)."""
        self._require_fit()
        ui = u.detach().cpu().numpy().ravel().astype(int)
        vi = v.detach().cpu().numpy().ravel().astype(int)
        out = np.asarray(self.M[ui].multiply(self.P[vi]).sum(1)).ravel()
        if self.normalized:
            out = out / (self._mass[ui] * self._mass[vi])
            out[~(self._reaches_gene[ui] & self._reaches_gene[vi])] = 0.0
        return torch.as_tensor(out.reshape(u.shape), dtype=torch.float32,
                               device=self.device)

    def neighbours(self, node_id: int, k: int = 10):
        """The k entities with the strongest propagated link to `node_id`."""
        self._require_fit()
        s = np.asarray((self.M[int(node_id)] @ self.P.T).todense()).ravel()
        if self.normalized:
            s = s / (self._mass[int(node_id)] * self._mass)
            # an entity that reaches no gene has no propagated signal at all
            s[~self._reaches_gene] = 0.0
            if not self._reaches_gene[int(node_id)]:
                s[:] = 0.0
        s[int(node_id)] = -np.inf
        idx = np.argpartition(-s, min(k, self.V - 1))[:k]
        idx = idx[np.argsort(-s[idx])]
        return torch.as_tensor(idx.copy()), torch.as_tensor(s[idx].copy())

    def top_k_network(self, k: int = 16, chunk: int = 512,
                      min_score: float = 0.0) -> Tensor:
        """
        The propagated network as a sparse (V, V) tensor, keeping each
        entity's k strongest links. Symmetrised, self-loops dropped.
        """
        self._require_fit()
        rows, cols, vals = [], [], []
        for start in range(0, self.V, chunk):
            stop = min(start + chunk, self.V)
            block = np.asarray((self.M[start:stop] @ self.P.T).todense())
            if self.normalized:
                block = block / (self._mass[start:stop, None] * self._mass[None, :])
                block[~self._reaches_gene[start:stop], :] = 0.0
                block[:, ~self._reaches_gene] = 0.0
            block[np.arange(stop - start), np.arange(start, stop)] = -np.inf
            kk = min(k, self.V - 1)
            top = np.argpartition(-block, kk - 1, axis=1)[:, :kk]
            r = np.repeat(np.arange(start, stop), kk)
            c = top.reshape(-1)
            w = block[np.repeat(np.arange(stop - start), kk), c]
            keep = w > min_score
            rows.append(r[keep]); cols.append(c[keep]); vals.append(w[keep])
        r = torch.as_tensor(np.concatenate(rows), dtype=torch.long)
        c = torch.as_tensor(np.concatenate(cols), dtype=torch.long)
        w = torch.as_tensor(np.concatenate(vals), dtype=torch.float32)
        rr, cc, ww = torch.cat([r, c]), torch.cat([c, r]), torch.cat([w, w])
        sp = torch.sparse_coo_tensor(torch.stack([rr, cc]), ww,
                                     size=(self.V, self.V)).coalesce()
        # coalesce sums the mirrored duplicates; halve them back
        return torch.sparse_coo_tensor(sp.indices(), sp.values() / 2,
                                       size=sp.shape).coalesce().to(self.device)


def propagate_all_tissues(expr_loader, adjacency, k: int = 16, **kwargs) -> Tensor:
    """
    Run the propagation per tissue and stack the results into the sparse
    (n_tissues, V, V) tensor the model conditioning expects — so a
    conditioned model can see co-expression structure on metabolites and
    reactions, not only on genes.
    """
    idx, val = [], []
    for t_i, t in enumerate(expr_loader.tissues):
        prop = CoExpressionPropagator(adjacency, expr_loader.co_expression[t],
                                      vocab_size=expr_loader.vocab_size, **kwargs)
        sp = prop.top_k_network(k=k).coalesce()
        u, v = sp.indices()
        idx.append(torch.stack([torch.full_like(u, t_i), u, v]))
        val.append(sp.values())
    return torch.sparse_coo_tensor(
        torch.cat(idx, dim=1), torch.cat(val),
        size=(len(expr_loader.tissues), expr_loader.vocab_size, expr_loader.vocab_size),
    ).coalesce()


class PropagatedExpression:
    """
    A drop-in replacement for `ExpressionLoader` whose co-expression values
    are propagated across the metabolic network.

    It exposes the same two methods everything downstream depends on —
    `within_list_co_exp(node_ids) -> (n_tissues, 4)` and
    `_lookup(tissue, u, v)` — so `CoExpressionBenchmark`, `DiscoveryFinder`
    and the figures work with it unchanged:

        raw        = ExpressionLoader(path=..., device=device)
        propagated = PropagatedExpression(raw, ref.adj_sparse, hops=3)

        CoExpressionBenchmark(dataset, expression_loader=propagated, ...)

    With `hops=0` it reduces exactly to the raw loader, which is the natural
    control: any change in a benchmark number is then attributable to the
    propagation and nothing else.

    The spread operator is built once and shared across tissues — only the
    co-expression matrix differs between them.
    """

    def __init__(self, expr_loader, adjacency, hops: int = 2, decay: float = 0.5,
                 normalize: str = "sym", prune: float = 1e-6, device=None,
                 verbose: bool = True):
        self.base = expr_loader
        self.tissues = list(expr_loader.tissues)
        self.vocab_size = expr_loader.vocab_size
        self.expression = expr_loader.expression
        self.device = torch.device(device) if device is not None else expr_loader.device
        self.hops, self.decay = hops, decay

        seed = CoExpressionPropagator(
            adjacency, expr_loader.co_expression[self.tissues[0]],
            vocab_size=self.vocab_size, hops=hops, decay=decay,
            normalize=normalize, prune=prune, device=self.device, verbose=False)
        spread = seed.build_spread_operator()
        if verbose:
            print(f"propagating co-expression: {hops} hops, decay {decay}, "
                  f"{len(self.tissues)} tissues")
            print(f"  spread operator: {spread.nnz} nonzeros "
                  f"({spread.nnz / max(self.vocab_size, 1):.1f} per entity)")

        self.prop = {}
        for t in self.tissues:
            p = CoExpressionPropagator(
                adjacency, expr_loader.co_expression[t], vocab_size=self.vocab_size,
                hops=hops, decay=decay, normalize=normalize, prune=prune,
                device=self.device, verbose=False)
            p.fit(spread=spread)
            self.prop[t] = p
        if verbose:
            reach = np.mean([p._reaches_gene.mean() for p in self.prop.values()])
            print(f"  entities reaching a gene: {reach:.0%} of the vocabulary")

    # ── the ExpressionLoader interface ─────────────────────────────────────

    def _lookup(self, tissue: str, u: Tensor, v: Tensor) -> Tensor:
        return self.prop[tissue].score(u, v)

    def within_list_co_exp(self, node_ids: Tensor) -> Tensor:
        """(n_tissues, 4): expr_count, expr_mean, coexpr_count, coexpr_mean."""
        node_ids = node_ids.to(self.device).long()
        res = torch.zeros(len(self.tissues), 4, device=self.device)
        pairs = torch.combinations(node_ids, r=2)
        for t_idx, t in enumerate(self.tissues):
            expr = self.expression[t][node_ids]
            matched = (self._lookup(t, pairs[:, 0], pairs[:, 1]) if pairs.numel()
                       else torch.zeros(0, device=self.device))
            res[t_idx] = torch.tensor([
                float(expr.count_nonzero()),
                float(expr.mean()) if expr.numel() else 0.0,
                float(matched.count_nonzero()),
                float(matched.mean()) if matched.numel() else 0.0,
            ], device=self.device)
        return res
