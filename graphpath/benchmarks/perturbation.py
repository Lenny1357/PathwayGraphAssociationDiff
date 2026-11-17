"""Do the genes a model groups together respond alike when perturbed?

The causal counterpart of `CoExpressionBenchmark`. Co-expression asks whether
two genes move together across conditions, which turned out to say very little
about pathway membership — same-subsystem pairs scored 1.06x random, Cohen's
d = 0.13. Perturb-seq asks a different question: knock the gene down and read
the whole transcriptome. On the same shared-subsystem test that reaches
d = 0.46.

Two scores, and they are not the same measurement:

  PROFILE SIMILARITY   corr(profile_i, profile_j) for two perturbed genes.
                       "knocking these two down has the same downstream
                       consequence", which is the classic Perturb-seq notion
                       of functional relatedness. Needs both genes PERTURBED.

  CROSS EFFECT         |effect[i, j]| — how much knocking down i moves j.
                       A directed, causal statement about one pair, and the
                       thing a correlation cannot express. Needs i perturbed
                       and j measured.

Both are reported against a SIZE-MATCHED RANDOM control drawn from the same
metabolic gene pool, because the co-expression work established that the raw
number is nearly meaningless on its own — random gene sets reached 70-100% of
the pathway score there. `*_excess` is the quantity to read.

Coverage is the other thing to watch. Only 847 of Human-GEM's 2,848 genes have
a perturbation profile that survived Replogle's phenotype filter, so a graph
can easily contribute fewer than two scorable genes; `coverage` says how often
that happened and a high score over three graphs is not a result.
"""

from typing import Optional, Sequence

import json
import numpy as np
import torch
from torch import Tensor

from .base import Benchmark, BenchmarkResult


class PerturbationBenchmark(Benchmark):
    """
    perturb_dir : the `perturb_ref/` written by perturbseq_setup/build_perturbseq.py
    dataset     : supplies `id_to_node`, so node tokens can be resolved to ENSG.
    min_genes   : graphs with fewer scorable gene nodes are skipped, not scored
                  as zero — a mean over one pair is noise, and counting it as a
                  result would hide that.
    n_null      : size-matched random gene sets drawn per graph size, pooled
                  into the control.
    """

    name = "perturbation response"

    def __init__(self, perturb_dir, dataset, min_genes: int = 3,
                 n_null: int = 200, seed: int = 0, pad_id: int = 0,
                 device="cpu", direction: str = "row", mask_self: bool = True):
        from pathlib import Path
        p = Path(perturb_dir)
        meta = json.loads((p / "meta.json").read_text())
        self.E = np.load(p / "effect_matrix.npy").astype(np.float64)
        self.pert = {g: i for i, g in enumerate(meta["genes"])}
        self.meas = {g: j for j, g in enumerate(meta["measured_genes"])}
        self.min_genes, self.n_null, self.seed = min_genes, n_null, seed
        self.pad_id = pad_id
        self.id_to_node = dataset.id_to_node

        # ── which direction of inference ─────────────────────────────────
        #
        # effect_matrix is (n_perturbed x n_measured) and the two axes answer
        # different questions:
        #
        #   "row"     correlate ROWS. Knock down A, knock down B — does the
        #             cell respond the same way? Two genes breaking the same
        #             thing. Scores only the PERTURBED genes.
        #   "column"  correlate COLUMNS. Across the knockdowns, do A and B move
        #             together? Co-regulation, the Perturb-seq analogue of
        #             co-expression. Scores the MEASURED genes, of which there
        #             are far more — which is the practical reason to want it,
        #             since thin coverage is what makes the row direction
        #             unscoreable on most graphs.
        #
        # They are different relations, not two estimates of one quantity, so
        # a disagreement between them is a finding rather than an error.
        if direction not in ("row", "column"):
            raise ValueError("direction must be 'row' or 'column'")
        self.direction = direction

        # A gene that was both perturbed and measured carries its own
        # knockdown in the matrix: ~9x a typical entry and always negative.
        # That is the intervention, not a finding, and it is a large same-sign
        # outlier in every profile containing it — which distorts the COLUMN
        # direction badly, since a column has only n_perturbed entries.
        E = self.E.astype(np.float64).copy()
        self.n_self_masked = 0
        if mask_self:
            for g, i in self.pert.items():
                j = self.meas.get(g)
                if j is not None:
                    E[i, j] = np.nan
                    self.n_self_masked += 1
        self.E_masked = E

        with np.errstate(invalid="ignore"):
            if direction == "row":
                C = E - np.nanmean(E, axis=1, keepdims=True)
                C = C / np.nan_to_num(np.nanstd(E, axis=1, keepdims=True),
                                      nan=1.0).clip(1e-12)
                self.n_readout = E.shape[1]
            else:
                C = E - np.nanmean(E, axis=0, keepdims=True)
                C = C / np.nan_to_num(np.nanstd(E, axis=0, keepdims=True),
                                      nan=1.0).clip(1e-12)
                self.n_readout = E.shape[0]
        # a masked cell contributes nothing to any dot product
        self.Z = np.nan_to_num(C, nan=0.0)
        # the pool for the SCORED axis — perturbed rows, or measured columns
        self.pool = np.array(sorted((self.pert if direction == "row"
                                     else self.meas).values()))
        # ...and one that is always the perturbed axis. `_cross_effect` indexes
        # E's axis 0 whatever the direction, so its null has to be drawn from
        # here rather than from `pool`, which in column mode holds measured
        # indices running past the end of that axis.
        self.pert_pool = np.array(sorted(self.pert.values()))

    # ── the two scores, on one gene set ────────────────────────────────────

    def _pairs(self, rows: np.ndarray) -> np.ndarray:
        """Every within-set pair's profile similarity, not their mean.

        `rows` indexes whichever axis `direction` selected — perturbed-gene
        rows for "row", measured-gene columns for "column"."""
        if len(rows) < 2:
            return np.zeros(0)
        Zi = self.Z[rows] if self.direction == "row" else self.Z[:, rows].T
        S = (Zi @ Zi.T) / self.n_readout
        return S[np.triu_indices(len(rows), 1)]

    def _profile_similarity(self, rows: np.ndarray) -> Optional[float]:
        v = self._pairs(rows)
        return float(v.mean()) if v.size else None

    def _cross_effect(self, rows: np.ndarray, cols: np.ndarray) -> Optional[float]:
        """Mean |effect| over ordered (perturbed, measured) pairs, excluding
        a gene's effect on itself — which is the knockdown, not a finding."""
        if len(rows) < 2 or len(cols) < 2:
            return None
        B = np.abs(self.E[np.ix_(rows, cols)])
        same = rows[:, None] == np.array([self._row_of_col(c) for c in cols])[None, :]
        vals = B[~same]
        return float(vals.mean()) if vals.size else None

    def _row_of_col(self, col: int) -> int:
        if not hasattr(self, "_col2row"):
            inv = {j: g for g, j in self.meas.items()}
            self._col2row = {j: self.pert.get(inv[j], -1) for j in self.meas.values()}
        return self._col2row.get(int(col), -1)

    def _resolve(self, toks: Tensor):
        """node tokens -> (perturbed rows, measured columns), de-duplicated."""
        names = [str(self.id_to_node.get(int(t), "")) for t in toks
                 if int(t) != self.pad_id]
        seen_r, seen_c = {}, {}
        for n in names:
            if n in self.pert:
                seen_r.setdefault(n, self.pert[n])
            if n in self.meas:
                seen_c.setdefault(n, self.meas[n])
        return (np.array(sorted(seen_r.values()), dtype=int),
                np.array(sorted(seen_c.values()), dtype=int))

    # ── the benchmark ──────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, node_toks: Tensor, adjacency: Optional[Tensor] = None,
                 **kw) -> BenchmarkResult:
        rng = np.random.default_rng(self.seed)
        B = node_toks.shape[0]
        sims, xeff, sizes, n_gene, skipped = [], [], [], [], 0
        pert_sizes = []          # perturbed genes per graph, for the cross null
        # Pooled pairs, not per-graph means. Coverage is thin — only 847 of
        # Human-GEM's genes have a profile, so a graph contributes ~2-3
        # scorable genes and its mean is over one or two pairs. Averaging
        # within a graph before comparing throws that power away; pooling every
        # within-graph pair and comparing the DISTRIBUTIONS recovers it, and
        # gives an effect size directly comparable to the d = 0.46 measured on
        # the shared-subsystem test.
        pooled = []

        for b in range(B):
            pert_rows, cols = self._resolve(node_toks[b])
            # TWO DIFFERENT AXES, and conflating them is what broke the column
            # direction: `scored` is whichever axis this direction correlates
            # over, while `pert_rows` always indexes E's perturbed axis. They
            # coincide in row mode, which is why reusing one name for both went
            # unnoticed until column mode indexed axis 0 with a measured id.
            scored = cols if self.direction == "column" else pert_rows
            n_gene.append(len(scored))
            if len(scored) < self.min_genes:
                skipped += 1
                continue
            sizes.append(len(scored))
            pert_sizes.append(len(pert_rows))
            pv = self._pairs(scored)
            if pv.size:
                pooled.append(pv)
            s = self._profile_similarity(scored)
            if s is not None:
                sims.append(s)
            # cross-effect is inherently a (perturbed x measured) quantity, so
            # it keeps the perturbed rows in BOTH directions. In column mode a
            # graph often has fewer than two perturbed genes and this returns
            # None — thin coverage, reported as such, not a wrong number.
            x = self._cross_effect(pert_rows, cols)
            if x is not None:
                xeff.append(x)

        # size-matched null: the same set sizes, genes drawn from the same
        # metabolic pool. Without this the numbers are uninterpretable — see
        # the module docstring.
        null_s, null_x = [], []
        for k, kp in zip(sizes, pert_sizes) if sizes else []:
            for _ in range(max(1, self.n_null // max(len(sizes), 1))):
                r = rng.choice(self.pool, size=min(k, len(self.pool)), replace=False)
                v = self._profile_similarity(r)
                if v is not None:
                    null_s.append(v)
                # In row mode `r` is already a perturbed draw and is reused, so
                # the RNG stream — and therefore every number this direction
                # reported before the fix — is unchanged. Column mode needs its
                # own draw, size-matched to the perturbed genes actually seen.
                if self.direction == "row":
                    rp = r
                elif kp >= 2:
                    rp = rng.choice(self.pert_pool,
                                    size=min(kp, len(self.pert_pool)),
                                    replace=False)
                else:
                    continue
                c = np.array([self.meas[g] for g, i in self.pert.items()
                              if i in set(rp.tolist()) and g in self.meas], dtype=int)
                if len(c) >= 2:
                    w = self._cross_effect(rp, c)
                    if w is not None:
                        null_x.append(w)

        def agg(obs, null):
            if not obs:
                return float("nan"), float("nan"), float("nan"), float("nan")
            o = float(np.mean(obs))
            if not null:
                return o, float("nan"), float("nan"), float("nan")
            n = np.asarray(null)
            z = (o - n.mean()) / max(n.std(), 1e-12)
            p = (1 + (n >= o).sum()) / (len(n) + 1)
            return o, float(n.mean()), float(z), float(p)

        s_o, s_n, s_z, s_p = agg(sims, null_s)
        x_o, x_n, x_z, x_p = agg(xeff, null_x)

        # pooled-pair statistics: the powered version of the same question
        obs_p = np.concatenate(pooled) if pooled else np.zeros(0)
        null_p = []
        for k in (sizes if sizes else []):
            r = rng.choice(self.pool, size=min(k, len(self.pool)), replace=False)
            null_p.append(self._pairs(r))
        null_p = np.concatenate(null_p) if null_p else np.zeros(0)
        if obs_p.size > 1 and null_p.size > 1:
            both = np.concatenate([obs_p, null_p])
            pooled_d = float((obs_p.mean() - null_p.mean()) / max(both.std(), 1e-12))
            allv = np.concatenate([obs_p, null_p])
            o = np.argsort(allv); rk = np.empty(len(allv)); rk[o] = np.arange(1, len(allv) + 1)
            n1 = len(obs_p)
            pooled_auc = float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(null_p)))
        else:
            pooled_d = pooled_auc = float("nan")

        summary = {
            "direction": self.direction,
            # NOTE: averaged over ALL graphs, including those skipped for
            # having too few — the SCORED graphs all have >= min_genes by
            # construction. `genes_per_scored_graph` is the other one.
            "genes_per_graph": float(np.mean(n_gene)) if n_gene else 0.0,
            "genes_per_scored_graph": float(np.mean(sizes)) if sizes else 0.0,
            "n_self_masked": float(self.n_self_masked),
            "n_pairs_pooled": float(obs_p.size),
            "pooled_d": pooled_d,
            "pooled_auc": pooled_auc,
            "profile_similarity": s_o,
            "profile_similarity_null": s_n,
            "profile_similarity_excess": s_o - s_n,
            "profile_similarity_z": s_z,
            "profile_similarity_p": s_p,
            "cross_effect": x_o,
            "cross_effect_null": x_n,
            "cross_effect_excess": x_o - x_n,
            "cross_effect_z": x_z,
        }
        notes = []
        if skipped:
            notes.append(f"{skipped} graph(s) skipped for having < {self.min_genes} "
                         f"genes with a perturbation profile")
        if len(sims) and len(sims) < 8:
            notes.append(f"only {len(sims)} graphs scored — treat the excess as "
                         f"indicative, not measured")
        return BenchmarkResult(
            name=self.name, values=None, summary=summary,
            n_graphs=B, n_scored=len(sims),
            per_graph={"profile_similarity": torch.tensor(sims, dtype=torch.float),
                       "genes_scored": torch.tensor(sizes, dtype=torch.float)},
            distributions={"pairs": obs_p, "pairs_null": null_p,
                           "similarity": np.asarray(sims),
                           "similarity_null": np.asarray(null_s),
                           "cross_effect": np.asarray(xeff),
                           "cross_effect_null": np.asarray(null_x)},
            notes=notes)


def perturbation_table(results: dict) -> str:
    """Several BenchmarkResults side by side, excess first."""
    names = list(results)
    keys = [("pooled_d", "POOLED Cohen's d"),
            ("pooled_auc", "  pooled AUC"),
            ("n_pairs_pooled", "  gene pairs pooled"),
            ("profile_similarity", "profile similarity"),
            ("profile_similarity_null", "  size-matched null"),
            ("profile_similarity_excess", "  EXCESS"),
            ("profile_similarity_z", "  z"),
            ("cross_effect", "cross effect |E|"),
            ("cross_effect_null", "  size-matched null"),
            ("cross_effect_excess", "  EXCESS"),
            ("genes_per_graph", "genes per graph")]
    w = max(len(n) for n in names) + 2
    out = [f"{'':<26}" + "".join(f"{n:>{w}}" for n in names)]
    for k, lab in keys:
        out.append(f"{lab:<26}" +
                   "".join(f"{results[n].summary.get(k, float('nan')):>{w}.4f}"
                           for n in names))
    out.append(f"{'graphs scored':<26}" +
               "".join(f"{results[n].n_scored:>{w}}" for n in names))
    return "\n".join(out)
