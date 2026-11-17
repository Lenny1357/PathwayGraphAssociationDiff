"""Is the conditioning doing anything?

The trap: condition a model on co-expression, then score it with the
co-expression benchmark, and a high number may only mean the model echoed
its own input. Everything here is built to avoid that.

The primary test is RELATIVE. Generate conditioned on tissue T, then score
the graph against every tissue and ask whether T wins. A model emitting
globally co-expressed genes scores well everywhere and fails; only output
that is specific to the tissue it was told about can succeed. Chance is
1/n_tissues.
"""

from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor


def assign_tissues(expr_loader, node_toks: Tensor,
                   allowed: Optional[Sequence[int]] = None) -> Tensor:
    """
    Label each graph with the tissue its nodes are most expressed in.

    Used to give training graphs a conditioning label when the dataset does
    not carry one. `allowed` restricts the choice — that is how a held-out
    tissue is kept out of training.
    """
    expr_all = expr_loader.expression_stack[:, node_toks.to(expr_loader.device)]
    sums = expr_all.permute(1, 0, 2).sum(dim=2)              # (B, n_tissues)
    if allowed is not None:
        mask = torch.full((sums.shape[1],), float("-inf"), device=sums.device)
        mask[torch.as_tensor(list(allowed), device=sums.device)] = 0.0
        sums = sums + mask
    return sums.argmax(dim=1)


def tissue_recovery(result, tissue_ids: Tensor, tissues: Sequence[str]) -> dict:
    """
    Can the conditioning tissue be read back off the generated graph?

    `result` is a CoExpressionBenchmark BenchmarkResult; `tissue_ids` are the
    conditioning labels for the graphs that were handed to it. Graphs the
    benchmark skipped are dropped here too, via its `graph_index`.
    """
    pg = result.per_graph
    if "per_tissue_coexpr" not in pg:
        raise ValueError("result carries no per-tissue scores")
    scores = pg["per_tissue_coexpr"].cpu()                   # (n_scored, T)
    idx = pg.get("graph_index")
    labels = tissue_ids.cpu()[idx.cpu()] if idx is not None else tissue_ids.cpu()

    # NaN wins argmax but loses every comparison, so an unsanitised score
    # matrix reports high accuracy alongside zero margin. Treat it as absent.
    n_nan = int(torch.isnan(scores).sum())
    scores = torch.nan_to_num(scores, nan=float("-inf"))
    all_bad = torch.isinf(scores).all(dim=1)
    pred = scores.argmax(1)
    hit = (pred == labels) & ~all_bad
    T = len(tissues)
    conf = np.zeros((T, T), dtype=int)
    for a, b in zip(labels.tolist(), pred.tolist()):
        conf[a, b] += 1

    # the margin between the conditioned tissue and the best alternative:
    # accuracy alone hides whether it barely won or won clearly
    own = scores.gather(1, labels[:, None]).squeeze(1)
    other = scores.clone()
    other.scatter_(1, labels[:, None], float("-inf"))
    margin = own - other.max(1).values

    return {"accuracy": float(hit.float().mean()),
            "n_nan_scores": n_nan,
            "n_unscoreable": int(all_bad.sum()),
            "chance": 1.0 / T,
            "n": int(len(labels)),
            "margin_mean": float(margin.mean()),
            "margin_positive": float((margin > 0).float().mean()),
            "confusion": conf,
            "per_graph_hit": hit,
            "tissues": list(tissues)}


def recovery_table(arms: dict) -> str:
    """Render several tissue_recovery results side by side."""
    names = list(arms)
    rows = [("accuracy", "accuracy"), ("chance", "chance"),
            ("margin_mean", "score margin"), ("margin_positive", "share T wins"),
            ("n", "graphs")]
    w = 16
    out = [f"{'metric':<16}" + "".join(f"{n:>16}" for n in names),
           "-" * (16 + w * len(names))]
    for key, label in rows:
        line = f"{label:<16}"
        for n in names:
            v = arms[n][key]
            line += f"{v:>16.3f}" if isinstance(v, float) else f"{v:>16}"
        out.append(line)
    return "\n".join(out)
