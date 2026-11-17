"""How each model builds a graph: the reverse trajectory, and the absorbing
arm's reveal / remask accounting.

    trajectories.json  {"dataset|arm": {"t": [...], "density": [...], "density_resolved",
                        "edge_masked", "edge_reveal_present", "masked", "node_churn",
                        "edge_churn", "panels": [{t, nodes, edges}, ...], "mask_id", "T"}}
    remask.json        {"dataset": {per_step, node_remask_count, ..., "cut": t}}

Snapshots are taken at EVERY step, so churn is exact. `panels` keeps a few
full states for drawing the graph as it evolves.
"""

from __future__ import annotations

import numpy as np
import torch

from graphpath.pipeline import arms as A
from graphpath.pipeline.common import DatasetBundle, RunLayout, dump_json, free, key
from graphpath.processes.marginal import _mirror_upper

EDGE_MASK_ID = 2          # masked model: 0 = absent, 1 = present, 2 = <MASK>
TRAJ_ARMS = ("masked", "transition", "continuous")   # the VAE has no trajectory


@torch.no_grad()
def _traj_callback(model, N, device, seed, graph=0, **kw):
    snaps = []
    def cb(t, n, e):
        snaps.append((int(t), n[graph].detach().cpu().clone(), e[graph].detach().cpu().clone()))
    torch.manual_seed(seed)
    n_, a_ = model.sample(N=N, B=1, device=device, callback=cb, **kw)
    return n_, a_, snaps


@torch.no_grad()
def _traj_marginal(model, N, device, seed):
    """`sample_marginal` with intermediate states kept. A hand copy of the
    sampler loop, so `collect` asserts it reproduces model.sample() exactly."""
    cfg, T = model.cfg, model.cfg.T
    diag = torch.eye(N, device=device, dtype=torch.bool)
    torch.manual_seed(seed)
    shift = model._edge_logit_shift()
    node_t = model.schedule.sample_prior(model.m_node, (1, N), device)
    edge_t = model.schedule.sample_prior(model.m_edge, (1, N, N), device)
    edge_t = _mirror_upper(edge_t).masked_fill(diag[None], 0)
    snaps = [(T, node_t[0].cpu().clone(), edge_t[0].cpu().clone())]
    for t_val in range(T - 1, -1, -1):
        t = torch.full((1,), t_val, dtype=torch.long, device=device)
        nl, el = model.forward(node_t, edge_t, t, t)
        if t_val > 0:
            node_t = torch.distributions.Categorical(logits=model.schedule.posterior_logits(
                nl, node_t, t_val, model.m_node, exact=cfg.exact_posterior)).sample()
        else:
            node_t = nl.argmax(-1)
        if shift is not None:
            el = el - shift
        if t_val > 0:
            NN = N * N
            ple = model.schedule.posterior_logits(el.reshape(NN, model.edge_vocab), edge_t.reshape(NN),
                                                  t_val, model.m_edge, exact=cfg.exact_posterior
                                                  ).reshape(1, N, N, model.edge_vocab)
            edge_t = torch.distributions.Categorical(logits=ple).sample()
        else:
            edge_t = el.argmax(-1)
        if cfg.symmetric_edges:
            edge_t = _mirror_upper(edge_t)
        edge_t = edge_t.masked_fill(diag[None], 0)
        snaps.append((t_val, node_t[0].cpu().clone(), edge_t[0].cpu().clone()))
    return node_t, edge_t, snaps


def traces(snaps, mask_id, N) -> dict:
    """Four traces per step. For the absorbing arm 0 means ABSENT and 2 means
    NOT YET DECIDED: `density` counts confirmed edges over all pairs,
    `density_resolved` is P(edge | decided) — the cross-arm comparable one."""
    iu = torch.triu_indices(N, N, 1)
    ts, dens, dens_res, masked, e_masked, nch, ech, rev_pres = ([] for _ in range(8))
    prev = None
    for t, n, e in snaps:
        up = e[iu[0], iu[1]]
        pres, und = (up == 1), (up == EDGE_MASK_ID)
        dec = ~und
        ts.append(max(t, 0))
        dens.append(float(pres.float().mean()))
        dens_res.append(float(pres[dec].float().mean()) if dec.any() else np.nan)
        e_masked.append(float(und.float().mean()))
        masked.append(float((n == mask_id).float().mean()) if mask_id is not None else np.nan)
        if prev is None:
            nch.append(np.nan); ech.append(np.nan); rev_pres.append(np.nan)
        else:
            nch.append(float((prev[0] != n).float().mean()))
            ech.append(float((prev[1] != up).float().mean()))
            newly = (prev[1] == EDGE_MASK_ID) & dec
            rev_pres.append(float(pres[newly].float().mean()) if newly.any() else np.nan)
        prev = (n, up)
    return dict(t=ts, density=dens, density_resolved=dens_res, edge_masked=e_masked,
                edge_reveal_present=rev_pres, masked=masked, node_churn=nch, edge_churn=ech)


def collect(run: RunLayout, dname: str, kind: str, variant, device, N: int, seed: int = 7,
            n_panels: int = 8, size_prior=None) -> dict:
    model, _ = A.load_arm(run, dname, kind, variant, device)
    if kind == "transition" and device.type == "mps":
        model = model.to("cpu")
    dev = next(model.parameters()).device
    mask_id = getattr(model, "node_mask", None)
    if kind == "transition":
        n_, a_, snaps = _traj_marginal(model, N, dev, seed)
        torch.manual_seed(seed)
        with torch.no_grad():
            rn, ra = model.sample(N=N, B=1, device=dev)
        reproduces = bool((rn.cpu() == n_.cpu()).all() and (ra.cpu() == a_.cpu()).all())
    else:
        # the masked arm is drawn as it was scored (size clamp for the pad variant). The
        # transition trace is a hand copy of its sampler without node_clamp, so it stays unclamped.
        kw, _ = (A.clamp_kwargs(kind, variant, size_prior, 1, N, seed, A.pad_scale_for(run, dname)) if size_prior is not None else ({}, None))
        n_, a_, snaps = _traj_callback(model, N, dev, seed, **A._to(kw, dev))
        reproduces = True
    out = traces(snaps, mask_id, N)
    pick = np.linspace(0, len(snaps) - 1, n_panels).astype(int)
    out["panels"] = [dict(t=int(snaps[i][0]), nodes=snaps[i][1].tolist(),
                          edges=snaps[i][2].to(torch.int8).tolist()) for i in pick]
    out.update(mask_id=mask_id, T=int(model.cfg.T), n_snapshots=len(snaps),
               reproduces_sample=reproduces, seed=seed)
    free(model)
    return out


def trajectories(run: RunLayout, data: dict, device, max_nodes: int = 64, seed: int = 7,
                 arms=TRAJ_ARMS) -> dict:
    out = {}
    for d, arm, v in run.all_weights():
        if d not in data or arm not in arms:
            continue
        try:
            tr = collect(run, d, arm, v, device, max_nodes, seed, size_prior=data[d].sizes())
        except Exception as e:  # noqa: BLE001
            print(f"  {key(d, arm, v):<22} trajectory FAILED ({type(e).__name__}: {e})")
            continue
        out[key(d, arm, v)] = tr
        late = np.asarray(tr["t"]) <= max(1, int(0.1 * max(tr["t"])))
        print(f"  {key(d, arm, v):<22}{tr['n_snapshots']:>4} snapshots  final density "
              f"{tr['density'][-1]:.4f} (data {data[d].density:.4f})  late node churn "
              f"{np.nanmean(np.asarray(tr['node_churn'], float)[late]):.3f}"
              + ("" if tr["reproduces_sample"] else "  ! does NOT reproduce model.sample()"))
    dump_json(out, run.analysis_dir() / "trajectories.json")
    return out


@torch.no_grad()
def remask(run: RunLayout, data: dict, device, max_nodes: int = 64, B: int = 8, seed: int = 0) -> dict:
    """model.sample(stats=...) on the absorbing arm: reveals, remasks and how
    often a remasked slot came back as a different token (the work) rather
    than the same one (a wasted step)."""
    out = {}
    for d, arm, v in run.all_weights():
        if d not in data or arm != "masked":
            continue
        model, _ = A.load_arm(run, d, arm, v, device)
        stats = {}
        torch.manual_seed(seed)
        kw, _ = A.clamp_kwargs(arm, v, data[d].sizes(), B, max_nodes, seed, A.pad_scale_for(run, d))
        model.sample(N=max_nodes, B=B, device=device, stats=stats, **A._to(kw, device))
        cfg = model.cfg
        out[key(d, arm, v)] = dict(
            per_step=stats["per_step"], node_remask_count=stats["node_remask_count"].tolist(),
            node_reveal_count=stats["node_reveal_count"].tolist(),
            node_remask_total=stats["node_remask_total"], node_reveal_total=stats["node_reveal_total"],
            edge_remask_total=stats["edge_remask_total"], revisions_changed=stats["revisions_changed"],
            revisions_same=stats["revisions_same"], still_pending=stats["still_pending"],
            T=int(cfg.T), remask_frac=cfg.remask_frac, cut=float(cfg.remask_stop_frac * cfg.T), B=B)
        r = out[key(d, arm, v)]
        redone = r["revisions_changed"] + r["revisions_same"]
        print(f"  {key(d, arm, v):<22} remasks {r['node_remask_total']:,} ({r['node_remask_total'] / B:.1f}/graph)"
              f"  changed on return {r['revisions_changed']}/{redone}")
        free(model)
    dump_json(out, run.analysis_dir() / "remask.json")
    return out
