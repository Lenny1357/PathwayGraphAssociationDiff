"""Conditioned completion: reveal K nodes of a held-out pathway, hand the
model an evidence network, ask for the rest.

The pipeline behind conditioning_test.ipynb and conditioning_neighbourhood.ipynb.

    channels   gene-gene evidence (co-expression, Perturb-seq response) propagated
               over the metabolic graph into one sparse (n_channels, V, V) tensor
    arms       masked-plain / masked-cond / masked-nbr / transition-plain
    completion node recall of the hidden nodes, edge F1 among recovered nodes,
               novel fraction — all against a frequency floor, revealed nodes excluded
    validate   wiring (zero-init equals base), ablation (encoder on/off on the same
               weights), attribution (did the attended partners appear)

Outputs: run/weights/<dataset>_<arm>.pt, run/cond/<dataset>_channels.pt,
run/history.json, analysis/<run>/{completion,ablation,attribution,wiring}.json.
"""

from __future__ import annotations

import collections
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from graphpath import Config, GraphDiffBERT
from graphpath.benchmarks import ReferenceNetwork
from graphpath.data.propagation import CoExpressionPropagator
from graphpath.layers.neighbourhood import NeighbourhoodConditioned
from graphpath.pipeline import arms as A
from graphpath.pipeline.common import DatasetBundle, RunLayout, dump_json, free, load_bundle
from graphpath.training.loop import train_step

CHANNELS = ["coexpr", "perturb", "both"]


@dataclass
class ConditioningConfig:
    datasets: list = field(default_factory=lambda: ["human"])
    arms: list = field(default_factory=lambda: ["masked-plain", "masked-cond", "transition-plain"])
    steps: int = 20_000
    batch_size: int = 32
    max_nodes: int = 64
    lr: float = 3e-4
    seed: int = 0
    cond_dropout_p: float = 0.15
    hops: int = 2
    decay: float = 0.5
    cond_topk: int = 16
    k_revealed: int = 4
    n_eval: int = 64
    eval_seed: int = 0
    guidance: list = field(default_factory=lambda: [1.0, 1.5, 2.0, 3.0])
    # the seed-neighbourhood variant
    rand_dim: int = 32
    nbr_dim: int = 32
    nbr_hops: int = 2
    nbr_heads: int = 4
    k_train: int = 4


class NbrGraphDiffBERT(NeighbourhoodConditioned, GraphDiffBERT):
    """GraphDiffBERT whose forward fills node_id_feat from the seed context."""
    _seed_ctx = None

    def set_seed_context(self, known_nodes, channel_ids):
        self._seed_ctx = None if known_nodes is None else (known_nodes, channel_ids)

    def forward(self, node_toks, edge_toks, t_node, t_edge, node_id_feat=None, **kw):
        if self._seed_ctx is not None and node_id_feat is None:
            known, chan = self._seed_ctx
            if known.shape[0] != node_toks.shape[0]:
                known, chan = known[: node_toks.shape[0]], chan[: node_toks.shape[0]]
            node_id_feat, pooled = self.build_inputs(node_toks, known, chan)
            if kw.get("cond_pooled") is None:
                kw["cond_pooled"] = pooled
        return super().forward(node_toks, edge_toks, t_node, t_edge, node_id_feat=node_id_feat, **kw)


# ── channels ──────────────────────────────────────────────────────────────

def _rank_norm(vals):
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return v.astype(np.float32)
    r = np.empty(v.size); r[v.argsort()] = np.arange(v.size)
    return (r / max(v.size - 1, 1)).astype(np.float32)


def build_channels(bundle: DatasetBundle, ref: ReferenceNetwork, cfg: ConditioningConfig) -> Optional[dict]:
    """{"tensor": sparse (C, V, V), "channels": [...], "V": V, "log": [...]}.
    Each source is propagated over the metabolic graph, rank-normalised to
    [0, 1], and `both` is the elementwise max of the two."""
    V, log, per = bundle.vocab_size, [], {}
    node_of = {v: k for k, v in bundle.id_to_node.items()}

    def propagated(gene_gene):
        return CoExpressionPropagator(ref.adj_sparse, gene_gene, vocab_size=V, hops=cfg.hops,
                                      decay=cfg.decay, normalize="sym").fit().top_k_network(k=cfg.cond_topk).coalesce()

    from graphpath.pipeline.score import expression_source
    ex, t0 = expression_source(bundle, "cpu")
    if ex is not None:
        per["coexpr"] = propagated(ex.co_expression[t0])      # one tissue: the source is what varies
        log.append(f"coexpr: tissue {t0}, {per['coexpr']._nnz():,} propagated entries")
    else:
        log.append("coexpr: no expression source")
    pdir = Path(bundle.perturb) if bundle.perturb else None
    rpath = pdir / "similarity_response.npz" if pdir else None
    if rpath and rpath.exists():
        z = np.load(rpath, allow_pickle=True)
        meta = json.loads((pdir / "meta.json").read_text())
        names = [str(x) for x in z["names"]] if "names" in z.files else meta["response_genes"]
        ri, rj, rv, miss = [], [], [], 0
        for a, b_, w in zip(z["row"], z["col"], z["val"]):
            ua, ub = node_of.get(names[int(a)]), node_of.get(names[int(b_)])
            if ua is None or ub is None or ua >= V or ub >= V:
                miss += 1; continue
            ri.append(ua); rj.append(ub); rv.append(abs(float(w)))
        log.append(f"perturb: {len(rv):,} of {len(z['val']):,} edges mapped into node space ({miss:,} dropped)")
        if rv:
            gg = torch.sparse_coo_tensor(torch.tensor(np.stack([ri, rj])), torch.tensor(rv, dtype=torch.float32),
                                         size=(V, V)).coalesce()
            per["perturb"] = propagated(gg)
    else:
        log.append("perturb: no similarity_response.npz")
    if not per:
        return None
    normed = {k: torch.sparse_coo_tensor(m.indices(), torch.tensor(_rank_norm(m.values().cpu().numpy())),
                                         size=(V, V)).coalesce() for k, m in per.items()}
    if len(normed) > 1:
        st = torch.sparse_coo_tensor(torch.cat([m.indices() for m in normed.values()], 1),
                                     torch.cat([m.values() for m in normed.values()]), size=(V, V)).coalesce()
        normed["both"] = torch.sparse_coo_tensor(st.indices(), st.values().clamp(max=1.0), size=(V, V)).coalesce()
    order = [c for c in CHANNELS if c in normed]
    idx, val = [], []
    for ci, c in enumerate(order):
        u, v_ = normed[c].indices()
        idx.append(torch.stack([torch.full_like(u, ci), u, v_])); val.append(normed[c].values())
    tens = torch.sparse_coo_tensor(torch.cat(idx, 1), torch.cat(val), size=(len(order), V, V)).coalesce()
    return {"tensor": tens, "channels": order, "V": V, "log": log}


# ── arms ──────────────────────────────────────────────────────────────────

def build_arm(kind: str, bundle: DatasetBundle, cfg: ConditioningConfig, n_channels: int, device):
    V = bundle.vocab_size
    if kind == "transition-plain":
        jc = A.JointConfig(**A.SHARED, **A.TRANSITION)
        from graphpath.models.joint import GraphDiffBERTJoint
        from graphpath.processes.marginal import estimate_marginals
        loader = torch.utils.data.DataLoader(bundle.train, batch_size=cfg.batch_size)
        m_node, m_edge = estimate_marginals(loader, node_vocab=V, edge_vocab=2)
        return GraphDiffBERTJoint(jc, vocab_size=V + 1, m_node=m_node, m_edge=m_edge).to(device)
    cond = kind in ("masked-cond", "masked-nbr")
    nbr = kind == "masked-nbr"
    c = Config(**A.SHARED, use_coexpr=cond, n_tissues=max(n_channels, 1), cond_d_model=64,
               cond_dropout_p=cfg.cond_dropout_p, use_pairwise_bias=cond, pairwise_hidden=32,
               use_coexpr_attn=cond, coexpr_attn_k=cfg.cond_topk, coexpr_attn_heads=4,
               coexpr_attn_layers="all", use_random_node_features=True,
               rand_feat_dim=cfg.rand_dim + (cfg.nbr_dim if nbr else 0))
    m = (NbrGraphDiffBERT if nbr else GraphDiffBERT)(c, vocab_size=V + 1).to(device)
    if nbr:
        m.attach_neighbourhood(hops=cfg.nbr_hops, n_heads=cfg.nbr_heads, nbr_dim=cfg.nbr_dim)
    return m


# ── the experiment ────────────────────────────────────────────────────────

class ConditioningExperiment:
    def __init__(self, cfg: ConditioningConfig, run: RunLayout, device):
        self.cfg, self.run, self.device = cfg, run, device
        self.data, self.ref, self.cond = {}, {}, {}
        for sub in ("weights", "cond", "tables"):
            (run.path / sub).mkdir(parents=True, exist_ok=True)
        self.out = run.analysis_dir()

    def load_data(self):
        for n in self.cfg.datasets:
            b = load_bundle(n, self.cfg.max_nodes, device=self.device)
            ref = ReferenceNetwork(b.train.node_to_id, device="cpu", num_nodes=b.vocab_size)
            ref.load_edges(b.edges); ref.build_adjacency()
            self.data[n], self.ref[n] = b, ref
            print(f"  {n:<7} train {len(b.train_graphs):,} / test {len(b.test_graphs):,}  vocab {b.vocab_size:,}")
        return self.data

    def channels(self):
        for n, b in self.data.items():
            cache = self.run.path / "cond" / f"{n}_channels.pt"
            if cache.exists():
                self.cond[n] = torch.load(cache, map_location="cpu", weights_only=False)
            else:
                c = build_channels(b, self.ref[n], self.cfg)
                if c is None:
                    print(f"  {n}: no conditioning source — conditioned arms will be skipped"); continue
                torch.save(c, cache); self.cond[n] = c
            print(f"  {n}: channels {self.cond[n]['channels']} ({self.cond[n]['tensor']._nnz():,} entries)")
        return self.cond

    def _wpath(self, dname, kind):
        return self.run.path / "weights" / f"{dname}_{kind}.pt"

    def _draw_seeds(self, nf, k, gen):
        known = torch.full_like(nf, -1)
        for b in range(nf.shape[0]):
            real = torch.nonzero(nf[b] != 0).flatten()
            if real.numel():
                sel = real[torch.randperm(real.numel(), generator=gen, device=real.device)[:k]]
                known[b, sel] = nf[b, sel]
        return known

    def train(self) -> dict:
        cfg, device, hist_all = self.cfg, self.device, {}
        for dname, b in self.data.items():
            V = b.vocab_size
            chans = self.cond.get(dname, {}).get("channels", []); cond_t = self.cond.get(dname, {}).get("tensor")
            for kind in cfg.arms:
                wpath = self._wpath(dname, kind)
                if wpath.exists():
                    hist_all[f"{dname}|{kind}"] = torch.load(wpath, map_location="cpu", weights_only=False).get("history", [])
                    print(f"  {dname}/{kind:<17} already trained — loaded"); continue
                cond_on = kind in ("masked-cond", "masked-nbr") and cond_t is not None
                if kind == "masked-nbr" and not cond_on:
                    print(f"  ! {dname}/{kind}: no evidence tensor — skipped"); continue
                model = build_arm(kind, b, cfg, len(chans) or 1, device)
                if cond_on:
                    model.set_coexpression(cond_t.to(device), top_k=cfg.cond_topk, verbose=False)
                if kind == "transition-plain" and device.type == "mps":
                    model = model.to("cpu")
                dev = next(model.parameters()).device
                opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
                g = torch.Generator().manual_seed(cfg.seed); gch = torch.Generator().manual_seed(cfg.seed)
                gseed = torch.Generator(device=dev).manual_seed(cfg.seed) if dev.type != "mps" else torch.Generator().manual_seed(cfg.seed)
                loader = torch.utils.data.DataLoader(b.train, batch_size=cfg.batch_size, shuffle=True, generator=g)
                it, hist, t0 = iter(loader), [], time.time()
                for step in range(cfg.steps):
                    try:
                        batch = next(it)
                    except StopIteration:
                        it = iter(loader); batch = next(it)
                    adj, nf, msk = batch[0].long().to(dev), batch[1].long().to(dev), batch[2].bool().to(dev)
                    co = None
                    if cond_on:   # a random channel per graph, so single-source channels are in-distribution
                        ids = torch.randint(0, len(chans), (nf.shape[0],), generator=gch).to(dev)
                        co = {"coexpr_sparse": cond_t, "tissue_ids": ids}
                    nbr_on = kind == "masked-nbr" and cond_on
                    if nbr_on:
                        model.set_seed_context(self._draw_seeds(nf, cfg.k_train, gseed), ids)
                    out = train_step(model, opt, nf, adj, V, V + 1, mask=msk, co_expr=co)
                    if nbr_on:
                        model.set_seed_context(None, None)
                    if step % 100 == 0 or step == cfg.steps - 1:
                        hist.append({"step": step, **{k: float(v) for k, v in out.items() if isinstance(v, (int, float))}})
                    if step % 2000 == 0:
                        print(f"    {kind} step {step:>6}  loss {out['loss']:.4f}  ({(time.time() - t0) / 60:.1f}m)")
                torch.save({"state_dict": model.state_dict(), "cfg": asdict(model.cfg), "history": hist,
                            "steps": cfg.steps, "kind": kind, "dataset": dname, "channels": chans,
                            "conditioned": bool(cond_on), "neighbourhood": kind == "masked-nbr",
                            "options": asdict(cfg)}, wpath)
                hist_all[f"{dname}|{kind}"] = hist
                print(f"  {dname}/{kind:<17} done in {(time.time() - t0) / 60:.1f}m")
                free(model, opt)
        dump_json(hist_all, self.run.path / "history.json")
        self.run.write_manifest(config=asdict(self.cfg), kind="conditioning", datasets=list(self.data))
        return hist_all

    def load_model(self, dname, kind):
        b = self.data[dname]
        chans = self.cond.get(dname, {}).get("channels", []); cond_t = self.cond.get(dname, {}).get("tensor")
        ck = torch.load(self._wpath(dname, kind), map_location="cpu", weights_only=False)
        m = build_arm(kind, b, self.cfg, len(chans) or 1, self.device)
        m.load_state_dict(ck["state_dict"]); m.eval()
        if kind in ("masked-cond", "masked-nbr") and cond_t is not None:
            m.set_coexpression(cond_t.to(self.device), top_k=self.cfg.cond_topk, verbose=False)
        if kind == "transition-plain" and self.device.type == "mps":
            m = m.to("cpu")
        return m

    # ── the completion task ──────────────────────────────────────────────
    def reveal(self, ids, k, seed=None):
        g = torch.Generator().manual_seed(self.cfg.eval_seed if seed is None else seed)
        known = torch.full_like(ids, -1); rev = torch.zeros_like(ids, dtype=torch.bool)
        for b in range(ids.shape[0]):
            real = torch.nonzero(ids[b] != 0).flatten()
            if real.numel():
                sel = real[torch.randperm(real.numel(), generator=g)[:k]]
                known[b, sel] = ids[b, sel]; rev[b, sel] = True
        return known, rev

    @staticmethod
    def score(gen_ids, gen_adj, true_ids, true_adj, rev) -> pd.DataFrame:
        """Per graph: recall of the hidden nodes, edge F1 among recovered nodes,
        share of completed nodes not in the true graph. Revealed nodes excluded."""
        out = []
        for b in range(true_ids.shape[0]):
            true_set = {int(v) for v in true_ids[b] if int(v) != 0}
            rev_set = {int(v) for v in true_ids[b][rev[b]]}
            hidden = true_set - rev_set
            gen_set = {int(v) for v in gen_ids[b] if int(v) != 0} - rev_set
            if not hidden:
                continue
            hit = hidden & gen_set
            t_pos = {i for i, v in enumerate(true_ids[b]) if int(v) in hit}
            g_pos = {i for i, v in enumerate(gen_ids[b]) if int(v) in hit}
            te = {(i, j) for i in t_pos for j in t_pos if i < j and true_adj[b, i, j] > 0}
            ge = {(i, j) for i in g_pos for j in g_pos if i < j and gen_adj[b, i, j] > 0}
            inter = len(te & ge)
            p = inter / len(ge) if ge else np.nan; r = inter / len(te) if te else np.nan
            f1 = 2 * p * r / (p + r) if (np.isfinite(p) and np.isfinite(r) and p + r > 0) else np.nan
            out.append(dict(node_recall=len(hit) / len(hidden), n_hidden=len(hidden), n_completed=len(gen_set),
                            novel_frac=(1 - len(hit) / len(gen_set)) if gen_set else np.nan, edge_f1=f1))
        return pd.DataFrame(out)

    def frequency_floor(self, b: DatasetBundle, true_ids, rev):
        cnt = collections.Counter(int(v) for g in b.train_graphs for v in g["node_ids"][:g["num_nodes"]] if int(v) != 0)
        common = [v for v, _ in cnt.most_common()]
        out = true_ids.clone()
        for g in range(true_ids.shape[0]):
            slots = torch.nonzero(~rev[g] & (true_ids[g] != 0)).flatten()
            revealed = {int(x) for x in true_ids[g][rev[g]]}
            for s, v in zip(slots, (v for v in common if v not in revealed)):
                out[g, s] = v
        return out, torch.zeros_like(true_ids[:, :, None].expand(-1, -1, true_ids.shape[1]))

    @torch.no_grad()
    def complete(self) -> dict:
        cfg, out = self.cfg, {}
        for dname, b in self.data.items():
            chans = self.cond.get(dname, {}).get("channels", []); cond_t = self.cond.get(dname, {}).get("tensor")
            true_ids, true_adj = b.heldout_tensors(cfg.n_eval, cfg.max_nodes, seed=cfg.eval_seed)
            known, rev = self.reveal(true_ids, cfg.k_revealed)
            rows = {}
            f_ids, f_adj = self.frequency_floor(b, true_ids, rev)
            rows["freq"] = self.score(f_ids, f_adj, true_ids, true_adj, rev)
            for kind in cfg.arms:
                if not self._wpath(dname, kind).exists():
                    continue
                model = self.load_model(dname, kind)
                dev = next(model.parameters()).device

                def run(**kw):
                    torch.manual_seed(cfg.eval_seed)
                    if kind == "transition-plain":
                        nv = torch.where(rev, true_ids, torch.zeros_like(true_ids))
                        n_, a_ = model.sample(N=cfg.max_nodes, B=true_ids.shape[0], device=dev,
                                              node_clamp=(rev.to(dev), nv.to(dev)))
                    else:
                        n_, a_ = model.sample(N=cfg.max_nodes, B=true_ids.shape[0], device=dev,
                                              known_nodes=known.to(dev), **kw)
                    return n_.long().cpu(), (a_ > 0).long().cpu()

                if kind in ("masked-cond", "masked-nbr") and cond_t is not None:
                    for ci, ch in enumerate(chans):
                        ids = torch.full((true_ids.shape[0],), ci, dtype=torch.long, device=dev)
                        if kind == "masked-nbr":
                            model.set_seed_context(known.to(dev), ids)
                        for gs in cfg.guidance:
                            n_, a_ = run(co_expr={"coexpr_sparse": cond_t, "tissue_ids": ids}, guidance_scale=gs)
                            rows[f"{kind} {ch} g{gs:g}"] = self.score(n_, a_, true_ids, true_adj, rev)
                        if kind == "masked-nbr":
                            model.set_seed_context(None, None)
                else:
                    n_, a_ = run()
                    rows[kind] = self.score(n_, a_, true_ids, true_adj, rev)
                free(model)
            out[dname] = dict(k_revealed=cfg.k_revealed, n_eval=int(true_ids.shape[0]),
                              hidden_per_graph=float(((true_ids != 0).sum(1) - rev.sum(1)).float().mean()),
                              arms={k: dict(mean=df.mean(numeric_only=True).to_dict(), n_graphs=len(df),
                                            per_graph=df.to_dict(orient="list")) for k, df in rows.items()})
            tab = pd.DataFrame({k: v["mean"] for k, v in out[dname]["arms"].items()}).T
            print(f"\n{dname}\n{tab.to_string(float_format=lambda v: f'{v:9.4f}')}")
            tab.to_csv(self.run.path / "tables" / f"{dname}_completion.csv")
        dump_json(out, self.out / "completion.json")
        return out

    # ── validation of the neighbourhood variant ──────────────────────────
    @torch.no_grad()
    def validate(self, seeds=(1, 2, 4, 8)) -> dict:
        cfg, out = self.cfg, {"wiring": {}, "ablation": {}, "attribution": {}}
        for dname, b in self.data.items():
            cond_t = self.cond.get(dname, {}).get("tensor")
            if cond_t is None or "masked-nbr" not in cfg.arms:
                continue
            nch = len(self.cond[dname]["channels"])
            # 1. wiring: at zero-init the attached model is the base model
            torch.manual_seed(0); m = build_arm("masked-nbr", b, cfg, nch, self.device)
            m.set_coexpression(cond_t.to(self.device), top_k=cfg.cond_topk, verbose=False); m.eval()
            B, N = 2, cfg.max_nodes
            nt = torch.randint(1, b.vocab_size, (B, N), device=self.device)
            et = torch.randint(0, 2, (B, N, N), device=self.device)
            t = torch.full((B,), 5, device=self.device)
            known = torch.full_like(nt, -1); known[:, :4] = nt[:, :4]
            chan = torch.zeros(B, dtype=torch.long, device=self.device)
            rf = torch.randn(B, N, cfg.rand_dim, device=self.device)
            nid, pooled = m.build_inputs(nt, known, chan, rand_feat=rf)
            base_nid = torch.cat([rf, torch.zeros(B, N, cfg.nbr_dim, device=self.device)], -1)
            co = {"coexpr_sparse": cond_t, "tissue_ids": chan}
            torch.manual_seed(1); a = m.forward(nt, et, t, t, node_id_feat=nid, co_expr=co)
            torch.manual_seed(1); bb = m.forward(nt, et, t, t, node_id_feat=base_nid, co_expr=co)
            out["wiring"][dname] = dict(neighbourhood_half_zero=float(nid[..., cfg.rand_dim:].abs().max()) == 0.0,
                                        logits_identical=bool(torch.allclose(a[0], bb[0]) and torch.allclose(a[1], bb[1])),
                                        non_seed_slots_zero=float(nid[:, 4:, cfg.rand_dim:].abs().max()) == 0.0)
            print(f"  {dname} wiring: {out['wiring'][dname]}")
            free(m)
            if not self._wpath(dname, "masked-nbr").exists():
                continue
            # 2. ablation: same weights, encoder on vs off
            m = self.load_model(dname, "masked-nbr")
            true_ids, true_adj = b.heldout_tensors(cfg.n_eval, cfg.max_nodes, seed=cfg.eval_seed)
            rows = {}
            for k in seeds:
                known, rev = self.reveal(true_ids, k)
                for on in (True, False):
                    ids = torch.zeros(true_ids.shape[0], dtype=torch.long, device=self.device)
                    m.set_seed_context(known.to(self.device) if on else None, ids if on else None)
                    torch.manual_seed(cfg.eval_seed)
                    n_, a_ = m.sample(N=cfg.max_nodes, B=true_ids.shape[0], device=self.device,
                                      known_nodes=known.to(self.device), co_expr={"coexpr_sparse": cond_t, "tissue_ids": ids})
                    m.set_seed_context(None, None)
                    d = self.score(n_.long().cpu(), (a_ > 0).long().cpu(), true_ids, true_adj, rev)
                    rows[f"k={k} {'on' if on else 'off'}"] = dict(k=k, encoder=on, **d.mean(numeric_only=True).to_dict())
            out["ablation"][dname] = rows
            # 3. attribution: evidence partners of each seed, and whether they appeared
            true_ids, true_adj = b.heldout_tensors(8, cfg.max_nodes, seed=cfg.eval_seed)
            known, rev = self.reveal(true_ids, cfg.k_revealed)
            ids = torch.zeros(true_ids.shape[0], dtype=torch.long, device=self.device)
            m.set_seed_context(known.to(self.device), ids)
            torch.manual_seed(cfg.eval_seed)
            n_, _ = m.sample(N=cfg.max_nodes, B=true_ids.shape[0], device=self.device, known_nodes=known.to(self.device),
                             co_expr={"coexpr_sparse": cond_t, "tissue_ids": ids})
            m.set_seed_context(None, None)
            gen = n_.long().cpu(); attr = []
            for g in range(min(4, true_ids.shape[0])):
                for s in torch.nonzero(rev[g]).flatten()[:2]:
                    sid = int(true_ids[g, s])
                    nb_idx, nb_w = m.coexpr_lookup.neighbours(ids[g:g + 1], torch.tensor([[sid]], device=self.device))
                    order = nb_w[0, 0].argsort(descending=True)[:5]
                    part = [int(nb_idx[0, 0, i]) for i in order if float(nb_w[0, 0, i]) > 0]
                    gset = {int(v) for v in gen[g]}
                    attr.append(dict(graph=g, seed=str(b.common(b.id_to_node.get(sid, sid))),
                                     partners=[str(b.common(b.id_to_node.get(p, p))) for p in part[:4]],
                                     partners_in_completion=sum(p in gset for p in part), n_partners=len(part)))
            out["attribution"][dname] = attr
            free(m)
        dump_json(out, self.out / "validation.json")
        return out
