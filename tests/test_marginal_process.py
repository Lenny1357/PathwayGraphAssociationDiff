"""The marginal transition process against brute force.

Q̄ composition, the exact O(K) posterior, forward marginal statistics, prior
sampling, triangle mirroring, and an end-to-end loss + sample + clamp check.
Ported from the self-test the model shipped with.
"""

import torch
import torch.nn.functional as F

from graphpath.config import MarginalConfig as Config
from graphpath.models.marginal import GraphDiffBERTMarginal as GraphDiffBERT
from graphpath.processes.marginal import MarginalTransitionSchedule, _mirror_upper


def test_marginal_process(verbose: bool = False):
    torch.manual_seed(0)
    cfg = Config(T=60, node_vocab=6, edge_vocab=2, d_model=32, n_heads=2,
                 n_layers=2, d_ff=64)
    sch = MarginalTransitionSchedule(cfg)

    for m_np in ([0.998, 0.002], [0.5, 0.5],
                 [0.9975] + [0.0025 / 8] * 8):
        m = torch.tensor(m_np, dtype=torch.float)
        K = m.numel()

        # (a) Q-bar composition matches the closed form
        Qbar = torch.eye(K, dtype=torch.double)
        for t in range(cfg.T):
            at = sch.alpha_t[t].double()
            Q  = at * torch.eye(K, dtype=torch.double) + \
                 (1 - at) * torch.outer(torch.ones(K, dtype=torch.double),
                                        m.double())
            Qbar = Qbar @ Q
            ac = sch.ac[t].double()
            closed = ac * torch.eye(K, dtype=torch.double) + \
                     (1 - ac) * torch.outer(torch.ones(K, dtype=torch.double),
                                            m.double())
            assert torch.allclose(Qbar, closed, atol=1e-10), f"Qbar t={t}"

        # (b) exact posterior matches brute-force marginalisation
        for t_idx in [1, cfg.T // 3, cfg.T - 1]:
            at  = sch.alpha_t[t_idx].double()
            acp = sch.ac_prev[t_idx].double()
            for j in range(K):
                logits = torch.randn(1, K)
                x_t    = torch.tensor([j])
                got = sch.posterior_logits(logits, x_t, t_idx, m,
                                           exact=True).exp()[0].double()

                p0 = F.softmax(logits.double(), -1)[0]
                Qt = at * torch.eye(K, dtype=torch.double) + \
                     (1 - at) * torch.outer(torch.ones(K, dtype=torch.double),
                                            m.double())
                Qb = acp * torch.eye(K, dtype=torch.double) + \
                     (1 - acp) * torch.outer(torch.ones(K, dtype=torch.double),
                                             m.double())
                brute = torch.zeros(K, dtype=torch.double)
                for c in range(K):
                    num = Qt[:, j] * Qb[c, :]
                    brute += p0[c] * num / num.sum()
                assert torch.allclose(got, brute, atol=1e-8), \
                    f"posterior K={K} t={t_idx} j={j}\n{got}\n{brute}"
                # posterior_logits returns float32 (the caller's dtype), so the
                # sum is only exact to ~6e-8; the original 1e-8 could never pass
                assert abs(got.sum().item() - 1) < 1e-6

        # (c) forward marginal statistics
        t_idx = cfg.T // 3
        x0 = torch.zeros(200_000, dtype=torch.long)
        xt = sch.q_sample(x0, torch.full((200_000,), t_idx), m)
        emp = torch.bincount(xt, minlength=K).float() / xt.numel()
        ac  = sch.ac[t_idx]
        theo = ac * F.one_hot(torch.tensor(0), K).float() + (1 - ac) * m
        err = (emp - theo).abs().max().item()
        assert err < 0.01, f"forward marginal err {err}"
        if verbose:
            print(f"  K={K:2d}  Qbar ok · posterior ok · fwd marginal err {err:.4f}")

    # (d) prior sampling hits the marginal
    m = torch.tensor([0.99, 0.01])
    s = MarginalTransitionSchedule.sample_prior(m, (200_000,), torch.device("cpu"))
    emp = torch.bincount(s, minlength=2).float() / s.numel()
    assert (emp - m).abs().max() < 0.005, emp
    if verbose:
        print(f"  prior draw: {emp.tolist()} vs m {m.tolist()}")

    # (e) mirror_upper is symmetric and preserves the upper triangle
    x = torch.randint(0, 2, (2, 7, 7))
    y = _mirror_upper(x)
    assert torch.equal(y, y.transpose(1, 2)), "mirror not symmetric"
    iu = torch.triu(torch.ones(7, 7, dtype=torch.bool), 1)
    assert torch.equal(y[:, iu], x[:, iu]), "mirror changed upper triangle"

    # (f) end-to-end shapes: loss backward + short sampling run
    model = GraphDiffBERT(
        cfg, vocab_size=cfg.node_vocab + 1,
        m_node=torch.full((cfg.node_vocab,), 1.0 / cfg.node_vocab),
        m_edge=torch.tensor([0.97, 0.03]),
    )
    B, N = 3, 9
    nf = torch.randint(0, cfg.node_vocab, (B, N))
    aj = (torch.rand(B, N, N) < 0.15).long()
    aj = _mirror_upper(aj).masked_fill(
        torch.eye(N, dtype=torch.bool)[None], 0)
    pm = torch.ones(B, N, dtype=torch.bool); pm[2, 7:] = False

    out = model.compute_loss(nf, aj, pm)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]), out
    if verbose:
        print(f"  loss ok: {out['loss'].item():.4f} "
              f"(node {out['loss_node']:.3f} | edge {out['loss_edge']:.3f})")

    model.eval()
    nt, at_ = model.sample(N=N, B=2)
    assert nt.shape == (2, N) and at_.shape == (2, N, N)
    assert torch.equal(at_, at_.transpose(1, 2)), "sample not symmetric"
    assert at_.diagonal(dim1=1, dim2=2).sum() == 0, "self-loops present"

    # (g) clamped context is respected end-to-end
    em = torch.zeros(2, N, N, dtype=torch.bool); em[:, :4, :4] = True
    ev = torch.zeros(2, N, N, dtype=torch.long); ev[:, 0, 1] = 1; ev[:, 1, 0] = 1
    em = em & ~torch.eye(N, dtype=torch.bool)[None]
    _, ac_ = model.sample(N=N, B=2, edge_clamp=(em, ev))
    assert torch.equal(ac_[em], ev[em]), "clamp not respected"
    if verbose:
        print("  sampling ok · symmetry ok · clamp ok")

