"""Checks for PerturbationBenchmark, on a synthetic effect matrix with a
planted pathway so the right answer is known."""
import sys, numpy as np, torch, json, tempfile
from pathlib import Path
from graphpath.benchmarks.perturbation import PerturbationBenchmark, perturbation_table
OK=FAIL=0
def check(l,g,w,tol=1e-9):
    global OK,FAIL
    ok=(abs(g-w)<=tol) if isinstance(w,float) else (g==w)
    print(f"  {'ok  ' if ok else 'FAIL'}  {l:<56} got {g!r}")
    OK,FAIL=OK+ok,FAIL+(not ok)

# 40 genes, 200 readout dims. Genes 0-9 are a PLANTED pathway: their profiles
# share a latent factor. The rest are independent noise.
rng=np.random.default_rng(0)
NG,ND=40,200
E=rng.normal(size=(NG,ND))
shared=rng.normal(size=ND)
E[:10]+=2.2*shared
d=Path(tempfile.mkdtemp())
np.save(d/'effect_matrix.npy',E.astype(np.float32))
np.savez_compressed(d/'similarity.npz',row=np.array([0]),col=np.array([1]),val=np.array([0.5]))
names=[f"ENSG{i:05d}" for i in range(NG)]
(d/'meta.json').write_text(json.dumps({"genes":names,"measured_genes":names,
    "n_measured":ND,"top_k":32,"min_abs":0.1,"dataset":"synthetic","source":"-"}))

class DS:
    id_to_node={i+1:n for i,n in enumerate(names)}   # 0 = PAD
DS.node_to_id={n:i for i,n in DS.id_to_node.items()}
b=PerturbationBenchmark(d, DS(), min_genes=3, n_null=300, seed=0)

def graph(idx,N=24):
    t=torch.zeros(1,N,dtype=torch.long)
    for k,i in enumerate(idx): t[0,k]=i+1
    return t
print("planted pathway vs random set")
r_path=b.evaluate(graph(range(10)))
r_rand=b.evaluate(graph(range(20,30)))
print(f"        pathway: sim {r_path.summary['profile_similarity']:.3f}  "
      f"null {r_path.summary['profile_similarity_null']:.3f}  "
      f"z {r_path.summary['profile_similarity_z']:.1f}")
print(f"        random : sim {r_rand.summary['profile_similarity']:.3f}  "
      f"z {r_rand.summary['profile_similarity_z']:.1f}")
check("planted pathway scores above its null",
      r_path.summary['profile_similarity_excess']>0.1, True)
check("random set does not", abs(r_rand.summary['profile_similarity_excess'])<0.1, True)
check("planted z is large", r_path.summary['profile_similarity_z']>5, True)
check("planted p is significant", r_path.summary['profile_similarity_p']<0.02, True)

print("\ncoverage and skipping")
r2=b.evaluate(graph([0,1]))                     # only 2 genes -> below min_genes
check("too-small graph is skipped, not scored 0", r2.n_scored, 0)
check("and noted", len(r2.notes)>0, True)
t=torch.zeros(1,24,dtype=torch.long)            # all PAD
check("all-padding graph scores nothing", b.evaluate(t).n_scored, 0)
mix=torch.zeros(1,24,dtype=torch.long); mix[0,:6]=torch.tensor([1,2,3,0,0,0])
check("PAD tokens are ignored", b.evaluate(mix).n_scored, 1)

print("\nde-duplication")
dup=torch.zeros(1,24,dtype=torch.long); dup[0,:6]=torch.tensor([1,1,2,2,3,3])
uniq=torch.zeros(1,24,dtype=torch.long); uniq[0,:3]=torch.tensor([1,2,3])
check("repeated tokens do not inflate the score",
      abs(b.evaluate(dup).summary['profile_similarity']
          -b.evaluate(uniq).summary['profile_similarity'])<1e-9, True)
check("genes_per_graph counts distinct genes",
      b.evaluate(dup).summary['genes_per_graph'], 3.0)

print("\nbatching")
big=torch.zeros(4,24,dtype=torch.long)
for r_ in range(4): big[r_,:10]=torch.arange(1,11)
rb=b.evaluate(big)
check("four identical graphs -> 4 scored", rb.n_scored, 4)
check("per-graph tensor length", len(rb.per_graph['profile_similarity']), 4)
check("cross effect reported", np.isfinite(rb.summary['cross_effect']), True)

# ── both directions of inference ──────────────────────────────────────────
#
# The fixture above is SQUARE (genes == measured_genes), which cannot catch a
# confusion between the two axes: every index is valid on both. The real screen
# is 847 x 8175, so this block rebuilds a rectangular one.
#
# The regression this guards: `evaluate` used one name for both "the axis being
# scored" and "the perturbed axis". They coincide in row mode, so row worked and
# column indexed E's axis 0 with a measured id — IndexError on every graph.
print("\ndirections of inference (rectangular matrix)")
NP2, NM2 = 20, 100
E2 = rng.normal(size=(NP2, NM2))
# plant a co-response block: measured genes 30-39 move together across all the
# knockdowns, which is the relation the COLUMN direction is supposed to find
E2[:, 30:40] += 1.8 * rng.normal(size=(NP2, 1))
d2 = Path(tempfile.mkdtemp())
np.save(d2/'effect_matrix.npy', E2.astype(np.float32))
names2 = [f"ENSG{i:05d}" for i in range(NM2)]      # perturbed are the first 20
(d2/'meta.json').write_text(json.dumps({"genes":names2[:NP2],"measured_genes":names2,
    "n_measured":NM2,"top_k":32,"min_abs":0.1,"dataset":"synthetic","source":"-"}))
class DS2:
    id_to_node={i+1:n for i,n in enumerate(names2)}

def graph2(idx, N=24):
    t=torch.zeros(1,N,dtype=torch.long)
    for k,i in enumerate(idx): t[0,k]=i+1
    return t

bench2 = {}
for dirn in ("row", "column"):
    try:
        bench2[dirn] = PerturbationBenchmark(d2, DS2(), direction=dirn,
                                             min_genes=2, n_null=60, seed=0)
        ok = True
    except Exception as e:
        ok = False; print(f"    construction failed: {e}")
    check(f"{dirn}: constructs", ok, True)

# a graph of mostly-unperturbed measured genes — the case that used to raise
mixed = graph2([2, 4, 6, 33, 34, 35, 36, 55, 70])
for dirn, bb in bench2.items():
    try:
        r = bb.evaluate(mixed); err = None
    except Exception as e:
        r = None; err = f"{type(e).__name__}: {e}"
    check(f"{dirn}: evaluates a graph of mostly-measured genes", err, None)
    if r is not None:
        check(f"{dirn}: scores the graph", r.n_scored, 1)
        check(f"{dirn}: pooled_d is finite",
              bool(np.isfinite(r.summary.get('pooled_d', np.nan))), True)
        check(f"{dirn}: reports its direction", r.summary.get('direction'), dirn)

if len(bench2) == 2:
    rr = bench2["row"].evaluate(mixed); rc = bench2["column"].evaluate(mixed)
    # 3 perturbed genes in `mixed` vs 9 measured — the coverage gap that is the
    # practical reason the column direction exists
    check("column scores more genes than row on the same graph",
          rc.summary['genes_per_scored_graph'] > rr.summary['genes_per_scored_graph'],
          True)
    # the planted co-response block must beat an arbitrary measured set
    planted = bench2["column"].evaluate(graph2(range(30, 40)))
    arb     = bench2["column"].evaluate(graph2(range(60, 70)))
    check("column direction finds the planted co-response block",
          planted.summary['pooled_d'] > arb.summary['pooled_d'], True)
    # cross-effect indexes E's perturbed axis in BOTH directions
    for dirn, bb in bench2.items():
        v = bb.evaluate(graph2([0, 1, 2, 3, 40, 41, 42])).summary.get('cross_effect')
        check(f"{dirn}: cross_effect computed from the perturbed axis",
              bool(np.isfinite(v)), True)

print("\n"+perturbation_table({"pathway":r_path,"random":r_rand}))
print(f"\n{OK} passed, {FAIL} failed")


def test_perturbation():
    assert FAIL == 0, f"{FAIL} checks failed (see output above)"


if __name__ == "__main__":
    sys.exit(1 if FAIL else 0)
