"""Checks for NovelEdgeBenchmark on a network whose hop distances are known."""
import sys, torch, numpy as np
from graphpath.benchmarks.novel_edges import NovelEdgeBenchmark, novel_edge_table
OK=FAIL=0
def check(l,g,w,tol=1e-9):
    global OK,FAIL
    ok=(abs(g-w)<=tol) if isinstance(w,float) else (g==w)
    print(f"  {'ok  ' if ok else 'FAIL'}  {l:<52} got {g!r}")
    OK,FAIL=OK+ok,FAIL+(not ok)

# a path graph 1-2-3-4-5-6 plus an isolated pair 7-8, so hops are hand-checkable
V, N = 12, 8
EDGES=[(1,2),(2,3),(3,4),(4,5),(5,6),(7,8)]
TYPES={0:'metabolite',1:'gene',2:'reaction',3:'metabolite',4:'reaction',5:'gene',
       6:'reaction',7:'metabolite',8:'reaction',9:'gene',10:'reaction',11:'metabolite'}
class Ref:
    num_nodes=V
    def __init__(self):
        u=torch.tensor([e[0] for e in EDGES]+[e[1] for e in EDGES])
        v=torch.tensor([e[1] for e in EDGES]+[e[0] for e in EDGES])
        self.adj_sparse=torch.sparse_coo_tensor(torch.stack([u,v]),
                        torch.ones(len(u)),(V,V)).coalesce()
class DS:
    """training sees ONLY the edge (1,2)."""
    id_to_node={i:f"n{i}" for i in range(V)}
    def __len__(self): return 1
    def __getitem__(self,i):
        a=torch.zeros(N,N); a[0,1]=a[1,0]=1
        nodes=torch.zeros(N,dtype=torch.long); nodes[0]=1; nodes[1]=2
        m=torch.zeros(N); m[:2]=1
        return a,nodes,m,0,0

b=NovelEdgeBenchmark(DS(),Ref(),node_type_fn=lambda n:TYPES[int(str(n)[1:])],max_hops=6)
check("training edge set collected",b.train_edges,{(1,2)})

print("\nhop distances")
d=b._hops([(1,2),(1,3),(1,4),(1,6),(1,7),(3,3)])
check("adjacent -> 1",float(d[0]),1.0)
check("two apart -> 2",float(d[1]),2.0)
check("three apart -> 3",float(d[2]),3.0)
check("five apart -> 5",float(d[3]),5.0)
check("other component -> inf",bool(np.isinf(d[4])),True)
check("self -> 0",float(d[5]),0.0)

print("\nevaluate")
# generated edges, by slot: (1,2) seen in training
#                            (2,3) novel and IN the reference   -> 1 hop
#                            (1,3) novel, shares a neighbour    -> 2 hops
#                            (1,7) novel, other component       -> unreachable
nodes=torch.zeros(1,N,dtype=torch.long); nodes[0,:4]=torch.tensor([1,2,3,7])
adj=torch.zeros(1,N,N)
for i,j in ((0,1),(1,2),(0,2),(0,3)): adj[0,i,j]=adj[0,j,i]=1
r=b.evaluate(nodes,adj)
check("4 edges counted",r.summary["edges_total"],4.0)
check("3 of them novel",r.summary["novel_rate"],0.75)
check("one novel edge is IN the reference",r.summary["novel_in_reference"],1/3)
check("two thirds are <=2 hops",r.summary["hops_le2"],2/3)
check("one third unreachable",r.summary["unreachable"],1/3)
check("per-graph count",float(r.per_graph["novel_edges"][0]),3.0)
bk=r.distributions["hop_buckets"]
check("bucket labels",set(bk),
      {"1 (in reference)","2 (shares a neighbour)","unreachable"})
check("the 1-hop bucket is the recovered edge",bk["1 (in reference)"],1)
check("type-pair breakdown present",len(r.distributions["type_pair_hops"])>0,True)

print("\nno-novel-edge case")
n2=torch.zeros(1,N,dtype=torch.long); n2[0,:2]=torch.tensor([1,2])
a2=torch.zeros(1,N,N); a2[0,0,1]=a2[0,1,0]=1
r2=b.evaluate(n2,a2)
check("novel_rate 0",r2.summary["novel_rate"],0.0)
check("noted",len(r2.notes)>0,True)
print("\n"+novel_edge_table(r))
print(f"\n{OK} passed, {FAIL} failed")


def test_novel_edges():
    assert FAIL == 0, f"{FAIL} checks failed (see output above)"


if __name__ == "__main__":
    sys.exit(1 if FAIL else 0)
