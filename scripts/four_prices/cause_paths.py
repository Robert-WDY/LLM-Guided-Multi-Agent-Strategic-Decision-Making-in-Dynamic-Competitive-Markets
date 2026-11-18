"""Controlled post-contract quote tapes. Market responses to each tape remain endogenous."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from world import World
from cause_trials import OUT,archive,jobs
from run import read,save
from saturation_report import ci

def contracts(path):
    return {d["t"]:d["requested"] for d in path["decisions"]}

def effects(task):
    _,m,_,_=task
    r=read(archive(task))
    s,j,t=r["seed"],r["masks"]["1"][0],r["probe"]
    base=r["paths"]["keep"]
    bp=contracts(base)
    world=World(m,s)
    target=None
    while world.t<t:
        ix=world.t
        target=bp.get(ix,target)
        rows,_=world.settle(target)
        assert rows==base["rows"][ix-1]
    out=[]
    for arm in ("audit_cut","audit_raise"):
        tr=r["paths"][arm]
        tp=contracts(tr)
        z={}
        weights=.99**np.arange(61-t)
        for own,other in ((0,0),(0,1),(1,0),(1,1)):
            w=world.clone()
            flow=[]
            for ix in range(t,61):
                if ix in tp:
                    target=tp[ix] if ix<t+4 else [(tp if (own if k==j else other) else bp)[ix][k] for k in range(4)]
                rows,_=w.settle(target)
                if own==other==1:
                    assert rows==tr["rows"][ix-1]
                flow.append(rows[j][2]-base["rows"][ix-1][j][2])
            z[own,other]=float(weights@flow)
        focal=(z[1,0]-z[0,0]+z[1,1]-z[0,1])/2
        peers=(z[0,1]-z[0,0]+z[1,1]-z[1,0])/2
        assert abs(z[0,0]+focal+peers-z[1,1])<1e-6
        out.append(dict(seed=s,seat=j,arm=arm,market=m,total=z[1,1],market_only=z[0,0],focal=focal,rivals=peers))
    return out

if __name__=="__main__":
    tasks=[t for t in jobs("test") if t[-1]%4==0]
    with ProcessPoolExecutor(8) as pool:
        rows=[r for group in pool.map(effects,tasks) for r in group]
    out={}
    for m in ("econ","native"):
        v=[r for r in rows if r["market"]==m]
        losses=[r for r in v if r["total"]< -1e-7]
        out[m]=dict(seeds=len(v)//2,actions=len(v),losses=len(losses),
                    on_losses={k:float(np.mean([r[k] for r in losses])) for k in ("total","market_only","focal","rivals")},
                    all_actions={k:ci([np.mean([r[k] for r in v if r["seed"]==s]) for s in sorted({r["seed"] for r in v})]) for k in ("total","market_only","focal","rivals")})
    save(OUT/"path-cases.json",rows);save(OUT/"path-effects.json",out)

