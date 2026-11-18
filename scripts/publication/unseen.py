"""Out-of-family check with Exp3 observing only its own settled economic reward."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from delayed import RAW, OUT, decision, draw, limits, prospective, read, save, world
from delayed_analysis import summary
from duopoly import settle


class Bandit:
    def __init__(self,w,seed,learns):
        self.grid=np.linspace(max(w.kw["alpha"][1]*w.cost[1],.3*w.high,w.low),.85*w.high,9)
        assert self.grid[-1]>self.grid[0]
        self.uniform=np.random.default_rng(seed+68123).random(60)
        self.learns=learns
        self.logw=np.zeros(9)

    def clone(self):
        b=object.__new__(Bandit)
        b.__dict__=self.__dict__.copy()
        b.logw=self.logw.copy()
        return b

    def distribution(self):
        z=np.exp(self.logw-self.logw.max())
        return .85*z/z.sum()+.15/len(z)

    def step(self,w,price):
        prob=self.distribution()
        idx=min(len(prob)-1,int(np.searchsorted(prob.cumsum(),self.uniform[w.t-1])))
        rival=self.grid[idx]
        q,profit=settle(w.kw,w.cost,(price,rival))
        cap=w.kw["multiplier"]*(self.grid[-1]/w.kw["alpha"][1]-w.cost[1])
        reward=profit[1]/cap
        assert -1e-8<=reward<=1+1e-4
        if self.learns:
            self.logw[idx]+=.15/len(prob)*np.clip(reward,0,1)/prob[idx]
        w.last,w.peer=float(price),float(rival)
        w.t+=1
        return [w.last,w.peer,q[0],profit[0],0,0,price,w.low,w.high]


def trial(args):
    seed,learns=args
    w=world("econ",seed,"stable")
    b=Bandit(w,seed,learns)
    t0=27+4*(draw(seed,"opportunity")%6)
    h=[]
    old=None
    for t in range(1,t0+1):
        d=decision(w,h,old,None,seed,*limits(w),[])
        if t<t0:
            h.append(b.step(w,d["price"]))
            old=d
    assert d["start"]==t0 and d["end"]==t0+3
    public=dict(prices=[v[:2] for v in h],round=t0,draft=d["draft"])
    choice=prospective("econ",public)
    seal=RAW/f"unseen-settlement-units-{seed}-{int(learns)}.json"
    if seal.exists():
        assert read(seal)==choice
    else:
        save(seal,choice)
    paths=[]
    dist=[]
    for mult in (1.,.96,1.04):
        other=w.clone()
        bandit=b.clone()
        rows=list(h)
        term=dict(d,price=float(np.clip(d["draft"]*mult,w.low,w.high)))
        for t in range(t0,61):
            if t>t0:
                term=decision(other,rows,term,None,seed,*limits(other),[])
            rows.append(bandit.step(other,term["price"]))
        paths.append(rows[t0-1:])
        dist.append(bandit.distribution().tolist())
    path=np.array(paths)
    assert np.all(path[:,0,1]==path[0,0,1]) and np.all(path[:,:4,0]==path[:,0,0,None])
    if not learns:
        assert np.all(path[:,:,1]==path[0,:,1]) and np.allclose(dist,1/9)
    return dict(seed=seed,learns=learns,public=public,choice=choice,rival=path[:,:,1].tolist(),
                profit=path[:,:,3].tolist(),final_distribution=dist)


def results(rows):
    out={}
    for learns in (False,True):
        part=[r for r in rows if r["learns"]==learns]
        assert [r["seed"] for r in part]==list(range(109000,109192))
        values=[]
        changed=0
        prob_changed=0
        for r in part:
            p=np.array(r["profit"])
            g=(p-p[:1])@(.99**np.arange(p.shape[1]))
            d=r["choice"]
            a=g[d["inferred"]]
            values.append([a,a-g[d["uniform"]],a-g[d["direct"]],a-g[2],a-g[1]])
            changed+=len({tuple(v) for v in r["rival"]})>1
            prob_changed+=np.ptp(r["final_distribution"],axis=0).max()>1e-10
        out["exp3" if learns else "uniform_quotes"]=dict(
            effects={k:summary(v) for k,v in zip(("keep","prior","direct","up","down"),np.asarray(values).T)},
            peer_path_changed=changed,peer_distribution_changed=int(prob_changed))
    return out


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows=list(ex.map(trial,[(s,b) for s in range(109000,109192) for b in (False,True)]))
    save(OUT/"unseen-paths.json.gz",rows)
    save(OUT/"unseen.json",results(rows))
