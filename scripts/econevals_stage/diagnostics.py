"""Common-state search-width, continuation and forecast diagnostics (fresh seed clusters)."""
from concurrent.futures import ProcessPoolExecutor
import itertools
import numpy as np
from common import OUT,save,market
from duopoly import FAMILIES,ALPHAS,Opponent,environment,settle,simple,startup,fitted,response,lower_bound


def simulate(history,hi,prices,mode,persist,draws,params):
    b,m,model=params
    peer=np.full((len(prices),draws.shape[1]),history[-1][1])
    last=np.full((len(prices),1),history[-1][0])
    movement=np.full((len(prices),1),response.bucket(history[-1][0]/history[-2][0]-1))
    ps=[response.predict(model,(k,0,0),mode) for k in (-1,0,1)]
    prob=np.array([x['probabilities'] for x in ps]); ch=np.array([x['changes'] for x in ps])
    gains=[]
    for j,u in enumerate(draws):
        k=movement+1
        label=(u>=prob[k,0]).astype(int)+(u>=prob[k,0]+prob[k,1]).astype(int)
        peer=np.clip(peer*(1+ch[k,label]),.01*hi,hi)
        own=prices[:,None] if j==0 or persist else np.full((len(prices),1),history[-1][0])
        x=own/hi
        gains.append(.95**j*np.exp(np.clip(b[0]+b[1]*x+b[2]*peer/hi,-6,6))*(m[0]+m[1]*x))
        movement=np.where(own/last<.99,-1,np.where(own/last>1.01,1,0));last=own
    return np.cumsum(gains,axis=0)


def realized(h,kw,c,op,t,price,hold):
    draft,peer=h[-1][:2]
    last=draft; total=0.
    for i in range(5):
        rival=op.price(t+i,last,peer)
        own=price if hold or i==0 else draft
        total+=.95**i*market.get_profits(p=(own,rival),c=c,**kw)[0]
        last,peer=own,rival
    return total


def cases(seed):
    rows=[]
    for family in FAMILIES:
        kw,c,_,_,hi=environment(ALPHAS[seed%3]);op=Opponent(seed,family,hi)
        h=[];p=peer=hi*.55
        for t in range(41):
            rival=op.price(t,p,peer)
            own=startup(t,hi,seed) if t<10 else p
            if t in (20,40):
                params=fitted(h,hi); bucket=response.bucket(h[-1][0]/h[-2][0]-1)
                label=response.bucket(rival/peer-1)+1
                for mode,width,hold in itertools.product(('generic','conditional'),(3,9),(False,True)):
                    fractions=[1,.95,1.05] if width==3 else [1,.9,.925,.95,.975,1.025,1.05,1.075,1.1]
                    candidates=np.clip(p*np.array(fractions),.01*hi,hi)
                    rng=np.random.default_rng(seed*1000+t)
                    rank_u=rng.random((5,3));valid_u=rng.random((5,8))
                    train=simulate(h,hi,candidates,mode,hold,rank_u[:,:9//width],params)
                    best=1+np.argmax(train[2,1:].mean(axis=1))
                    final=simulate(h,hi,candidates[[0,best]],mode,hold,valid_u,params)
                    gains=final[[2,4],1,:]-final[[2,4],0,:]
                    gate=all(lower_bound(list(g))>0 and np.all(g>=0) for g in gains)
                    raw=candidates[best] if train[2,best].mean()>train[2,0].mean() else p
                    v=realized(h,kw,c,op,t,p,hold)
                    raw_gain=realized(h,kw,c,op,t,raw,hold)-v
                    g=realized(h,kw,c,op,t,candidates[best],hold)-v if gate else 0.
                    pr=np.array(response.predict(params[2],(bucket,0,0),mode)['probabilities'])
                    rows.append(dict(seed=seed,family=family,t=t,mode=mode,width=width,hold=hold,raw=raw_gain,
                        gated=g,brier=float(np.square(pr-np.eye(3)[label]).sum()),logloss=float(-np.log(pr[label]))))
            if t and t%10==0:own=simple(h,hi)
            q,pi=settle(kw,c,(own,rival));h.append((own,rival,q[0],pi[0]));p,peer=own,rival
    return rows


def compare(rows,left,right,col):
    s=range(2200,2248)
    def avg(i,spec):
        return np.mean([r[col] for r in rows if r['seed']==i and all(r[k]==v for k,v in spec.items())])
    d=[avg(i,left)-avg(i,right) for i in s]
    boot=np.random.default_rng(72).choice(d,(5000,48)).mean(axis=1)
    return dict(mean=float(np.mean(d)),ci=np.quantile(boot,[.025,.975]).tolist())


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=5) as pool:rows=sum(pool.map(cases,range(2200,2248)),[])
    stats={}
    for hold in (False,True):
        for col in ('raw','gated'):
            for label,m,w in [('same-width','generic',3),('wide-search','generic',9)]:
                l=dict(mode='conditional',width=3,hold=hold); r=dict(mode=m,width=w,hold=hold)
                stats[f'{label}/{hold}/{col}']=compare(rows,l,r,col)
    for col in ('brier','logloss'):
        stats[col]=compare(rows,dict(mode='conditional',width=3,hold=False),dict(mode='generic',width=3,hold=False),col)
    save(OUT/'diagnostic-windows.json.gz',rows)
    save(OUT/'diagnostics.json',stats)
    print(stats,flush=True)

