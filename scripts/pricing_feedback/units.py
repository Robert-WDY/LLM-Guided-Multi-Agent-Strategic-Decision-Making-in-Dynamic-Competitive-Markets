"""Counterfactual control for the conversion from quoted prices to unit-profit currency."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from study import World,WARM,OUT,RAW,read,save,replay,decide
from analyze import cluster


def exchange(history):
    h=np.asarray(history[-24:],float)
    p,q,profit=h[:,0],h[:,2],h[:,3]
    v=profit/np.maximum(q,.01)
    slopes=[]
    for j in range(1,len(p)):
        for i in range(max(0,j-8),j):
            if min(q[i],q[j])>0 and abs(p[j]-p[i])>.015*max(p[i],p[j]):
                slopes.append((v[j]-v[i])/(p[j]-p[i]))
    slope=np.median(slopes) if slopes else 1.
    return float(slope) if np.isfinite(slope) and slope>1.e-6 else 1.


def episode(m,s,c,mode,agent=None):
    w=World(m,s,c)
    h=[];prev=None;terms=[]
    for t in range(1,w.length+1):
        left=w.length-t+1
        if prev is None or t>prev['end']:
            d=decide(t,left,w.ref,w.low,w.high,h,prev)
            if t>WARM:
                if agent and w.active():
                    original=agent(m,h,w.low,w.high,d['draft'],min(left,4),left)
                    d=decide(t,left,w.ref,w.low,w.high,h,prev,original=original)
                k=exchange(h)
                converted=[[v[0]*k,v[1]*k,v[2],v[3]] for v in h]
                e=decide(t,left,w.ref*k,w.low*k,w.high*k,converted,prev,mode=mode,gate=0,original=d['draft']*k)
                d.update(price=e['price']/k,proposed=e['proposed']/k,score=e['score'],transitions=e['transitions'],scale=k)
            d['active']=w.active()
            d['actual_override']=(abs(d['price']-d['draft'])>1.e-8 and abs(w.legal(d['price'])-w.legal(d['draft']))>1.e-8)
            terms.append(d)
            prev=d
        h.append(w.execute(prev['price']))
    return dict(seed=s,market=m,condition=c,arm=mode,rows=h,contracts=terms)


def worker(task):
    m,s=task;out=[]
    for c in ('stable','demand','rival','reverse'):
        base=replay(m,s,c,'feedback')
        original=replay(m,s,c,'conditional')
        original['arm']='original'
        out.extend([base,original])
        out.extend(episode(m,s,c,a) for a in ('generic','conditional','wide'))
    save(RAW/f'units-{m}-{s}.json.gz',out)


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=10) as ex:
        list(ex.map(worker,[(m,s) for m in ('econ','native') for s in range(59100,59148)]))
    result={}
    for m in ('econ','native'):
        r=[v for f in RAW.glob('units-'+m+'-*.gz') for v in read(f)]
        unit=100 if m=='native' else 1
        profit={(v['seed'],v['condition'],v['arm']):sum(.99**i*x[3]/unit for i,x in enumerate(v['rows'])) for v in r}
        pairs={}
        for a,b in [('conditional','original'),('conditional','generic'),('conditional','wide'),('conditional','feedback')]:
            d=[[profit[s,c,a]-profit[s,c,b] for c in ('stable','demand','rival','reverse')] for s in range(59100,59148)]
            pairs[a+'-'+b]=cluster(np.mean(d,axis=1))
        result[m]=dict(episodes=len(r),pairs=pairs)
    save(OUT/'units.json',result)

