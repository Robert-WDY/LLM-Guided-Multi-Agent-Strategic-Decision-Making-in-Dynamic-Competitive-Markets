"""Test incremental value beyond the successful feedback rule, at actual persistence."""
import itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import beta
from common import OUT,save,read
from duopoly import FAMILIES,ALPHAS,Opponent,environment,startup,simple,fitted,settle,lower_bound
from diagnostics import simulate,realized


def run(seed,family,arm,threshold=None,training=False):
    kw,c,_,_,hi=environment(ALPHAS[seed%3]);op=Opponent(seed,family,hi)
    h=[];errs=[];count=0;total=0.;last=peer=.55*hi
    for t in range(300):
        rival=op.price(t,last,peer)
        p=startup(t,hi,seed) if t<10 else last
        if t and t%10==0:
            draft=simple(h,hi);p=draft
            if arm!='B':
                mode='conditional' if arm.startswith('D') else 'generic'
                values=np.clip(draft*np.array([1,.95,1.05]),.01*hi,hi)
                rng=np.random.default_rng(seed*1000+t)
                fit=fitted(h,hi)
                search=simulate(h,hi,values,mode,True,rng.random((5,3)),fit)[2].mean(axis=1)
                best=1+np.argmax(search[1:])
                validation=simulate(h,hi,values[[0,best]],mode,True,rng.random((5,8)),fit)
                gains=validation[[2,4],1]-validation[[2,4],0]
                bounds=np.array([lower_bound(list(g)) for g in gains])
                passed=min(bounds)>0 and np.all(gains>=0)
                if arm.endswith('raw'):
                    if search[best]>search[0]:p=values[best]
                elif passed and (threshold is None or np.all(bounds>threshold)):p=values[best]
                if training and passed:
                    actual=[]
                    for price in [draft,values[best]]:
                        l,r=last,peer;v=0.;cum=[]
                        for j in range(5):
                            opp=op.price(t+j,l,r)
                            v+=.95**j*settle(kw,c,(price,opp))[1][0]
                            if j in (2,4):cum.append(v)
                            l,r=price,opp
                        actual.append(cum)
                    errs.append((bounds-(np.array(actual[1])-actual[0])).tolist())
                count+=int(abs(p-draft)>1e-9)
        q,pi=settle(kw,c,(p,rival));h.append((p,rival,q[0],pi[0]));last,peer=p,rival
        total+=.99**t*pi[0]
    return dict(seed=seed,family=family,arm=arm,profit=total,interventions=count,errors=errs)


def worker(args):
    s,cal=args
    return [run(s,f,a,cal.get(a)) for f,a in itertools.product(FAMILIES,('B','C','D','Craw','Draw'))]


if __name__=='__main__':
    cal={}
    for a in ('C','D'):
        rows=[run(s,f,a,training=True) for s,f in itertools.product(range(3500,3512),FAMILIES[:3])]
        errors=sum([r['errors'] for r in rows],[])
        assert len({r['seed'] for r in rows if r['errors']})>=8 and len(errors)>=12
        cal[a]=np.maximum(0,np.quantile(errors,.9,axis=0)).tolist()
    path=OUT/'strong-calibration.json'
    if path.exists():assert read(path)==cal
    else:save(path,cal)
    with ProcessPoolExecutor(max_workers=5) as ex:
        rows=sum(ex.map(worker,((s,cal) for s in range(3600,3648))),[])
    lookup={(r['seed'],r['family'],r['arm']):r for r in rows};stats={}
    for a,b in [('C','B'),('D','B'),('D','C'),('Draw','Craw'),('Draw','B'),('Craw','B')]:
        diffs=[np.mean([lookup[s,f,a]['profit']-lookup[s,f,b]['profit'] for f in FAMILIES]) for s in range(3600,3648)]
        ci=np.quantile(np.random.default_rng(61).choice(diffs,(5000,48)).mean(axis=1),[.025,.975])
        sub=[r for r in rows if r['arm']==a]
        acted={r['seed'] for r in sub if r['interventions']}
        loss={r['seed'] for r in sub if r['interventions'] and r['profit']<lookup[r['seed'],r['family'],'B']['profit']-1e-8}
        upper=None if not acted else 1 if len(loss)==len(acted) else float(beta.ppf(.95,len(loss)+1,len(acted)-len(loss)))
        stats[a+'-'+b]=dict(mean=float(np.mean(diffs)),ci=ci.tolist(),coverage=sum(r['interventions'] for r in sub)/(len(sub)*29),loss_seeds=len(loss),acted_seeds=len(acted),risk_upper_vs_B=upper)
    save(OUT/'strong-episodes.json.gz',rows);save(OUT/'strong-draft.json',stats)
    print(stats)

