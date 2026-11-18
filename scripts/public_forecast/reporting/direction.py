"""Replicate only the promising market while directly controlling for the direction of price changes."""
import hashlib
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from challenger import propose
from collect import OUT,RAW,ROOT,world,limits,decision,settle,draw,read,save,cases
from gates import metric
from analyze import cluster
from paid import initialize


def trial(task):
    s,p=task;m='econ'
    path=RAW/f'direction-{m}-{s}-{p}.json.gz'
    if path.exists():return
    c=('stable','demand','rival','reverse')[(s-80200)%4];w=world(m,s,c)
    h=[];old=None;branches={};point=None;failures=[]
    t0=23+4*(draw(s,'opportunity')%6);cut=read(OUT/'offer-freeze.json')['cuts'][m]
    directory=RAW/'direction-prospective';directory.mkdir(exist_ok=True)
    for t in range(1,61):
        lo,hi=limits(w)
        d=decision(w,h,old,p,s,lo,hi,failures)
        if t==t0:
            point=propose(m,h,d,lo,hi)
            point.update(seed=s,provider=p,t=t,condition=c,accept=bool(point['distinct'] and point['score']>cut and point['both_better']))
            seal=directory/f'{p}-{s}.json'
            if seal.exists():assert point==read(seal)
            else:save(seal,point)
            for name,mult in [('up',1.04),('down',.96)]:
                price=float(np.clip(d['draft']*mult,lo,hi))
                branches[name]=[w.clone(),list(h),dict(d,price=price)]
        for name,(other,rows,term) in branches.items():
            low,high=limits(other)
            if t!=t0:term=decision(other,rows,term,p,s,low,high,failures)
            rows.append(settle(other,term['price'],low,high));branches[name][2]=term
        h.append(settle(w,d['price'],lo,hi));old=d
    save(path,[dict(seed=s,provider=p,condition=c,opportunity=point,history=h,
              branches={k:v[1][t0-1:] for k,v in branches.items()},fallbacks=failures)])


def report():
    results=[]
    for r in cases('direction','econ'):
        v=r['opportunity'];t=v['t']
        assert v==read(RAW/'direction-prospective'/f"{r['provider']}-{r['seed']}.json")
        base=np.asarray(r['history'])[t-1:,3]
        actual={}
        for k,z in r['branches'].items():
            a=np.asarray(z);assert len(a)==61-t and len(r['history'])==60
            difference=a[:,3]-base
            actual[k]=float(difference@(.99**np.arange(len(difference))))
        chosen='down' if v['proposed']<v['draft'] else 'up'
        if v['distinct']:assert abs(r['branches'][chosen][0][6]-v['proposed'])<1e-8
        results.append(dict(**v,**actual,delta=actual[chosen] if v['distinct'] else 0,chosen=chosen))
    assert len(results)==80
    out={}
    for key in ('keep','up','down','random'):
        deltas=[]
        for v in results:
            other=0 if key=='keep' else .5*(v['up']+v['down']) if key=='random' else v[key]
            deltas.append(dict(v,delta=v['delta']-other))
        out['forecast_minus_'+key]=metric(deltas,[True]*len(deltas))
    for k in ('up','down'):
        out[k+'_minus_keep']=metric([dict(r,delta=r[k]) for r in results],[True]*len(results))
    out['gate']=metric(results,[r['accept'] for r in results])
    out['directions']={k:sum(v['chosen']==k for v in results) for k in ('up','down')}
    save(OUT/'direction-opportunities.json.gz',results);save(OUT/'direction-results.json',out)


if __name__=='__main__':
    code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    snapshot=dict(code=code,design=hashlib.sha256((ROOT/'experiment-specs/public-forecast-v1/followup/DIRECTION.md').read_bytes()).hexdigest(),
           offer_freeze=hashlib.sha256((OUT/'offer-freeze.json').read_bytes()).hexdigest(),seeds=[80200,80239])
    path=OUT/'direction-freeze.json'
    if path.exists():assert read(path)==snapshot
    else:save(path,snapshot)
    initialize()
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(trial,[(s,p) for s in range(80200,80240) for p in ('deepseek','doubao')]))
    report()

