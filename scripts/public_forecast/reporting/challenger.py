"""Prospective terminal evaluation of candidates generated directly by the sales forecaster."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ThreadPoolExecutor
import hashlib
import numpy as np
from scipy.stats import spearmanr
from collect import OUT,RAW,ROOT,world,limits,decision,settle,draw,exchange,cases,read,save
from models import bundle,diagnosis,realized_skill
from gates import metric,truth_delta
from paid import initialize
from analyze import cluster


def propose(m,h,d,lo,hi):
    a=np.asarray(h);draft=d['draft'];base=float(np.clip(draft,lo,hi))
    # The hint bounds and enforceable floor/ceiling limit what this company can submit.
    grid=np.clip(draft*np.array([1.,.96,1.04]),lo,hi)
    context='full' if m=='native' else 'basic'
    p,sc,b,_=diagnosis(bundle(m),h,context,[(v,lo,hi) for v in grid])
    margin=np.median(a[-4:,3]/np.maximum(.0001,a[-4:,2]))
    margin_hat=margin+exchange(h)*(grid-a[-1,0])
    values=p[-3:,0]*margin_hat
    idx=[i for i in (1,2) if abs(grid[i]-base)>1.e-7]
    best=max(idx,key=lambda i:values[i]) if idx else 0
    units=a[-8:,2]
    residual=np.mean(abs(units-p[-11:-3,0]))
    score=(values[best]-values[0])/max(1.e-8,abs(margin_hat[best])*residual,.02*np.mean(abs(a[-8:,3])))
    skills=realized_skill(a[-8:][:,[2,1]],b[-11:-3],p[-11:-3],sc[-11:-3])
    return dict(draft=draft,proposed=float(grid[best]) if idx else draft,score=float(score),
                both_better=bool(min(skills)>0),distinct=bool(idx),skills=skills.tolist())


def freeze():
    cuts={}
    for m in ('econ','native'):
        xs=[]
        for r in cases('rcal',m):
            d=r['opportunity'];xs.append(propose(m,r['history'][:d['t']-1],d,d['low'],d['high'])['score'])
        cuts[m]=max(0.,float(np.quantile(xs,.75)))
    obj=dict(cuts=cuts,code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        design=hashlib.sha256((ROOT/'experiment-specs/public-forecast-v1/followup/FORECAST-OFFER.md').read_bytes()).hexdigest())
    path=OUT/'offer-freeze.json'
    if path.exists():assert read(path)==obj
    else:save(path,obj)


def one(task):
    m,p,s=task
    path=RAW/f'offer-{m}-{s}-{p}.json.gz'
    if path.exists():return
    cs=('stable','demand','rival','reverse')
    c=cs[(s-79200)%4];w=world(m,s,c)
    h=[];old=None;other=None;alt=None;branch=[];point=None;failed=[]
    t0=23+4*(draw(s,'opportunity')%6);cut=read(OUT/'offer-freeze.json')['cuts'][m]
    (RAW/'offer-prospective').mkdir(exist_ok=True)
    for t in range(1,61):
        lo,hi=limits(w)
        d=decision(w,h,old,p,s,lo,hi,failed)
        if t==t0:
            point=propose(m,h,d,lo,hi);point.update(t=t,seed=s,provider=p,condition=c,
                 accept=bool(point['distinct'] and point['score']>cut and point['both_better']))
            seal=RAW/'offer-prospective'/f'offer-{m}-{p}-{s}.json'
            if seal.exists():assert read(seal)==point
            else:save(seal,point)
            alt=w.clone();branch=list(h);other=dict(d,price=point['proposed'])
        if alt is not None:
            low,high=limits(alt)
            if t!=t0:other=decision(alt,branch,other,p,s,low,high,failed)
            branch.append(settle(alt,other['price'],low,high))
        h.append(settle(w,d['price'],lo,hi));old=d
    save(path,[dict(seed=s,provider=p,market=m,condition=c,history=h,opportunity=point,alternative=branch[t0-1:],fallbacks=failed)])


def report():
    out={}
    for m in ('econ','native'):
        gs=[]
        for r in cases('offer',m):
            d=r['opportunity'];assert len(r['history'])==60 and len(r['alternative'])==61-d['t']
            assert d==read(RAW/'offer-prospective'/f"offer-{m}-{r['provider']}-{r['seed']}.json")
            gs.append(dict(**d,delta=truth_delta(r)/(100 if m=='native' else 1)))
        assert len(gs)==64
        out[m]=dict(all=metric(gs,[r['distinct'] for r in gs]),positive=metric(gs,[r['score']>0 for r in gs]),
                    gate=metric(gs,[r['accept'] for r in gs]),score_rank=float(spearmanr([r['score'] for r in gs],[r['delta'] for r in gs]).statistic))
        save(OUT/f'{m}-offer-opportunities.json.gz',gs)
    save(OUT/'offer-results.json',out)


if __name__=='__main__':
    if 'freeze' in sys.argv:freeze()
    else:
        initialize();assert read(OUT/'offer-freeze.json')['code']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(one,[(m,p,s) for s in range(79200,79232) for m in ('econ','native') for p in ('deepseek','doubao')]))
        report()

