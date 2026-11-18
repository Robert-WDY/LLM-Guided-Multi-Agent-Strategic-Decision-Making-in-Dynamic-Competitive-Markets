"""Opportunity scoring with an explicit settlement cutoff and frozen thresholds."""
from functools import lru_cache
import hashlib
import json
import numpy as np
import joblib
from scipy.stats import beta
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from collect import OUT,RAW,ROOT,read,save,cases,exchange
from models import bundle,diagnosis,realized_skill
from analyze import cluster

KINDS=('draft','forecast','full')


def features(m,h,point):
    a=np.asarray(h);ctx='full' if m=='native' else 'basic'
    d,o=point['draft'],point['proposed']
    prices=[(v,point['low'],point['high']) for v in (d,o)]
    pred,scale,carry,pr=diagnosis(bundle(m),h,ctx,prices)
    target=a[6:,[2,1]];window=min(8,len(target))
    skl=realized_skill(target[-window:],carry[:-2][-window:],pred[:-2][-window:],scale[:-2][-window:])
    base=[point['score'],o/d-1,d/a[-1,0]-1,a[-1,1]/a[-1,0]-1,
          (o-d)*(a[np.argmax(a[-8:,3])-8,0]-d)/(d*d)]
    margin=np.median(a[-4:,3]/np.maximum(1.e-4,a[-4:,2]))
    k=exchange(h)
    eff=np.clip([d,o],point['low'],point['high'])
    margin_hat=margin+k*(eff-a[-1,0])
    implied=(pred[-1,0]*margin_hat[1]-pred[-2,0]*margin_hat[0])/max(1,np.mean(abs(a[-8:,3])))
    fc=[(pred[-2,0]-carry[-2,0])/scale[-2,0],(pred[-1,0]-pred[-2,0])/scale[-1,0],
        (pred[-2,1]-carry[-2,1])/scale[-2,1],implied,pr[-1,1],pr[-1,2],(eff[1]-eff[0])/d]
    full=[*base,*fc,*skl,implied*min(skl)]
    return {key:np.clip(arr,-12,12).tolist() for key,arr in
            [('draft',base),('forecast',base+fc),('full',full),('skill',skl)]}


def truth_delta(r):
    t=r['opportunity']['t']
    diff=np.asarray(r['alternative'])[:,3]-np.asarray(r['history'])[t-1:,3]
    return float(diff@(.99**np.arange(len(diff))))


def metric(rows,selected):
    seed=sorted({r['seed'] for r in rows})
    values={s:[] for s in seed};bad=set();exposed=set()
    for r,yes in zip(rows,selected):
        delta=r['delta']*.99**(r['t']-1)
        values[r['seed']].append(delta if yes else 0)
        if yes:
            exposed.add(r['seed'])
            if delta< -1.e-8:bad.add(r['seed'])
    n,k=len(exposed),len(bad)
    def subgroup(yes):
        a=[r['delta'] for r,p in zip(rows,selected) if p==yes]
        return dict(n=len(a),mean=float(np.mean(a)),positive=float(np.mean(np.array(a)>1.e-8)),
                    loss=float(np.mean(np.array(a)<-1.e-8))) if a else None
    return dict(impact=cluster([np.mean(values[s]) for s in seed]),accepted=subgroup(True),rejected=subgroup(False),
                coverage=float(np.mean(selected)),harmed=k,exposed=n,
                risk_upper=None if not n else 1 if n==k else float(beta.ppf(.95,k+1,n-k)))


def inputs_for(phase,m):
    rs=cases(phase,m);unit=100 if m=='native' else 1
    return [dict(seed=r['seed'],provider=r['provider'],t=r['opportunity']['t'],
                 delta=truth_delta(r)/unit,x=features(m,r['history'][:r['opportunity']['t']-1],r['opportunity'])) for r in rs]


def train(m):
    tr,cal=[inputs_for(p,m) for p in ('rtrain','rcal')]
    assert len(tr)==48 and len(cal)==32
    fitted={};cuts={};calibration={}
    for kind in KINDS:
        mod=make_pipeline(StandardScaler(),LogisticRegression(C=.12,max_iter=500))
        mod.fit([r['x'][kind] for r in tr],[r['delta']>1.e-8 for r in tr])
        scores=mod.predict_proba([r['x'][kind] for r in cal])[:,1]
        grid=[]
        for cut in np.quantile(scores,[.5,.75,.9]):
            grid.append(dict(cut=float(cut),**metric(cal,scores>cut)))
        ok=[g for g in grid if g['coverage']>=.10 and g['exposed'] and g['harmed']/g['exposed']<=.10 and g['impact']['ci'][0]>0]
        cuts[kind]=dict(strict=max(ok,key=lambda g:g['impact']['mean'])['cut'] if ok else 2.,
                        diagnostic=float(np.quantile(scores,.75)))
        fitted[kind]=mod
        calibration[kind]=grid
    joblib.dump((fitted,cuts),OUT/f'{m}-gates.joblib')
    save(OUT/f'{m}-thresholds.json',dict(cuts=cuts,calibration=calibration))


@lru_cache
def gate(m):
    return joblib.load(OUT/f'{m}-gates.joblib')


def record(m,provider,seed,h,point):
    x=features(m,h,point);mods,cuts=gate(m)
    scores={k:float(mods[k].predict_proba([x[k]])[0,1]) for k in KINDS}
    return dict(seed=seed,provider=provider,market=m,t=point['t'],scores=scores,skill=x['skill'],
                both_better=bool(min(x['skill'])>0),decisions={k:{name:bool(scores[k]>val) for name,val in cuts[k].items()} for k in KINDS},
                history_sha256=hashlib.sha256(json.dumps(h,separators=(',',':')).encode()).hexdigest(),
                draft=point['draft'],proposed=point['proposed'])


def seal(m,provider,seed,h,point):
    folder=RAW/'prospective';folder.mkdir(exist_ok=True)
    path=folder/f'{m}-{provider}-{seed}.json'
    v=record(m,provider,seed,h,point)
    if path.exists():assert v==read(path),'prospective record disagrees on resume'
    else:save(path,v)


if __name__=='__main__':
    import sys
    train(sys.argv[1])

