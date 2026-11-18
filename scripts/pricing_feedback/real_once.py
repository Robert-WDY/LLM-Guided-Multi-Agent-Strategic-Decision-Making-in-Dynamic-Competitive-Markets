"""Actual-agent terminal outcomes for a single prospectively scheduled substitution."""
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.stats import beta
from study import RAW,OUT,World,WARM,HOLD,decide,read,save
from units import exchange
from paid import initialize,ask,summary
from selector import fit,predict
from analyze import cluster,selection


def trial(task):
    phase,p,m,s=task
    output=RAW/f'once-{phase}-{p}-{m}-{s}.json.gz'
    if output.exists():return
    c=('stable','demand','rival','reverse')[s%4]
    t0=15+4*((s//4)%5)
    fallbacks=[]
    def draft(*args):
        try:return ask(p,s,*args)
        except RuntimeError as exc:
            if 'valid draft after bounded attempts' not in str(exc):raise
            fallbacks.append(len(args[1]))
            return args[4]
    from study import replay
    base=read(RAW/f'real2-{p}-{m}-{s}-feedback.json.gz') if phase=='train' else replay(m,s,c,'feedback',agent=draft)
    w=World(m,s,c)
    history=[];old=None;obs=None;alternate=[]
    for t in range(1,61):
        left=61-t
        if t<t0:
            contract=next(d for d in base['contracts'] if d['start']<=t<=d['end'])
            old=contract
            row=w.execute(old['price']);assert row==base['rows'][t-1]
        else:
            d=decide(t,left,w.ref,w.low,w.high,history,old)
            if t==t0:
                original=next(d['draft'] for d in base['contracts'] if d['start']==t)
                k=exchange(history)
                h=[[v[0]*k,v[1]*k,v[2],v[3]] for v in history]
                cand=decide(t,left,w.ref*k,w.low*k,w.high*k,h,old,mode='conditional',gate=-1.e30,original=original*k)
                d=dict(d,price=cand['proposed']/k,draft=original,proposed=cand['proposed']/k,score=cand['score'])
                obs=dict(t=t,score=d['score'],draft=original,proposed=d['price'],
                         executable=w.active() and abs(w.legal(d['price'])-w.legal(original))>1.e-8)
            elif d['start']==t and w.active():
                original=draft(m,history,w.low,w.high,d['draft'],min(left,HOLD),left)
                d=decide(t,left,w.ref,w.low,w.high,history,old,original=original)
            old=d;row=w.execute(d['price']);alternate.append(row)
        history.append(row)
    diff=[a[3]-b[3] for a,b in zip(alternate,base['rows'][t0-1:])]
    obs.update(delta=sum(.99**i*v for i,v in enumerate(diff)),short=sum(.99**i*v for i,v in enumerate(diff[:4])))
    save(output,dict(seed=s,provider=p,market=m,condition=c,rows=base['rows'],alternate=alternate,
                     labels=[obs],fallbacks=fallbacks))


def rows(phase,m):
    return [read(f) for f in sorted(RAW.glob(f'once-{phase}-*-{m}-*.gz'))]


def score(r,model):
    v=r['labels'][0]
    return dict(v,seed=r['seed'],provider=r['provider'],score=predict(model,r['rows'][:v['t']-1],v))


def policy(data,cutoff,unit):
    seeds=sorted({v['seed'] for v in data})
    chosen=[v for v in data if v['executable'] and v['score']>cutoff]
    exposed={v['seed'] for v in chosen};harmed={v['seed'] for v in chosen if v['delta']<-1.e-8}
    gains=[sum(.99**(v['t']-1)*v['delta']/unit for v in chosen if v['seed']==s)/sum(v['seed']==s for v in data) for s in seeds]
    n,k=len(exposed),len(harmed)
    return dict(impact=cluster(gains),selected=len(chosen),opportunities=len(data),exposed=n,harmed=k,
            risk_upper=None if n==0 else 1 if n==k else beta.ppf(.95,k+1,n-k))


def evaluate(phase):
    output={}
    for m in ('econ','native'):
        rs=rows(phase,m);unit=100 if m=='native' else 1
        if phase=='train':
            assert len(rs)==64
            oof=[]
            for seed in sorted({r['seed'] for r in rs}):
                model=fit([r for r in rs if r['seed']!=seed])
                oof.extend(score(r,model) for r in rs if r['seed']==seed)
            cuts=np.quantile([v['score'] for v in oof],[.5,.75,.9])
            options=[dict(cut=c,**policy(oof,c,unit)) for c in cuts]
            ok=[v for v in options if v['selected']>=.1*len(rs) and v['exposed'] and v['harmed']/v['exposed']<=.1 and v['impact']['ci'][0]>0]
            output[m]=dict(model=fit(rs),options=options,diagnostic=cuts[1],
                           strict=max(ok,key=lambda v:v['impact']['mean'])['cut'] if ok else 1.e30)
        else:
            assert len(rs)==96
            rule=read(OUT/'once-train.json')[m]
            data=[score(r,rule['model']) for r in rs]
            output[m]=dict(diagnostic=policy(data,rule['diagnostic'],unit),strict=policy(data,rule['strict'],unit),
                      all=policy(data,-1.e30,unit),ranking=selection(data,rule['diagnostic'],unit),
                      sign_flip=np.mean([(v['short']>1.e-8)!=(v['delta']>1.e-8) for v in data]),
                      fallbacks=sum(len(r['fallbacks']) for r in rs),total=len(rs))
    save(OUT/f'once-{phase}.json',output)


if __name__=='__main__':
    initialize()
    for phase,start,n in [('train',58100,32),('confirm',60100,48)]:
        with ThreadPoolExecutor(max_workers=20) as ex:
            list(ex.map(trial,[(phase,p,m,s) for s in range(start,start+n)
                               for p in ('deepseek','doubao') for m in ('econ','native')]))
        evaluate(phase)
    summary()

