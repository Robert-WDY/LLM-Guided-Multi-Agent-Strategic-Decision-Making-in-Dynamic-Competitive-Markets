"""One-cycle deviations labelled by the subsequent feedback policy's entire remaining profit."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from study import World,decide,RAW,OUT,HOLD,WARM,GATE_OFF,read,save,replay
from selector import fit,predict
from analyze import cluster,selection
from scipy.stats import beta


def study_labels(m,s,c):
    baseline=replay(m,s,c,'feedback')
    world=World(m,s,c)
    history=[];prev=None
    for i,row in enumerate(baseline['rows']):
        t=i+1;left=world.length-i
        plan=decide(t,left,world.ref,world.low,world.high,history,prev)
        if plan['start']==t and t>WARM and left>=HOLD and world.active():
            cand=decide(t,left,world.ref,world.low,world.high,history,prev,mode='conditional',gate=-1.e30)
            shadow=world.clone()
            alt=dict(cand,price=cand['proposed'])
            other=list(history)
            delta=short=0.
            for j,b in enumerate(baseline['rows'][i:]):
                if j:
                    alt=decide(t+j,left-j,world.ref,world.low,world.high,other,alt)
                y=shadow.execute(alt['price'])
                other.append(y)
                delta+=.99**j*(y[3]-b[3])
                if j==HOLD-1:short=delta
            baseline['labels'].append(dict(t=t,score=cand['score'],proposed=cand['proposed'],
                                           draft=cand['draft'],short=short,delta=delta))
        assert world.execute(plan['price'])==row
        history.append(row);prev=plan
    return baseline


def profit(row):
    return sum(.99**j*v[3] for j,v in enumerate(row['rows']))


def one(task):
    phase,m,s=task
    cond=['stable','demand','rival']+(['reverse'] if phase=='confirm' else [])
    result=[]
    if phase!='train':
        spec=read(OUT/'long-model.json')[m]
        thresholds=spec['thresholds']
        if phase=='confirm':
            gate=read(OUT/'long-gates.json')[m]['strict']
            thresholds=[gate,spec['diagnostic']]
            old=read(OUT/'learned-thresholds.json')[m]['diagnostic']
            short_model=read(OUT/'selectors.json')[m]
    for c in cond:
        base=study_labels(m,s,c) if phase!='calibrate' else replay(m,s,c,'feedback')
        result.append(base)
        if phase=='train':continue
        for i,threshold in enumerate(thresholds):
            r=replay(m,s,c,'diagnostic',threshold,selector=spec['model'])
            r['arm']='gate'+str(i)
            result.append(r)
        if phase=='confirm':
            for a in ['conditional','generic','wide','fixed','short']:
                r=replay(m,s,c,'diagnostic',old,selector=short_model) if a=='short' else replay(m,s,c,a)
                r['arm']=a
                result.append(r)
    save(RAW/f'long-{phase}-{m}-{s}.json.gz',result)


def cases(phase,m):
    return [r for f in sorted(RAW.glob(f'long-{phase}-{m}-*.gz')) for r in read(f)]


def policy(rows,arm,unit):
    base={(r['seed'],r['condition']):profit(r) for r in rows if r['arm']=='feedback'}
    gains={};exposed=set();loss=set();cover=actions=0
    for r in rows:
        if r['arm']!=arm:continue
        seed=r['seed'];key=seed,r['condition']
        diff=(profit(r)-base[key])/unit
        gains.setdefault(seed,[]).append(diff)
        plan=[d for d in r['contracts'] if d['start']>WARM and d['active']]
        count=sum(d['actual_override'] for d in plan)
        cover+=len(plan);actions+=count
        if count:
            exposed.add(seed)
            if diff < -1.e-8:loss.add(seed)
    n,k=len(exposed),len(loss)
    return dict(profit=cluster([np.mean(v) for v in gains.values()]),coverage=actions/cover,
                actions=actions,opportunities=cover,harmed=k,exposed=n,risk_upper=None if n==0 else 1 if n==k else beta.ppf(.95,k+1,n-k))


def process(phase):
    markets=('econ','native');res={}
    for m in markets:
        rows=cases(phase,m);unit=100 if m=='native' else 1
        if phase=='train':
            model=fit(rows)
            scores=[predict(model,r['rows'][:v['t']-1],v) for r in rows for v in r['labels']]
            thresholds=list(np.quantile(scores,[.5,.75,.9]))+[.9]
            res[m]=dict(model=model,thresholds=thresholds,diagnostic=thresholds[1])
        elif phase=='calibrate':
            spec=read(OUT/'long-model.json')[m]
            options=[dict(threshold=th,**policy(rows,'gate'+str(i),unit)) for i,th in enumerate(spec['thresholds'])]
            ok=[v for v in options if v['coverage']>=.10 and v['exposed'] and v['harmed']/v['exposed']<=.1 and v['profit']['ci'][0]>0]
            res[m]=dict(strict=max(ok,key=lambda v:v['profit']['mean'])['threshold'] if ok else GATE_OFF,options=options)
        else:
            spec=read(OUT/'long-model.json')[m]
            raw=[dict(v,score=predict(spec['model'],r['rows'][:v['t']-1],v)) for r in rows for v in r['labels']]
            flip=np.mean([(v['short']>1.e-8)!=(v['delta']>1.e-8) for v in raw])
            results={a:policy(rows,a,unit) for a in ('gate0','gate1','short','conditional','generic','wide','fixed')}
            res[m]=dict(episodes=len(rows),long_vs_short_label_flip=flip,diagnostic=selection(raw,spec['diagnostic'],unit),
                        strict=selection(raw,read(OUT/'long-gates.json')[m]['strict'],unit),policies=results)
    save(OUT/('long-'+{'train':'model','calibrate':'gates','confirm':'confirm'}[phase]+'.json'),res)


if __name__=='__main__':
    phase=sys.argv[1]
    start,n={'train':(57000,24),'calibrate':(57100,24),'confirm':(57200,48)}[phase]
    with ProcessPoolExecutor(max_workers=10) as ex:
        list(ex.map(one,[(phase,m,s) for m in ('econ','native') for s in range(start,start+n)]))
    process(phase)

