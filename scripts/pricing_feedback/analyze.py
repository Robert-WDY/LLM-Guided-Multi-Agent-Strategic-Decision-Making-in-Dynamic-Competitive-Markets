"""Recompute paired estimates and prospective selection diagnostics from settled traces."""
import sys
from collections import defaultdict
import hashlib
import numpy as np
from scipy.stats import beta, rankdata
from study import RAW, OUT, ROOT, GATE_OFF, read, save


def cluster(values):
    x = np.asarray(values)
    draws = np.random.default_rng(67319).choice(x, size=(4000,len(x))).mean(axis=1)
    return dict(mean=float(x.mean()),ci=np.quantile(draws,[.025,.975]).tolist(),n=len(x))


def load(phase, market):
    return [r for f in sorted(RAW.glob(f'{phase}-{market}-*.json.gz')) for r in read(f)]


def prepare_labels(rows):
    return [dict(v,seed=r['seed'],condition=r['condition']) for r in rows for v in r['labels']]


def selection(labels, gate, unit):
    d=np.array([v['delta']/unit for v in labels])
    s=np.array([v['score'] for v in labels])
    use=s>gate
    pos=d>1.e-8
    n_pos,n_neg=pos.sum(),(~pos).sum()
    auc=(rankdata(s)[pos].sum()-n_pos*(n_pos+1)/2)/(n_pos*n_neg) if n_pos and n_neg else None
    out=dict(n=len(d),accepted=int(use.sum()),auc=auc,
             positive_missed=float((pos & ~use).sum()/max(1,n_pos)))
    for label,mask in [('accepted',use),('rejected',~use),('all',s==s)]:
        out[label+'_outcomes']=(dict(loss=float((d[mask]<-1.e-8).mean()),gain=float(d[mask].mean()),
                                    positive=float(pos[mask].mean()),p10=float(np.quantile(d[mask],.1)))
                               if mask.any() else None)
    return out


def calibrate(learned=False):
    metrics={}
    for m in ('econ','native'):
        rows=load('cal',m)
        labels=prepare_labels(rows)
        if learned:
            from selector import predict
            model=read(OUT/'selectors.json')[m]
            labels=[dict(v,seed=r['seed'],score=predict(model,r['rows'][:v['t']-1],v)) for r in rows for v in r['labels']]
        scores=[v['score'] for v in labels]
        seeds=sorted({v['seed'] for v in labels})
        candidates=[max(0.,float(np.quantile(scores,q))) for q in (0,.25,.5,.65,.75,.85,.9)]
        all_trials=[]
        for th in sorted(set(candidates)):
            select=[v for v in labels if v['score']>th]
            if not select:
                continue
            impact=cluster([sum(v['delta'] for v in select if v['seed']==s)/sum(v['seed']==s for v in labels) for s in seeds])
            coverage=len(select)/len(labels)
            loss=sum(v['delta']< -1.e-8 for v in select)/len(select)
            all_trials.append(dict(threshold=th,impact=impact,coverage=coverage,loss=loss))
        ok=[v for v in all_trials if v['coverage']>=.10 and v['loss']<=.10 and v['impact']['ci'][0]>0]
        gate=max(ok,key=lambda v:v['impact']['mean'])['threshold'] if ok else GATE_OFF
        metrics[m]=dict(threshold=gate,diagnostic=max(0.,float(np.quantile(scores,.75))),
                        n_seeds=len(seeds),n_labels=len(scores),grid=all_trials)
    save(OUT/('learned-thresholds.json' if learned else 'thresholds.json'),metrics)


def summarize(rows, gate, diagnostic):
    seeds=sorted({r['seed'] for r in rows})
    conditions=sorted({r['condition'] for r in rows})
    unit=100 if rows[0]['market']=='native' else 1
    index={(r['seed'],r['condition'],r['arm']):r for r in rows}
    def total(r,field=3,post=False):
        offset=r['when']-1 if post else 0
        return sum(.99**i*v[field] for i,v in enumerate(r['rows'][offset:]))/unit
    values={key:total(v) for key,v in index.items()}
    pairs=[('feedback','fixed'),('feedback','rule'),('conditional','feedback'),('conditional','generic'),
           ('conditional','wide'),('gated','feedback'),('gated','fixed'),('gated','wide'),('wide','feedback')]
    if any(r['arm']=='learned' for r in rows):
        pairs += [(a,b) for a in ('learned','diagnostic') for b in ('feedback','wide','conditional','fixed')]
    diff={}
    for a,b in pairs:
        per=[[values[s,c,a]-values[s,c,b] for c in conditions] for s in seeds]
        diff[a+'-'+b]=dict(cluster=cluster(np.mean(per,axis=1)),
                            by_condition={c:cluster(np.asarray(per)[:,i]) for i,c in enumerate(conditions)})
    adaptation={}
    for arm in ('feedback','fixed','generic','conditional','gated'):
        for c in set(conditions)-{'stable'}:
            delays=[];post=[];first_visible=[];observed=[];requested=[]
            for s in seeds:
                r,stable=index[s,c,arm],index[s,'stable',arm]
                offset=r['when']-1
                dif=[abs(v[0]/u[0]-1)>=.01 for v,u in zip(r['rows'][offset:],stable['rows'][offset:])]
                delays += [next((i for i in range(len(dif)-1) if dif[i] and dif[i+1]),None)]
                shock_rows=list(zip(r['rows'][offset:],stable['rows'][offset:]))
                first_visible.append(next((i+1 for i,(v,u) in enumerate(shock_rows) if v[:4]!=u[:4]),None))
                nxt=next((x['start']-r['when'] for x,y in zip(r['contracts'],stable['contracts'])
                          if x['start']>=r['when'] and abs(x['price']/y['price']-1)>=.01),None)
                requested.append(nxt)
                post.append(total(r,post=True)-total(index[s,c,'freeze'],post=True))
            observed=[d for d in delays if d is not None]
            med=lambda x:float(np.median([i for i in x if i is not None])) if any(i is not None for i in x) else None
            adaptation[arm+'-'+c]=dict(n=len(seeds),responded=len(observed),median_responders=med(delays),
                    median_requested_responders=med(requested),requested_responded=sum(x is not None for x in requested),
                    median_first_observable=med(first_visible),post_vs_frozen=cluster(post))
    risk={}
    for arm in sorted({r['arm'] for r in rows}-{'feedback','fixed','rule','freeze'}):
        active=set();loss=set();opportunities=actions=0
        for s in seeds:
            for c in conditions:
                r=index[s,c,arm]
                ds=[d for d in r['contracts'] if d['start']>6 and d['active']]
                accepted=sum(d.get('actual_override',abs(d['price']-d['draft'])>1.e-8) for d in ds)
                opportunities+=len(ds);actions+=accepted
                if accepted:
                    active.add(s)
                    if values[s,c,arm]<values[s,c,'feedback']-1.e-8:loss.add(s)
        n,k=len(active),len(loss)
        risk[arm]=dict(opportunities=opportunities,actions=actions,coverage=actions/max(1,opportunities),
                       exposed_seeds=n,harmed_seeds=k,upper95=None if n==0 else 1 if n==k else beta.ppf(.95,k+1,n-k))
    labels=prepare_labels(rows)
    return dict(episodes=len(rows),rounds=sum(len(r['rows']) for r in rows),seeds=len(seeds),units='yuan' if unit==100 else 'benchmark',
        pairs=diff,adaptation=adaptation,risk=risk,
        means={a:np.mean([v for (s,c,arm),v in values.items() if arm==a]) for a in {r['arm'] for r in rows}},
        surplus={a:np.mean([total(r,field=4) for r in rows if r['arm']==a]) for a in {r['arm'] for r in rows}},
        gate=selection(labels,gate,unit), diagnostic_gate=selection(labels,diagnostic,unit),
        label_curve=[dict(quantile=q,**selection(labels,np.quantile([v['score'] for v in labels],q),unit))
                     for q in (.25,.5,.75,.9)])


if __name__=='__main__':
    phase=sys.argv[1]
    if phase=='fit':
        from selector import fit
        save(OUT/'selectors.json',{m:fit(load('dev',m)) for m in ('econ','native')})
    elif phase=='cal':
        calibrate()
        calibrate(True)
    else:
        th=read(OUT/'thresholds.json') if phase=='confirm' else {}
        result={}
        for m in ('econ','native'):
            thresholds=th.get(m,dict(threshold=0,diagnostic=0))
            result[m]=summarize(load(phase,m),thresholds['threshold'],thresholds['diagnostic'])
            if phase=='confirm':
                from selector import predict
                model=read(OUT/'selectors.json')[m]
                rules=read(OUT/'learned-thresholds.json')[m]
                labs=[dict(v,score=predict(model,r['rows'][:v['t']-1],v)) for r in load(phase,m) for v in r['labels']]
                result[m]['learned_selection']={k:selection(labs,rules[k],100 if m=='native' else 1) for k in ('threshold','diagnostic')}
        save(OUT/(phase+'.json'),result)

