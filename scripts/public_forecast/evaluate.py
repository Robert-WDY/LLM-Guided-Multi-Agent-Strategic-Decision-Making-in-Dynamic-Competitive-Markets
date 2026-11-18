"""Independent seed-clustered forecast, attribution and prospective policy readout."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix,roc_auc_score
from collect import RAW,OUT,ROOT,read,save,cases
from models import bundle,diagnosis,truth,flags,design,attr_features
from gates import metric,record,KINDS,truth_delta
from analyze import cluster


def batch(phase,m):
    market=bundle(m);active='full' if m=='native' else 'basic'
    controls,naive=joblib.load(OUT/f'{m}-controls.joblib')
    results=[];gates=[]
    for r in cases(phase,m):
        a=np.asarray(r['history']);target=a[6:,[2,1]];y=truth(r)
        pred,sc,base,probs=diagnosis(market,a,active)
        predictions={'carry':base,'full':pred}
        if m=='native':
            predictions['basic']=diagnosis(market,a,'basic')[0]
            designs=[design(a,True,quote)[0] for quote in (False,True)]
            for key in ('demand','no_limits','no_stockout'):
                v=[]
                for j in (0,1):
                    model,take=controls[key,j]
                    v.append(np.maximum(0,base[:,j]+sc[:,j]*model.predict(designs[j][:,take])))
                predictions[key]=np.column_stack(v)
        z=probs.argmax(axis=1)
        conservative=flags(probs,market['alert'][active])
        simple=naive.predict(attr_features(a,base,sc,base,m=='native'))
        start=r['when']-7
        row=dict(seed=r['seed'],provider=r['provider'],condition=r['condition'],when=r['when'],up=r['up'],
                 ae={k:abs(v-target).mean(axis=0).tolist() for k,v in predictions.items()},
                 actual=y.tolist(),argmax=z.tolist(),alert=conservative.tolist(),simple=simple.tolist())
        results.append(row)
        if phase=='rtest':
            s=r['seed'];p=r['provider'];d=r['opportunity'];t=d['t']
            sealed=read(RAW/'prospective'/f'{m}-{p}-{s}.json')
            assert sealed==record(m,p,s,r['history'][:t-1],d)
            scaled=100 if m=='native' else 1
            delta=truth_delta(r)/scaled
            future=np.mean(abs(target[t-7:t-3]-base[t-7:t-3])/sc[t-7:t-3],axis=0)-np.mean(abs(target[t-7:t-3]-pred[t-7:t-3])/sc[t-7:t-3],axis=0)
            gates.append(dict(**sealed,delta=delta,future_both=bool(np.min(future)>0),
                    term_changed=any(v[0]!=u[0] for v,u in zip(a[t-1:t+3],r['alternative'][:4])),
                    fallback_events=len(r['fallbacks'])))
    return results,gates


def summarize_forecast(rows):
    def rates(rs):
        def average(vals):return np.asarray(vals).mean(axis=0)
        seeds=sorted({r['seed'] for r in rs})
        stat={}
        for key in rs[0]['ae']:
            agg=np.array([average([r['ae'][key] for r in rs if r['seed']==s]) for s in seeds])
            carry=np.array([average([r['ae']['carry'] for r in rs if r['seed']==s]) for s in seeds])
            stat[key]={label:dict(mae=agg[:,j].mean(),delta=cluster(agg[:,j]-carry[:,j]),
                                  reduction=1-agg[:,j].mean()/carry[:,j].mean()) for j,label in enumerate(('sales','quote'))}
        return stat
    out=dict(all=rates(rows),by_condition={c:rates([r for r in rows if r['condition']==c]) for c in sorted({r['condition'] for r in rows})})
    by=defaultdict(list)
    for r in rows:by[r['seed']].append(r)
    if 'no_stockout' in rows[0]['ae']:
        out['ablation_delta']={k:[cluster([np.mean([v['ae'][k][j]-v['ae']['full'][j] for v in group]) for group in by.values()]) for j in (0,1)]
                               for k in ('demand','basic','no_stockout','no_limits')}
    return out


def attribution(rows):
    output={}
    for mode in ('argmax','alert','simple'):
        stats={}
        for lag in (1,4,8):
            pairs=[];classes={}
            for r in rows:
                i=r['when']-7+lag-1
                pairs.append((r['actual'][i],r[mode][i]))
            a,b=zip(*pairs)
            cm=confusion_matrix(a,b,labels=[0,1,2])
            stats[str(lag)]=dict(matrix=cm.tolist(),balanced_accuracy=float(np.mean(cm.diagonal()/cm.sum(axis=1))),
                  by_condition={c:float(np.mean([r[mode][r['when']-7+lag-1]==r['actual'][r['when']-7+lag-1] for r in rows if r['condition']==c]))
                                for c in sorted({r['condition'] for r in rows})})
        stable=[r for r in rows if r['condition']=='stable']
        stats['false_stable_episode']=np.mean([any(r[mode]) for r in stable])
        stats['false_prechange_episode']=np.mean([any(r[mode][:r['when']-7]) for r in rows])
        stats['stable_false_round']=np.mean([np.mean(np.array(r[mode])!=0) for r in stable])
        for c in ('demand','rival','reverse'):
            d=[]
            for r in rows:
                if r['condition']!=c:continue
                j=r['when']-7
                d.append(next((k+1 for k in range(len(r[mode])-j-1) if r[mode][j+k]==r[mode][j+k+1]==r['actual'][j]),None))
            good=[v for v in d if v is not None]
            stats[c+'_delay']=dict(detected=len(good),total=len(d),median=float(np.median(good)) if good else None)
        output[mode]=stats
    return output


def gate_evaluation(rows):
    sel={}
    for kind in KINDS:
        for cut in ('diagnostic','strict'):
            sel[kind+'_'+cut]=[r['decisions'][kind][cut] for r in rows]
    sel['both_recent']=[r['both_better'] for r in rows]
    sel['both_future_oracle']=[r['future_both'] for r in rows]
    sel['all']=[True]*len(rows)
    res={k:metric(rows,v) for k,v in sel.items()}
    seeds=sorted({r['seed'] for r in rows})
    by={s:[i for i,r in enumerate(rows) if r['seed']==s] for s in seeds}
    y=np.array([r['delta']>1.e-8 for r in rows])
    pred={k:np.array([r['scores'][k] for r in rows]) for k in KINDS}
    auc={k:float(roc_auc_score(y,x)) for k,x in pred.items()}
    from scipy.stats import rankdata
    def score(v,i):
        z=y[i];n=z.sum();b=len(i)-n
        if not (n and b):return np.nan
        return (rankdata(v[i])[z].sum()-n*(n+1)/2)/(n*b)
    rng=np.random.default_rng(317)
    ds=[]
    for _ in range(2000):
        idx=[i for s in rng.choice(seeds,len(seeds)) for i in by[s]]
        ds.append(score(pred['full'],idx)-score(pred['draft'],idx))
    diff=cluster([np.mean([.99**(rows[i]['t']-1)*rows[i]['delta']*(int(sel['full_diagnostic'][i])-int(sel['draft_diagnostic'][i])) for i in by[s]]) for s in seeds])
    return dict(policies=res,auc=auc,auc_full_minus_draft_ci=np.nanquantile(ds,[.025,.975]).tolist(),
                profit_full_minus_draft=diff,nonexecuted=sum(not r['term_changed'] for r in rows),
                providers={p:metric([r for r in rows if r['provider']==p],
                    [r['decisions']['full']['diagnostic'] for r in rows if r['provider']==p]) for p in ('deepseek','doubao')})


def one(args):
    phase,m=args
    rs,g=batch(phase,m)
    obj=dict(episodes=len(rs),n_forecasts=len(rs)*54,forecast=summarize_forecast(rs),attribution=attribution(rs))
    if g:
        assert len(g)==96
        obj['gate']=gate_evaluation(g)
        save(OUT/f'{m}-opportunities.json.gz',g)
    save(OUT/f'{phase}-{m}-episodes.json.gz',rs)
    save(OUT/f'{phase}-{m}.json',obj)


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(one,[(p,m) for p in ('test','rtest') for m in ('econ','native')]))

