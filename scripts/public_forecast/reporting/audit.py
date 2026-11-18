"""Reconstruct the forecast clock, price contracts and evidence for the final restricted study."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collect import RAW,OUT,ROOT,read,save,cases,SPLITS
from models import bundle,diagnosis
from gates import metric,truth_delta
from analyze import cluster
from scipy.stats import spearmanr


def worker(task):
    phase,m=task;data=[];total=rounds=0
    by=defaultdict(list)
    for r in cases(phase,m):
        s=r['seed'];c=r['condition'];h=np.asarray(r['history']);w=r['when']
        assert h.shape==(60,9)
        fc,sc,carry,pr=diagnosis(bundle(m),h,'full' if m=='native' else 'basic')
        np.testing.assert_array_equal(carry,h[5:-1][:,[2,1]])
        assert np.isfinite(fc).all() and np.isfinite(pr).all() and len(fc)==54
        j=w-7
        for name,sl in [('pre',slice(0,j)),('first4',slice(j,j+4)),('next8',slice(j+4,j+12))]:
            y=h[6:][sl][:,[2,1]]
            by[c,name].append((s,np.mean(abs(fc[sl]-y)-abs(carry[sl]-y),axis=0)))
        data.append(dict(seed=s,provider=r['provider'],scenario=c,shock=w,
              columns=['forecast_round','last_settled_round','request','floor','ceiling','last_sales','last_quote',
                       'forecast_sales','forecast_quote','actual_sales','actual_quote','stable_after','demand_after','rival_after'],
              rounds=[[i+7,i+6,*h[i+6,6:9],*carry[i],*fc[i],h[i+6,2],h[i+6,1],*pr[i]] for i in range(54)]))
        if phase=='rtest':
            t=r['opportunity']['t'];alt=np.asarray(r['alternative'])
            assert len(alt)==61-t and alt.shape[1]==9
            np.testing.assert_allclose(alt[:4,6],r['opportunity']['proposed'],rtol=0,atol=1.e-9)
            np.testing.assert_allclose(h[t-1:t+3,6],r['opportunity']['draft'],rtol=0,atol=1.e-9)
            # If all four settlements and submitted requests match, future decisions are shared too.
            if np.array_equal(alt[:4],h[t-1:t+3]):np.testing.assert_array_equal(alt,h[t-1:])
            saved=read(RAW/'prospective'/f"{m}-{r['provider']}-{s}.json")
            assert saved['history_sha256']==hashlib.sha256(json.dumps(r['history'][:t-1],separators=(',',':')).encode()).hexdigest()
        total+=1;rounds+=len(fc)
    if phase=='test':
        # Identical shock-free observations for scenario twins until the hidden onset.
        for s in range(*[SPLITS['test'][0],sum(SPLITS['test'])]):
            rs=read(RAW/f'test-{m}-{s}-rule.json.gz');stable=rs[0]
            assert stable['condition']=='stable'
            for r in rs[1:]:
                assert r['history'][:r['when']-1]==stable['history'][:r['when']-1]
    save(OUT/f'{phase}-{m}-predictions.json.gz',data)
    impact={}
    for (c,k),xs in by.items():
        seeds=sorted({i for i,_ in xs})
        arr=np.array([np.mean([v for i,v in xs if i==s],axis=0) for s in seeds])
        impact[c+'-'+k]=[cluster(arr[:,j]) for j in range(2)]
    result=dict(episodes=total,paired_forecasts=rounds,windows=impact)
    if phase=='rtest':
        g=read(OUT/f'{m}-opportunities.json.gz')
        result['skill_validation']={str(b):dict(n=len(v),future_both_rate=float(np.mean([x['future_both'] for x in v])),
                 mean_terminal=float(np.mean([x['delta'] for x in v])))
                for b in (False,True) if (v:=[x for x in g if x['both_better']==b])}
        result['skill_rank_terminal']=float(spearmanr([min(x['skill']) for x in g],[x['delta'] for x in g]).statistic)
    return phase+'-'+m,result


if __name__=='__main__':
    freeze=read(OUT/'freeze.json')
    for name,expected in freeze['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    with ProcessPoolExecutor(max_workers=4) as ex:
        result=dict(ex.map(worker,[(p,m) for p in ('test','rtest') for m in ('econ','native')]))
    with sqlite3.connect(ROOT/'runs/pricing-feedback-v2/model-ledger.sqlite') as db:
        previous=db.execute('select sum(cost) from calls where id<=12460').fetchone()[0]
        entries=db.execute('select provider,status,count(*),sum(cost),sum(input),sum(output) from calls where id>12460 group by provider,status').fetchall()
    assert previous==23808940 and all(r[1]!='reserved' for r in entries)
    cost=sum(r[3] for r in entries)
    assert previous+cost<=50000000
    counts={p:len(list(RAW.glob(p+'-*.json.gz'))) for p in SPLITS}
    for p,(_,n) in SPLITS.items():assert counts[p]==n*(4 if p.startswith('r') else 2)
    assert len(list((RAW/'prospective').glob('*.json')))==192
    save(OUT/'audit.json',dict(results=result,raw_counts=counts,prospective_records=192,freeze_entries=len(freeze['files']),
          ledger_rows=entries,previous=previous/1e6,additional=cost/1e6,cumulative=(previous+cost)/1e6,
          remaining=(50000000-previous-cost)/1e6,new_attempts=sum(r[2] for r in entries)))

