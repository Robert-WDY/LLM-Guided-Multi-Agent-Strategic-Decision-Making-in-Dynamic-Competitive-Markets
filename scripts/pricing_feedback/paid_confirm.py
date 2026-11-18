"""Independent larger matched real-agent cohort. Shares the phase-wide hard budget."""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import numpy as np
from scipy.stats import beta
from paid import MODELS,initialize,ask,connect,summary as ledger_summary
from study import RAW,OUT,read,save,replay
from analyze import cluster


def case(task):
    provider,m,s=task
    def draft(*args):
        try:
            return ask(provider,s,*args)
        except RuntimeError as exc:
            if 'valid draft after bounded attempts' not in str(exc):
                raise
            fallbacks.append(len(args[1]))
            return args[4]
    for arm in ('feedback','conditional','wide','diagnostic'):
        path=RAW/f'real2-{provider}-{m}-{s}-{arm}.json.gz'
        if path.exists():continue
        fallbacks=[]
        rule=read(OUT/'long-model.json')[m]
        cond=('stable','demand','rival','reverse')[s%4]
        r=replay(m,s,cond,arm,rule['diagnostic'],agent=draft,selector=rule['model'])
        r['fallbacks']=fallbacks
        save(path,r)


def report():
    results={}
    for provider in MODELS:
        for m in ('econ','native'):
            rows=[read(f) for f in RAW.glob(f'real2-{provider}-{m}-*.gz')]
            assert len(rows)==128
            entries={(r['seed'],r['arm']):r for r in rows}
            seeds=sorted({r['seed'] for r in rows})
            def total(r):
                return sum(.99**i*x[3] for i,x in enumerate(r['rows']))/(100 if m=='native' else 1)
            stats={}
            for a,b in [('conditional','feedback'),('conditional','wide'),('wide','feedback'),('diagnostic','feedback')]:
                dif=[];actions=times=n=k=0
                for s in seeds:
                    r=entries[s,a]
                    delta=total(r)-total(entries[s,b]);dif.append(delta)
                    ds=[d for d in r['contracts'] if d['start']>6 and d['active']]
                    act=sum(d['actual_override'] for d in ds)
                    times+=len(ds);actions+=act
                    if act:n+=1;k+=int(delta < -1.e-8)
                stats[a+'-'+b]=dict(profit=cluster(dif),coverage=actions/times,exposed=n,harmed=k,
                                    upper95=1 if k==n and n else beta.ppf(.95,k+1,n-k) if n else None)
            results[provider+'-'+m]=dict(comparisons=stats,fallbacks=sum(len(r['fallbacks']) for r in rows))
    save(OUT/'real-confirm.json',results)
    ledger_summary()


if __name__=='__main__':
    initialize()
    if 'summary' not in sys.argv:
        with ThreadPoolExecutor(max_workers=20) as ex:
            list(ex.map(case,[(p,m,s) for s in range(58100,58132) for p in MODELS for m in ('econ','native')]))
    report()

