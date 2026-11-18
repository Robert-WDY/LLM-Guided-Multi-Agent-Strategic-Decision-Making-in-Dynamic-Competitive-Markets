"""Quoted-price unit control applied to the same independent real-agent cohort."""
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from paid import ask,initialize,summary
from units import episode
from analyze import cluster
from study import RAW,OUT,save,read
from scipy.stats import beta


def job(task):
    provider,m,seed=task
    for mode in ('conditional','wide'):
        path=RAW/f'realunits-{provider}-{m}-{seed}-{mode}.json.gz'
        if path.exists():continue
        fallbacks=[]
        def draft(*args):
            try:return ask(provider,seed,*args)
            except RuntimeError as exc:
                if 'valid draft after bounded attempts' not in str(exc):raise
                fallbacks.append(len(args[1]))
                return args[4]
        cond=('stable','demand','rival','reverse')[seed%4]
        r=episode(m,seed,cond,mode,agent=draft)
        r['fallbacks']=fallbacks
        save(path,r)


def report():
    out={}
    for p in ('deepseek','doubao'):
        for m in ('econ','native'):
            by_arm={}
            for a,prefix in [('feedback','real2'),('conditional','realunits'),('wide','realunits')]:
                by_arm[a]={r['seed']:r for r in (read(f) for f in RAW.glob(f'{prefix}-{p}-{m}-*-{a}.json.gz'))}
                assert len(by_arm[a])==32
            rows={}
            for a,b in [('conditional','feedback'),('conditional','wide'),('wide','feedback')]:
                ds=[];harmed=exposed=actions=opps=0
                for s,r in sorted(by_arm[a].items()):
                    delta=sum(.99**i*(x[3]-y[3]) for i,(x,y) in enumerate(zip(r['rows'],by_arm[b][s]['rows'])))/(100 if m=='native' else 1)
                    ds.append(delta)
                    terms=[v for v in r['contracts'] if v['start']>6 and v['active']]
                    hit=sum(v['actual_override'] for v in terms)
                    if hit:exposed+=1;harmed+=int(delta< -1.e-8)
                    actions+=hit;opps+=len(terms)
                rows[a+'-'+b]=dict(profit=cluster(ds),coverage=actions/opps,exposed=exposed,harmed=harmed,
                                    risk_upper=1 if harmed==exposed else beta.ppf(.95,harmed+1,exposed-harmed))
            out[p+'-'+m]=dict(results=rows,fallbacks=sum(len(r['fallbacks']) for a in ('conditional','wide') for r in by_arm[a].values()))
    save(OUT/'real-units.json',out)
    summary()


if __name__=='__main__':
    initialize()
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(job,[(p,m,s) for s in range(58100,58132) for p in ('deepseek','doubao') for m in ('econ','native')]))
    report()

