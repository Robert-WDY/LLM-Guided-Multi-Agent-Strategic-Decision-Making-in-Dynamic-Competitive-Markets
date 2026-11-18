"""Freeze external calibration, run paired markets, retain paths and cluster estimates."""
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import numpy as np
from scipy.stats import beta
from common import RAW, OUT, ROOT, REVISION, digest, read, save
from duopoly import ALPHAS, FAMILIES, environment, episode, settle


def refs():
    rows=[]
    for a in ALPHAS:
        for n in (1,2):
            kw,c,nash,mono,hi=environment(a,n)
            for name,p in (("nash",nash),("joint",mono)):
                qs,ps=zip(*(settle(kw,c,(p,)*n) for _ in range(300)))
                # Independent smooth FOC at the candidate, evaluated at actual upstream demand.
                from common import market
                step=1e-4*a
                def value(delta):
                    prices=[p]*n
                    if name=="joint": prices=[p+delta]*n
                    else:prices[0]+=delta
                    v=market.get_profits(p=prices,c=c,**kw)
                    return sum(v) if name=="joint" else v[0]
                slope=(value(step)-value(-step))/(2*step)
                assert abs(slope)<.015
                index=None if n==1 else float(np.clip((np.mean([p]*50)-nash)/(mono-nash),0,1))
                rows.append(dict(alpha=a,players=n,reference=name,price=p,profit=ps[-1][0],foc=slope,index=index,periods=len(ps)))
    save(OUT/'references.json',rows)


def calibration():
    fitted={}
    for arm in ('C','D'):
        errors=[]; ids=set()
        for seed,family in itertools.product(range(100,112),FAMILIES[:3]):
            r=episode(seed,family,arm,calibration=True)
            errors.extend(r['errors'])
            if r['errors']:ids.add(seed)
        assert len(errors)>=12 and len(ids)>=8
        fitted[arm]=dict(budget=np.maximum(0,np.quantile(errors,.90,axis=0)).tolist(),cases=len(errors),clusters=len(ids))
    hashes={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('duopoly.py'),Path(__file__).with_name('common.py'),
        ROOT/'experiment-specs/econevals-pricing-v1/PLAN.md',ROOT/'src/game_theory_agent/game_theory/reliable_response.py',ROOT/'src/game_theory_agent/game_theory/reliable_advisor.py')}
    frozen=dict(upstream=REVISION,fit=fitted,hashes=hashes)
    path=OUT/'freeze.json'
    if path.exists():assert read(path)==frozen, 'calibration changed'
    else:save(path,frozen)
    return fitted


def worker(args):
    seed,cal=args
    rows=[]
    for fam,arm in itertools.product(FAMILIES,('A','B','C','D','Craw','Draw')):
        r=episode(seed,fam,arm,cal.get(arm,{}).get('budget'))
        rows.append(r)
    save(RAW/f'holdout-{seed}.json.gz',rows)
    return [{k:v for k,v in r.items() if k not in ('history','decisions','errors')} for r in rows]


def summarize(rows):
    lookup={(r['seed'],r['family'],r['arm']):r for r in rows}
    def delta(left,right,fams,col='profit'):
        diffs=np.array([np.mean([lookup[s,f,left][col]-lookup[s,f,right][col] for f in fams]) for s in range(1200,1248)])
        boot=np.random.default_rng(20261118).choice(diffs,(5000,len(diffs))).mean(axis=1)
        return dict(mean=float(diffs.mean()),ci=np.quantile(boot,[.025,.975]).tolist(),seed_wins=int((diffs>1e-8).sum()))
    metrics={}
    for arm in ('A','B','C','D','Craw','Draw'):
        subset=[r for r in rows if r['arm']==arm]
        acted={r['seed'] for r in subset if r['interventions']}
        lost={r['seed'] for r in subset if r['interventions'] and r['profit']<lookup[r['seed'],r['family'],'A']['profit']-1e-8}
        upper=(1. if len(lost)==len(acted) else float(beta.ppf(.95,len(lost)+1,len(acted)-len(lost)))) if acted else None
        metrics[arm]={k:float(np.mean([r[k] for r in subset])) for k in ['profit','cs','index','steps']}
        metrics[arm].update(coverage=sum(r['interventions'] for r in subset)/sum(r['opportunities'] for r in subset),acted_seeds=len(acted),loss_seeds=len(lost),upper=upper)
    contrasts={f'{l}-{r}':delta(l,r,FAMILIES) for l,r in [('D','C'),('D','B'),('D','A'),('C','B'),('D','Draw'),('Draw','Craw'),('B','A')]}
    families={f:delta('D','C',[f]) for f in FAMILIES}
    gate=contrasts['D-C']['ci'][0]>0 and contrasts['D-B']['ci'][0]>0 and metrics['D']['coverage']>=.1 and (metrics['D']['upper'] or 1)<=.10
    save(OUT/'summary.json',dict(arms=metrics,contrasts=contrasts,families=families,passed=bool(gate),seeds=48,episodes=len(rows),rounds=300*len(rows),cs_delta=delta('D','C',FAMILIES,'cs')))
    save(OUT/'episodes.json.gz',rows)


if __name__=='__main__':
    multiprocessing.freeze_support()
    refs()
    fit=calibration()
    print('frozen',fit,flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        batches=list(pool.map(worker,((s,fit) for s in range(1200,1248))))
    summarize(sum(batches,[]))
    print(read(OUT/'summary.json')['contrasts'],flush=True)

