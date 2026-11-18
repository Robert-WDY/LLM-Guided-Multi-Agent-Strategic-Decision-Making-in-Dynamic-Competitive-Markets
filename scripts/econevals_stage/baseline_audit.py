"""Identify the economic source of B's performance under 10-period decisions."""
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from common import OUT,read,save
from duopoly import FAMILIES,ALPHAS,environment,Opponent,settle,startup,episode


def evaluate(seed):
    out=[];bud=read(OUT/'freeze.json')['fit']['D']['budget']
    for f in FAMILIES:
        for arm in ('A','B','D','anchor','adaptive'):
            if arm in ('A','B','D'):
                row=episode(seed,f,arm,bud if arm=='D' else None)
                if arm=='B':
                    prices=[d[2] for d in row['decisions']]
                    assert len(set(round(float(p),8) for p in prices))<=2
                out.append(dict(seed=seed,family=f,arm=arm,profit=row['profit']));continue
            kw,c,_,_,hi=environment(ALPHAS[seed%3]);op=Opponent(seed,f,hi)
            hist=[];last=peer=hi*.55;sign=1;total=0.;previous=None
            for t in range(300):
                own=startup(t,hi,seed) if t<10 else last
                rival=op.price(t,last,peer)
                if t==10:
                    anchor=max(hist,key=lambda x:x[1])[0]
                    own=anchor if arm=='anchor' else anchor+sign*.025*hi
                elif t>10 and t%10==0 and arm=='adaptive':
                    block=np.mean([p for _,p in hist[-10:]])
                    if previous is not None and block<previous:sign*=-1
                    previous=block
                    own=float(np.clip(last+sign*.025*hi,.12*hi,.96*hi))
                _,p=settle(kw,c,(own,rival));hist.append((own,p[0]));last,peer=own,rival
                total+=.99**t*p[0]
            out.append(dict(seed=seed,family=f,arm=arm,profit=total))
    return out


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=5) as pool:rows=sum(pool.map(evaluate,range(4200,4248)),[])
    stats={}
    for a,b in [('anchor','A'),('B','anchor'),('adaptive','B'),('D','adaptive'),('D','anchor'),('D','B')]:
        vals=[np.mean([r['profit'] for r in rows if r['arm']==a and r['seed']==s])-np.mean([r['profit'] for r in rows if r['arm']==b and r['seed']==s]) for s in range(4200,4248)]
        stats[a+'-'+b]=dict(mean=float(np.mean(vals)),ci=np.quantile(np.random.default_rng(4).choice(vals,(5000,48)).mean(axis=1),[.025,.975]).tolist())
    save(OUT/'baseline-audit-episodes.json.gz',rows);save(OUT/'baseline-audit.json',stats)
    print(stats)

