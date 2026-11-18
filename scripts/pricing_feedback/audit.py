"""Independent accounting, horizon checks and source-state diagnostics."""
import hashlib
import math
import numpy as np
from scipy.stats import beta
from study import RAW,OUT,ROOT,read,save,ALPHAS
from analyze import cluster
from selector import predict


def once(m):
    model=read(OUT/'long-model.json')[m]
    scale=100 if m=='native' else 1
    benefits=[];randoms=[];exposed=set();harmed=set();acted=possible=0
    for f in sorted(RAW.glob('long-confirm-'+m+'-*.gz')):
        seed_gains=[];random_gains=[]
        for r in read(f):
            if r['arm']!='feedback':continue
            eligible=r['labels'];possible+=len(eligible)
            scores=[predict(model['model'],r['rows'][:v['t']-1],v) for v in eligible]
            winner=next((v for v,s in zip(eligible,scores) if s>model['diagnostic']),None)
            profit=0.;blind=0.
            if winner:
                seed=r['seed'];acted+=1;exposed.add(seed)
                profit=.99**(winner['t']-1)*winner['delta']/scale
                if profit< -1.e-8:harmed.add(seed)
                # Oracle-matched episode coverage for context; exact average over uniform dates.
                blind=np.mean([.99**(v['t']-1)*v['delta']/scale for v in eligible])
            seed_gains.append(profit);random_gains.append(blind)
        benefits.append(np.mean(seed_gains));randoms.append(np.mean(random_gains))
    n,k=len(exposed),len(harmed)
    return dict(versus_original=cluster(benefits),matched_episode_uniform=cluster(randoms),
                selection_minus_uniform=cluster(np.array(benefits)-randoms),exposed=n,harmed=k,
                risk_upper=1 if n==k else beta.ppf(.95,k+1,n-k),acted=acted,eligible=possible)


def run():
    frozen=read(OUT/'freeze.json')
    for name,digest in frozen.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    count={'econ':0,'native':0};max_error=0.;evidence=[];native_clips=0;label_rows=[]
    for f in sorted(RAW.glob('confirm-*.json.gz')):
        cases=read(f)
        assert len(cases)==40
        index={(r['condition'],r['arm']):r for r in cases}
        for r in cases:
            m,s,c,a=[r[k] for k in ('market','seed','condition','arm')]
            assert len(r['rows'])==60
            count[m]+=60
            stable=index['stable',a]
            assert r['rows'][:r['when']-1]==stable['rows'][:r['when']-1]
            if a in ('gated','learned'):assert r['rows']==index[c,'feedback']['rows']
            contracts=r['contracts']
            assert [i for d in contracts for i in range(d['start'],d['end']+1)]==list(range(1,61))
            for d in contracts:
                assert d['next_decision']==(d['end']+1 if d['end']<60 else None)
                if a in ('generic','conditional','wide') and d['start']>6:
                    assert d['transitions']==45*(1+d['end']-d['start'])
                if m=='native' and a=='feedback':
                    native_clips+=sum(abs(v[0]-round(d['price']))>.01 for v in r['rows'][d['start']-1:d['end']])
            if m=='econ':
                alpha=ALPHAS[s%3]
                for t,(p,peer,q,profit,surplus) in enumerate(r['rows'],1):
                    taste=(2.45 if s%2==0 else 1.55) if c=='demand' and t>=r['when'] else 2.
                    x,y=[math.exp((taste-v/alpha)/.25) for v in (p,peer)]
                    expect=100*x/(1+x+y)
                    max_error=max(max_error,abs(expect-q),abs(expect*(p/alpha-1)-profit))
                    assert abs(surplus-25*math.log(1+x+y))<1.e-8
            unit=100 if m=='native' else 1
            evidence.append(dict(market=m,seed=s,scenario=c,arm=a,discounted_profit=sum(.99**i*v[3]/unit for i,v in enumerate(r['rows'])),
                             overrides=sum(d['actual_override'] for d in contracts)))
            for v in r['labels']:
                label_rows.append(dict(market=m,seed=s,scenario=c,**v))
    assert count=={'econ':115200,'native':115200} and max_error<=.00500001
    assert len(label_rows)==4992 and len(evidence)==3840
    save(OUT/'episode-profits.json.gz',evidence)
    save(OUT/'prospective-labels.json.gz',label_rows)
    save(OUT/'audit.json',dict(rounds=count,max_econ_rounding_error=max_error,
        native_feedback_clamped_rounds=native_clips,strict_gate_rows_match=True,frozen_hashes=len(frozen),
        once={m:once(m) for m in count}))


if __name__=='__main__':run()

