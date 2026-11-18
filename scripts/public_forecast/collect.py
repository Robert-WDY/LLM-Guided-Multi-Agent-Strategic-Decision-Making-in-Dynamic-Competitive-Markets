"""Generate settled public histories and matched terminal paths, with evaluator labels kept separate."""
import hashlib
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/pricing_feedback'))
from study import World, decide, HOLD, WARM, read, save
from units import exchange
from paid import ask, initialize
from game_theory_agent.market.actor_policies import observation

OUT=ROOT/'experiment-results/public-forecast-v1'
RAW=ROOT/'runs/public-forecast-v1'
for folder in (RAW,OUT):
    folder.mkdir(parents=True,exist_ok=True)
SPLITS={'train':(71200,64),'dev':(72200,24),'cal':(73200,32),'test':(74200,64),
        'rtrain':(75200,24),'rcal':(76200,16),'rtest':(77200,48)}


def draw(seed,tag):
    return int.from_bytes(hashlib.sha256(f'forecast-v1:{tag}:{seed}'.encode()).digest()[:8],'big')


def world(m,s,c):
    w=World(m,s,c)
    w.when=15+draw(s,'onset')%25
    w.up=bool(draw(s,'direction')%2)
    return w


def limits(w):
    return (w.legal(w.low),w.legal(w.high)) if w.market=='native' else (w.low,w.high)


def settle(w,price,lo,hi):
    p,r,q,profit,_=w.execute(price)
    demand=stockout=0
    if w.market=='native':
        o=observation(w.state,'company_A')
        demand,stockout=o['demand'],o['stockout']
    # p, rival price, own sales, own profit, public demand, public stockout, requested p, known floor/ceiling
    return [p,r,q,profit,demand,stockout,price,lo,hi]


def decision(w,h,prev,provider,seed,lo,hi,failures):
    t=w.t;left=61-t
    d=decide(t,left,w.ref,w.low,w.high,h,prev)
    if d['start']==t and t>WARM and provider and w.active():
        context=w.market
        if w.market=='native':
            extra=[[round(v[4]),round(v[5]),round(v[6]-v[0])] for v in h[-4:]]
            context+=f" fresh market (current admissible own price [{lo},{hi}]; prior settled [market demand, stockouts, own requested minus executed price] {extra})"
        try:
            original=ask(provider,seed,context,h,w.low,w.high,d['draft'],min(left,HOLD),left)
        except RuntimeError as exc:
            if 'valid draft after bounded attempts' not in str(exc):raise
            failures.append(t)
            original=d['draft']
        d=decide(t,left,w.ref,w.low,w.high,h,prev,original=original)
    return d


def candidate(w,h,d):
    k=exchange(h)
    normalized=[[v[0]*k,v[1]*k,v[2],v[3]] for v in h]
    search=decide(w.t,61-w.t,w.ref*k,w.low*k,w.high*k,normalized,dummy_previous(d),mode='conditional',
                  original=d['draft']*k,gate=-1.e30)
    return search['proposed']/k,search['score']


def dummy_previous(d):
    # With an explicit draft the search needs only the prior exploration direction.
    return {'end':d['start']-1,'direction':d['direction']}


def episode(m,s,c,provider,*,prospective=False):
    w=world(m,s,c)
    probe=23+4*(draw(s,'opportunity')%6)
    h=[];prev=None;alt=None;alternative=[];other=None;point=None;failures=[]
    for t in range(1,61):
        lo,hi=limits(w)
        d=decision(w,h,prev,provider,s,lo,hi,failures)
        if t==probe and provider:
            offer,score=candidate(w,h,d)
            point=dict(t=t,draft=d['draft'],proposed=offer,score=score,active=w.active(),low=lo,high=hi)
            if prospective:
                from gates import seal
                seal(m,provider,s,h,point)
            alt=w.clone()
            alternative=list(h)
            other=dict(d,price=offer)
        if alt is not None:
            low,high=limits(alt)
            if t!=probe:
                other=decision(alt,alternative,other,provider,s,low,high,failures)
            alternative.append(settle(alt,other['price'],low,high))
        # Different ordinary controllers generate the training price support.
        if not provider and t>WARM and d['start']==t and s%3!=0:
            step=(draw(s,str(t))%2001-1000)/10000
            d=dict(d,price=float(np.clip(d['price']*(1+step),w.low,w.high)))
        h.append(settle(w,d['price'],lo,hi));prev=d
    return dict(seed=s,market=m,condition=c,when=w.when,up=w.up,provider=provider,
                history=h,opportunity=point,alternative=alternative[probe-1:],fallbacks=failures)


def one(task):
    phase,m,s,p=task
    path=RAW/f'{phase}-{m}-{s}-{p or "rule"}.json.gz'
    if path.exists():return
    cs=['stable','demand','rival']+(['reverse'] if phase in ('test','rtest') else [])
    if p:cs=[cs[(s-SPLITS[phase][0])%len(cs)]]
    save(path,[episode(m,s,c,p,prospective=phase=="rtest") for c in cs])


def cases(phase,m):
    return [r for f in sorted(RAW.glob(f'{phase}-{m}-*.json.gz')) for r in read(f)]


if __name__=='__main__':
    phase=sys.argv[1];start,n=SPLITS[phase];real=phase.startswith('r')
    if phase in ('rtest','test') and not (OUT/'freeze.json').exists():raise RuntimeError('freeze required')
    if real:initialize()
    pool=ThreadPoolExecutor if real else ProcessPoolExecutor
    with pool(max_workers=16 if real else 8) as ex:
        list(ex.map(one,[(phase,m,s,p) for s in range(start,start+n) for m in ('econ','native')
                         for p in (('deepseek','doubao') if real else (None,))]))

