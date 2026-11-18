"""Common-clock terminal profit and same-state price interventions."""
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/publication"))
from paired import decision, read, save, world
from delayed import observable

RAW=ROOT/"runs/independent-deployment-v1"
OUT=ROOT/"experiment-results/independent-deployment-v1"
for folder in (OUT,RAW):
    folder.mkdir(exist_ok=True,parents=True)
DATES=tuple(range(19,48,4))
KINDS=("stable","demand","rival","reverse")


def draw(seed,tag):
    return int.from_bytes(hashlib.sha256(f"deployment-v1:{seed}:{tag}".encode()).digest()[:8],"big")


def source(m,seed,blind=False):
    w=world(m,seed,KINDS[seed%4])
    if blind:
        w.when=8+draw(seed,"switch")%46
    return w


def contract(w,h,previous,model=None):
    if previous is not None and w.t<=previous["end"]:
        return previous
    d=decision(w,h,previous,None,0,w.low,w.high,[])
    if model is not None and w.t>6 and w.active():
        from local_prices import context_key
        key=context_key(w.market,w.t,h)
        choice=model[key]["action"]
        price=float(np.clip(d["draft"]*(.96,1.,1.04)[choice],w.low,w.high))
        return dict(d,price=price,draft=price,model_key=key)
    return d


def suffix(w,h,d,mult,model=None):
    b=w.clone()
    past=list(h)
    term=dict(d,price=float(np.clip(d["draft"]*mult,w.low,w.high)))
    if b.t!=term["start"] or term["end"]!=b.t+3:
        raise ValueError("intervention is not at a full contract boundary")
    original=b.t
    profits=[]
    lookups=[]
    while b.t<=60:
        if b.t>original:
            term=contract(b,past,term,model)
        if b.t==term["start"] and "model_key" in term:
            lookups.append(term["model_key"])
        row=b.execute(term["price"])
        profits.append(row[3])
        past.append(row)
    return profits,lookups


def public(h,d,t):
    return dict(prices=[row[:2] for row in h],draft=d["draft"],round=t)


def observation(h,d,t):
    return observable(public(h,d,t))


def terminal(profits,market,origin=1,discount=.99):
    return float(np.dot(profits,discount**np.arange(origin-1,origin-1+len(profits)))/(100 if market=="native" else 1))

