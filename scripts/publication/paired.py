"""Generate matched hidden-response continuations with a shared predecision state."""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/public_price_value"))
from trials import decision, draw, limits, read, save, settle, world

OUT = ROOT / "experiment-results/publication-v1"
RAW = ROOT / "runs/publication-v1"
for folder in (OUT, RAW):
    folder.mkdir(exist_ok=True, parents=True)
SPLITS = {"train": (101000,128), "cal": (102000,64), "test": (103000,192)}
ACTIONS = ("keep","down","up")
TYPES = ("rival","reverse")


def episode(task):
    phase, market, seed = task
    w = world(market,seed,"stable")
    probe = 23 + 4*(draw(seed,"opportunity")%6)
    h, term = [], None
    for t in range(1,probe+1):
        lo,hi=limits(w)
        term=decision(w,h,term,None,seed,lo,hi,[])
        if t<probe:
            h.append(settle(w,term["price"],lo,hi))
    assert term["start"]==probe and term["end"]==probe+3
    public={"prices":[row[:2] for row in h],"draft":term["draft"],"round":probe}
    prospective=None
    if phase=="test":
        from signal_policy import policy
        prospective=policy(market,public)
        seal=RAW/f"seal-{market}-{seed}.json"
        if seal.exists():
            assert read(seal)==prospective
        else:
            save(seal,prospective)
    path_profits=[]
    first=[]
    hashes=[]
    for kind in TYPES:
        path_by_action=[]
        first_by_action=[]
        digest_by_action=[]
        for multiplier in (1.,.96,1.04):
            b=w.clone()
            b.change,b.when=kind,probe+1
            rows=list(h)
            d=dict(term,price=float(np.clip(term["draft"]*multiplier,w.low,w.high)))
            for t in range(probe,61):
                lo,hi=limits(b)
                if t>probe:
                    d=decision(b,rows,d,None,seed,lo,hi,[])
                rows.append(settle(b,d["price"],lo,hi))
            suffix=np.asarray(rows[probe-1:],dtype="<f8")
            assert np.isfinite(suffix).all() and b.t==61
            assert np.all(suffix[:4,6]==suffix[0,6])
            path_by_action.append(suffix[:,3].tolist())
            first_by_action.append(suffix[0].tolist())
            digest_by_action.append(hashlib.sha256(suffix.tobytes()).hexdigest())
        path_profits.append(path_by_action)
        first.append(first_by_action)
        hashes.append(digest_by_action)
    assert first[0]==first[1] and len(set(v[1] for v in first[0]))==1
    return dict(market=market,seed=seed,active=w.active(),public=public,profit=path_profits,
                first=first[0],trace_sha256=hashes,prospective=prospective)


def gains(row,discount=.99):
    p=np.asarray(row["profit"],float)
    unit=100 if row["market"]=="native" else 1
    return (p-p[:,:1,:])@(discount**np.arange(p.shape[-1]))/unit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=SPLITS)
    parser.add_argument("--workers",type=int,default=9)
    args=parser.parse_args()
    start,n=SPLITS[args.phase]
    if args.phase=="test":
        from signal_policy import verify
        verify()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for market in ("econ","native"):
            rows=list(pool.map(episode,[(args.phase,market,s) for s in range(start,start+n)]))
            save(OUT/f"{args.phase}-{market}.json.gz",rows)


if __name__=="__main__":
    main()
