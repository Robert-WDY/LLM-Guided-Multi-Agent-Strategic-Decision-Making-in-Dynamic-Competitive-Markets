"""A single intervention with optional stopping; onset and event labels are evaluator-only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from markets import DATES,OUT,RAW,contract,observation,public,read,save,source,suffix,terminal
from public_value_selector import checksum


def sample(task):
    phase,m,s=task
    w=source(m,s,True)
    history=[]
    d=None
    snapshots=[]
    for t in range(1,61):
        d=contract(w,history,d)
        if t in DATES:
            snapshots.append((w.clone(),list(history),dict(d)))
        history.append(w.execute(d["price"]))
    base=[v[3] for v in history]
    xs=[observation(h,d,b.t) for b,h,d in snapshots]
    decisions=None
    if phase=="test":
        decisions=select(m,xs)
        seal=RAW/f"blind-{m}-{s}.json"
        if seal.exists():
            assert read(seal)==decisions
        else:
            save(seal,decisions)
    paths=[]
    for b,h,d in snapshots:
        down,up=[suffix(b,h,d,v)[0] for v in (.96,1.04)]
        paths.append([down,up])
    return dict(seed=s,market=m,condition=w.change,onset=None if w.change=="stable" else w.when,
                x=xs,base=base,paths=paths,decisions=decisions)


def label(row,discount=.99):
    y=[]
    for t,paths in zip(DATES,row["paths"]):
        keep=np.array(row["base"][t-1:])
        y.append([terminal(np.array(p)-keep,row["market"],t,discount) for p in paths])
    return np.asarray(y)


def action(values):
    # Permanent abstention (zero), the two immediate moves, or another observation.
    return np.column_stack([np.zeros(len(values)),values]).argmax(axis=1)


def fit(m):
    rows=read(OUT/f"blind-train-{m}.json.gz")
    xs=np.array([r["x"] for r in rows])
    y=np.array([label(r) for r in rows])
    # Stride four would split by regime. Alternate entire balanced four-seed blocks.
    fold=np.array([(r["seed"]//4)%2 for r in rows])
    continuation=np.zeros(len(rows))
    fitted={}
    for j in reversed(range(len(DATES))):
        targets=np.column_stack([y[:,j,:],continuation])
        held=np.zeros_like(targets)
        def estimator():
            return ExtraTreesRegressor(n_estimators=80,min_samples_leaf=10,random_state=1971+j,n_jobs=1)
        for v in (0,1):
            a=fold!=v
            held[~a]=estimator().fit(xs[a,j],targets[a]).predict(xs[~a,j])
        fitted[DATES[j]]=estimator().fit(xs[:,j],targets)
        choices=action(held)
        continuation=np.column_stack([np.zeros(len(rows)),targets])[np.arange(len(rows)),choices]
    joblib.dump(fitted,OUT/f"stopping-{m}.joblib",compress=3)


@lru_cache(maxsize=2)
def models(m):
    return joblib.load(OUT/f"stopping-{m}.joblib")


def select(m,xs):
    # Each indexed instruction can depend only on the current snapshot; stop at the first non-wait.
    out=[]
    for t,x in zip(DATES,xs):
        pred=models(m)[t].predict([x])[0]
        a=int(action(pred[None,:])[0])
        out.append(dict(action=a,immediate=int(np.r_[0.,pred[:2]].argmax()),values=pred.tolist()))
    return out


def fingerprint():
    paths=[Path(__file__),Path(__file__).with_name("markets.py"),*[OUT/f"stopping-{m}.joblib" for m in ("econ","native")]]
    return {p.name:checksum(p) for p in paths}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("train","fit","test"))
    p.add_argument("--workers",type=int,default=9)
    a=p.parse_args()
    if a.phase=="fit":
        f=OUT/"stopping-freeze.json"
        if f.exists():
            assert read(f)==fingerprint()
        else:
            for market in ("econ","native"):
                fit(market)
            save(f,fingerprint())
    else:
        if a.phase=="test":
            assert read(OUT/"stopping-freeze.json")==fingerprint()
        start,n=(130000,384) if a.phase=="train" else (131000,512)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for market in ("econ","native"):
                rows=list(ex.map(sample,[(a.phase,market,s) for s in range(start,start+n)]))
                save(OUT/f"blind-{a.phase}-{market}.json.gz",rows)

