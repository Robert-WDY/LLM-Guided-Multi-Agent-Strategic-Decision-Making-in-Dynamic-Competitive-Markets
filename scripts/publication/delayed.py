"""One full posted contract before a public-only response classification and price decision."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from paired import OUT, RAW, ROOT, decision, draw, limits, read, save, settle, world
from public_value_selector import checksum, choose, features

KINDS=("stable","demand","rival","reverse")


def observable(public):
    # Classifier and value model see the same legal price information.
    h=np.array(public["prices"],float)[-12:]
    return [*features(public,"generic")[0][:16],*(h/h[-1,0]).ravel()]


def xrows(public):
    obs=observable(public)
    return [[*obs,m] for m in (.96,1.04)]


@lru_cache(maxsize=2)
def models(market):
    return joblib.load(OUT/f"delayed-{market}.joblib")


def prospective(market,public):
    clf,values,direct=models(market)
    prob=clf.predict_proba([observable(public)])[0]
    actions=xrows(public)
    x=[[*a,k] for k in range(4) for a in actions]
    pred=values.predict(x).reshape(4,2)
    table=np.column_stack((np.zeros(4),pred))
    direct_pred=np.r_[0.,direct.predict(actions)]
    old=choose(market,public)["conditional"]
    return dict(prob=prob.tolist(),uniform=int(table.mean(0).argmax()),inferred=int((prob@table).argmax()),
                disclosed=table.argmax(1).tolist(),direct=int(direct_pred.argmax()),
                legacy=(("keep","down","up").index(old["arm"]) if old["use"] else 0))


def trial(args):
    phase,m,seed=args
    w=world(m,seed,"stable")
    probe=23+4*(draw(seed,"opportunity")%6)
    h=[]
    d=None
    for t in range(1,probe+1):
        lo,hi=limits(w)
        d=decision(w,h,d,None,seed,lo,hi,[])
        if t<probe:
            h.append(settle(w,d["price"],lo,hi))
    result=[]
    for k,kind in enumerate(KINDS):
        b=w.clone()
        b.change,b.when=kind,probe+1
        past=list(h)
        for _ in range(4):
            past.append(settle(b,d["price"],*limits(b)))
        t0=b.t
        lo,hi=limits(b)
        draft=decision(b,past,d,None,seed,lo,hi,[])
        assert draft["start"]==t0 and t0==probe+4
        p=dict(prices=[r[:2] for r in past],draft=draft["draft"],round=t0)
        selected=None
        if phase=="test":
            selected=prospective(m,p)
            seal=RAW/f"delayed-{m}-{seed}-{k}.json"
            if seal.exists():
                assert read(seal)==selected
            else:
                save(seal,selected)
        profits=[]
        for multiplier in (1.,.96,1.04):
            b2=b.clone()
            arm=dict(draft, price=float(np.clip(draft["draft"]*multiplier,b.low,b.high)))
            history=list(past)
            for t in range(t0,61):
                lo,hi=limits(b2)
                if t>t0:
                    arm=decision(b2,history,arm,None,seed,lo,hi,[])
                history.append(settle(b2,arm["price"],lo,hi))
            assert len(history)==60 and len({v[6] for v in history[t0-1:t0+3]})==1
            profits.append([r[3] for r in history[t0-1:]])
        result.append(dict(public=p,active=b.active(),profit=profits,selected=selected))
    return dict(seed=seed,market=m,cases=result)


def payoffs(case,market):
    p=np.array(case["profit"],float)
    return (p-p[:1])@(.99**np.arange(p.shape[1]))/(100 if market=="native" else 1)


def fit_market(m):
    rows=read(OUT/f"delayed-train-{m}.json.gz")
    z,labels,x,xc,y=[],[],[],[],[]
    for r in rows:
        for k,c in enumerate(r["cases"]):
            z.append(observable(c["public"]))
            labels.append(k)
            a=xrows(c["public"])
            x.extend(a)
            xc.extend([v+[k] for v in a])
            y.extend(payoffs(c,m)[1:])
    opts=dict(n_estimators=128,min_samples_leaf=8,random_state=78031,n_jobs=1)
    clf=ExtraTreesClassifier(**opts).fit(z,labels)
    cond=ExtraTreesRegressor(**opts).fit(xc,y)
    direct=ExtraTreesRegressor(**opts).fit(x,y)
    assert list(clf.classes_)==list(range(4))
    joblib.dump((clf,cond,direct),OUT/f"delayed-{m}.joblib",compress=3)


def snapshot():
    paths=[Path(__file__),ROOT/"experiment-specs/publication-v1/DELAY.md",*[OUT/f"delayed-{m}.joblib" for m in ("econ","native")]]
    return {p.relative_to(ROOT).as_posix():checksum(p) for p in paths}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("train","fit","test"))
    phase=parser.parse_args().phase
    freeze=OUT/"delayed-freeze.json"
    if phase=="fit":
        if freeze.exists():
            assert snapshot()==read(freeze)
            return
        for m in ("econ","native"):
            fit_market(m)
        save(freeze,snapshot())
        return
    if phase=="test":
        assert read(freeze)==snapshot()
    start,n=(106000,128) if phase=="train" else (108000,192)
    with ProcessPoolExecutor(max_workers=9) as ex:
        for m in ("econ","native"):
            rows=list(ex.map(trial,[(phase,m,s) for s in range(start,start+n)]))
            save(OUT/f"delayed-{phase}-{m}.json.gz",rows)


if __name__=="__main__":
    main()
