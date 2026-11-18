"""Estimate the same candidate payoffs with or without disclosure of the rival type."""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from paired import OUT, ROOT, gains, read, save
from public_value_selector import checksum, choose, features, verify as verify_previous

QUALITIES=(.5,.75,1.)


def inputs(public):
    actions=features(public,"generic")
    return [[*row,kind] for kind in (0,1) for row in actions]


def posterior(values,q):
    """Rows are the two possible signals, columns the three actions."""
    return q*values+(1-q)*values[::-1]


def prediction(model, public):
    v=model.predict(inputs(public)).reshape(2,2)
    return np.column_stack((np.zeros(2),v))


@lru_cache(maxsize=2)
def bundle(market):
    return joblib.load(OUT/f"{market}-signal.joblib")


def policy(market,public):
    model,cuts=bundle(market)
    pred=prediction(model,public)
    output={}
    for q in QUALITIES:
        expected=posterior(pred,q)
        actions=expected.argmax(axis=1)
        filtered=np.where(expected.max(axis=1)>cuts[q],actions,0)
        output[str(q)]={"act":actions.tolist(),"gated":filtered.tolist()}
    return dict(signal=output,prior=choose(market,public),scores=pred.tolist())


def hashes():
    files=[Path(__file__), Path(__file__).with_name("paired.py"),ROOT/"experiment-specs/publication-v1/PLAN.md",
           *[OUT/f"{m}-signal.joblib" for m in ("econ","native")]]
    return {p.relative_to(ROOT).as_posix():checksum(p) for p in files}


def verify():
    verify_previous()
    assert read(OUT/"freeze.json")==hashes()


def train():
    if (OUT/"freeze.json").exists():
        verify()
        return
    for market in ("econ","native"):
        rows=read(OUT/f"train-{market}.json.gz")
        x=[item for r in rows for item in inputs(r["public"])]
        y=np.concatenate([gains(r)[:,1:].ravel() for r in rows])
        model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=8,n_jobs=1,random_state=59171)
        model.fit(x,y)
        cal=read(OUT/f"cal-{market}.json.gz")
        preds=[prediction(model,r["public"]) for r in cal]
        cuts={q:max(0.,float(np.quantile([v for p in preds for v in posterior(p,q).max(axis=1)],.75)))
              for q in QUALITIES}
        joblib.dump((model,cuts),OUT/f"{market}-signal.joblib",compress=3)
    save(OUT/"freeze.json",hashes())


if __name__=="__main__":
    train()
