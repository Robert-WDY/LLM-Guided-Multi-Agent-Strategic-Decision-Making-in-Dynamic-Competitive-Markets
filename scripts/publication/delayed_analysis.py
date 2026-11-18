"""Inspect selection, probability accuracy and terminal value without counting twins as independent."""
import argparse
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from delayed import OUT, RAW, KINDS, models, observable, xrows, payoffs, read, save, prospective, checksum


def just_prices(p):
    # Replace the pending draft before constructing any feature for the ancillary classifier.
    return observable(dict(p,draft=p["prices"][-1][0]))


def fit():
    freeze=OUT/"price-only-freeze.json"
    files=[Path(__file__), Path(__file__).resolve().parents[2]/"experiment-specs/publication-v1/DRAFT-ABLATION.md"]
    paths=[OUT/f"only-prices-{m}.joblib" for m in ("econ","native")]
    def fingerprint():
        return {p.name:checksum(p) for p in files+paths}
    if freeze.exists():
        assert fingerprint()==read(freeze)
        return
    for m,path in zip(("econ","native"),paths):
        rows=read(OUT/f"delayed-train-{m}.json.gz")
        x=[just_prices(c["public"]) for r in rows for c in r["cases"]]
        labels=np.tile(np.arange(4),len(rows))
        clf=ExtraTreesClassifier(n_estimators=128,min_samples_leaf=8,random_state=78031,n_jobs=1)
        joblib.dump(clf.fit(x,labels),path,compress=3)
    save(freeze,fingerprint())


def summary(values):
    v=np.array(values,float)
    rng=np.random.default_rng(4129)
    idx=rng.integers(len(v),size=(12000,len(v)))
    qs=np.quantile(v[idx].mean(1),[.0125,.025,.975,.9875])
    return dict(mean=float(v.mean()),ci95=qs[1:3].tolist(),ci975=qs[[0,3]].tolist(),seeds=len(v))


def evaluate(m):
    rows=read(OUT/f"delayed-test-{m}.json.gz")
    assert [r["seed"] for r in rows]==list(range(108000,108192))
    clf,values,direct=models(m)
    quote_clf=joblib.load(OUT/f"only-prices-{m}.joblib")
    cases=[c for r in rows for c in r["cases"]]
    p=[c["public"] for c in cases]
    obs=[observable(o) for o in p]
    prob=clf.predict_proba(obs)
    quotes=quote_clf.predict_proba([just_prices(o) for o in p])
    x=[xrows(o) for o in p]
    alt=values.predict([a+[k] for offers in x for k in range(4) for a in offers]).reshape(-1,4,2)
    table=np.concatenate((np.zeros((len(cases),4,1)),alt),axis=2)
    chosen=(quotes[:,:,None]*table).sum(1).argmax(1)
    direct_action=np.column_stack((np.zeros(len(cases)),direct.predict([a for offers in x for a in offers]).reshape(-1,2))).argmax(1)
    actual=np.array([payoffs(c,m) for c in cases])
    labels=np.tile(np.arange(4),len(rows))
    true=np.eye(4)[labels]
    selections={key:np.array([c["selected"][key] for c in cases]) for key in ("uniform","inferred","direct","legacy")}
    selections.update(disclosed=table[np.arange(len(cases)),labels].argmax(1),only_prices=chosen,
                      up=np.full(len(cases),2),down=np.ones(len(cases),dtype=int))
    assert np.array_equal(direct_action,selections["direct"])
    assert np.array_equal(table.mean(1).argmax(1),selections["uniform"])
    assert np.array_equal((prob[:,:,None]*table).sum(1).argmax(1),selections["inferred"])
    assert np.allclose(prob,[c["selected"]["prob"] for c in cases])
    assert np.array_equal(selections["disclosed"],[c["selected"]["disclosed"][k] for k,c in zip(labels,cases)])
    y={k:actual[np.arange(len(cases)),v] for k,v in selections.items()}
    byseed=lambda x:summary(np.asarray(x).reshape(-1,4).mean(1))
    out={"policy":{k:byseed(v) for k,v in y.items()}}
    for a,b in (("inferred","uniform"),("inferred","direct"),("disclosed","inferred"),("only_prices","uniform"),
                ("only_prices","inferred"),("inferred","up"),("inferred","legacy")):
        out[a+"-"+b]=byseed(y[a]-y[b])
    out["disagree_inferred_uniform"]=int((selections["inferred"]!=selections["uniform"]).sum())
    out["by_kind"]={name:summary((y["inferred"]-y["uniform"])[labels==i]) for i,name in enumerate(KINDS)}
    for name,pr in (("with_draft",prob),("only_prices",quotes)):
        pred=pr.argmax(1)
        confusion=np.zeros((4,4),int)
        np.add.at(confusion,(labels,pred),1)
        out[name]=dict(accuracy=byseed(pred==labels),brier=byseed(((pr-true)**2).sum(1)),
                       confusion=confusion.tolist(),false_rival=float((pred[labels<2]>=2).mean()))
    shared=sum(rows[i]["cases"][0]["public"]["prices"]==rows[i]["cases"][1]["public"]["prices"] for i in range(len(rows)))
    if m=="econ":
        assert shared==len(rows) and np.array_equal(quotes[::4],quotes[1::4])
    out["identical_stable_demand_prices"]=shared
    out["active_cases"]=sum(c["active"] for c in cases)
    return out


def transfer(m):
    hist=read(OUT/("archive-econ.json.gz" if m=="econ" else "fresh-native.json.gz"))
    obs=defaultdict(list)
    for r in hist:
        v=prospective(m,r["public"])
        profits=np.array(r["profit"]) if m=="econ" else np.array([r["paths"][k]["settled"] for k in ("keep","down","up")])[:,:,3]
        y=(profits-profits[:1])@(.99**np.arange(profits.shape[1]))/(100 if m=="native" else 1)
        obs[r["seed"]].append([y[v["inferred"]],y[v["inferred"]]-y[v["uniform"]],y[v["inferred"]]-y[v["direct"]]])
    x=np.array([np.mean(a,axis=0) for a in obs.values()])
    return {k:summary(a) for k,a in zip(("keep","uniform","direct"),x.T)}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("fit","report"))
    phase=parser.parse_args().phase
    fit()
    if phase=="report":
        save(OUT/"delayed-results.json",{m:dict(test=evaluate(m),transport=transfer(m)) for m in ("econ","native")})
