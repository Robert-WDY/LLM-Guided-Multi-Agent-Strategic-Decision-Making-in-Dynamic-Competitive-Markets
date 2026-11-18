"""Independent seeded terminal paths under the locked real-model price policies."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np

from markets import KINDS,OUT,RAW,contract,draw,public,read,save,source,suffix
from delayed import models,observable,snapshot,xrows
from local_prices import SOURCES

SIZES={"econ":(121000,192),"native":(122000,4608)}


@lru_cache(maxsize=2)
def menu(name):
    return read(OUT/f"llm-{name}.json.gz")["entries"]


def frozen_actions(m,points):
    clf,model,direct=models(m)
    probs=clf.predict_proba([observable(p) for p in points])
    actions=[xrows(p) for p in points]
    x=[a+[k] for v in actions for k in range(4) for a in v]
    values=model.predict(x).reshape(-1,4,2)
    weighted=(probs[:,:,None]*values).sum(1)
    zeros=np.zeros(len(points))
    inferred=np.column_stack([zeros,weighted]).argmax(1)
    uniform=np.column_stack([zeros,values.mean(1)]).argmax(1)
    direct_arm=np.column_stack([zeros,direct.predict([a for v in actions for a in v]).reshape(-1,2)]).argmax(1)
    return [dict(inferred=int(a),uniform=int(b),direct=int(c),offer=int(d+1)) for a,b,c,d in
            zip(inferred,uniform,direct_arm,weighted.argmax(1))]


def work(task):
    m,start,count=task
    f=RAW/f"independent-{m}-{start}.json.gz"
    if f.exists():
        return
    snapshots=[]
    for seed in range(start,start+count):
        for name in SOURCES:
            model=menu(name)
            w=source(m,seed)
            probe=23+4*(draw(seed,"intervention")%6)
            h=[];keys=[];prev=None
            for t in range(1,probe+1):
                d=contract(w,h,prev,model)
                if t==probe:
                    break
                if d["start"]==t and "model_key" in d:
                    keys.append(d["model_key"])
                h.append(w.execute(d["price"]))
                prev=d
            snapshots.append((seed,name,w,h,keys,d))
    points=[public(h,d,w.t) for _,_,w,h,_,d in snapshots]
    decisions=frozen_actions(m,points)
    seal=RAW/f"independent-seal-{m}-{start}.json"
    if seal.exists():
        assert decisions==read(seal)
    else:
        save(seal,decisions)
    rows=[]
    for (seed,name,w,h,keys,d),p,choice in zip(snapshots,points,decisions):
        vals=[suffix(w,h,d,scale,menu(name)) for scale in (1.,.96,1.04)]
        rows.append(dict(seed=seed,model=name,active=w.active(),condition=w.change,public=p,decision=choice,
                    prefix_keys=keys,branch_keys=[v[1] for v in vals],profit=[v[0] for v in vals]))
    save(f,rows)


def run(workers):
    from paired import OUT as earlier
    assert read(earlier/"delayed-freeze.json")==snapshot()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for m,(start,n) in SIZES.items():
            jobs=[(m,s,min(32,start+n-s)) for s in range(start,start+n,32)]
            list(ex.map(work,jobs))
            rows=[r for _,s,_ in jobs for r in read(RAW/f"independent-{m}-{s}.json.gz")]
            assert len(rows)==2*n
            save(OUT/f"independent-{m}.json.gz",rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=12)
    run(parser.parse_args().workers)

