"""Evaluate the frozen stopping controller with the independent model price policies."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from markets import OUT,RAW,DATES,contract,observation,read,save,source,suffix,terminal
from blind import action,models,fingerprint
from independent import menu
from results import bootstrap


def batch(task):
    m,start=task
    sources=[]
    for s in range(start,start+16):
        for name in ("qwen","smol"):
            policy=menu(name);w=source(m,s,True);h=[];d=None;states=[]
            for t in range(1,61):
                d=contract(w,h,d,policy)
                if t in DATES:
                    states.append((w.clone(),list(h),dict(d)))
                h.append(w.execute(d["price"]))
            sources.append((name,s,states,[v[3] for v in h]))
    rules=[{} for _ in sources]
    for j,t in enumerate(DATES):
        xs=[observation(states[j][1],states[j][2],t) for _,_,states,_ in sources]
        preds=models(m)[t].predict(xs)
        for r,a,now in zip(rules,action(preds),np.column_stack([np.zeros(len(preds)),preds[:,:2]]).argmax(1)):
            if "dynamic" not in r and a!=3:
                r["dynamic"]=[j,int(a)]
            if j<2:
                r[("immediate","fixed_wait")[j]]=[j,int(now)]
    seal=RAW/f"transfer-{m}-{start}.json"
    if seal.exists():
        assert read(seal)==rules
    else:
        save(seal,rules)
    rows=[]
    for (name,s,states,base),dec in zip(sources,rules):
        cache={};vs={}
        for k,(j,a) in dec.items():
            if a and (j,a) not in cache:
                w,h,d=states[j]
                path=suffix(w,h,d,(1.,.96,1.04)[a],menu(name))[0]
                cache[j,a]=terminal(np.array(path)-np.array(base[DATES[j]-1:]),m,DATES[j])
            vs[k]=cache.get((j,a),0.)
        rows.append(dict(seed=s,model=name,market=m,decision=dec,values=vs))
    return rows


def report(rows):
    result={}
    for m in ("econ","native"):
        for name in ("qwen","smol"):
            g=[v for v in rows if v["market"]==m and v["model"]==name]
            assert [r["seed"] for r in g]==list(range(143000,143512))
            y=np.array([[r["values"][k] for k in ("dynamic","immediate","fixed_wait")] for r in g])
            result[m+"-"+name]=dict(keep=bootstrap(y[:,0]),immediate=bootstrap(y[:,0]-y[:,1]),fixed_wait=bootstrap(y[:,0]-y[:,2]),
                                    wait=sum(r["decision"]["dynamic"][0]>0 for r in g),
                                    abstain=sum(r["decision"]["dynamic"][1]==0 for r in g))
    return result


if __name__=="__main__":
    assert fingerprint()==read(OUT/"stopping-freeze.json")
    with ProcessPoolExecutor(max_workers=6) as ex:
        groups=list(ex.map(batch,[(m,s) for m in ("econ","native") for s in range(143000,143512,16)]))
    rows=[r for group in groups for r in group]
    save(OUT/"blind-transfer.json.gz",rows)
    save(OUT/"blind-transfer-results.json",report(rows))

