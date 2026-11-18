"""Apply the frozen stopping rule to the numeric-price cohort, with live branch continuations."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from blind import DATES,select
from markets import OUT,RAW,observation,read,save,source,terminal
from numeric import renew
from results import bootstrap


def one(task):
    m,r=task;name=r["model"];s=r["seed"]
    path=RAW/f"number-stop-{m}-{name}-{s}.json.gz"
    if path.exists():
        return read(path)
    w=source(m,s);h=[];term=None;states=[];requests=[]
    for t in range(1,61):
        term,event=renew(w,h,term,name)
        if event: requests.append(event[1])
        if t in DATES:
            states.append((w.clone(),list(h),dict(term)))
        h.append(w.execute(term["price"]))
    # The original-price branch must be identical to the archived market before adding more counterfactuals.
    assert [v[3] for v in h[r["public"]["round"]-1:]]==r["profit"][0]
    d=select(m,[observation(rows,original,w.t) for w,rows,original in states])
    j=next(i for i,k in enumerate(d) if k["action"]!=3)
    choices={"dynamic":[j,d[j]["action"]],"immediate":[0,d[0]["immediate"]],"fixed_wait":[1,d[1]["immediate"]],
             "down":[0,1],"up":[0,2]}
    seal=RAW/f"number-stop-seal-{m}-{name}-{s}.json"
    if seal.exists():
        assert read(seal)==choices
    else:
        save(seal,choices)
    cache={};values={}
    for policy,(j,a) in choices.items():
        if not a:
            values[policy]=0.
            continue
        if (j,a) not in cache:
            original,past,draft=states[j];b=original.clone();rows=list(past)
            term=dict(draft,price=float(np.clip(draft["draft"]*(1.,.96,1.04)[a],b.low,b.high)))
            for t in range(b.t,61):
                if t>DATES[j]:
                    term,event=renew(b,rows,term,name)
                    if event:requests.append(event[1])
                rows.append(b.execute(term["price"]))
            diff=[x[3]-y[3] for x,y in zip(rows,h)]
            cache[j,a]=terminal(diff,m)
        values[policy]=cache[j,a]
    item=dict(seed=s,market=m,model=name,choices=choices,values=values,requests=requests)
    save(path,item)
    return item


def report(rows):
    result={}
    for m in ("econ","native"):
        for name in ("qwen","smol"):
            selected=[r for r in rows if r["market"]==m and r["model"]==name]
            assert len(selected)==64
            v=lambda k:np.array([r["values"][k] for r in selected])
            result[m+"-"+name]=dict(keep=bootstrap(v("dynamic")),**{k:bootstrap(v("dynamic")-v(k)) for k in ("immediate","fixed_wait","up","down")})
    return result


if __name__=="__main__":
    jobs=[(m,r) for m,start in (("econ",140000),("native",141000)) for r in read(OUT/f"numeric-{m}.json.gz") if r["seed"]<start+64]
    with ThreadPoolExecutor(max_workers=12) as ex:
        rows=list(ex.map(one,jobs))
    save(OUT/"numeric-stopping.json.gz",rows)
    save(OUT/"numeric-stopping-results.json",report(rows))
    old=read(OUT/"numeric-requests.json.gz")
    keys={k for r in rows for k in r["requests"] if k not in old}
    save(OUT/"numeric-stopping-requests.json.gz",{k:read(RAW/"numeric-requests"/(k+".json")) for k in sorted(keys)})

