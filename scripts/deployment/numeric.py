"""Direct numeric model pricing at every contract renewal with exact-request caching."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from threading import Lock
from time import sleep
from urllib.error import URLError
from urllib.request import Request,urlopen

import numpy as np

from markets import OUT,RAW,contract,draw,public,read,save,source
from independent import frozen_actions
from local_prices import SOURCES

CACHE=RAW/"numeric-requests"
CACHE.mkdir(exist_ok=True)
LOCKS=[Lock() for _ in range(257)]


def build_request(name,m,t,h,d,lo,hi):
    history=[[round(float(v),3) for v in row[:4]] for row in h[-4:]]
    prompt=(f"Market {m}. Set your own selling price for the next {min(4,61-t)} rounds to maximize "
            f"your profit over the remaining {61-t} rounds. Feasible interval [{lo:.3f}, {hi:.3f}]. "
            f"Feedback reference {d['draft']:.3f}. Settled rows [your price, competitor price, your sales, your profit]: "
            +json.dumps(history,separators=(",",":"))+". Return only your numeric selling price.")
    msgs=[dict(role="system",content="You are the pricing manager. Choose a selling price using the given business data."),
          dict(role="user",content=prompt)]
    body=dict(model=name,messages=msgs,temperature=0,seed=1741,max_tokens=16,
              grammar='root ::= ("0" | [1-9] [0-9]{0,6}) ("." [0-9]{1,3})?',stream=False)
    return body


def number_request(name,m,t,h,d,lo,hi):
    body=build_request(name,m,t,h,d,lo,hi)
    data=json.dumps(body,separators=(",",":")).encode()
    key=hashlib.sha256(data).hexdigest()
    f=CACHE/(key+".json")
    with LOCKS[int(key[:6],16)%len(LOCKS)]:
        if f.exists():
            return key,read(f)["price"]
        req=Request(f"http://127.0.0.1:{SOURCES[name][3]}/v1/chat/completions",data=data,headers={"Content-Type":"application/json"})
        for attempt in range(4):
            try:
                with urlopen(req,timeout=120) as resp:
                    result=json.load(resp)
                break
            except (OSError,URLError):
                if attempt==3:
                    raise
                sleep(1+attempt)
        price=float(result["choices"][0]["message"]["content"].strip())
        assert 0<=price<1e7 and result["model"]==name
        save(f,dict(body=body,price=price,usage=result["usage"]))
        return key,price


def renew(w,h,old,name):
    if old is not None and w.t<=old["end"]:
        return old,None
    d=contract(w,h,old)
    if w.t<=6 or not w.active():
        return d,None
    lo,hi=(w.legal(w.low),w.legal(w.high)) if w.market=="native" else (w.low,w.high)
    key,raw=number_request(name,w.market,w.t,h,d,lo,hi)
    price=float(np.clip(raw,lo,hi))
    return dict(d,draft=price,price=price),[w.t,key,raw,price,d["draft"],lo,hi]


def trial(task):
    m,name,seed=task
    path=RAW/f"numeric-{m}-{name}-{seed}.json.gz"
    if path.exists():
        return
    w=source(m,seed)
    t0=23+4*(draw(seed,"intervention")%6)
    h=[];events=[];d=None
    for t in range(1,t0+1):
        d,event=renew(w,h,d,name)
        if event is not None:
            events.append(event)
        if t<t0:
            h.append(w.execute(d["price"]))
    p=public(h,d,t0)
    choice=frozen_actions(m,[p])[0]
    seal=RAW/f"numeric-seal-{m}-{name}-{seed}.json"
    if seal.exists():
        assert read(seal)==choice
    else:
        save(seal,choice)
    profits=[];traces=[];requests=[]
    for mult in (1.,.96,1.04):
        b=w.clone();past=list(h);calls=[]
        term=dict(d,price=float(np.clip(d["draft"]*mult,b.low,b.high)))
        for t in range(t0,61):
            if t>t0:
                term,event=renew(b,past,term,name)
                if event is not None:
                    calls.append(event)
            past.append(b.execute(term["price"]))
        tail=past[t0-1:]
        traces.append([r[:4] for r in tail])
        profits.append([r[3] for r in tail]);requests.append(calls)
    save(path,dict(seed=seed,model=name,active=w.active(),condition=w.change,public=p,decision=choice,
                   prefix_calls=events,branch_calls=requests,settled=traces,profit=profits))


def main():
    tasks=[(m,name,start+i) for i in range(192) for m,start in (("econ",140000),("native",141000)) for name in SOURCES]
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(trial,tasks))
    for m in ("econ","native"):
        rows=[read(RAW/f"numeric-{m}-{name}-{s}.json.gz") for market,name,s in tasks if market==m]
        save(OUT/f"numeric-{m}.json.gz",rows)
    records={p.stem:read(p) for p in sorted(CACHE.glob("*.json"))}
    save(OUT/"numeric-requests.json.gz",records)


if __name__=="__main__":
    main()
