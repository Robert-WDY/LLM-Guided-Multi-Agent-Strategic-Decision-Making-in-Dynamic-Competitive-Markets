"""Voluntary advice review by the original LLM; all prices and profits remain simulated."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from urllib.request import Request,urlopen

import numpy as np

from markets import OUT,RAW,draw,read,save
from local_prices import SOURCES
from numeric import LOCKS
from results import bootstrap


def choose(row,m,reverse):
    name=row["model"]
    t=row["public"]["round"]
    original=row["public"]["draft"]
    arm=row["decision"]["inferred"]
    if not arm or not row["active"]:
        return dict(accepted=False,offered=False)
    event=row["prefix_calls"][-1]
    assert event[0]==t
    proposed=float(np.clip(original*(.96 if arm==1 else 1.04),event[5],event[6]))
    if abs(proposed-original)<1.e-8:
        return dict(accepted=False,offered=False)
    old_request=read(RAW/"numeric-requests"/(event[1]+".json"))
    context=old_request["body"]["messages"][1]["content"].split(". Return only your numeric selling price.")[0]
    options=[f"your original price {original:.4f}",f"the advisor's proposed price {proposed:.4f}"]
    if reverse:
        options.reverse()
    msgs=[dict(role="system",content="You remain the pricing manager. You have final authority. Maximize your own remaining cumulative profit."),
          dict(role="user",content=f"{context}. You already submitted a draft. A public-price advisor now offers a revision. Choose the action you prefer: A = {options[0]}; B = {options[1]}. Return just A or B.")]
    body=dict(model=name,messages=msgs,temperature=0,seed=1741,max_tokens=3,grammar='root ::= [AB]')
    raw=json.dumps(body,separators=(",",":")).encode()
    key=hashlib.sha256(raw).hexdigest()
    path=RAW/f"adoption-{key}.json"
    with LOCKS[int(key[:6],16)%len(LOCKS)]:
        if path.exists():
            return read(path)
        request=Request(f"http://127.0.0.1:{SOURCES[name][3]}/v1/chat/completions",data=raw,headers={"Content-Type":"application/json"})
        with urlopen(request,timeout=150) as resp:
            result=json.load(resp)
        letter=result["choices"][0]["message"]["content"].strip()
        assert letter in ("A","B") and result["model"]==name
        d=dict(accepted=letter==("A" if reverse else "B"),offered=True,letter=letter,request_sha256=key,
               original=original,proposed=proposed,reverse=reverse)
        save(path,d)
        return d



def trial(task):
    m,row=task
    choices=[choose(row,m,r) for r in (False,True)]
    order=draw(row["seed"],"adoption-letter-order")%2
    use=choices[order]["accepted"]
    p=np.array(row["profit"]);arm=row["decision"]["inferred"]
    gain=float((p[arm]-p[0])@(.99**np.arange(p.shape[1]))/(100 if m=="native" else 1))
    return dict(market=m,seed=row["seed"],model=row["model"],choices=choices,order=order,
                gain=gain,adopted=use,value=gain*use)


def report(rows):
    result={}
    for m in ("econ","native"):
        for name in SOURCES:
            g=[r for r in rows if r["market"]==m and r["model"]==name]
            assert len(g)==192
            result[m+"-"+name]=dict(offered=sum(r["choices"][0]["offered"] for r in g),
                adopted=sum(r["adopted"] for r in g),order_sensitive=sum(r["choices"][0]["accepted"]!=r["choices"][1]["accepted"] for r in g),
                voluntary_keep=bootstrap([r["value"] for r in g]),
                voluntary_forced=bootstrap([r["value"]-r["gain"] for r in g]))
    return result


if __name__=="__main__":
    tasks=[(m,r) for m in ("econ","native") for r in read(OUT/f"numeric-{m}.json.gz")]
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows=list(ex.map(trial,tasks))
    save(OUT/"adoption.json.gz",rows)
    save(OUT/"adoption-results.json",report(rows))

