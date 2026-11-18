"""Reverse the economic meanings of A and C on the exact same market observations."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from urllib.request import ProxyHandler,Request,build_opener

from local_prices import OUT,RAW,SOURCES,messages,read,save


def one(args):
    name,key=args
    msgs=messages(key)
    before="A = 4% below the reference. B = reference. C = 4% above the reference."
    after="A = 4% above the reference. B = reference. C = 4% below the reference."
    assert before in msgs[1]["content"]
    msgs[1]["content"]=msgs[1]["content"].replace(before,after)
    body=dict(model=name,messages=msgs,temperature=0,seed=1741,max_tokens=3,grammar='root ::= [ABC]',stream=False)
    payload=json.dumps(body,ensure_ascii=False,separators=(",",":")).encode()
    keyhash=hashlib.sha256(payload).hexdigest()
    path=RAW/f"reversed-{keyhash}.json"
    if not path.exists():
        req=Request(f"http://127.0.0.1:{SOURCES[name][3]}/v1/chat/completions",data=payload,headers={"Content-Type":"application/json"})
        with build_opener(ProxyHandler({})).open(req,timeout=180) as res:
            ans=json.load(res)
        l=ans["choices"][0]["message"]["content"].strip()
        assert l in ("A","B","C") and ans["model"]==name
        save(path,dict(prompt=msgs,letter=l,action="CBA".index(l),hash=keyhash))
    return key,read(path)


if __name__=="__main__":
    out={}
    for name in SOURCES:
        original=read(OUT/f"llm-{name}.json.gz")["entries"]
        with ThreadPoolExecutor(max_workers=2) as ex:
            new=dict(ex.map(one,[(name,k) for k in original]))
        save(OUT/f"menu-reversal-{name}.json.gz",new)
        out[name]=dict(n=len(new),same_letter=sum(v["letter"]==original[k]["letter"] for k,v in new.items()),
                       same_price=sum(v["action"]==original[k]["action"] for k,v in new.items()))
    save(OUT/"menu-reversal.json",out)
