"""Four independent pricing managers using the finite distribution of a real local LLM."""
import hashlib
import itertools
import json
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiment-results/four-prices-v1"
RAW = ROOT / "runs/four-prices-v1"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
ROLES = ("Maximize your accumulated trading profit.",
         "Maintain a cash buffer to meet future payroll; weigh immediate cash against later earnings.",
         "Grow the repeat customer base while keeping the business solvent.",
         "Prioritize steady service and avoid sudden swings in profit.")
MARKETS = ("econ","native","appointments")
GROUPS = ("balanced","unseen","random")
FACTORS = (.92,.94,.96,.98,1.,1.02,1.04,1.06,1.08)
MODEL_SHA = "decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33"


def number(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str,("four-prices-v1",*parts))).encode()).digest()[:8],"big")


def key(m,role,h):
    def trend(x,y):
        tol=.025*max(abs(x),abs(y),1.e-8)
        return 1 if x>y+tol else -1 if x<y-tol else 0
    width=min(4,len(h)//2)
    return (m,role,trend(h[-1][0],h[-1][1]),*[trend(sum(r[col] for r in h[-width:]),
           sum(r[col] for r in h[-2*width:-width])) for col in (2,3)])


def request(k,seed):
    m,role,rel,sales,profit=k
    tags=("lower","similar","higher")
    prompt=(f"<|im_start|>system\nYou are an independent pricing manager. {ROLES[role]}<|im_end|>\n"
            f"<|im_start|>user\nFour competing {dict(econ='retail sellers',native='fresh food sellers',appointments='appointment services')[m]}. "
            f"Your price versus other sellers: {tags[rel+1]}. Your recent sales: {tags[sales+1]}. "
            f"Recent profit: {tags[profit+1]}. Choose a price adjustment for the next contract relative to "
            "your feedback reference. 0=-8%, 1=-6%, 2=-4%, 3=-2%, 4=0%, 5=+2%, 6=+4%, 7=+6%, 8=+8%. "
            "Return its digit only.<|im_end|>\n<|im_start|>assistant\n")
    body=dict(prompt=prompt,temperature=.85,top_p=1.,min_p=0.,top_k=0,n_probs=9,
              post_sampling_probs=True,n_predict=1,grammar="root ::= [0-8]",seed=seed,cache_prompt=True)
    data=json.dumps(body,separators=(",",":")).encode()
    req=Request("http://127.0.0.1:19943/completion",data=data,headers={"Content-Type":"application/json"})
    with build_opener(ProxyHandler({})).open(req,timeout=180) as resp:
        ans=json.load(resp)
    probs={v["token"]:v["prob"] for v in ans["completion_probabilities"][0]["top_probs"]}
    p=[probs[str(i)] for i in range(9)]
    assert abs(sum(p)-1)<2.e-5 and min(p)>=0 and ans["content"] in tuple(map(str,range(9)))
    return dict(action=int(ans["content"]),p=p,sha=hashlib.sha256(data).hexdigest())


def label(k):
    return "/".join(map(str,k))


@lru_cache(maxsize=1)
def table():
    return json.loads((OUT/"model.json").read_text(encoding="utf8"))["entries"]


def sample(k,seed,seat,t,live,greedy,tape):
    p=table()[label(k)]["p"]
    if greedy:
        return int(np.argmax(p))
    rnd=number("model",seed,seat,t)
    if live:
        backend_seed=rnd%2147483646+1
        tag=f"{seat}:{t}:{label(k)}"
        if tag not in tape:
            tape[tag]=request(k,backend_seed)
        return tape[tag]["action"]
    u=(rnd+.5)/2**64
    return min(8,int(np.searchsorted(np.cumsum(p)/sum(p),u,side="right")))


def build():
    keys=list(itertools.product(MARKETS,range(4),range(-1,2),range(-1,2),range(-1,2)))
    with ThreadPoolExecutor(8) as pool:
        entries=dict(zip(map(label,keys),pool.map(lambda k:request(k,123),keys)))
    (OUT/"model.json").write_text(json.dumps(dict(model_sha256=MODEL_SHA,entries=entries)),encoding="utf8")


if __name__=="__main__":
    build()
