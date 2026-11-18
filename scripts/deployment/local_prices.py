"""Real deterministic price-menu drafts. Identical prompt bytes reuse the stored response."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
from urllib.request import Request,urlopen

from markets import OUT,RAW,read,save

SOURCES={
    "qwen":("Qwen/Qwen2.5-0.5B-Instruct-GGUF","9217f5db79a29953eb74d5343926648285ec7e67","74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",19867),
    "smol":("HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF","2d4a76a30b4af41ecd395c35725ac11688d4cfe4","decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33",19868),
}


def bucket(a,b,tolerance):
    bound=tolerance*max(abs(a),abs(b),1.e-8)
    return -1 if a<b-bound else 1 if a>b+bound else 0


def context_key(m,t,h):
    # The latest two equal-length settled windows, including the short warmup.
    width=min(4,len(h)//2)
    def avg(window,col):
        return sum(v[col] for v in window)/len(window)
    recent,prior=h[-width:],h[-2*width:-width]
    relative=bucket(h[-1][0],h[-1][1],.02)
    sales=bucket(avg(recent,2),avg(prior,2),.05)
    profit=bucket(avg(recent,3),avg(prior,3),.05)
    time=0 if 61-t>32 else 1 if 61-t>12 else 2
    return "|".join(map(str,(m,time,relative,sales,profit)))


def messages(key):
    market,time,relative,sales,profit=key.split("|")
    def descriptor(i,words):
        return words[int(i)+1]
    return [dict(role="system",content="You run one seller in a repeated competitive market. Choose your price to maximize your own cumulative profit through the end. Return exactly one price option letter."),
            dict(role="user",content=(
                f"Market: {'fresh groceries with possible stockouts' if market=='native' else 'differentiated products'}. "
                f"Time remaining: {('over 32 rounds','13 to 32 rounds','at most 12 rounds')[int(time)]}. "
                f"Your latest selling price is {descriptor(relative,('below','close to','above'))} your rival's. "
                f"Your recent units sold are {descriptor(sales,('falling','steady','rising'))}; "
                f"your recent profit is {descriptor(profit,('falling','steady','rising'))}. "
                "A feedback rule supplies a reference price. Your next request lasts up to four rounds. "
                "A = 4% below the reference. B = reference. C = 4% above the reference. Choose A, B, or C."))]


def inference(task):
    name,key=task
    repo,revision,digest,port=SOURCES[name]
    msgs=messages(key)
    body=dict(model=name,messages=msgs,temperature=0,seed=1741,max_tokens=3,
              grammar='root ::= [ABC]',stream=False)
    payload=json.dumps(body,ensure_ascii=False,separators=(",",":")).encode()
    sha=hashlib.sha256(payload).hexdigest()
    f=RAW/f"llm-{name}-{sha}.json"
    if f.exists():
        return key,read(f)
    req=Request(f"http://127.0.0.1:{port}/v1/chat/completions",data=payload,
                headers={"Content-Type":"application/json"})
    with urlopen(req,timeout=120) as resp:
        v=json.load(resp)
    letter=v["choices"][0]["message"]["content"].strip()
    if letter not in ("A","B","C") or v["model"]!=name:
        raise ValueError("independent model failed the pricing contract")
    row=dict(action="ABC".index(letter),letter=letter,prompt=msgs,request_sha256=sha,
             usage=v["usage"],finish=v["choices"][0]["finish_reason"])
    save(f,row)
    return key,row


if __name__=="__main__":
    keys=["|".join(map(str,v)) for v in itertools.product(("econ","native"),range(3),(-1,0,1),(-1,0,1),(-1,0,1))]
    for name,source in SOURCES.items():
        path=RAW.parents[1]/".tmp"/"independent-models"/(name+".gguf")
        with path.open("rb") as f:
            assert hashlib.file_digest(f,"sha256").hexdigest()==source[2]
        with ThreadPoolExecutor(max_workers=2) as ex:
            entries=dict(ex.map(inference,[(name,key) for key in keys]))
        save(OUT/f"llm-{name}.json.gz",dict(repo=source[0],revision=source[1],sha256=source[2],entries=entries))

