"""Closed-loop trials with real price drafts, cached matched prompts and a shared RMB cap."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import sqlite3
from threading import local
import time
from openai import OpenAI
from study import OUT, RAW, read, replay, save
from analyze import cluster
import numpy as np

DB = RAW/'model-ledger.sqlite'
LOCAL = local()
MODELS = {'deepseek':('deepseek-flash','https://api.deepseek.com','DEEPSEEK_API_KEY'),
          'doubao':('doubao-seed-2-0-lite-260428','https://ark.cn-beijing.volces.com/api/v3','ARK_API_KEY')}
LIMIT = 50_000_000  # integer micro-yuan
MAX_OUTPUT = 48


def connect():
    return sqlite3.connect(DB,timeout=90)


def initialize():
    with connect() as db:
        db.execute('pragma journal_mode=WAL')
        db.execute('create table if not exists calls (id integer primary key, key text, provider text, model text, cost integer, status text, price real, input integer, output integer)')
        db.execute('create index if not exists lookup on calls(key,status)')


def request_data(provider,seed,market,history,low,high,hint,hold,left):
    model,url,key=MODELS[provider]
    compact=[[round(v,3) for v in row[:4]] for row in history[-16:]]
    prompt=(f"Set your company's selling price to maximize total own profit over the remaining {left} periods "
            f"in a {market} duopoly. Your price request applies for {hold} periods. Bounds [{low:.6f}, {high:.6f}]. "
            f"Your feedback rule suggests {hint:.6f}. You may keep it or choose another price. "
            "Settled history [your price, public rival price, units sold by you, your profit]: "
            +json.dumps(compact,separators=(',',':'))+". Return only a single number within bounds.")
    identity=hashlib.sha256(f'{provider}|{model}|{seed}|{prompt}'.encode()).hexdigest()
    return prompt, identity


def ask(provider,seed,market,history,low,high,hint,hold,left):
    model,url,key=MODELS[provider]
    prompt,identity=request_data(provider,seed,market,history,low,high,hint,hold,left)
    worst=(len(prompt.encode())+256)*4+MAX_OUTPUT*16
    for attempt in range(2):
        with connect() as db:
            db.execute('begin immediate')
            first=db.execute("select status,price from calls where key=? order by id limit 2",(identity,)).fetchall()
            hit=next((price for status,price in first if status=='ok'),None)
            if hit is not None:return hit
            if len(first)>=2:
                raise RuntimeError(f'{provider} did not supply a valid draft after bounded attempts')
            if db.execute('select coalesce(sum(cost),0) from calls').fetchone()[0]+worst>LIMIT:
                raise RuntimeError('shared provider budget exhausted')
            ident=db.execute("insert into calls(key,provider,model,cost,status) values(?,?,?,?,'reserved')",
                             (identity,provider,model,worst)).lastrowid
        if not hasattr(LOCAL,provider):
            setattr(LOCAL,provider,OpenAI(base_url=url,api_key=os.environ[key],timeout=25,max_retries=0))
        try:
            obj=getattr(LOCAL,provider).chat.completions.create(model=model,temperature=0,max_tokens=MAX_OUTPUT,
                    messages=[{'role':'user','content':prompt}],extra_body={'thinking':{'type':'disabled'}})
        except Exception as exc:
            code=getattr(exc,'status_code',None)
            with connect() as db:
                db.execute('update calls set status=? where id=?',(f'error-{code or type(exc).__name__}',ident))
            if code in (401,402,403,404):raise RuntimeError(f'{provider} returned {code}') from None
            time.sleep(.2)
            continue
        try:
            price=float(obj.choices[0].message.content.strip())
            valid=math.isfinite(price) and low<=price<=high
        except (AttributeError,ValueError,IndexError,TypeError):
            price,valid=None,False
        usage=obj.usage
        tin,tout=(usage.prompt_tokens,usage.completion_tokens) if usage else (None,None)
        measured=tin*4+tout*16 if usage else worst
        assert measured<=worst,'token reservation bound exceeded'
        with connect() as db:
            db.execute('update calls set status=?,price=?,cost=?,input=?,output=? where id=?',
                       ('ok' if valid else 'invalid',price if valid else None,measured,tin,tout,ident))
        if valid:return price
    raise RuntimeError(f'{provider} did not supply a valid draft after bounded attempts')


def case(task):
    provider,market,seed=task
    path=RAW/f'paid-{provider}-{market}-{seed}.json.gz'
    if path.exists():return
    condition=('demand','rival','reverse')[seed%3]
    model=read(OUT/'selectors.json')[market]
    gates=read(OUT/'learned-thresholds.json')[market]
    episodes=[]
    def original(*args):return ask(provider,seed,*args)
    for arm in ('feedback','learned','diagnostic','conditional'):
        gate=gates['diagnostic' if arm=='diagnostic' else 'threshold']
        r=replay(market,seed,condition,arm,gate,agent=original,selector=model,labels=arm=='feedback')
        episodes.append(r)
    save(path,episodes)


def summary():
    stats={}
    for provider in MODELS:
        for m in ('econ','native'):
            cases=[read(p) for p in sorted(RAW.glob(f'paid-{provider}-{m}-*.gz'))]
            if not cases:continue
            def total(r):
                return sum(.99**i*v[3] for i,v in enumerate(r['rows']))/(100 if m=='native' else 1)
            stats[provider+'-'+m]={arm:dict(profit_difference=cluster([total(rows[i])-total(rows[0]) for rows in cases]),
                    active=sum(sum(d['actual_override'] for d in rows[i]['contracts']) for rows in cases),
                    lower_profit=sum(total(rows[i])<total(rows[0])-1.e-8 for rows in cases),
                    opportunities=sum(sum(d['start']>6 and d['active'] for d in rows[i]['contracts']) for rows in cases))
                    for i,arm in enumerate(('feedback','learned','diagnostic','conditional'))}
    with connect() as db:
        ledger=[dict(provider=p,status=s,attempts=n,micro_yuan=k,input=t,output=u) for p,s,n,k,t,u in
                db.execute("select provider,status,count(*),sum(cost),sum(input),sum(output) from calls group by provider,status")]
    save(OUT/'paid.json',dict(results=stats,ledger=ledger,conservative_yuan=sum(r['micro_yuan'] for r in ledger)/1.e6,
                              limit_yuan=50))
    print(ledger,flush=True)


if __name__=='__main__':
    import sys
    initialize()
    if 'summary' not in sys.argv:
        tasks=[(p,m,s) for s in range(54800,54806) for p in MODELS for m in ('econ','native')]
        with ThreadPoolExecutor(max_workers=12) as ex:
            list(ex.map(case,tasks))
    summary()

