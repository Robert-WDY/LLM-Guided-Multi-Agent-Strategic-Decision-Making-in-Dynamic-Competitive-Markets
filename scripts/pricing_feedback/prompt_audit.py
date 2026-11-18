"""Check that identical model-visible histories within a seed get the identical draft."""
from collections import defaultdict
import hashlib
import sys
from study import RAW,OUT,World,read,save,decide
from paid import MODELS,connect


def enumerate_cases():
    for family in ('real2','realunits'):
        for file in sorted(RAW.glob(family+'-*.json.gz')):
            _,provider,market,seed,_=file.stem.split('-')
            r=read(file)
            world=World(market,int(seed),r['condition'])
            for i,d in enumerate(r['contracts']):
                if d['start']<=6 or not d['active']:continue
                t=d['start'];history=r['rows'][:t-1];left=61-t
                hint=decide(t,left,world.ref,world.low,world.high,history,r['contracts'][i-1])['draft']
                from paid import request_data
                _,key=request_data(provider,int(seed),market,history,world.low,world.high,hint,min(4,left),left)
                yield file,key,d['draft'],hint,t


def check():
    bad=set();mixed=defaultdict(set);seen=defaultdict(set)
    with connect() as db:
        lookup=defaultdict(list)
        for key,status,p in db.execute("select key,status,price from calls order by id"):
            lookup[key].append((status,p))
    count=0
    for file,key,draft,hint,t in enumerate_cases():
        count+=1
        assert key in lookup,(file,t)
        first=lookup[key][:2]
        canonical=next((p for status,p in first if status=='ok'),hint)
        seen[key].add(round(draft,8))
        if abs(draft-canonical)>1.e-7:bad.add(file.name)
    return dict(decisions=count,prompts=len(seen),multi_draft_histories=sum(len(v)>1 for v in seen.values()),affected=sorted(bad))


if __name__=='__main__':
    result=check()
    print(result)
    save(OUT/'prompt-audit.json',result)

