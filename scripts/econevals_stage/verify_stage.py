"""Independently replay primary rows, calculate official index variants and inventory evidence."""
from collections import defaultdict
from pathlib import Path
import json
import math
import numpy as np
from common import RAW,OUT,ROOT,read,save,digest
from duopoly import FAMILIES,ALPHAS,Opponent,environment
from provider import MODELS


def verify():
    compact={(r['seed'],r['family'],r['arm']):r for r in read(OUT/'episodes.json.gz')}
    assert len(compact)==48*4*6
    indices=defaultdict(list); max_error=0.;rounds=0
    for seed in range(1200,1248):
        a=ALPHAS[seed%3]
        _,_,n,m,hi=environment(a)
        batch=read(RAW/f'holdout-{seed}.json.gz')
        assert len(batch)==24
        for r in batch:
            assert r['seed']==seed
            op=Opponent(seed,r['family'],hi)
            o=compact[seed,r['family'],r['arm']]
            h=np.array(r['history']);assert h.shape==(300,4)
            x,y,q,profit=h.T
            ours=np.exp((2-x/a)/.25);theirs=np.exp((2-y/a)/.25)
            expected=100*ours/(1+ours+theirs)
            err=max(np.abs(q-expected).max(),np.abs(profit-expected*(x/a-1)).max())
            assert err<=.00500001
            max_error=max(max_error,err)
            last=peer=.55*hi
            for t,(p,opp,_,_) in enumerate(h):
                assert abs(op.price(t,last,peer)-opp)<1e-10
                last,peer=p,opp
            assert abs(np.sum(.99**np.arange(300)*profit)-o['profit'])<1e-7
            assert abs(np.mean(25*np.log(1+ours+theirs))-o['cs'])<1e-9
            for count in (50,51):
                idx=float(np.clip((np.mean(h[-count:,:2])-n)/(m-n),0,1))
                indices[r['arm']].append((count,idx))
                if count==50:assert abs(idx-o['index'])<1e-9
            decisions=r['decisions']
            assert len(decisions)==29 and [x[0] for x in decisions]==list(range(10,300,10))
            if r['arm'][0] in 'CD':assert all(d[3]==165 for d in decisions)
            rounds+=len(h)
    fixed=read(OUT/'freeze.json')
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'experiment-specs/econevals-pricing-v1/PLAN.md',
        ROOT/'src/game_theory_agent/game_theory/reliable_response.py',ROOT/'src/game_theory_agent/game_theory/reliable_advisor.py']
    names={p.name:p for p in paths}
    assert all(digest(names[k])==v for k,v in fixed['hashes'].items())
    logs={}
    for p in sorted(RAW.glob('*-calls.jsonl')):
        entries=[json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
        sent=[e for e in entries if e['event']=='sent']
        out=[e for e in entries if e['event']=='response']
        errors=[dict(status=e.get('status'),code=e.get('code')) for e in entries if e['event']=='error']
        logs[p.stem]=dict(requests=len(sent),responses=len(out),valid=sum(e['valid'] for e in out),errors=errors,
            models=sorted({e['model'] for e in sent}),input_tokens=sum(e['usage'].get('prompt_tokens',0) for e in out),
            output_tokens=sum(e['usage'].get('completion_tokens',0) for e in out))
        assert len(out)+len(errors)==len(sent)
    assert sum(r['requests'] for r in logs.values())<=500
    assert len(read(OUT/'official-replay.json'))==6
    assert all(r['max_profit_difference']==0 for r in read(OUT/'official-replay.json'))
    assert len(read(OUT/'official-real.json'))==1
    assert len(read(OUT/'native.json')['cases'])==24
    assert len(read(OUT/'strong-episodes.json.gz'))==960
    assert len(read(OUT/'diagnostic-windows.json.gz'))==3072
    for provider in MODELS:assert read(OUT/f'windows-{provider}.json')['windows']==24
    assert len(read(OUT/'pilot-deepseek.json'))==20
    save(OUT/'audit.json',dict(rounds_replayed=rounds,max_rounding_error=max_error,calibration_hashes_verified=True,
         index={arm:{str(k):float(np.mean([v for c,v in values if c==k])) for k in (50,51)} for arm,values in indices.items()},
         usage=logs))
    hashes={str(p.relative_to(ROOT)).replace(chr(92),'/'):digest(p) for base in (OUT,RAW,Path(__file__).parent,ROOT/'experiment-specs/econevals-pricing-v1')
         for p in sorted(base.iterdir()) if p.is_file() and p.name!='manifest.json'}
    save(OUT/'manifest.json',hashes)
    print('Validated',rounds,'settlements;',len(hashes),'files')


if __name__=='__main__':verify()

