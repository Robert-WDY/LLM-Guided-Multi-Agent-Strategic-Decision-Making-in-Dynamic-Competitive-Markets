"""Paired Agent Draft, fixed-candidate ablations and evaluator-only oracle."""
from reliable_v19_common import *

def case(args):
    m,n,seed,goal=args;path=OUT/'advice'/f'{m}-{n}-{seed}-{goal}.json.gz'
    if path.exists():return read(path)['summary']
    c,s,memory,draft,history=source(m,n,seed);raw=draft.to_dict();pool=local_candidates(c,s,'company_A',draft,SPEC['candidates'])
    actual={p['id']:rollout(c,s,memory,seed,p['action'],raw) for p in pool}
    rule=company_action(c,s,'company_A','balanced').to_dict();actual['rule']=rollout(c,s,memory,seed,rule,raw)
    rows=[];details={};spec=objective(ReliableRequest(draft_action=raw,goal=goal),s)
    history=[dict(round=x['state']['round'],prices=x['state']['prices']) for x in history]
    for mode in SPEC['modes']:
        req=ReliableRequest(draft_action=raw,goal=goal,mode='research',response_model=mode,max_candidates=SPEC['candidates'],step_budget=SPEC['budget'],seed=seed,public_history=history)
        r=advise(c,s,req);assert r['source_state_hash']==s.state_hash and r['candidates']==pool
        selected=r['recommended_id'];out=actual[selected];base=actual['agent_draft'];delta={}
        for h in (1,3,5):
            a=out['metrics'][h];b=base['metrics'][h];key='profit' if goal=='profit' else 'welfare'
            delta[h]=dict(value=a[key]-b[key],profit=a['profit']-b['profit'],welfare=a['welfare']-b['welfare'],worst_cash=a['minimum_cash'],exited=a['exited'],oracle_regret=max(v['metrics'][h][key] for k,v in actual.items() if k!='rule')-a[key])
        rows.append(dict(mode=mode,disposition=r['disposition'],selected=selected,delta=delta,seconds=r['search']['elapsed_seconds'],steps=r['search']['used_steps'],complete=r['search']['complete']))
        details[mode]=r
    oracle={h:max(actual[p['id']]['metrics'][h]['profit' if goal=='profit' else 'welfare'] for p in pool) for h in (1,3,5)}
    checks=sum(replay(c,s,x['transitions']) for x in actual.values())
    summary=dict(market=m,companies=n,seed=seed,goal=goal,results=rows,oracle=oracle,rule_delta={h:actual['rule']['metrics'][h]['profit' if goal=='profit' else 'welfare']-actual['agent_draft']['metrics'][h]['profit' if goal=='profit' else 'welfare'] for h in (1,3,5)},checks=checks)
    save(path,dict(summary=summary,source_state=s.to_dict(),config=c.to_dict(),draft=raw,history=history,actual=actual,details=details));return summary
def main(smoke=False):
    freeze_market()
    if smoke:
        for n in (2,5,10):
            r=case(('normal',n,518010+n,'profit'));print(n,[(x['mode'],x['selected'],x['delta'][3]['value']) for x in r['results']],flush=True)
        return
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [*ROOT.glob('src/game_theory_agent/game_theory/reliable*.py'),ROOT/'configs/reliable-response-v1.json',Path(__file__),ROOT/'scripts/reliable_v19_common.py']}
    path=OUT/'implementation-before-holdout.json'
    if path.exists():assert read_json(path)==hashes,'implementation changed after holdout start'
    else:write_json(path,hashes)
    cases=[(m,n,s,g) for m in REGIMES for n in (2,5,10) for s in SPEC['seeds'] for g in SPEC['goals']];rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(case,cases):
            rows.append(row)
            if len(rows)%12==0:print('advice',len(rows),'/720',flush=True)
    write_json(OUT/'advice.json',dict(passed=True,cases=len(rows),results=rows));print('advice complete',flush=True)
if __name__=='__main__':
    import sys
    try:main('--smoke' in sys.argv)
    except Exception as exc:
        write_json(OUT/f'failure-advice-{time.time_ns()}.json',dict(type=type(exc).__name__,error=str(exc)));raise
