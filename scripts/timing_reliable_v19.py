"""Serial local latency after batch research ends; distinct development-only seeds."""
from reliable_v19_common import *

def main():
    freeze_market();assert (OUT/'advice.json').exists(),'Measure after CPU-intensive primary batch completes'
    rows=[]
    for n in (2,5,10):
        for seed in (527001,527002,527003):
            c,s,_,draft,history=source('normal',n,seed)
            hist=[dict(round=x['state']['round'],prices=x['state']['prices']) for x in history]
            for mode in ('interactive','research') if seed==527001 else ('interactive',):
                start=time.perf_counter();r=advise(c,s,ReliableRequest(draft_action=draft.to_dict(),public_history=hist,goal='profit',mode=mode,max_candidates=12,seed=seed));elapsed=time.perf_counter()-start
                assert r['search']['complete'] and r['search']['used_steps']<=1800
                rows.append(dict(companies=n,seed=seed,mode=mode,wall_seconds=elapsed,steps=r['search']['used_steps'],disposition=r['disposition']))
                print(n,seed,mode,round(elapsed,3),flush=True)
    write_json(OUT/'timing.json',dict(passed=True,rows=rows,interactive_under_5_seconds=all(x['wall_seconds']<5 for x in rows if x['mode']=='interactive'),scope='Serial warm local process; 9 interactive/3 research samples, not a production p99 or concurrency guarantee.'))
if __name__=='__main__':main()
