"""Compare simpler frozen v19 response models on the same new holdout states."""
from evaluate_reliable_v20 import *

def ablate(path):
    v=read(path);s=v['summary'];dest=DEST/'response-ablations'/path.name
    if dest.exists():return read(dest)['summary']
    c=MarketConfig.from_mapping(v['config']);state=MarketState.from_dict(v['source']);rows=[]
    for mode in ('none','generic'):
        req=ReliableRequest(draft_action=v['draft'],goal=s['goal'],mode='research',response_model=mode,seed=s['seed'],public_history=v['history'])
        r=advise(c,state,req);out=rollout(c,state,v['memory'],s['seed'],r['action'],v['draft'])
        rows.append(dict(mode=mode,advice=r,actual=out,gain=out['metrics'][5][s['goal']]-v['baseline']['metrics']['5'][s['goal']],intervention=economic_change(r['action'],v['draft'])))
    summary=dict(case_id=s['case_id'],seed=s['seed'],goal=s['goal'],market=s['market'],companies=s['companies'],results=[{k:r[k] for k in ('mode','gain','intervention')} for r in rows])
    save(dest,dict(summary=summary,results=rows));return summary

if __name__=='__main__':
    results=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for r in pool.map(ablate,sorted((DEST/'holdout').glob('*.gz'))):
            results.append(r)
            if len(results)%6==0:print('response ablations',len(results),'/180',flush=True)
    assert len(results)==180
    write_json(DEST/'response-ablations.json',dict(results=results))
