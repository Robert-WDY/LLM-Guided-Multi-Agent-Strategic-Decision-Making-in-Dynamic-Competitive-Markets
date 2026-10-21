"""Fresh confirmation after failed v20 equal-coverage tests; never reuses holdout."""
from evaluate_reliable_v20 import *
from game_theory_agent.game_theory import empirical_advisor as empirical

DEST201=ROOT/'runs/reliable-advisor-v201'

def case201(args):
    split,m,n,seed,g,burn=args;p=DEST201/split/f'{m}-{n}-{seed}-{g}.json.gz'
    if p.exists():return read(p)['summary']
    c,s,mem,d,hist=snapshot(m,n,seed,burn);raw=d.to_dict();req=ReliableRequest(draft_action=raw,goal=g,mode='research',seed=seed,public_history=hist)
    r=empirical.proposal(c,s,req);a=rollout(c,s,mem,seed,r['action'],raw);b=rollout(c,s,mem,seed,raw,raw)
    delta={str(h):a['metrics'][h][g]-b['metrics'][h][g] for h in (3,5)}
    summary=dict(case_id=p.name.removesuffix('.json.gz'),seed=seed,market=m,companies=n,goal=g,intervention=economic_change(r['action'],raw),gain=delta['5'],gains=delta,config_sha256=c.config_sha256,
        features=r['public_features'],selected=r['recommended_id'],bounds={str(h):next((x['lower_bound'] for x in r['validation'] if x['horizon']==h),0) for h in (3,5)})
    save(p,dict(summary=summary,source=s.to_dict(),config=c.to_dict(),memory=mem,history=hist,draft=raw,proposal=r,actual=a,baseline=b));return summary

def freeze201():
    paths=[Path(empirical.__file__),Path(__file__),empirical.PATH,ROOT/'scripts/reliable_v19_common.py',ROOT/'scripts/evaluate_reliable_v20.py',ROOT/'src/game_theory_agent/game_theory/reliable_advisor.py',ROOT/'configs/reliable-response-v1.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths};p=DEST201/'frozen.json'
    if p.exists():assert read_json(p)==hashes
    else:write_json(p,hashes)
    freeze_market()

def main201(split):
    if split=='holdout':freeze201()
    start,count=(640001,30) if split=='calibration' else (650001,50)
    jobs=[(split,REGIMES[(i+j)%6],n,start+i,g,4+(i*2+j)%6) for i in range(count) for j,n in enumerate((2,5,10)) for g in ('profit','welfare')];rows=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for r in pool.map(case201,jobs):
            rows.append(r)
            if len(rows)%12==0:print('v201',split,len(rows),'/',len(jobs),flush=True)
    write_json(DEST201/(split+'.json'),dict(results=rows,cases=len(rows)))
    if split=='calibration':
        groups={}
        for g in ('profit','welfare'):
            selected=[r for r in rows if r['goal']==g and r['intervention']]
            groups[g]=dict(selected_cases=len(selected),seed_clusters=len({r['seed'] for r in selected}),overprediction_budget={str(h):max(0,quantile([r['bounds'][str(h)]-r['gains'][str(h)]/10000000 for r in selected],.9)) if selected else 0 for h in (3,5)})
        cal=dict(policy=empirical.VERSION,goals=groups,support={k:[min(r['features'][k] for r in rows),max(r['features'][k] for r in rows)] for k in rows[0]['features']},config_sha256=sorted({r['config_sha256'] for r in rows}),seeds=list(range(start,start+count)),quantile=.9,cases=len(rows))
        if empirical.PATH.exists():assert read_json(empirical.PATH)==cal
        else:write_json(empirical.PATH,cal)
        freeze201();print(json.dumps(groups),flush=True)

if __name__=='__main__':main201(sys.argv[1])
