"""Separate calibration/holdout and closed-loop experiments; zero paid calls."""
import sys
from diagnose_reliable_v20 import *
from game_theory_agent.game_theory.calibrated_advisor import proposal, apply_calibration, load_calibration, economic_change, POLICY_VERSION, CALIBRATION_PATH

def snapshots(split):
    start=610001 if split=='calibration' else 620001
    return [(split,REGIMES[(i+j)%6],n,start+i,g,4+(i*2+j)%6) for i in range(30) for j,n in enumerate((2,5,10)) for g in ('profit','welfare')]

def snapshot(m,n,seed,burn):
    c,e,s=make(m,n,seed);mem={};hist=[]
    for _ in range(burn):
        aa,mem=actions(c,s,mem,seed);hist.append(dict(round=s.round,prices=frame(s)['prices']));s=step(e,s,aa)
    aa,_=actions(c,s,mem,seed);return c,s,mem,aa['company_A'],hist

def case(args):
    split,m,n,seed,g,burn=args;cid=f'{m}-{n}-{seed}-{g}';p=DEST/split/(cid+'.json.gz')
    if p.exists():return read(p)['summary']
    c,s,mem,d,hist=snapshot(m,n,seed,burn);raw=d.to_dict();req=ReliableRequest(draft_action=raw,goal=g,mode='research',seed=seed,public_history=hist)
    proposed=proposal(c,s,req);a=rollout(c,s,mem,seed,proposed['action'],raw);b=rollout(c,s,mem,seed,raw,raw)
    delta={str(h):a['metrics'][h][g]-b['metrics'][h][g] for h in (1,3,5)}
    minimum_means={str(h):min((x['mean_gain'] for x in proposed['validation']+proposed['uncertainty_validation'] if x['horizon']==h),default=0) for h in (3,5)}
    summary=dict(case_id=cid,market=m,companies=n,seed=seed,goal=g,round=s.round,intervention=economic_change(proposed['action'],raw),gain=delta['5'],gains=delta,minimum_means=minimum_means,config_sha256=c.config_sha256,
                 selected=proposed['recommended_id'],features=proposed['public_features'],seconds=proposed['search']['elapsed_seconds'])
    save(p,dict(summary=summary,config=c.to_dict(),source=s.to_dict(),memory=mem,history=hist,draft=raw,proposal=proposed,actual=a,baseline=b))
    return summary

def quantile(values,q):
    v=sorted(values);return v[min(len(v)-1,max(0,math.ceil((len(v)+1)*q)-1))]

def calibrate(rows):
    assert all(610001<=r['seed']<=610030 for r in rows)
    groups={}
    for g in ('profit','welfare'):
        selected=[r for r in rows if r['goal']==g and r['intervention']]
        groups[g]=dict(selected_cases=len(selected),seed_clusters=len({r['seed'] for r in selected}),
                       overprediction_budget={str(h):max(0,quantile([r['minimum_means'][str(h)]-r['gains'][str(h)]/10000000 for r in selected],.9)) if selected else 0 for h in (3,5)})
    support={k:[min(r['features'][k] for r in rows),max(r['features'][k] for r in rows)] for k in rows[0]['features']}
    payload=dict(policy=POLICY_VERSION,protocol='selected-policy-empirical-overprediction-v1',quantile=.9,minimum_gain=0.,config_sha256=sorted({r['config_sha256'] for r in rows}),
                 goals=groups,support=support,seeds=list(range(610001,610031)),cases=len(rows),independent_unit='seed cluster; multiple market conditions within a seed are correlated',
                 caveat='Empirical selected-action error budget, not a predictive safety guarantee or proof of calibration under policy-induced shift.')
    if CALIBRATION_PATH.exists():assert read_json(CALIBRATION_PATH)==payload,'refusing to retune frozen calibration'
    else:write_json(CALIBRATION_PATH,payload)
    return payload

def freeze():
    paths=[ROOT/'src/game_theory_agent/game_theory/calibrated_advisor.py',Path(__file__),CALIBRATION_PATH,ROOT/'scripts/diagnose_reliable_v20.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    p=DEST/'implementation-before-holdout.json'
    if p.exists():assert read_json(p)==hashes,'implementation changed after holdout freeze'
    else:write_json(p,hashes)
    freeze_market()

def rolling(args):
    m,n,seed,g=args;cid=f'{m}-{n}-{seed}-{g}';p=DEST/'rolling'/(cid+'.json.gz')
    if p.exists():return read(p)['summary']
    c,initial,initial_memory,_,history=snapshot(m,n,seed,4);cal=load_calibration();branches={}
    for mode in ('none','once','every'):
        env=MarketEnv(c);env.load_state(initial);s=initial;mem=deepcopy(initial_memory);hist=deepcopy(history);total=0.;trace=[]
        for t in range(6):
            aa,mem=actions(c,s,mem,seed);draft=aa['company_A'].to_dict();r=None;changed=False
            if 'company_A' in s.strategic_market.active_company_ids and (mode=='every' or (mode=='once' and t==0)):
                req=ReliableRequest(draft_action=draft,goal=g,mode='research',seed=seed,public_history=hist)
                r=apply_calibration(proposal(c,s,req),c.config_sha256,cal);aa['company_A']=CompanyAction.from_dict(r['action']);changed=economic_change(r['action'],draft)
            hist.append(dict(round=s.round,prices=frame(s)['prices']));after=step(env,s,aa);audit_settlement(s.to_dict(),after.to_dict(),c.to_dict())
            value=after.company('company_A').financial.round_profit_cents if g=='profit' else after.welfare_accounting.round_total_economic_welfare_cents;total+=.95**t*value
            trace.append(dict(round=s.round,draft=draft,intervention=changed,advice=r,actions={k:v.to_dict() for k,v in aa.items()},state_before_hash=s.state_hash,state_after_hash=after.state_hash,round_goal_cents=value,discounted_cumulative_cents=total))
            s=after
        replay(c,initial,[dict(round=x['round'],actions=x['actions'],state_hash=x['state_after_hash']) for x in trace])
        branches[mode]=dict(total=total,trace=trace,interventions=sum(x['intervention'] for x in trace),final_state=s.to_dict())
    summary=dict(case_id=cid,market=m,companies=n,seed=seed,goal=g,once_gain=branches['once']['total']-branches['none']['total'],every_gain=branches['every']['total']-branches['none']['total'],
                 once_interventions=branches['once']['interventions'],every_interventions=branches['every']['interventions'],replay_passed=True)
    save(p,dict(summary=summary,source=initial.to_dict(),config=c.to_dict(),initial_memory=initial_memory,branches=branches));return summary

def main(split):
    if split=='rolling':
        freeze();jobs=[(REGIMES[(i+j)%6],n,630001+i,g) for i in range(10) for j,n in enumerate((2,5,10)) for g in ('profit','welfare')];fn=rolling
    else:
        if split=='holdout':freeze()
        jobs=snapshots(split);fn=case
    results=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for r in pool.map(fn,jobs):
            results.append(r)
            if len(results)%6==0:print(split,len(results),'/',len(jobs),flush=True)
    write_json(DEST/(split+'.json'),dict(results=results,cases=len(results)))
    if split=='calibration':
        print(json.dumps(calibrate(results)['goals']),flush=True);freeze()

if __name__=='__main__':main(sys.argv[1])
