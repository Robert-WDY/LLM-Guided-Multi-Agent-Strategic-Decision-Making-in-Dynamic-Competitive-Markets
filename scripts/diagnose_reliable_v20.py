"""Researcher-only fixed-action 2x2 diagnosis. Never imported by serving code."""
import argparse
import random
from collections import defaultdict
from statistics import mean
from reliable_v19_common import *
from game_theory_agent.game_theory.advisor_market import economic_key, forecast
from game_theory_agent.game_theory.reliable_advisor import DraftRollouts
from game_theory_agent.market.protocols import state_hash

DEST=ROOT/'runs/reliable-advisor-v20'

def stats(rows):
    intervened=[r for r in rows if r['intervention']]
    losses=[r['gain'] for r in intervened if r['gain']<0]
    byseed=defaultdict(list)
    for r in rows: byseed[r['seed']].append(r['gain'])
    rng=random.Random(2009); clusters=list(byseed.values())
    means=sorted(mean(mean(rng.choice(clusters)) for _ in clusters) for _ in range(1500)) if clusters else [0]*1500
    values=sorted(r['gain'] for r in intervened)
    return dict(cases=len(rows),independent_seed_clusters=len(clusters),interventions=len(intervened),coverage=len(intervened)/max(1,len(rows)),
                losses=len(losses),intervention_loss_rate=len(losses)/len(intervened) if intervened else None,
                all_case_mean_gain_cents=mean(r['gain'] for r in rows) if rows else None,
                cluster_bootstrap_mean_ci95_cents=[means[37],means[1462]],
                mean_loss_cents=mean(losses) if losses else 0,worst_gain_cents=min(values) if values else None,
                lower_ten_percent_mean_cents=mean(values[:max(1,math.ceil(len(values)*.1))]) if values else None)

def inventory():
    rows=[]
    for p in sorted((OUT/'advice').glob('*.json.gz')):
        v=read(p);s=v['summary']
        if s['seed'] not in SPEC['seeds']:continue
        r=v['details']['conditional'];selected=r['recommended_id'];goal=s['goal']
        changed=economic_key(CompanyAction.from_dict(r['action']))!=economic_key(CompanyAction.from_dict(r['draft_action']))
        assert changed==(r['disposition']=='recommend')
        gain=v['actual'][selected]['metrics']['5'][goal]-v['actual']['agent_draft']['metrics']['5'][goal]
        rows.append(dict(case_id=p.name.removesuffix('.json.gz'),market=s['market'],companies=s['companies'],seed=s['seed'],goal=goal,selected=selected,intervention=changed,gain=gain))
    assert len(rows)==720
    write_json(DEST/'intervention-risk-v19.json',dict(results={g:stats([r for r in rows if r['goal']==g]) for g in ('profit','welfare')},rows=rows,action_disposition_mismatches=0))
    negatives=[r for r in rows if r['gain']<0];chosen=list(negatives);used={r['case_id'] for r in chosen};matches=[]
    for r in negatives:
        options=[x for x in rows if x['case_id'] not in used and x['intervention'] and x['gain']>=0 and all(x[k]==r[k] for k in ('market','companies','goal','selected'))]
        match=min(options,key=lambda x:x['seed']) if options else None
        if match:chosen.append(match);used.add(match['case_id'])
        matches.append(dict(negative=r['case_id'],matched=match['case_id'] if match else None,criteria=['market','companies','goal','selected']))
    write_json(DEST/'diagnostic-selection.json',dict(cases=chosen,matches=matches))
    return chosen

def simulate(c,initial,memory,policy_seed,request,model,first,draft,scenario,horizon,policy,random_seed):
    s=replace(initial,episode_seed=random_seed);s=replace(s,state_hash=state_hash(s.to_dict()))
    e=MarketEnv(c);e.load_state(s);mem=deepcopy(memory);runner=DraftRollouts(c,initial,request,model)
    previous=frame(s);service=0;profit=welfare=0.;components=defaultdict(float);trace=[];rounds=[]
    for t in range(min(horizon,s.rounds_remaining)):
        current=frame(s)
        if policy=='actual':
            # Future market randomness changes; established seeded operating styles do not.
            policy_state=replace(s,episode_seed=policy_seed)
            policy_state=replace(policy_state,state_hash=state_hash(policy_state.to_dict()))
            aa,mem=actions(c,policy_state,mem,policy_seed)
        else:
            aa={cid:company_action(c,s,cid,'balanced') for cid in s.company_ids}
            for peer in s.company_ids:
                if peer=='company_A' or peer not in s.strategic_market.active_company_ids:continue
                mode='generic' if t==0 else request.response_model
                p=runner.response(features(previous,current,peer,service),mode,peer)
                raw=aa[peer].to_dict();raw['price_cents']=round(s.company(peer).commercial.price_cents*(1+sample(p,request.seed,scenario,s.round,peer)))
                aa[peer]=rebind(c,s,peer,raw)
        aa['company_A']=rebind(c,s,'company_A',first if t==0 else draft)
        a=step(e,s,aa);audit_settlement(s.to_dict(),a.to_dict(),c.to_dict())
        profit+=.95**t*a.company('company_A').financial.round_profit_cents;welfare+=.95**t*a.welfare_accounting.round_total_economic_welfare_cents
        wc={k:v for k,v in a.welfare_accounting.to_dict().items() if k.startswith('round_') and isinstance(v,(int,float))}
        for k,v in wc.items():components[k]+=.95**t*v
        rounds.append(dict(profit=profit,welfare=welfare,components=dict(components)))
        trace.append(dict(round=s.round,actions={k:v.to_dict() for k,v in aa.items()},state_hash=a.state_hash))
        previous=current;service=aa['company_A'].service_budget_cents;s=a
    return dict(profit=profit,welfare=welfare,components=dict(components),rounds=rounds,transitions=trace)

def diagnose(row):
    path=DEST/'diagnosis'/(row['case_id']+'.json.gz')
    if path.exists():return read(path)['summary']
    saved=read(OUT/'advice'/(row['case_id']+'.json.gz'));c,s,memory,_,_=source(row['market'],row['companies'],row['seed'])
    assert s.to_dict()==saved['source_state']
    raw=saved['draft'];selected=saved['details']['conditional']['action'];req=ReliableRequest(draft_action=raw,goal=row['goal'],mode='research',seed=row['seed'],public_history=saved['history']);model=load_model()
    public,_=forecast(c,s,'company_A');goal=row['goal']
    controls=[]
    for action,cid in ((raw,'agent_draft'),(selected,row['selected'])):
        out=simulate(c,s,memory,row['seed'],req,model,action,raw,0,5,'actual',s.episode_seed)
        original=saved['actual'][cid]
        assert out['transitions']==original['transitions'],'D consistency failure'
        assert abs(out[goal]-original['metrics']['5'][goal])<1e-5
        controls.append(out)
    groups={k:[] for k in 'ABCD'}
    # 8 fresh random paths shared across all groups, independent of selection paths.
    for scenario in range(2000,2008):
        seed=int(hashlib.sha256(f'v20-diagnostic:{row["seed"]}:{scenario}'.encode()).hexdigest()[:15],16)
        for group,initial,policy in [('A',public,'model'),('B',s,'model'),('C',public,'actual'),('D',s,'actual')]:
            b=simulate(c,initial,memory,row['seed'],req,model,raw,raw,scenario,5,policy,seed)
            a=simulate(c,initial,memory,row['seed'],req,model,selected,raw,scenario,5,policy,seed)
            groups[group].append(dict(scenario=scenario,random_seed=seed,baseline=b,candidate=a,gain=a[goal]-b[goal],components={k:a['components'][k]-b['components'][k] for k in a['components']}))
    means={k:mean(x['gain'] for x in v) for k,v in groups.items()};errors={k:means[k]-means['D'] for k in 'ABCD'}
    summary=dict(**row,means_cents=means,errors_cents=errors,state_correction_cents=means['B']-means['A'],policy_correction_cents=means['C']-means['A'],interaction_cents=means['D']-means['B']-means['C']+means['A'],control_exact=True,policy_identity_fixed=True)
    save(path,dict(summary=summary,groups=groups,original_controls=controls,scope='Privileged diagnosis only; true initial states and actual branch-updated policy memory must not enter advisor'))
    return summary

def main():
    from game_theory_agent.game_theory.reliable_response import load_model,sample
    freeze_market();DEST.mkdir(parents=True,exist_ok=True);rows=inventory();print('risk complete; diagnostic cases',len(rows),flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        results=[]
        for r in pool.map(diagnose,rows):
            results.append(r);print('diagnosis',len(results),'/',len(rows),r['case_id'],flush=True)
    write_json(DEST/'diagnosis.json',dict(cases=len(results),results=results,all_controls_exact=all(r['control_exact'] for r in results)))

from game_theory_agent.game_theory.reliable_response import load_model,sample
if __name__=='__main__':main()
