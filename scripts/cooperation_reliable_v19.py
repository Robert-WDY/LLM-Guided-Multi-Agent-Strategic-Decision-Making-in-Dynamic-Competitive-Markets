"""Threshold × size × institution × defection: promises enter actual public-goods/aid settlement."""
from reliable_v19_common import *
from game_theory_agent.cooperation.institution import DirectedCredibility
from game_theory_agent.market.cooperation_personas import apply_cooperation_persona

def episode(case):
    threshold,n,institution,defection,seed=case;path=OUT/'cooperation'/f'{threshold}-{n}-{institution}-{defection}-{seed}.json.gz'
    if path.exists():return read(path)['summary']
    c,_,_=make('normal',n,seed,10);raw=c.to_dict();raw['strategic_market']['threshold_project']['required_total_contribution_cents']=n*300000 if threshold=='reachable' else n*100000*4+1
    c=MarketConfig.from_mapping(raw);e=MarketEnv(c);s=e.reset(company_ids=[f'company_{chr(65+i)}' for i in range(n)],episode_id=f'institution-{seed}',episode_seed=seed,max_rounds=10,cooperation_mode='combined_v1');initial=s.to_dict();ledger=DirectedCredibility(s.company_ids)
    defectors=set(s.company_ids[:0 if defection=='none' else 1 if defection=='one' else max(2,n//2)])
    rows=[];accepted_bad=proposed_bad=accepted_good=proposed_good=aid=0
    for t in range(10):
        cooperative={a:apply_cooperation_persona(c,s,a,company_action(c,s,a,'balanced'),'cooperator') for a in s.company_ids}
        previous={a.agent_id:(a.shared_resilience_contribution_cents or 0)+(a.threshold_project_contribution_cents or 0) for a in s.last_joint_action}
        responses={}
        active=s.strategic_market.active_company_ids
        willing={observer:{proposer:(True if institution=='none' else (not t or any(v>0 for a,v in previous.items() if a!=observer)) if institution=='aggregate' else ledger.accepts(observer,proposer,s.round)) for proposer in active if proposer!=observer} for observer in active}
        # Honest actors decide whether to participate before issuing a promise.
        # Refusing participation is not a broken promise; only defectors announce a contribution they intend not to make.
        promises={a:((x.shared_resilience_contribution_cents or 0)+(x.threshold_project_contribution_cents or 0)) if a in defectors or any(willing.get(a,{}).values()) else 0 for a,x in cooperative.items()}
        for observer in active:
            for proposer in active:
                if observer==proposer or promises[proposer]<=0:continue
                accepted=willing[observer][proposer]
                responses[f'{observer}>{proposer}']=accepted
                if proposer in defectors:proposed_bad+=1;accepted_bad+=accepted
                else:proposed_good+=1;accepted_good+=accepted
        aa={}
        for actor in s.company_ids:
            contribute=actor not in defectors and promises[actor]>0
            aa[actor]=cooperative[actor] if contribute else apply_cooperation_persona(c,s,actor,company_action(c,s,actor,'balanced'),'free_rider')
        # Public-good benefits remain non-excludable. Only an existing bilateral aid offer can be withheld from a specific proposer.
        for i in range(0,n-1,2):
            recipient,donor=s.company_ids[i:i+2]
            if recipient not in active or donor not in active:continue
            if responses.get(f'{donor}>{recipient}',False):
                # Aid fields do not change the already announced contribution budget.
                # Settlement caps the transfer by capacity and recipient cash.
                aa[donor]=replace(aa[donor],mutual_aid_partner_company_id=recipient,mutual_aid_capacity_offer_orders=1000,mutual_aid_capacity_request_orders=0)
                aa[recipient]=replace(aa[recipient],mutual_aid_partner_company_id=donor,mutual_aid_capacity_request_orders=1000,mutual_aid_capacity_offer_orders=0)
        after=step(e,s,aa);audit_settlement(s.to_dict(),after.to_dict(),c.to_dict())
        realized={a:(x.shared_resilience_contribution_cents or 0)+(x.threshold_project_contribution_cents or 0) for a,x in aa.items()}
        for key,accepted in responses.items():
            observer,proposer=key.split('>');ledger.verify(observer,proposer,promises[proposer],realized[proposer],s.round,accepted=accepted)
        aid+=sum(x['fulfilled_orders'] for x in after.to_dict()['strategic_market']['mutual_aid']['last_transfers'])
        rows.append(dict(round=s.round,actions={a:x.to_dict() for a,x in aa.items()},state_hash=after.state_hash,promises=promises,responses=responses,realized=realized));s=after
    beliefs=[(a,b,ledger.belief(a,b)['credibility']) for a in s.company_ids for b in s.company_ids if a!=b]
    summary=dict(threshold=threshold,companies=n,institution=institution,defection=defection,seed=seed,project_success=s.strategic_market.threshold_project.status.value=='succeeded',profit=sum(x.financial.cumulative_profit_cents for x in s.companies),focal_profit=s.company('company_A').financial.cumulative_profit_cents,welfare=s.welfare_accounting.cumulative_total_economic_welfare_cents,aid_orders=aid,
                 defectors=len(defectors),bad_acceptance=accepted_bad/proposed_bad if proposed_bad else None,good_acceptance=accepted_good/proposed_good if proposed_good else None,bad_credibility=mean(v for a,b,v in beliefs if b in defectors) if defectors else None,checks=replay(c,MarketState.from_dict(initial),rows))
    assert DirectedCredibility.restore(ledger.export()).export()==ledger.export()
    save(path,dict(summary=summary,initial_state=initial,config=c.to_dict(),transitions=rows,ledger=ledger.export()));return summary
def main():
    freeze_market();cases=[(t,n,i,d,s) for t in ('reachable','unreachable') for n in (2,5,10) for i in ('none','aggregate','individual') for d in ('none','one','multiple') for s in SPEC['cooperation_seeds']];rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(episode,cases):
            rows.append(row)
            if len(rows)%120==0:print('cooperation',len(rows),'/1080',flush=True)
    write_json(OUT/'cooperation.json',dict(passed=True,cases=len(rows),rounds=10800,results=rows));print('cooperation complete',flush=True)
if __name__=='__main__':main()
