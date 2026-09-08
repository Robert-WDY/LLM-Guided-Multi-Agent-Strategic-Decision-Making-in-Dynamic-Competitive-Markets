"""All preregistered cells, paired seed statistics and explicit paid-experiment gate."""
from collections import defaultdict
from reliable_v19_common import *

def stats(values):return paired_summary(values,draws=2000)
def delta(row,h):return row['delta'][str(h)]
def main():
    freeze_market();a=read_json(OUT/'advice.json');c=read_json(OUT/'cooperation.json');s=read_json(OUT/'scaling.json');cal=read_json(OUT/'response-calibration.json');assert len(a['results'])==720 and len(c['results'])==1080 and len(s['results'])==120
    for p,h in read_json(OUT/'implementation-before-holdout.json').items():assert sha(ROOT/p)==h
    groups=[];pooled=defaultdict(lambda:defaultdict(list));checks=sum(r['checks'] for r in a['results'])+sum(r['checks'] for r in c['results'])+sum(r['checks'] for r in s['results'])
    for m in REGIMES:
        for n in (2,5,10):
            for goal in SPEC['goals']:
                cases=[x for x in a['results'] if (x['market'],x['companies'],x['goal'])==(m,n,goal)]
                for mode in SPEC['modes']:
                    rr=[next(x for x in r['results'] if x['mode']==mode) for r in cases];comparison=[next(x for x in r['results'] if x['mode']=='none') for r in cases]
                    for horizon in (3,5):
                        gains=[delta(r,horizon)['value'] for r in rr];st=stats(gains);record=dict(market=m,companies=n,goal=goal,mode=mode,horizon=horizon,value=st,profit=stats([delta(r,horizon)['profit'] for r in rr]),welfare=stats([delta(r,horizon)['welfare'] for r in rr]),negative_rate=st['negative']/20,abstentions=sum(x['disposition']=='abstain' for x in rr),complete=sum(x['complete'] for x in rr),effectiveness_pass=st['interval'][0]>0 and st['negative']/20<=.1,oracle_regret=mean(delta(x,horizon)['oracle_regret'] for x in rr),vs_search=stats([delta(x,horizon)['value']-delta(y,horizon)['value'] for x,y in zip(rr,comparison)]))
                        groups.append(record)
                        for case,row in zip(cases,rr):pooled[f'{mode}-h{horizon}'][case['seed']].append(delta(row,horizon)['value'])
    pooled={k:stats([mean(v) for v in byseed.values()]) for k,byseed in pooled.items()}
    institution=[]
    for threshold in ('reachable','unreachable'):
        for n in (2,5,10):
            for mode in ('none','aggregate','individual'):
                for defection in ('none','one','multiple'):
                    rows=[r for r in c['results'] if (r['threshold'],r['companies'],r['institution'],r['defection'])==(threshold,n,mode,defection)];base=[next(x for x in c['results'] if (x['threshold'],x['companies'],x['institution'],x['defection'],x['seed'])==(threshold,n,mode,'none',r['seed'])) for r in rows]
                    institution.append(dict(threshold=threshold,companies=n,institution=mode,defection=defection,project_success=sum(r['project_success'] for r in rows),focal_advantage=stats([r['focal_profit']-b['focal_profit'] for r,b in zip(rows,base)]),welfare=mean(r['welfare'] for r in rows),bad_acceptance=mean(r['bad_acceptance'] for r in rows) if defection!='none' else None,good_acceptance=mean(r['good_acceptance'] for r in rows if r['good_acceptance'] is not None) if any(r['good_acceptance'] is not None for r in rows) else None,credibility=mean(r['bad_credibility'] for r in rows) if defection!='none' else None,aid=mean(r['aid_orders'] for r in rows)))
    scales=[]
    for mode in ('normal','scaled'):
        for n in (2,5,10):
            rows=[r for r in s['results'] if r['regime']==mode and r['companies']==n];scales.append(dict(regime=mode,companies=n,margin=mean(r['margin'] for r in rows if r['margin'] is not None),price_dispersion=mean(r['price_dispersion'] for r in rows),exit_rate=sum(r['exits'] for r in rows)/(20*n),stockout=mean(r['stockout'] for r in rows),consumer_welfare=mean(r['consumer_welfare'] for r in rows)))
    primary=[r for r in groups if r['mode']=='conditional' and r['horizon']==3];five=[x for row in a['results'] for x in row['results'] if x['mode']=='conditional'];passed=sum(x['effectiveness_pass'] for x in primary);negative5=sum(delta(x,5)['value']<0 for x in five)
    reasons=[]
    if passed!=36:reasons.append(f'conditional primary groups passed {passed}/36')
    if negative5/720>.1:reasons.append(f'5-round negative rate {negative5/720:.3%} exceeds 10%')
    if not cal['conditional_no_worse']:reasons.append('held-out response Brier worse than generic')
    if any(not x['complete'] for x in five):reasons.append('incomplete search/validation cases')
    result=dict(passed=True,advice_cases=720,advisor_calls=2160,institution_episodes=1080,scaling_episodes=120,arithmetic_checks=checks,groups=groups,pooled=pooled,institution=institution,scaling=scales,conditional_primary_pass=passed,conditional_3round_losses=sum(delta(x,3)['value']<0 for x in five),conditional_5round_losses=negative5,conditional_abstentions=sum(x['disposition']=='abstain' for x in five),calibration={k:{a:b for a,b in v.items() if a!='rows'} for k,v in cal['results'].items()},paid_economic_gate=not reasons,paid_gate_reasons=reasons,world_calibrated=False,scope='Evaluation completed is not equivalent to reliable improvement. All margins and payoffs are synthetic. Conditions are repeated measures of 20 seeds.')
    write_json(OUT/'analysis.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('groups','institution','scaling')}),flush=True)
if __name__=='__main__':main()
