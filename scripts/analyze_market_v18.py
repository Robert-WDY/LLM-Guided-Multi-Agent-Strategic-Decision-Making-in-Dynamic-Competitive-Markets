"""Matched, seed-clustered analysis and final independent replay of all evidence."""
from collections import defaultdict
from statistics import mean,median
from evaluate_market_v18 import *

def summary(values):return paired_summary(values,draws=2000)
def p95(values):return sorted(values)[max(0,math.ceil(.95*len(values))-1)]
def interval(s):return f"{s['mean']/100:,.0f} [{s['interval'][0]/100:,.0f}, {s['interval'][1]/100:,.0f}]"

def verify_episode(row):
 key=f"{row['market']}-{row['companies']}-{row['seed']}-{row['rounds']}-{row['policy']}";p=OUT/'episodes'/(key+'.json.gz');v=read_compressed(p);c=MarketConfig.from_mapping(v['config']);e=MarketEnv(c);s=MarketState.from_dict(v['initial_state']);e.load_state(s);checks=0
 for t in v['transitions']:
  a=step(e,s,{cid:CompanyAction.from_dict(x) for cid,x in t['actions'].items()},t['extra']);assert a.state_hash==t['state_hash'];checks+=audit_settlement(s.to_dict(),a.to_dict(),v['config']);s=a
 return checks

def main():
 t=read_json(OUT/'tournament.json');a=read_json(OUT/'advisors.json');closed=read_json(OUT/'closed_loop.json');latency=read_json(OUT/'latency.json');probes=read_json(OUT/'probes.json')
 assert len(t['results'])==2376 and len(a['results'])==288 and len(closed['results'])==54
 index={(r['market'],r['companies'],r['seed'],r['policy']):r for r in t['results']};groups=[];contrasts=[]
 for market in MARKETS:
  for n in (2,5,10):
   for policy in POLICIES:
    rs=[index[(market,n,s,policy)] for s in SPEC['tournament_seeds']];b=[index[(market,n,s,'rule')] for s in SPEC['tournament_seeds']]
    groups.append(dict(market=market,companies=n,policy=policy,n=8,welfare_delta=summary([r['welfare']-v['welfare'] for r,v in zip(rs,b)]),profit_delta=summary([r['profit']-v['profit'] for r,v in zip(rs,b)]),mean_profit=mean(r['profit'] for r in rs),mean_welfare=mean(r['welfare'] for r in rs),mean_consumer_surplus=mean(r['consumer_surplus'] for r in rs),mean_price=mean(r['mean_price'] for r in rs),mean_margin=mean(r['margin'] for r in rs if r['margin'] is not None),mean_contributions=mean(r['contributions'] for r in rs),project_successes=sum(r['project_success'] for r in rs),mean_aid_orders=mean(r['aid_orders'] for r in rs),mean_fines=mean(r['fines'] for r in rs),exits=sum(r['exits'] for r in rs),mean_volatility=mean(r['price_volatility'] for r in rs),mean_stockout=mean(r['stockout'] for r in rs)))
   for treatment,baseline in (('cooperators','free_riders'),('one_defector','cooperators'),('reciprocity','one_defector')):
    rs=[index[(market,n,s,treatment)] for s in SPEC['tournament_seeds']];bs=[index[(market,n,s,baseline)] for s in SPEC['tournament_seeds']];contrasts.append(dict(market=market,companies=n,treatment=treatment,baseline=baseline,welfare=summary([r['welfare']-b['welfare'] for r,b in zip(rs,bs)]),focal_profit=summary([r['focal_profit']-b['focal_profit'] for r,b in zip(rs,bs)])))
 advisor_groups=[];pooled={};available=[r for r in a['results'] if r['available']]
 for market in ADVISOR_MARKETS:
  for n in (2,5,10):
   for goal in ('profit','welfare'):
    cases=[r for r in available if (r['market'],r['companies'],r['goal'])==(market,n,goal)]
    for variant in ('v16','v17','fast'):
     rs=[next(x for x in r['results'] if x['variant']==variant) for r in cases];bs=[next(x for x in r['results'] if x['variant']=='v16') for r in cases]
     if not rs:continue
     gains=summary([r['gain'] for r in rs]);incumbent=summary([r['incumbent_gain'] for r in rs]);advisor_groups.append(dict(market=market,companies=n,goal=goal,variant=variant,n=len(rs),vs_rule=gains,vs_incumbent=incumbent,vs_v16=summary([r['gain']-b['gain'] for r,b in zip(rs,bs)]),effectiveness_pass=gains['interval'][0]>0 and gains['negative']/len(rs)<=.1,incumbent_pass=incumbent['interval'][0]>0 and incumbent['negative']/len(rs)<=.1,mean_oracle_regret=mean(r['oracle_regret'] for r in rs),oracle_optimal=sum(r['oracle_regret']==0 for r in rs),complete=sum(r['complete'] for r in rs),median_latency=median(r['latency'] for r in rs),profit_mae=mean(abs(r['profit_error']) for r in rs)))
 for variant in ('v16','v17','fast'):
  for goal in ('profit','welfare'):
   byseed=defaultdict(list);losses=incumbent_losses=regret=0;total=0
   for row in available:
    if row['goal']!=goal:continue
    r=next(r for r in row['results'] if r['variant']==variant);byseed[row['seed']].append(r['gain']);losses+=r['gain']<0;incumbent_losses+=r['incumbent_gain']<0;regret+=r['oracle_regret'];total+=1
   pooled[f'{variant}-{goal}']=dict(seed_clustered_gain=summary([mean(v) for v in byseed.values()]),cases=total,losses=losses,incumbent_losses=incumbent_losses,mean_oracle_regret=regret/total)
 latency_groups=[]
 for n in (2,5,10):
  for variant in ('v16','v17','fast'):
   rs=[r for r in latency['results'] if r['companies']==n and r['variant']==variant];limit=SPEC['latency_p95_seconds'][str(n)];latency_groups.append(dict(companies=n,variant=variant,n=len(rs),median=median(r['seconds'] for r in rs),p95=p95([r['seconds'] for r in rs]),limit=limit,passed=all(r['complete'] for r in rs) and p95([r['seconds'] for r in rs])<=limit))
 loop=[]
 for r in closed['results']:
  if r['policy']=='closed_rule':continue
  b=next(x for x in closed['results'] if (x['market'],x['companies'],x['seed'],x['policy'])==(r['market'],r['companies'],r['seed'],'closed_rule'))
  loop.append(dict(market=r['market'],companies=r['companies'],seed=r['seed'],policy=r['policy'],profit_delta=r['profit']-b['profit'],focal_profit_delta=r['focal_profit']-b['focal_profit'],welfare_delta=r['welfare']-b['welfare'],exits=r['exits'],advice=r['advisor_decisions'],complete=r['response_complete']))
 # Final replay uses the patched policy-independent oracle, including exits and aid receipts.
 with ProcessPoolExecutor(max_workers=4) as pool:checks=sum(pool.map(verify_episode,t['results']+closed['results']))
 workload=defaultdict(list)
 for row in closed['results']:
  key=f"{row['market']}-{row['companies']}-{row['seed']}-{row['rounds']}-{row['policy']}";v=read_compressed(OUT/'episodes'/(key+'.json.gz'))
  for d in v['advice']:workload[row['companies']].append(d['result']['search']['elapsed_seconds'])
 workload_latency={str(n):dict(calls=len(values),median=median(values),p95=p95(values),maximum=max(values)) for n,values in workload.items()}
 advice_checks=0;negative_cash=actual_exits=0
 for row in available:
  v=read_compressed(OUT/'advisors'/f"{row['market']}-{row['companies']}-{row['seed']}-{row['goal']}.json.gz");c=MarketConfig.from_mapping(v['config']);s=MarketState.from_dict(v['source_state']);other={k:CompanyAction.from_dict(x) for k,x in v['actual_other_actions'].items()}
  for variant,r in v['details'].items():
   e=MarketEnv(c);e.load_state(s);after=step(e,s,{**other,'company_A':CompanyAction.from_dict(r['action'])},v['extra']);advice_checks+=audit_settlement(s.to_dict(),after.to_dict(),v['config']);saved=next(x for x in row['results'] if x['variant']==variant);real=after.company('company_A').financial.round_profit_cents if row['goal']=='profit' else after.welfare_accounting.round_total_economic_welfare_cents;assert real==saved['realized'];negative_cash+=after.company('company_A').financial.cash_balance_cents<0;actual_exits+='company_A' not in after.strategic_market.active_company_ids
 sources=read_json(OUT/'world-sources.json');retail=next(s for s in sources['sources'] if s['id']=='retail');anchors={x['quarter']:x['profit_billions']/x['sales_billions'] for x in retail['data']}
 scope=dict(agents='Scripted public-history policies and UCB learners, plus computed advice; not newly paid LLM free-form negotiation',independent_seeds=8,world_calibrated=False,global_optimality=False,oracle='One-round full-state evaluator, up to 48 generated legal candidates plus all proposed and incumbent actions; not an omniscient production advisor',latency='Serial after warmup; three repetitions per group; p95 is approximately maximum',scope_limits=['No entry/new firms','Two suppliers fixed across company scales','Consumer cohorts, not independent households','No wages, income dynamics or complete banking sector','No empirically estimated demand elasticities or cost distributions','No new free-form LLM bargaining experiment'])
 result=dict(passed=True,tournament_episodes=len(t['results']),tournament_rounds=sum(r['rounds'] for r in t['results']),advisor_cases=len(a['results']),advisor_available=len(available),advisor_decisions=sum(len(r['results']) for r in available),unavailable=[r for r in a['results'] if not r['available']],groups=groups,cooperation_contrasts=contrasts,advisor_groups=advisor_groups,pooled=pooled,latency=latency_groups,closed_loop_workload_latency=workload_latency,closed_loop=loop,closed_loop_episodes=54,closed_loop_rounds=540,closed_loop_advice=sum(r['advisor_decisions'] for r in closed['results']),negative_cash=negative_cash,actual_one_step_exits=actual_exits,arithmetic_checks=checks+advice_checks,final_replay=True,probes=probes,external_retail_margins=anchors,scope=scope,new_paid_calls=0)
 write_json(OUT/'analysis.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('groups','cooperation_contrasts','advisor_groups','closed_loop','probes','scope')}))

if __name__=='__main__':main()
