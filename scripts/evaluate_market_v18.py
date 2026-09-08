"""Preregistered cooperation/competition and advisor evaluation, no paid calls.

All interventions are evaluator scenarios. They never modify the shipped market.
Each trajectory is independently replayed and audited, then stored compressed.
"""
import argparse,gzip,hashlib,json,math,time
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
from statistics import mean
from concurrent.futures import ProcessPoolExecutor
from game_theory_agent.market import MarketEnv,MarketConfig,MarketState,CompanyAction,load_market_config
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.market.actor_policies import actor_ids,company_action
from game_theory_agent.market.actor_experiments import learning_choice,update_memory,read_json,write_json
from game_theory_agent.market.cooperation_personas import apply_cooperation_persona
from game_theory_agent.game_theory.advisor_market import recipe_action
from game_theory_agent.game_theory.advisor_candidates import expanded
from game_theory_agent.game_theory.advisor_beliefs import RobustRequest,public_frame
from game_theory_agent.game_theory.advisor_robust import advise
from game_theory_agent.game_theory.advisor_search import advise as advise16
from game_theory_agent.game_theory.objectives import AdviceRequest
from game_theory_agent.game_theory.advisor_audit import audit_settlement,paired_summary
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.local_budget import status

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/market-evaluation-v18';BASE=load_market_config(ROOT/'configs/market_v14_local.yaml')
MARKETS=('normal','recession','boom','tight_supply','cash_stress','price_sensitive','quality_sensitive','disruption','project_feasible','scaled','asymmetric')
POLICIES=('rule','cooperators','free_riders','one_defector','reciprocity','price_war','coordination','mutual_aid','learners')
ADVISOR_MARKETS=('normal','recession','tight_supply','price_sensitive','project_feasible','scaled')
SPEC=dict(version='evaluation-v18.0',markets=list(MARKETS),companies=[2,5,10],tournament_seeds=list(range(481001,481009)),policies=list(POLICIES),rounds=20,advisor_markets=list(ADVISOR_MARKETS),advisor_seeds=list(range(482001,482009)),goals=['profit','welfare'],advisor_variants=['v16','v17','fast'],advisor_horizon=1,scenarios=2,candidates=12,step_budget=1200,fast_candidates=8,fast_finalists=1,oracle_candidates=48,closed_loop_seeds=[483001,483002],closed_loop_markets=['normal','tight_supply','project_feasible'],closed_loop_rounds=10,closed_loop_variants=['rule','focal_v17','all_v17'],latency_seeds=[484001,484002,484003],latency_profiles=['v16','v17','fast'],latency_p95_seconds={'2':3,'5':5,'10':10},effectiveness='mean paired objective gain bootstrap lower bound >0 and at most 10% losses, separately by market/size/goal; report every failure, not a forced pass',independence='Eight seeds; conditions/rounds/companies are repeated measures. Aggregate within seed before pooled intervals. All intervals exploratory; no universal theorem.',world_alignment='Independent OECD/BIS/Ostrom qualitative checks and Census QFR external margin anchors; no fitted industry or real agent data, no claim of empirical calibration.',new_paid_calls=0)

def config_for(name,n):
 raw=BASE.to_dict()
 if name=='recession':raw['market']['base_demand_orders']=7200
 elif name=='boom':raw['market']['base_demand_orders']=18000
 elif name=='tight_supply':
  for s in raw['supply_chain']['suppliers'].values():s['base_capacity_orders']=3500
 elif name=='cash_stress':raw['company_initial']['cash_balance_cents']=9000000;raw['supply_chain']['transaction_accounting']['initial_supplier_cash_cents']=3000000
 elif name in ('price_sensitive','quality_sensitive'):
  for segment in raw['consumer_choice']['segments'].values():
   c=segment['coefficients_ppm'];c['price']=c['price']*2 if name=='price_sensitive' else c['price']//2
   if name=='quality_sensitive':c['service']*=2
 elif name=='disruption':
  for e in raw['events']['definitions'].values():e['signal_generation_probability_ppm']=min(800000,e['signal_generation_probability_ppm']*4)
 elif name=='project_feasible':raw['strategic_market']['threshold_project']['required_total_contribution_cents']=n*300000
 elif name=='scaled':
  raw['market']['base_demand_orders']=3000*n
  for s in raw['supply_chain']['suppliers'].values():s['base_capacity_orders']=2750*n
  raw['supply_chain']['transaction_accounting']['initial_supplier_cash_cents']=5000000*n
  raw['autonomous_market']['government']['initial_cash_cents']=12500000*n
  raw['strategic_market']['threshold_project']['required_total_contribution_cents']=2000000*n
 elif name not in ('normal','asymmetric'):raise ValueError(name)
 return MarketConfig.from_mapping(raw)

def initial(name,n,seed,rounds):
 config=config_for(name,n);env=MarketEnv(config);s=env.reset(company_ids=[f'company_{chr(65+i)}' for i in range(n)],episode_id=f'evaluation-v18-{seed}',episode_seed=seed,max_rounds=rounds,cooperation_mode='combined_v1')
 if name=='asymmetric':
  s=replace(s,companies=tuple(replace(c,operations=replace(c.operations,base_capacity_orders=1000 if i%2==0 else 6000)) for i,c in enumerate(s.companies)))
  s=replace(s,state_hash=state_hash(s.to_dict()));env.load_state(s)
 return config,env,s

def population(config,s,policy,memory):
 choices={a:learning_choice(a,memory) if policy=='learners' else 'balanced' for a in actor_ids(s)}
 # Native government policy chooses from its own public-history memory unless learners explicitly controls it.
 extra={a:v for a,v in choices.items() if a not in s.company_ids and (a!='government' or policy=='learners')}
 aa={a:company_action(config,s,a,choices[a]) for a in s.company_ids}
 for i,cid in enumerate(s.company_ids):
  persona='cooperator' if policy=='cooperators' else 'free_rider' if policy=='free_riders' else ('free_rider' if i==0 else 'cooperator') if policy=='one_defector' else ('free_rider' if i==0 else 'retaliator') if policy=='reciprocity' else None
  if persona:aa[cid]=apply_cooperation_persona(config,s,cid,aa[cid],persona)
  if policy=='price_war':aa[cid]=recipe_action(config,s,cid,{'price':850000,'reserve':True})
 if policy in ('coordination','mutual_aid'):
  for i in range(0,len(s.company_ids)-1,2):
   pair=s.company_ids[i:i+2];target=min(18000,max(7500,round(mean(s.company(c).commercial.price_cents for c in pair)*1.1)))
   for j,cid in enumerate(pair):
    if cid not in s.strategic_market.active_company_ids:continue
    request=aa[cid].to_dict()
    if policy=='coordination':request.update(price_cents=target,price_coordination_partner_company_id=pair[1-j],price_coordination_target_cents=target)
    else:request.update(mutual_aid_partner_company_id=pair[1-j],mutual_aid_capacity_offer_orders=1000 if j==1 else 0,mutual_aid_capacity_request_orders=1000 if j==0 else 0)
    aa[cid]=resolve_action_request(config,s,cid,request,source='evaluation-v18',action_id=f'eval:{s.round}:{cid}').action
 return aa,extra,choices

def step(env,s,aa,extra):return env.step(f'{s.episode_id}:{s.round}:{s.state_version}',aa,actor_choices=extra).state_after

def save_compressed(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
 with gzip.open(temporary,'wt',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,separators=(',',':'))
 temporary.replace(path)

def read_compressed(path):
 with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)

def episode(case):
 name,n,seed,rounds,policy=case;key=f'{name}-{n}-{seed}-{rounds}-{policy}';path=OUT/'episodes'/(key+'.json.gz')
 if path.exists():return read_compressed(path)['summary']
 config,env,s=initial(name,n,seed,rounds);start=s.to_dict();memory={};history=[];traces=[];revenues=contributions=aid_orders=fines=coordination=checks=0;advice_records=[];decisions=0
 while not s.terminal:
  resident='one_defector' if policy in ('focal_v17','all_v17','closed_rule') else policy
  aa,extra,choices=population(config,s,resident,memory)
  if policy in ('focal_v17','all_v17'):
   for cid in s.company_ids if policy=='all_v17' else s.company_ids[:1]:
    if cid not in s.strategic_market.active_company_ids:continue
    req=RobustRequest(company_id=cid,goal='profit',history=history,horizon=1,scenarios=2,max_candidates=8,finalists=1,step_budget=max(500,len(actor_ids(s))*50),diagnostics=False,backtest=False,seed=seed)
    r=advise(config,s,req);aa[cid]=CompanyAction.from_dict(r['action']);assert r['source_state_hash']==s.state_hash;decisions+=1
    advice_records.append(dict(round=s.round,company=cid,result=r))
  after=step(env,s,aa,extra);checks+=audit_settlement(s.to_dict(),after.to_dict(),config.to_dict());d=after.to_dict();st=d['strategic_market'];revenue=sum(c['financial']['round_revenue_cents'] for c in d['companies'].values());revenues+=revenue
  contrib={a:(v.shared_resilience_contribution_cents or 0)+(v.threshold_project_contribution_cents or 0) for a,v in aa.items()};contributions+=sum(contrib.values());aid_orders+=sum(t['fulfilled_orders'] for t in st['mutual_aid']['last_transfers']);fines+=after.government.round_fines_cents;coordination+=len(st['price_coordination']['last_outcomes'])
  traces.append(dict(round=s.round,actions={a:v.to_dict() for a,v in aa.items()},extra=extra,state_hash=after.state_hash,prices={c.company_id:c.commercial.price_cents for c in after.companies},contributions=contrib,sales=sum(c.commercial.sales_orders for c in after.companies),stockout=after.market.lost_after_stockout_orders,hhi=st['hhi_ppm'],active=st['active_company_count'],project=st['threshold_project']['status'],industry_resilience=d['shared_resilience']['industry_resilience_ppm'],supplier_prices={v.supplier_id:v.unit_price_cents for v in after.supply_chain.suppliers},profits={c.company_id:c.financial.round_profit_cents for c in after.companies},revenue=revenue,welfare=after.welfare_accounting.round_total_economic_welfare_cents))
  history.append(public_frame(s));s=after;update_memory(memory,s,choices)
 replay=MarketEnv(config);rs=MarketState.from_dict(start);replay.load_state(rs)
 for t in traces:
  rs=step(replay,rs,{a:CompanyAction.from_dict(v) for a,v in t['actions'].items()},t['extra']);assert rs.state_hash==t['state_hash']
 profit=sum(c.financial.cumulative_profit_cents for c in s.companies);shares=[max(0,c.financial.cumulative_profit_cents) for c in s.companies];total=sum(shares)
 volatility=mean(abs(b['prices'][cid]/a['prices'][cid]-1) for a,b in zip(traces,traces[1:]) for cid in s.company_ids)
 summary=dict(market=name,companies=n,seed=seed,rounds=rounds,policy=policy,profit=profit,focal_profit=s.company('company_A').financial.cumulative_profit_cents,welfare=s.welfare_accounting.cumulative_total_economic_welfare_cents,consumer_surplus=s.welfare_accounting.cumulative_consumer_surplus_cents,revenue=revenues,margin=profit/revenues if revenues else None,contributions=contributions,aid_orders=aid_orders,fines=fines,coordination=coordination,project_success=traces[-1]['project']=='succeeded',project_status=traces[-1]['project'],stockout=sum(t['stockout'] for t in traces),sales=sum(t['sales'] for t in traces),exits=n-s.strategic_market.active_company_count,hhi=mean(t['hhi'] for t in traces),price_volatility=volatility,mean_price=mean(p for t in traces for p in t['prices'].values()),arithmetic_checks=checks,advisor_decisions=decisions,response_complete=sum(r['result']['v17']['response_complete'] for r in advice_records),replay=True)
 save_compressed(path,dict(config=config.to_dict(),initial_state=start,transitions=traces,advice=advice_records,summary=summary));return summary

def evaluate_action(config,s,aa,extra):
 e=MarketEnv(config);e.load_state(s);a=step(e,s,aa,extra);audit_settlement(s.to_dict(),a.to_dict(),config.to_dict());return a

def advisor_case(case):
 name,n,seed,goal=case;key=f'{name}-{n}-{seed}-{goal}';path=OUT/'advisors'/(key+'.json.gz')
 if path.exists():return read_compressed(path)['summary']
 config,env,s=initial(name,n,seed,20);memory={};history=[];resident=('cooperators','price_war','learners')[seed%3]
 for _ in range(4):
  aa,extra,choices=population(config,s,resident,memory);history.append(public_frame(s));s=step(env,s,aa,extra);update_memory(memory,s,choices)
 if 'company_A' not in s.strategic_market.active_company_ids:
  row=dict(market=name,companies=n,seed=seed,goal=goal,available=False,reason='focal company exited before advice');save_compressed(path,dict(summary=row));return row
 actual,extra,_=population(config,s,resident,memory);baseline=dict(actual);baseline['company_A']=recipe_action(config,s,'company_A',{})
 def value(a):return a.company('company_A').financial.round_profit_cents if goal=='profit' else a.welfare_accounting.round_total_economic_welfare_cents
 base=evaluate_action(config,s,baseline,extra);incumbent=evaluate_action(config,s,actual,extra);oracle=[]
 for candidate in expanded(config,s,'company_A',48,seed):
  a=evaluate_action(config,s,{**actual,'company_A':CompanyAction.from_dict(candidate['action'])},extra);oracle.append(dict(id=candidate['id'],recipe=candidate['recipe'],value=value(a)))
 results=[];details={}
 for variant in ('v16','v17','fast'):
  common=dict(company_id='company_A',goal=goal,horizon=1,scenarios=2,max_candidates=12,step_budget=1200,diagnostics=False,backtest=False,seed=seed)
  if variant=='fast':common.update(max_candidates=8,finalists=1,step_budget=max(500,len(actor_ids(s))*50))
  req=AdviceRequest(**common) if variant=='v16' else RobustRequest(**common,history=history)
  r=(advise16 if variant=='v16' else advise)(config,s,req);a=evaluate_action(config,s,{**actual,'company_A':CompanyAction.from_dict(r['action'])},extra);real=value(a);best=max(x['value'] for x in oracle)
  best=max(best,real) # Evaluator reference includes every tested recommended action as well.
  results.append(dict(variant=variant,gain=real-value(base),incumbent_gain=real-value(incumbent),realized=real,profit_gain=a.company('company_A').financial.round_profit_cents-base.company('company_A').financial.round_profit_cents,welfare_gain=a.welfare_accounting.round_total_economic_welfare_cents-base.welfare_accounting.round_total_economic_welfare_cents,oracle_regret=best-real,latency=r['search']['elapsed_seconds'],used_steps=r['search']['used_steps'],complete=r.get('v17',{}).get('response_complete',True),baseline=r['recommended_id']=='baseline',profit_error=a.company('company_A').financial.round_profit_cents-next(x for x in r['rankings'] if x['candidate']['id']==r['recommended_id'])['metrics']['profit']))
  details[variant]=r
 best=max([value(base),value(incumbent)]+[x['value'] for x in oracle]+[x['realized'] for x in results])
 for r in results:r['oracle_regret']=best-r['realized']
 row=dict(market=name,companies=n,seed=seed,goal=goal,resident=resident,available=True,oracle_best=best,results=results)
 save_compressed(path,dict(summary=row,source_state=s.to_dict(),config=config.to_dict(),actual_other_actions={a:v.to_dict() for a,v in actual.items() if a!='company_A'},extra=extra,oracle=oracle,details=details));return row

def latency():
 rows=[]
 for n in SPEC['companies']:
  config,env,s=initial('normal',n,484000,20)
  # Warm import and one recommendation per size; not included in steady-state timings.
  advise(config,s,RobustRequest(horizon=1,scenarios=2,max_candidates=8,finalists=1,step_budget=1200,diagnostics=False,backtest=False))
  for seed in SPEC['latency_seeds']:
   for variant in SPEC['latency_profiles']:
    req=dict(horizon=1,scenarios=2,max_candidates=12,step_budget=1200,diagnostics=False,backtest=False,seed=seed)
    if variant=='fast':req.update(max_candidates=8,finalists=1,step_budget=max(500,len(actor_ids(s))*50))
    typed=AdviceRequest(**req) if variant=='v16' else RobustRequest(**req);start=time.perf_counter();r=(advise16 if variant=='v16' else advise)(config,s,typed);elapsed=time.perf_counter()-start
    rows.append(dict(companies=n,variant=variant,seed=seed,seconds=elapsed,complete=r.get('v17',{}).get('response_complete',True),steps=r['search']['used_steps']))
 return dict(passed=True,results=rows,scope='Serial isolated-process measurements after warmup, three repeats per group; sample p95 is essentially the maximum, not a production SLA. Includes full advise call, excludes HTTP/UI.')

def main(stage,smoke=False):
 OUT.mkdir(exist_ok=True);before=status()
 prereg=OUT/'preregistration.json'
 if prereg.exists():assert read_json(prereg)==SPEC
 else:write_json(prereg,SPEC)
 if stage=='smoke':
  for name in MARKETS:
   r=episode((name,2,480001,5,'mutual_aid'));print(name,'smoke passed; project:',r['project_status'],flush=True)
  print(advisor_case(('normal',2,480002,'profit')));return
 path=OUT/(stage+'.json')
 if path.exists():print(stage,'already complete');return
 if stage=='tournament':cases=[(m,n,s,20,p) for m in MARKETS for n in SPEC['companies'] for s in SPEC['tournament_seeds'] for p in POLICIES];fn=episode
 elif stage=='advisors':cases=[(m,n,s,g) for m in ADVISOR_MARKETS for n in SPEC['companies'] for s in SPEC['advisor_seeds'] for g in SPEC['goals']];fn=advisor_case
 elif stage=='closed_loop':cases=[(m,n,s,10,p) for m in SPEC['closed_loop_markets'] for n in SPEC['companies'] for s in SPEC['closed_loop_seeds'] for p in ('closed_rule','focal_v17','all_v17')];fn=episode
 elif stage=='latency':write_json(path,latency());assert status()==before;print('latency complete');return
 else:raise ValueError(stage)
 rows=[];start=time.perf_counter()
 with ProcessPoolExecutor(max_workers=4) as pool:
  for r in pool.map(fn,cases):
   rows.append(r)
   if len(rows)%12==0:print(json.dumps(dict(stage=stage,completed=len(rows),total=len(cases))),flush=True)
 assert status()==before
 write_json(path,dict(passed=True,cases=len(rows),seconds=time.perf_counter()-start,results=rows,new_paid_calls=0));print(stage,'complete',len(rows))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True);args=parser.parse_args()
 try:main(args.stage)
 except Exception as exc:
  write_json(OUT/f'failure-{time.time_ns()}.json',dict(stage=args.stage,type=type(exc).__name__,error=str(exc)));raise
