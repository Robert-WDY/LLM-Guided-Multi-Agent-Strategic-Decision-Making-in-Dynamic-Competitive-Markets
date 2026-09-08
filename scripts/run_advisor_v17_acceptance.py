"""Preregistered incremental v17 experiments; retain raw cases and failures."""
import argparse,asyncio,json,random,time,hashlib
from pathlib import Path
from statistics import mean
from copy import deepcopy
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from game_theory_agent.market import MarketEnv,MarketState,load_market_config
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.market.actor_policies import actions_for
from game_theory_agent.market.actor_experiments import run_episode,write_json,read_json
from game_theory_agent.game_theory.advisor_beliefs import RobustRequest,fit_prices,price_audit,public_frame
from game_theory_agent.game_theory.advisor_robust import advise,select_validated,LearnedRollouts
from game_theory_agent.game_theory.advisor_search import advise as old_advise
from game_theory_agent.game_theory.advisor_market import Rollouts,forecast
from game_theory_agent.game_theory.advisor_candidates import expanded
from game_theory_agent.game_theory.objectives import AdviceRequest,objective
from game_theory_agent.game_theory.advisor_audit import audit_settlement,paired_summary
from game_theory_agent.research_workbench import configured
from game_theory_agent.local_budget import status
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'runs/advisor-v17';CONFIG=load_market_config(ROOT/'configs/market_v14_local.yaml')
SPEC=dict(development_seeds=list(range(371001,371009)),holdout_seeds=list(range(379001,379033)),goal_assignment='even seed profit; odd seed welfare; 16 independent seeds per goal',pressures=['normal','supplier','government'],company_count=5,horizon=1,scenarios=2,steps=1200,max_candidates=12,multiplayer_seeds=[372101,372102],multiplayer_counts=[2,5,10],strong_subset=list(range(379001,379009)),strong_steps=3600,strong_candidates=36,strong_response_passes=2,acceptance='Arithmetic, executable actions, privacy, budget, full responses and truthful intervals must pass. Measure economic gains without deleting adverse outcomes; no universal or real-world claims.',new_real_calls=0)
def initial(count=5,seed=371001,config=CONFIG,rounds=10):return MarketEnv(config).reset(company_ids=[f'company_{chr(65+i)}' for i in range(count)],episode_id=f'v17-{seed}',episode_seed=seed,max_rounds=rounds,cooperation_mode='combined_v1')
def saved(name,compute):
 p=DEST/(name+'.json')
 if p.exists():return read_json(p)
 result=compute();write_json(p,result);return result
def replay(source,req,scenario,outcome,config):
 seed=int(hashlib.sha256(f'{source.episode_seed}:{req.seed}:{scenario}'.encode()).hexdigest()[:15],16);s=replace(source,episode_seed=seed);s=replace(s,state_hash=state_hash(s.to_dict()));e=MarketEnv(config);e.load_state(s);checks=0
 for row in outcome['trace']:
  a=e.step(f'{s.episode_id}:{s.round}:{s.state_version}',row['joint_actions'],actor_choices=row['other_actor_choices']).state_after
  assert a.state_hash==row['state_hash'];checks+=audit_settlement(s.to_dict(),a.to_dict());s=a
 return checks
def multiplayer_case(case):
 count,seed,rounds,mode=case
 folder=DEST/'multiplayer'/f'{count}-{seed}-{rounds}-{mode}';modes={f'company_{chr(65+i)}':mode for i in range(count)};res=asyncio.run(run_episode(CONFIG,directory=folder,seed=seed,rounds=rounds,company_count=count,modes=modes));cp=read_json(folder/'checkpoint.json');checks=sum(audit_settlement(t['state_before'],t['state_after']) for t in cp['transitions']);minimum=min(c['financial']['cash_balance_cents'] for t in cp['transitions'] for c in t['state_after']['companies'].values());prices=[t['state_after']['companies']['company_A']['commercial']['price_cents'] for t in cp['transitions']];volatility=mean(abs(b/a-1) for a,b in zip(prices,prices[1:]));decisions=[]
 if mode=='robust_profit':
  for i,t in enumerate(cp['transitions'],1):
   ds=read_json(folder/f'round-{i:03d}.json')['decisions']
   for cid in modes:
    d=ds[cid]
    if 'advisor_decision' in d:
     a=d['advisor_decision'];assert a['source_state_hash']==t['state_before_hash'] and a['action']==t['final_actions'][cid];assert len(a['request']['history'])==i-1;decisions.append(a)
 row=dict(companies=count,seed=seed,rounds=rounds,mode=mode,welfare=res['welfare_cents'],profit=res['company_profit_cents'],minimum_cash=minimum,price_volatility=volatility,closed_companies=count-len(MarketState.from_dict(cp['state']).strategic_market.active_company_ids),arithmetic_checks=checks,advisor_decisions=len(decisions),complete_responses=sum(d['v17']['response_complete'] for d in decisions),replay=res['replay_passed']);return row

def holdout_seed(seed):
 rows=[]
 goal='profit' if seed%2==0 else 'welfare'
 for pressure in SPEC['pressures']:
  config=CONFIG if pressure=='normal' else configured(CONFIG,dict(overrides={'supplier_cash' if pressure=='supplier' else 'government_cash':2000000}));s=initial(seed=seed,config=config);e=MarketEnv(config);e.load_state(s);history=[]
  for _ in range(3):history.append(public_frame(s));s=e.step(f'{s.episode_id}:{s.round}:{s.state_version}',actions_for(config,s,{})).state_after
  variants=['v16','v17']+(['strong'] if seed in SPEC['strong_subset'] else [])+(['no_response','no_learning'] if pressure=='normal' and seed in SPEC['strong_subset'] else [])
  for variant in variants:
   args=dict(goal=goal,horizon=1,scenarios=2,max_candidates=12,step_budget=1200,diagnostics=False,seed=seed)
   if variant=='strong':args.update(max_candidates=36,step_budget=3600,finalists=3,response_passes=2)
   if variant!='v16':args.update(history=history,response_aware=variant!='no_response',learn_opponents=variant!='no_learning')
   req=AdviceRequest(**args) if variant=='v16' else RobustRequest(**args);r=saved(f'holdout/{seed}-{pressure}-{variant}',lambda:(old_advise if variant=='v16' else advise)(config,s,req));b=r['backtest'];checks=0
   for i,sc in enumerate((9000,9001)):
    checks+=replay(s,req,sc,b['outcomes'][i],config)+replay(s,req,sc,b['baseline'][i],config)
   assert r['search']['used_steps']<=req.step_budget
   row=dict(seed=seed,pressure=pressure,goal=goal,variant=variant,selected=r['recommended_id'],gain=mean(b['paired_utility_gains']),profit_gain=b['profit_gain_cents'],welfare_gain=b['welfare_gain_cents'],profit_error=b['prediction_profit_error_cents'],relative_error=abs(b['prediction_profit_error_cents'])/max(10000,abs(mean(o['profit'] for o in b['outcomes']))),minimum_cash=min(o['minimum_cash'] for o in b['outcomes']),elapsed=r['search']['elapsed_seconds'],response_complete=r.get('v17',{}).get('response_complete'),sensitivity=r.get('v17',{}).get('sensitivity'),arithmetic_checks=checks);rows.append(row)
  print(json.dumps(dict(seed=seed,pressure=pressure,completed=len(rows))),flush=True)
 return rows

async def main(stage):
 DEST.mkdir(exist_ok=True);p=DEST/f'stage{stage}.json'
 if p.exists():print(json.dumps({k:v for k,v in read_json(p).items() if k!='results'}));return
 if stage>1:assert read_json(DEST/f'stage{stage-1}.json')['passed']
 prereg=DEST/'preregistration.json'
 if prereg.exists():assert read_json(prereg)==SPEC
 else:write_json(prereg,SPEC)
 before=status();start=time.perf_counter();rows=[];extra={}
 if stage==1:
  for seed in range(371001,371033):
   rng=random.Random(seed)
   for kind in ('flat','trend','reversal'):
    prices=[10000]
    for i in range(1,41):prices.append(max(1,round(prices[-1]*(1+(0 if kind=='flat' else .03 if kind=='trend' or i<21 else -.04)+rng.uniform(-.002,.002)))))
    audit=price_audit(prices);fit=fit_prices(prices)
    if kind=='trend':assert audit['mean_relative_error']<audit['flat_relative_error'] and fit['probabilities'][2]>.7
    if kind=='reversal':assert fit['probabilities'][0]>.7
    rows.append(dict(seed=seed,kind=kind,**audit,final_belief=fit))
  extra['calibration_by_kind']={k:dict(relative_mae=mean(r['mean_relative_error'] for r in rows if r['kind']==k),flat_mae=mean(r['flat_relative_error'] for r in rows if r['kind']==k),coverage=mean(r['empirical_coverage'] for r in rows if r['kind']==k)) for k in ('flat','trend','reversal')}
 elif stage==2:
  # Completely enumerated, independent 3x3 leader/response problems.
  rng=random.Random(371002)
  for case in range(64):
   matrix=[[(rng.randint(-10,10),rng.randint(-10,10)) for _ in range(3)] for _ in range(3)];responses=[max(range(3),key=lambda j:matrix[i][j][1]) for i in range(3)];values=[matrix[i][responses[i]][0] for i in range(3)];items=[dict(id='baseline' if i==0 else str(i),safe=True,score=v,paired_gains=[v-values[0]]) for i,v in enumerate(values)];selected=select_validated(items);index=0 if selected=='baseline' else int(selected);assert values[index]==max(values);rows.append(dict(case=case,payoffs=matrix,selected=selected,exhaustive_value=max(values)))
  for seed in SPEC['development_seeds'][:4]:
   s=initial(2,seed);req=RobustRequest(horizon=1,scenarios=2,max_candidates=48,step_budget=3000,search_method='grid',diagnostics=False,backtest=False,seed=seed);r=saved(f'grid-{seed}',lambda:advise(CONFIG,s,req));f,_=forecast(CONFIG,s,'company_A');oracle=LearnedRollouts(CONFIG,f,req,r['v17']['beliefs']);spec=r['objective'];values={}
   assert len(r['rankings'])==len(expanded(CONFIG,f,'company_A',48,seed,'grid'))
   for row in r['rankings']:
    out=[oracle.evaluate({'company_A':row['candidate']['recipe']},s) for s in (0,1)];scores=[sum(v.get(k,0)/spec['scales'][k]*w for k,w in spec['weights'].items()) for v in out];score=mean(scores)-req.risk_aversion*(mean(scores)-min(scores));assert abs(score-row['score'])<1e-8;values[row['candidate']['id']]=score
   assert values[r['v17']['nominal_choice']]==max(values.values());rows.append(dict(seed=seed,grid_candidates=len(values),all_scores_independent=True,response_complete=r['v17']['response_complete']))
 elif stage==3:
  cases=[(n,seed,10) for n in SPEC['multiplayer_counts'] for seed in SPEC['multiplayer_seeds']]+[(5,seed,20) for seed in SPEC['multiplayer_seeds']]
  with ProcessPoolExecutor(max_workers=4) as pool:
   for row in pool.map(multiplayer_case,[(n,seed,rounds,mode) for n,seed,rounds in cases for mode in ('rule','advisor_profit','robust_profit')]):rows.append(row);print(json.dumps(row),flush=True)
  extra.update(episodes=len(rows),rounds=sum(r['rounds'] for r in rows),advisor_decisions=sum(r['advisor_decisions'] for r in rows),interpretation='All companies decide against identical pre-round state. Preset comparison: v16 uses horizon2, v17 horizon1; this stage validates interaction, not isolated algorithm superiority.')
 elif stage==4:
  for phase in ('survival','expansion','mature','catch_up','closing'):
   for horizon in (1,3):
    req=RobustRequest(phase=phase,horizon=horizon,scenarios=2,max_candidates=12,step_budget=3000,diagnostics=False,seed=374001);r=saved(f'phase-{phase}-{horizon}',lambda:advise(CONFIG,initial(seed=374001),req));assert r['v17']['response_complete'];b=r['backtest'];rows.append(dict(phase=phase,horizon=horizon,selected=r['recommended_id'],recipe=r['recipe'],sensitivity=r['v17']['sensitivity'],profit_gain=b['profit_gain_cents'],welfare_gain=b['welfare_gain_cents'],utility_gain=mean(b['paired_utility_gains'])))
 elif stage==5:
  with ProcessPoolExecutor(max_workers=4) as pool:
   for seed_rows in pool.map(holdout_seed,SPEC['holdout_seeds']):rows.extend(seed_rows)
  groups={}
  for goal in ('profit','welfare'):
   for pressure in SPEC['pressures']:
    ref={r['seed']:r for r in rows if r['goal']==goal and r['pressure']==pressure and r['variant']=='v16'};new=[r for r in rows if r['goal']==goal and r['pressure']==pressure and r['variant']=='v17'];groups[goal+'-'+pressure]=dict(v17_vs_rule=paired_summary([r['gain'] for r in new]),v17_minus_v16=paired_summary([r['gain']-ref[r['seed']]['gain'] for r in new]),profit_mae=mean(abs(r['profit_error']) for r in new),mean_relative_error=mean(r['relative_error'] for r in new),response_completion=sum(r['response_complete'] for r in new))
  extra.update(groups=groups,independent_seeds=32,seeds_per_goal=16,experimental_only=True,real_economy_validated=False)
 else:raise ValueError('stages1..5 are experiments; stage6 is final integration acceptance')
 assert status()==before
 result=dict(stage=stage,passed=True,cases=len(rows),elapsed_seconds=time.perf_counter()-start,new_real_calls=0,results=rows,**extra);write_json(p,result);print(json.dumps({k:v for k,v in result.items() if k!='results'}))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',type=int,required=True);args=parser.parse_args()
 try:asyncio.run(main(args.stage))
 except Exception as e:
  DEST.mkdir(exist_ok=True);write_json(DEST/f'failure-{args.stage}-{time.time_ns()}.json',dict(error=type(e).__name__,message=str(e)));raise
