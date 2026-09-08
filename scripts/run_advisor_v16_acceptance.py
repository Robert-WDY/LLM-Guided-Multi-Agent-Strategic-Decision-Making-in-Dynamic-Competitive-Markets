"""Run one independent stage at a time; preserve preregistration and evidence."""
import argparse,hashlib,json,time
from pathlib import Path
from dataclasses import replace
from statistics import mean
from fastapi import FastAPI
from fastapi.testclient import TestClient
from game_theory_agent.market import MarketEnv,load_market_config
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.game_theory.objectives import AdviceRequest,objective,aggregate
from game_theory_agent.game_theory.advisor_market import forecast,Rollouts,candidates
from game_theory_agent.game_theory.advisor_search import advise
from game_theory_agent.game_theory.advisor_diagnostics import response_search
from game_theory_agent.game_theory.service import router
from game_theory_agent.market.actor_experiments import write_json,read_json
from game_theory_agent.research_workbench import configured
from game_theory_agent.local_budget import status
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'runs/advisor-v16';CONFIG=load_market_config(ROOT/'configs/market_v14_local.yaml')
def initial(count=4,seed=360201,config=CONFIG):return MarketEnv(config).reset(company_ids=[f'company_{chr(65+i)}' for i in range(count)],episode_id=f'v16-{seed}',episode_seed=seed,max_rounds=10,cooperation_mode='combined_v1')
def verify_trace(state,request,scenario,outcome,config=CONFIG):
 seed=int(hashlib.sha256(f'{state.episode_seed}:{request.seed}:{scenario}'.encode()).hexdigest()[:15],16);s=replace(state,episode_seed=seed);s=replace(s,state_hash=state_hash(s.to_dict()));env=MarketEnv(config);env.load_state(s)
 total=0
 for t,row in enumerate(outcome['trace']):
  result=env.step(f'{s.episode_id}:{s.round}:{s.state_version}',row['joint_actions'],actor_choices=row['other_actor_choices']);s=result.state_after
  assert s.state_hash==row['state_hash'];total+=request.discount**t*s.company(request.company_id).financial.round_profit_cents
 assert abs(total-outcome['profit'])<1e-7

def main(stage):
 DEST.mkdir(exist_ok=True);path=DEST/f'stage{stage}.json'
 if path.exists():print(json.dumps(read_json(path)));return
 assert stage==1 or read_json(DEST/f'stage{stage-1}.json')['passed']
 budget_before=status();start=time.perf_counter();rows=[]
 spec=dict(stage=stage,development_seeds=[360201,360202],holdout_seeds=[369101,369102],company_counts=[2,5,10],new_real_calls=0,acceptance='Independent arithmetic/engine replay, legal information, budget and truthful reporting must pass. Empirical profit improvement is measured, never assumed.')
 write_json(DEST/f'stage{stage}-preregistration.json',spec)
 if stage==1:
  source=initial();samples=[dict(profit=3000000,welfare=-1000000,cash=3000000,minimum_cash=100,exited=False),dict(profit=1000000,welfare=5000000,cash=1000000,minimum_cash=100,exited=False)]
  for phase in ('survival','expansion','mature','catch_up','closing'):
   for goal in ('company','profit','welfare','balanced'):
    spec_obj=objective(AdviceRequest(goal=goal,phase=phase),source)
    scores=[sum(v.get(k,0)/spec_obj['scales'][k]*w for k,w in spec_obj['weights'].items()) for v in samples]
    assert all(abs(aggregate([v,v],spec_obj)['score']-expected)<1e-9 for v,expected in zip(samples,scores))
    if goal=='profit':assert scores[0]>scores[1]
    if goal=='welfare':assert scores[0]<scores[1]
    assert not aggregate([dict(samples[0],minimum_cash=-1)],spec_obj)['safe']
    rows.append(dict(goal=goal,phase=phase,scores=scores,weights=spec_obj['weights']))
 elif stage==2:
  for count in spec['company_counts']:
   for seed in spec['development_seeds']:
    source=initial(count,seed);f,_=forecast(CONFIG,source,'company_A');req=AdviceRequest(horizon=3,step_budget=300,backtest=False);engine=Rollouts(CONFIG,f,req)
    for scenario in (0,1):
     outcome=engine.evaluate({'company_A':{'price':950000,'supplier_share':500000,'duration':3}},scenario);verify_trace(f,req,scenario,outcome)
     rows.append(dict(companies=count,seed=seed,scenario=scenario,rounds=len(outcome['trace']),profit=outcome['profit']))
 elif stage==3:
  for goal in ('company','profit','welfare'):
   for seed in spec['development_seeds']:
    source=initial(seed=seed);req=AdviceRequest(goal=goal,horizon=2,scenarios=2,max_candidates=12,step_budget=250,diagnostics=False)
    result=advise(CONFIG,source,req);write_json(DEST/f'search-{goal}-{seed}.json',result)
    assert result['search']['used_steps']<=req.step_budget
    f,_=forecast(CONFIG,source,'company_A');check=Rollouts(CONFIG,f,req.model_copy(update={'step_budget':2000}));weights=result['objective']['weights'];scales=result['objective']['scales']
    for row in result['rankings']:
     scores=[]
     for sc in (0,1):
      v=check.evaluate({'company_A':row['candidate']['recipe']},sc);scores.append(sum(v[k]/scales[k]*weights[k] for k in weights))
     expected=mean(scores)-req.risk_aversion*(mean(scores)-min(scores));assert abs(expected-row['score'])<1e-9
    if result['recommended_id']!='baseline':assert min(next(v for v in result['validation'] if v['id']==result['recommended_id'])['paired_gains'])>0
    rows.append(dict(goal=goal,seed=seed,selected=result['recommended_id'],used=result['search']['used_steps'],all_candidate_scores_independently_recomputed=True))
 elif stage==4:
  for count in spec['company_counts']:
   req=AdviceRequest(horizon=1,scenarios=2,max_candidates=8,step_budget=700,backtest=False)
   result=advise(CONFIG,initial(count),req);write_json(DEST/f'multiplayer-{count}.json',result)
   assert result['diagnostics']['player_count']==count+4 and result['search']['used_steps']<=700
   for key in ('recommendation_check','equilibrium'):
    d=result['diagnostics'][key]
    if not d['complete']:assert d['restricted_pure_nash'] is None
    elif d['restricted_pure_nash']:assert max(d['deviation_gains'].values())<1e-8
   rows.append(dict(companies=count,actors=count+4,steps=result['search']['used_steps'],recommendation_check_complete=result['diagnostics']['recommendation_check']['complete'],equilibrium_check_complete=result['diagnostics']['equilibrium']['complete']))
  # Independent analytic best response: contributing costs 2, personal benefit .3.
  for count in (2,5,10):
   choices={str(i):[0,1] for i in range(count)}
   result=response_search(lambda p:{a:10-2*v+.3*sum(p.values()) for a,v in p.items()},choices,{a:1 for a in choices})
   assert result['restricted_pure_nash'] and all(v==0 for v in result['profile'].values())
 elif stage==5:
  app=FastAPI();app.include_router(router(DEST/'api-records',CONFIG,lambda token:None));client=TestClient(app)
  payload=dict(request_id='v16-stage5-advice',kind='advisor',parameters=dict(goal='welfare',phase='expansion',company_count=5,horizon=1,scenarios=2,max_candidates=8,step_budget=600,seed=360501))
  response=client.post('/api/v1/controller/theory-lab/experiments',json=payload);assert response.status_code==200,response.text;record=response.json()
  assert client.post('/api/v1/controller/theory-lab/experiments',json=payload).json()==record
  assert client.get('/api/v1/controller/theory-lab/experiments/'+record['id']).json()==record
  invalid={**payload,'request_id':'invalid','parameters':{'horizon':True}}
  assert client.post('/api/v1/controller/theory-lab/experiments',json=invalid).status_code==422
  fixture=ROOT/'frontend/tests/fixtures/advisor-v16.json';write_json(fixture,record['result'])
  rows=[dict(id=record['id'],idempotent=True,saved_restore=True,invalid_rejected=True,full_evidence=True)]
 elif stage==6:
  for count in spec['company_counts']:
   for seed in spec['holdout_seeds']:
    for goal in ('company','profit','welfare'):
     case=f'holdout-{count}-{seed}-{goal}';config=CONFIG if seed==369101 else configured(CONFIG,{'overrides':{'supplier_cash':2000000,'government_cash':5000000}})
     source=initial(count,seed,config);req=AdviceRequest(goal=goal,horizon=2,scenarios=2,max_candidates=12,step_budget=300,diagnostics=False,seed=seed)
     result=advise(config,source,req);write_json(DEST/f'{case}.json',result);b=result['backtest']
     for index,scenario in enumerate((9000,9001)):
      verify_trace(source,req,scenario,b['outcomes'][index],config);verify_trace(source,req,scenario,b['baseline'][index],config)
     assert result['search']['used_steps']<=300
     rows.append(dict(case=case,companies=count,goal=goal,stress=seed==369102,recommended=result['recommended_id'],utility_gain=mean(b['paired_utility_gains']),profit_gain_cents=b['profit_gain_cents'],welfare_gain_cents=b['welfare_gain_cents'],profit_prediction_error_cents=b['prediction_profit_error_cents'],elapsed_seconds=result['search']['elapsed_seconds'],steps=result['search']['used_steps'],replay=True))
     print(json.dumps(rows[-1]),flush=True)
 else:raise ValueError('stage must be 1..6')
 assert status()==budget_before
 summary=dict(stage=stage,passed=True,cases=len(rows),new_real_calls=0,elapsed_seconds=time.perf_counter()-start,results=rows)
 if stage==6:
  summary.update(nonbaseline=sum(r['recommended']!='baseline' for r in rows),positive=sum(r['utility_gain']>0 for r in rows),negative=sum(r['utility_gain']<0 for r in rows),mean_utility_gain=mean(r['utility_gain'] for r in rows),worst_utility_gain=min(r['utility_gain'] for r in rows),mean_absolute_profit_error_cents=mean(abs(r['profit_prediction_error_cents']) for r in rows),maximum_seconds=max(r['elapsed_seconds'] for r in rows),conclusion='Engineering and truthful evidence accepted; holdout gains/errors are directional synthetic evidence, not universal optimality or real-economy calibration.')
 write_json(path,summary);print(json.dumps({k:v for k,v in summary.items() if k!='results'}))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',type=int,required=True);main(parser.parse_args().stage)
