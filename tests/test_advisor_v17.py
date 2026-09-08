from pathlib import Path
from dataclasses import replace
import pytest
from game_theory_agent.market import MarketEnv,load_market_config
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.game_theory.advisor_beliefs import fit_prices,price_audit,RobustRequest,learn
from game_theory_agent.game_theory.advisor_robust import advise,select_validated
from game_theory_agent.game_theory.advisor_candidates import expanded
CONFIG=load_market_config(Path(__file__).parents[1]/'configs/market_v14_local.yaml')

def initial():return MarketEnv(CONFIG).reset(episode_seed=370001,max_rounds=10,cooperation_mode='combined_v1')

def test_public_learning_prequential_shift_and_calibration():
 prices=[round(10000*1.04**i) for i in range(20)]
 fit=fit_prices(prices);assert fit['probabilities'][2]>.7
 audit=price_audit(prices);assert audit['mean_relative_error']<audit['flat_relative_error'];assert audit['interval_samples']==11
 shifted=prices+[round(prices[-1]*.95**i) for i in range(1,16)];assert fit_prices(shifted)['probabilities'][0]>.7
 with pytest.raises(ValueError):learn(initial(),'company_A',[dict(round=1,prices={'company_A':100})])

def test_counter_response_changes_analytic_recommendation():
 # Row0 appears optimal against column0 (10>6), but the column player chooses
 # column1 after row0 (8>0). Row1 then guarantees 6 instead of -4.
 pay=[[(10,0),(-4,8)],[(6,2),(5,1)]]
 reacted=[max(range(2),key=lambda j:pay[i][j][1]) for i in range(2)]
 assert reacted==[1,0]
 baseline=pay[0][reacted[0]][0]
 rows=[dict(id='baseline',safe=True,score=baseline,paired_gains=[0]),dict(id='safe',safe=True,score=pay[1][reacted[1]][0],paired_gains=[pay[1][reacted[1]][0]-baseline])]
 assert select_validated(rows)=='safe'
 rows[1]['safe']=False;assert select_validated(rows)=='baseline'

def test_grid_generation_exhausts_declared_small_space_legally():
 s=initial();rows=expanded(CONFIG,s,'company_A',48,1,'grid');assert len(rows)>=8
 env=MarketEnv(CONFIG);env.load_state(s)
 for r in rows:assert env.validate_action(r['action'],'company_A').valid
 assert rows==expanded(CONFIG,s,'company_A',48,999,'grid')

def test_response_search_privacy_budget_and_fair_profiles():
 s=initial();req=RobustRequest(horizon=1,scenarios=2,max_candidates=8,step_budget=650,diagnostics=False)
 r=advise(CONFIG,s,req);assert r['v17']['response_complete'];assert len(r['v17']['response_profiles'])==3
 assert r['search']['used_steps']<=650;assert len(r['v17']['sensitivity']['perturbations'])==12
 if r['recommended_id']!='baseline':assert min(next(v for v in r['validation'] if v['id']==r['recommended_id'])['paired_gains'])>0
 hidden=replace(s,companies=tuple(replace(c,financial=replace(c.financial,cash_balance_cents=123)) if c.company_id=='company_B' else c for c in s.companies));hidden=replace(hidden,state_hash=state_hash(hidden.to_dict()));other=advise(CONFIG,hidden,req)
 assert r['recipe']==other['recipe'] and r['v17']==other['v17']
 assert r['rankings']==other['rankings']

def test_insufficient_response_budget_abstains_without_certificate():
 r=advise(CONFIG,initial(),RobustRequest(horizon=1,scenarios=2,max_candidates=8,step_budget=100,diagnostics=False,backtest=False))
 assert not r['v17']['response_complete'] and r['recommended_id']=='baseline';assert r['search']['used_steps']<=100

def test_independent_arithmetic_oracle_and_corruption():
 from copy import deepcopy
 from game_theory_agent.game_theory.advisor_audit import audit_settlement
 from game_theory_agent.market.actor_policies import actions_for
 env=MarketEnv(CONFIG);s=env.reset(max_rounds=5,cooperation_mode='combined_v1');a=env.step(f'{s.episode_id}:{s.round}:{s.state_version}',actions_for(CONFIG,s,{})).state_after
 assert audit_settlement(s.to_dict(),a.to_dict())>20
 broken=deepcopy(a.to_dict());broken['companies']['company_A']['financial']['round_revenue_cents']+=1
 with pytest.raises(AssertionError,match='revenue'):audit_settlement(s.to_dict(),broken)

def test_all_company_advisors_history_resume(tmp_path):
 import asyncio
 from game_theory_agent.market.actor_experiments import run_episode,read_json
 modes={'company_A':'robust_profit','company_B':'robust_welfare'};progress=[];args=dict(company_count=2,seed=370102,rounds=5,modes=modes)
 assert asyncio.run(run_episode(CONFIG,directory=tmp_path/'a',**args,cancelled=lambda:bool(progress and progress[-1]==2),on_progress=lambda n,total:progress.append(n)))['status']=='cancelled'
 a=asyncio.run(run_episode(CONFIG,directory=tmp_path/'a',**args));b=asyncio.run(run_episode(CONFIG,directory=tmp_path/'b',**args));assert a==b
 assert len(a['memory']['_advisor_public_history'])==5
 for i in range(1,6):
  decisions=read_json(tmp_path/'a'/f'round-{i:03d}.json')['decisions']
  for cid in modes:
   r=decisions[cid]['advisor_decision'];assert len(r['request']['history'])==i-1
   assert r['source_round']==i and r['v17']['response_complete']

def test_v17_api_uses_persisted_public_history_and_restores(tmp_path):
 from fastapi import FastAPI
 from fastapi.testclient import TestClient
 from game_theory_agent.game_theory.service import router
 from game_theory_agent.game_theory.advisor_beliefs import public_frame
 from game_theory_agent.market.actor_policies import actions_for
 e=MarketEnv(CONFIG);s=e.reset(max_rounds=10,cooperation_mode='combined_v1');history=[]
 for _ in range(2):history.append(public_frame(s));s=e.step(f'{s.episode_id}:{s.round}:{s.state_version}',actions_for(CONFIG,s,{})).state_after
 app=FastAPI();app.include_router(router(tmp_path,CONFIG,lambda t:None,lambda ep:s,lambda ep,r:history));client=TestClient(app)
 request=dict(request_id='v17-api',kind='advisor',parameters=dict(advisor_version='v17',episode_id=s.episode_id,horizon=1,scenarios=2,max_candidates=8,step_budget=650,diagnostics=False,backtest=False))
 response=client.post('/api/v1/controller/theory-lab/experiments',json=request);assert response.status_code==200,response.text
 result=response.json();assert result['result']['request']['history']==history
 assert all(v['observations']==2 for v in result['result']['v17']['beliefs'].values())
 assert client.post('/api/v1/controller/theory-lab/experiments',json=request).json()==result
 assert client.get('/api/v1/controller/theory-lab/experiments/'+result['id']).json()==result

def test_workbench_accepts_ten_companies_and_preserves_default_identity(tmp_path):
 from game_theory_agent.research_workbench import Workbench,BatchRequest
 w=Workbench(tmp_path,CONFIG,start_worker=False)
 req=BatchRequest(request_id='ten',seeds=[370901],rounds=5,company_count=10,variants=[dict(label='v17',modes={'company_J':'robust_profit'})])
 a=w.submit(req);assert a==w.submit(req)
 with pytest.raises(ValueError):BatchRequest(request_id='bad',seeds=[1],variants=[dict(label='x')],company_count=True)

def test_action_memo_matches_uncached_entire_trajectories():
 from game_theory_agent.game_theory.advisor_market import Rollouts
 s=initial();req=RobustRequest(horizon=3,scenarios=2,step_budget=500)
 cached=Rollouts(CONFIG,s,req);reference=Rollouts(CONFIG,s,req);reference.cache_actions=False
 for profile in ({'company_A':{}},{'company_A':{'price':900000}},{'company_B':{'price':1100000}},{'company_A':{'reserve':True}}):
  for seed in (0,1,2000):assert cached.evaluate(profile,seed)==reference.evaluate(profile,seed)
 assert cached.used==reference.used==36
