from game_theory_agent.market.protocols import state_hash
from pathlib import Path
from dataclasses import replace
import hashlib
from game_theory_agent.market import MarketEnv,load_market_config
from game_theory_agent.game_theory.objectives import AdviceRequest
from game_theory_agent.game_theory.advisor_market import forecast,candidates,Rollouts,BudgetExhausted
import pytest
CONFIG=load_market_config(Path(__file__).parents[1]/'configs/market_v14_local.yaml')
def test_public_privacy_and_full_engine_trace():
 s=MarketEnv(CONFIG).reset(company_ids=['company_A','company_B','company_C','company_D','company_E'],episode_seed=360201,max_rounds=5,cooperation_mode='combined_v1')
 changed=replace(s,companies=tuple(replace(c,financial=replace(c.financial,cash_balance_cents=17)) if c.company_id=='company_B' else c for c in s.companies))
 f,record=forecast(CONFIG,s,'company_A');g,other=forecast(CONFIG,changed,'company_A');assert f==g and record==other
 request=AdviceRequest(horizon=3,step_budget=100);runner=Rollouts(CONFIG,f,request)
 result=runner.evaluate({'company_A':{'price':950000}},0)
 seed=int(hashlib.sha256(f'{f.episode_seed}:{request.seed}:0'.encode()).hexdigest()[:15],16)
 env=MarketEnv(CONFIG);changed=replace(f,episode_seed=seed);env.load_state(replace(changed,state_hash=state_hash(changed.to_dict())));profits=0
 for i,row in enumerate(result['trace']):
  state=env.get_state();after=env.step(f'{state.episode_id}:{state.round}:{state.state_version}',row['joint_actions'],actor_choices=row['other_actor_choices']).state_after
  assert after.state_hash==row['state_hash'];profits+=request.discount**i*after.company('company_A').financial.round_profit_cents
 assert profits==result['profit'];assert len(result['trace'])==3
 assert runner.evaluate({'company_A':{'price':950000}},0)==result and runner.used==3
 runner.used=99
 with pytest.raises(BudgetExhausted):runner.evaluate({'company_A':{}},1)
 assert runner.used==99

def test_all_candidates_respect_engine_constraints():
 state=MarketEnv(CONFIG).reset(max_rounds=5,cooperation_mode='combined_v1');env=MarketEnv(CONFIG);env.load_state(state)
 rows=candidates(CONFIG,state,'company_A',48);assert len(rows)>4
 for row in rows:assert env.validate_action(row['action'],'company_A').valid

