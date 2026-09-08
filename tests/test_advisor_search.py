import pytest
from dataclasses import replace
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.game_theory.objectives import AdviceRequest
from game_theory_agent.game_theory.advisor_search import advise
from game_theory_agent.game_theory.advisor_diagnostics import response_search
from game_theory_agent.game_theory.advisor_market import BudgetExhausted
from game_theory_agent.market import MarketEnv,load_market_config
CONFIG=load_market_config('configs/market_v14_local.yaml')
def test_bounded_search_has_independent_validation_and_private_invariance():
 s=MarketEnv(CONFIG).reset(max_rounds=5,cooperation_mode='combined_v1')
 req=AdviceRequest(horizon=2,scenarios=2,max_candidates=8,step_budget=200,diagnostics=False)
 a=advise(CONFIG,s,req)
 assert a['search']['used_steps']<=200 and set(a['search']['discovery_scenarios']).isdisjoint(a['search']['validation_scenarios'])
 if a['recommended_id']!='baseline':
  v=next(v for v in a['validation'] if v['id']==a['recommended_id']);assert min(v['paired_gains'])>0 and v['safe']
 changed=replace(s,companies=tuple(replace(c,financial=replace(c.financial,cash_balance_cents=17)) if c.company_id=='company_B' else c for c in s.companies))
 changed=replace(changed,state_hash=state_hash(changed.to_dict()))
 b=advise(CONFIG,changed,req)
 assert a['rankings']==b['rankings'] and a['recommended_id']==b['recommended_id'] and a['action']==b['action']
 env=MarketEnv(CONFIG);env.load_state(s);assert env.validate_action(a['action'],'company_A').valid
 assert not a['search']['global_optimality_proven']

def test_multiplayer_deviation_exact_known_games_and_budget_failure():
 choices={str(i):[0,1] for i in range(10)}
 # A ten-player public good with private cost 2 and benefit .3 per contributor.
 def payoff(p):return {a:10-2*v+.3*sum(p.values()) for a,v in p.items()}
 r=response_search(payoff,choices,{a:1 for a in choices})
 assert r['complete'] and r['restricted_pure_nash'] and set(r['profile'].values())=={0}
 def pennies(p):return {'a':1 if p['a']==p['b'] else -1,'b':-1 if p['a']==p['b'] else 1}
 r=response_search(pennies,{'a':[0,1],'b':[0,1]},{'a':0,'b':0})
 assert r['complete'] and not r['restricted_pure_nash'] and max(r['deviation_gains'].values())==2
 def stopped(p):raise BudgetExhausted()
 r=response_search(stopped,choices,{a:0 for a in choices});assert not r['complete'] and r['restricted_pure_nash'] is None
