from copy import deepcopy
from dataclasses import replace
import pytest
from game_theory_agent.market import MarketEnv,load_market_config
from game_theory_agent.market.actor_policies import company_action
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.game_theory.reliable_advisor import ReliableRequest,advise,local_candidates,DraftRollouts
from game_theory_agent.game_theory.reliable_response import fit
from game_theory_agent.cooperation.institution import DirectedCredibility
from game_theory_agent import local_budget

@pytest.fixture
def market():
    c=load_market_config('configs/market_v14_local.yaml');s=MarketEnv(c).reset(episode_id='reliable-test',episode_seed=518001,max_rounds=10,cooperation_mode='combined_v1');return c,s

def test_exact_draft_preserved_when_budget_incomplete(market):
    c,s=market;draft=company_action(c,s,'company_A','margin').to_dict();r=advise(c,s,ReliableRequest(draft_action=draft,step_budget=1),model=fit([]))
    assert r['disposition']=='abstain' and r['action']==draft and r['candidates'][0]['action']==draft
    assert not r['search']['complete'] and r['search']['used_steps']<=1

def test_candidate_is_local_and_metadata_mismatch_rejected(market):
    c,s=market;draft=company_action(c,s,'company_A','balanced');p=local_candidates(c,s,'company_A',draft)
    assert p[0]['action']==draft.to_dict()
    assert all(x['action']['agent_id']=='company_A' for x in p)
    raw=draft.to_dict();raw['state_version']+=1
    with pytest.raises(Exception):advise(c,s,ReliableRequest(draft_action=raw),model=fit([]))

def test_hidden_cash_and_future_history_boundaries(market):
    c,s=market;draft=company_action(c,s,'company_A','balanced').to_dict();req=ReliableRequest(draft_action=draft,max_candidates=2)
    a=advise(c,s,req,model=fit([]));peer=s.companies[1];changed=replace(peer,financial=replace(peer.financial,cash_balance_cents=peer.financial.cash_balance_cents+123456))
    altered=replace(s,companies=(s.companies[0],changed,*s.companies[2:]));altered=replace(altered,state_hash=state_hash(altered.to_dict()));b=advise(c,altered,req,model=fit([]))
    assert a['action']==b['action'] and a['validation']==b['validation']
    with pytest.raises(ValueError):advise(c,s,req.model_copy(update={'draft_action':{**draft,'round':s.round+1}}),model=fit([]))
    from game_theory_agent.game_theory.advisor_beliefs import PriceFrame
    with pytest.raises(ValueError):advise(c,s,req.model_copy(update={'public_history':[PriceFrame(round=s.round,prices={x.company_id:x.commercial.price_cents for x in s.companies})]}),model=fit([]))

def test_simultaneous_first_response_independent_of_candidate(market):
    c,s=market;draft=company_action(c,s,'company_A','balanced');pool=local_candidates(c,s,'company_A',draft,3);req=ReliableRequest(draft_action=draft.to_dict());runner=DraftRollouts(c,s,req,fit([]))
    a=runner.evaluate(pool[0]['action'],draft.to_dict(),101,3);b=runner.evaluate(pool[1]['action'],draft.to_dict(),101,3)
    for peer in s.company_ids[1:]:assert a['trace'][0]['actions'][peer]==b['trace'][0]['actions'][peer]

def test_directed_credibility_uses_only_accepted_verified_promises():
    x=DirectedCredibility(['A','B','C']);assert x.accepts('B','A',1)
    x.verify('B','A',100,0,1,accepted=True)
    assert not x.accepts('B','A',2) and x.accepts('C','A',2)
    before=x.export();x.verify('C','A',100,0,1,accepted=False);assert x.export()==before
    assert DirectedCredibility.restore(before).export()==before and x.accepts('B','A',5)

def test_budget_increase_keeps_old_calls_and_detects_unauthorized_total(tmp_path):
    p=tmp_path/'ledger.sqlite3';local_budget.initialize(p);local_budget.reserve(p);before=local_budget.status(p)
    result=local_budget.authorize_increase(30,'owner explicitly approved cumulative 30 CNY',p)
    assert result['total_cny']==30 and result['reserved_cny']==before['reserved_cny'] and result['new_calls']==before['new_calls']
    assert local_budget.authorize_increase(30,'same authorization',p)==result
    with pytest.raises(local_budget.LocalBudgetError):local_budget.authorize_increase(31,'not authorized',p)
