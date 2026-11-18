import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from game_theory_agent.game_theory.cycle_pricing import decide, native_decision
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.actor_experiments import read_json, run_episode
from game_theory_agent.market.actor_policies import company_action
from game_theory_agent.market.replay import MarketTransition
from game_theory_agent.market.models import MarketState

CONFIG = load_market_config(Path(__file__).parents[1]/"configs/market_v14_complete.yaml")


def test_hold_profit_feedback_and_terminal_truncation():
    history=[];old=None;contracts=[]
    for t in range(1,16):
        d=decide(t,16-t,100,50,150,history,old)
        if d['start']==t:contracts.append(d)
        # The middle warm price yields the highest profit. The second full cycle loses money.
        profit=200-abs(d['price']-100) if t<=6 else 100 if t<=10 else 60
        history.append([d['price'],100,2,profit]);old=d
    assert [(v['start'],v['end']) for v in contracts]==[(1,2),(3,4),(5,6),(7,10),(11,14),(15,15)]
    assert contracts[3]['price']==100 and contracts[4]['price']>100
    assert contracts[5]['price']<contracts[4]['price'] and contracts[5]['next_decision'] is None


def test_equal_search_budget_and_rejected_original():
    history=[[90+i*3,100-i,50-i,i*8+50] for i in range(6)]
    for mode in ('generic','wide','conditional'):
        d=decide(7,3,100,50,180,history,{},mode=mode,original=103)
        assert d['transitions']==135 and d['price']==103 and d['end']==9
        assert d['proposed']!=103
        assert decide(8,2,100,50,180,history,d,mode=mode,original=140)==d


def test_native_company_cannot_observe_private_competitor_accounts():
    env=MarketEnv(CONFIG);s=env.reset(episode_id='private-cycle')
    hidden=replace(s,companies=tuple(replace(x,financial=replace(x.financial,cash_balance_cents=100))
                   if x.company_id!='company_A' else x for x in s.companies))
    assert native_decision(CONFIG,s,'company_A',None,'pricing_feedback')==native_decision(CONFIG,hidden,'company_A',None,'pricing_feedback')


def test_workbench_price_contract_survives_resume_and_preserves_portfolio(tmp_path):
    cfg=dict(seed=825409,rounds=15,modes={'company_A':'pricing_feedback'})
    progress=[]
    checkpoint=tmp_path/'resume'
    r=asyncio.run(run_episode(CONFIG,directory=checkpoint,**cfg,on_progress=lambda n,k:progress.append(n),
                              cancelled=lambda:bool(progress and progress[-1]==8)))
    assert r['status']=='cancelled'
    a=asyncio.run(run_episode(CONFIG,directory=checkpoint,**cfg))
    b=asyncio.run(run_episode(CONFIG,directory=tmp_path/'fresh',**cfg))
    assert a==b and a['replay_passed']
    cp=read_json(checkpoint/'checkpoint.json')
    env=MarketEnv(CONFIG)
    state=MarketState.from_dict(cp['manifest']['initial_state']);env.load_state(state)
    for j,trace in enumerate(cp['transitions'],1):
        journal=read_json(checkpoint/f'round-{j:03d}.json')['decisions']
        decision=journal['company_A'];term=decision['price_decision']
        assert term['start']<=j<=term['end']<=15
        actual=dict(MarketTransition.from_dict(trace).joint_action)
        assert actual['company_A'].price_cents==decision['executed_price_cents']
        base=company_action(CONFIG,state,'company_A','balanced').to_dict()
        assert {k:v for k,v in actual['company_A'].to_dict().items() if k not in ('action_id','price_cents')}=={
            k:v for k,v in base.items() if k not in ('action_id','price_cents')}
        state=env.step(f'{state.episode_id}:{j}:{state.state_version}',actual,actor_choices={
            name:row['option'] for name,row in journal.items() if name not in state.company_ids}).state_after
    assert state.state_hash==a['state_hash']


def test_pricing_restricted_to_company(tmp_path):
    with pytest.raises(ValueError,match='companies only'):
        asyncio.run(run_episode(CONFIG,directory=tmp_path,seed=778,modes={'government':'pricing_feedback'}))

