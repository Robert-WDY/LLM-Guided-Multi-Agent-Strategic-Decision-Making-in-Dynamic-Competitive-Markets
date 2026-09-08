import asyncio
from dataclasses import replace
from pathlib import Path
from game_theory_agent.market import MarketEnv,load_market_config
from game_theory_agent.game_theory.market_strategies import choose,MODES,market_matrix
from game_theory_agent.market.actor_experiments import run_episode,read_json

CONFIG=load_market_config(Path(__file__).parents[1]/'configs/market_v14_local.yaml')


def test_strategies_do_not_read_opponent_cash_and_matrix_does_not_mutate():
    env=MarketEnv(CONFIG);s=env.reset(episode_id='theory-privacy',max_rounds=5)
    changed=replace(s,companies=tuple(replace(c,financial=replace(c.financial,cash_balance_cents=123)) if c.company_id=='company_B' else c for c in s.companies))
    for mode in MODES:
        assert choose(CONFIG,s,'company_A',mode,{})==choose(CONFIG,changed,'company_A',mode,{})
    r=market_matrix(CONFIG,s);assert r['source_state_hash']==s.state_hash and len(r['outcome_hashes'])==4


def test_theory_memory_survives_cancel_and_same_seed_replay(tmp_path):
    progress=[];modes={'company_A':'theory_wsls'}
    args=dict(seed=330101,rounds=5,modes=modes)
    r=asyncio.run(run_episode(CONFIG,directory=tmp_path/'resumed',**args,on_progress=lambda n,t:progress.append(n),cancelled=lambda:bool(progress and progress[-1]==2)))
    assert r['status']=='cancelled'
    resumed=asyncio.run(run_episode(CONFIG,directory=tmp_path/'resumed',**args))
    fresh=asyncio.run(run_episode(CONFIG,directory=tmp_path/'fresh',**args))
    assert resumed==fresh and resumed['replay_passed']
    assert read_json(tmp_path/'resumed/round-003.json')['decisions']['company_A']['theory_decision']['uses_hidden_state'] is False
