import asyncio
from pathlib import Path
from dataclasses import replace
import pytest
from game_theory_agent.market import MarketEnv,load_market_config
from game_theory_agent.market.actor_policies import actor_ids,observation
from game_theory_agent.market.actor_experiments import run_episode,read_json,write_json
CONFIG=load_market_config(Path(__file__).parents[1]/"configs/market_v14_complete.yaml")


def test_four_role_learning_resume_and_replay(tmp_path):
    e=MarketEnv(CONFIG);s=e.reset(episode_id="ids",max_rounds=10)
    modes={a:"learning" for a in actor_ids(s)};progress=[]
    first=asyncio.run(run_episode(CONFIG,directory=tmp_path,seed=240101,modes=modes,
        cancelled=lambda:bool(progress and progress[-1]==4),on_progress=lambda n,total:progress.append(n)))
    assert first["status"]=="cancelled"
    result=asyncio.run(run_episode(CONFIG,directory=tmp_path,seed=240101,modes=modes))
    assert result["replay_passed"] and result["rounds"]==10
    assert all(sum(v["counts"].values())==10 for v in result["memory"].values())
    assert asyncio.run(run_episode(CONFIG,directory=tmp_path,seed=240101,modes=modes))==result


def test_private_competitor_changes_do_not_enter_other_actor_prompts():
    e=MarketEnv(CONFIG);s=e.reset(episode_id="information")
    changed=replace(s,companies=tuple(replace(c,financial=replace(c.financial,cash_balance_cents=123)) if c.company_id=="company_B" else c for c in s.companies))
    for a in actor_ids(s):
        if a!="company_B":assert observation(s,a)==observation(changed,a)


def test_objective_advisor_executes_and_resumes_same_actions(tmp_path):
    from game_theory_agent.market import MarketState
    from game_theory_agent.game_theory.advisor_market import recipe_action
    progress=[];args=dict(seed=360601,rounds=5,modes={'company_A':'advisor_welfare'})
    interrupted=tmp_path/'resume'
    assert asyncio.run(run_episode(CONFIG,directory=interrupted,**args,cancelled=lambda:bool(progress and progress[-1]==2),on_progress=lambda n,total:progress.append(n)))['status']=='cancelled'
    resumed=asyncio.run(run_episode(CONFIG,directory=interrupted,**args))
    fresh=asyncio.run(run_episode(CONFIG,directory=tmp_path/'fresh',**args))
    assert resumed==fresh and resumed['replay_passed']
    cp=read_json(interrupted/'checkpoint.json');state=MarketState.from_dict(cp['manifest']['initial_state']);env=MarketEnv(CONFIG);env.load_state(state)
    for i in range(1,6):
        journal=read_json(interrupted/f'round-{i:03d}.json');d=journal['decisions']['company_A']
        assert d['advisor_decision']['objective']['goal']=='welfare'
        assert d['advisor_decision']['source_state_hash']==state.state_hash
        action=recipe_action(CONFIG,state,'company_A',d['advisor_recipe'])
        assert action.to_dict()==d['advisor_decision']['action']
        trace=cp['transitions'][i-1]
        # Replay assertion in run_episode checks every journal-overridden action against settlement.
        from game_theory_agent.market.replay import MarketTransition
        t=MarketTransition.from_dict(trace)
        actions=dict(t.joint_action)
        assert actions['company_A'].to_dict()==action.to_dict()
        state=env.step(f'{state.episode_id}:{state.round}:{state.state_version}',actions,actor_choices={a:v['option'] for a,v in journal['decisions'].items() if a not in state.company_ids}).state_after
    assert d['advisor_decision']['objective']['phase']=='closing'


def test_pending_paid_call_never_retried(tmp_path):
    asyncio.run(run_episode(CONFIG,directory=tmp_path,seed=240102,modes={"government":"model"},authorize_real=True,cancelled=lambda:True))
    s=read_json(tmp_path/"checkpoint.json")["state"]
    write_json(tmp_path/"round-001.json",dict(state_hash=s["state_hash"],decisions={"government":{"status":"pending_unknown"}}))
    with pytest.raises(RuntimeError,match="unknown"):
        asyncio.run(run_episode(CONFIG,directory=tmp_path,seed=240102,modes={"government":"model"},authorize_real=True))


def test_schema_failure_saved_and_only_explicit_review_allows_retry(tmp_path,monkeypatch):
    from game_theory_agent.market import actor_experiments as ae
    from game_theory_agent.market.actor_model import ActorModelValidationError,ActorModelChoice
    calls=[]
    async def choose(**kwargs):
        calls.append(kwargs)
        if len(calls)==1:raise ActorModelValidationError(dict(raw_response='{"bad":true}',input_tokens=10,output_tokens=5))
        return ActorModelChoice("government","reserve","test","fake","fake",10,5,'{"option_id":"reserve","reason":"test"}')
    monkeypatch.setattr(ae,"choose_option",choose)
    args=dict(directory=tmp_path,seed=240103,rounds=5,modes={"government":"model"},authorize_real=True)
    with pytest.raises(ActorModelValidationError):asyncio.run(run_episode(CONFIG,**args))
    journal=read_json(tmp_path/"round-001.json")
    assert journal["decisions"]["government"]["model_response"]["raw_response"]=='{"bad":true}'
    with pytest.raises(RuntimeError):asyncio.run(run_episode(CONFIG,**args))
    assert len(calls)==1
    ae.authorize_schema_retry(tmp_path,1,"government","explicit test review")
    assert asyncio.run(run_episode(CONFIG,**args))["status"]=="complete" and len(calls)==6
