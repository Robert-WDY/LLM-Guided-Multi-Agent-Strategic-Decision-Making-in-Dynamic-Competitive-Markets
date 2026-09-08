from dataclasses import replace
from pathlib import Path
import pytest
from game_theory_agent.market import MarketEnv, MarketState, load_market_config
from game_theory_agent.market.government_strategy import unpack
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition, verify_replay
from game_theory_agent.gameplay import build_rule_action

CONFIG=load_market_config(Path(__file__).parents[1]/"configs/market_v14_complete.yaml")


def test_independent_fine_multiplier_changes_actual_collections():
    from game_theory_agent.market import MarketConfig
    data=CONFIG.to_dict();data["autonomous_market"]["government"]["detection_boost_ppm"]=1000000
    c=MarketConfig.from_mapping(data);out={}
    for option in ("balanced","enforce"):
        e=MarketEnv(c);s=e.reset(episode_id="fine-paired",episode_seed=230107,max_rounds=5,cooperation_mode="combined_v1")
        aa={cid:build_rule_action(c,s,cid) for cid in s.company_ids};a,b=s.company_ids[:2]
        for cid,partner in ((a,b),(b,a)):
            aa[cid]=replace(aa[cid],price_cents=12000,price_coordination_partner_company_id=partner,price_coordination_target_cents=12000)
        out[option]=e.step(f"{s.episode_id}:1:0",aa,actor_choices={"government":option}).state_after.government.round_fines_cents
    assert out["balanced"]>0 and out["enforce"]==2*out["balanced"]


def run(option):
    e=MarketEnv(CONFIG);s=e.reset(episode_id="government-paired",episode_seed=230101,max_rounds=10,cooperation_mode="combined_v1")
    manifest=EpisodeManifest.create(e,s,cooperation_mode="combined_v1");traces=[]
    for i in range(10):
        if i==4:
            e=MarketEnv(CONFIG);e.load_state(MarketState.from_dict(s.to_dict()))
        actions={cid:replace(build_rule_action(CONFIG,s,cid),resilience_budget_cents=100000 if s.rounds_remaining>1 else 0) for cid in s.company_ids}
        result=e.step(f"{s.episode_id}:{s.round}:{s.state_version}",actions,actor_choices={"government":option} if option else {})
        traces.append(MarketTransition.create(s,actions,result));s=result.state_after
    verify_replay(MarketEnv(CONFIG),manifest,[MarketTransition.from_dict(t.to_dict()) for t in traces])
    return s,traces


def test_rebates_are_transfers_not_new_aggregate_welfare():
    reserve,rt=run("reserve");consumer,ct=run("consumer")
    # No household savings model: a post-purchase rebate changes distribution only.
    assert reserve.welfare_accounting.cumulative_total_economic_welfare_cents==consumer.welfare_accounting.cumulative_total_economic_welfare_cents
    assert consumer.government.cash_cents<reserve.government.cash_cents
    for t in ct:
        s=t.step_result.state_after
        assert sum(d.government_rebate_cents for d in s.consumer_decisions)==s.government.round_consumer_rebate_cents
        assert all(0<=d.government_rebate_cents<=d.spending_cents-d.refund_cents for d in s.consumer_decisions)


def test_targeted_grants_pay_actual_investment_and_stop_at_horizon():
    s,tr=run("resilience")
    for t in tr[:-1]:
        g=t.step_result.state_after.government
        assert g.round_matched_support_cents==100000
        assert sum(c.financial.round_government_support_cents>0 for c in t.step_result.state_after.companies)==1
    assert s.government.round_matched_support_cents==0


def test_policy_learning_and_corrupt_memory_rejected():
    s,tr=run(None)
    assert {t.step_result.state_after.government.last_decision.reason for t in tr}=={"reserve","enforce","resilience","consumer","balanced"}
    assert unpack(s.government.policy_memory)["observations"]==10
    bad=replace(s,government=replace(s.government,policy_memory='{}'))
    with pytest.raises(Exception):
        MarketEnv(CONFIG).load_state(bad)
