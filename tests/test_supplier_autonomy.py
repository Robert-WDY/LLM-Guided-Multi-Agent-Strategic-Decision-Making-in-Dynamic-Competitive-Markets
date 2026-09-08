from dataclasses import replace
from pathlib import Path

import pytest

from game_theory_agent.market import MarketEnv, MarketState, load_market_config
from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.exceptions import ConfigError, StateInvariantError
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition, verify_replay
from game_theory_agent.market.supplier_policy import advance_supplier_quotes, decide_quote, price_bounds
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.persistence import CODEC, SessionStore

ROOT=Path(__file__).resolve().parents[1]


def config(): return load_market_config(ROOT/"configs/market_v11_supplier.yaml")


@pytest.mark.parametrize("sales,capacity,reason,quote",[(11000,11000,"high_utilization",1890),(0,11000,"low_utilization",1710),(6000,11000,"target_band",1800),(0,0,"unavailable",1800)])
def test_decision_uses_only_own_settled_utilization(sales,capacity,reason,quote):
    c=config();s=MarketEnv(c).reset(episode_id="quote").supply_chain.suppliers[0]
    s=replace(s,round_sales_orders=sales,available_capacity_orders=capacity)
    decision=decide_quote(supplier=s,config=c.mapping("supply_chain"),observed_round=3)
    assert (decision.reason,decision.quoted_price_cents,decision.applies_round)==(reason,quote,4)


def test_prices_clip_to_cost_floor_and_cap_and_ignore_supplier_order():
    c=config();supply=MarketEnv(c).reset(episode_id="quote").supply_chain
    altered=replace(supply,suppliers=tuple(replace(s,unit_price_cents=price_bounds(s,c.mapping("supply_chain"))[0],round_sales_orders=0) for s in supply.suppliers))
    low=advance_supplier_quotes(settled=altered,config=c.mapping("supply_chain"),observed_round=4)
    assert all(s.unit_price_cents==price_bounds(s,c.mapping("supply_chain"))[0] for s in low.suppliers)
    altered=replace(supply,suppliers=tuple(replace(s,unit_price_cents=price_bounds(s,c.mapping("supply_chain"))[1],round_sales_orders=s.available_capacity_orders) for s in supply.suppliers))
    high=advance_supplier_quotes(settled=altered,config=c.mapping("supply_chain"),observed_round=4)
    reverse=advance_supplier_quotes(settled=replace(altered,suppliers=tuple(reversed(altered.suppliers))),config=c.mapping("supply_chain"),observed_round=4)
    assert {s.supplier_id:s for s in high.suppliers}=={s.supplier_id:s for s in reverse.suppliers}
    assert all(s.unit_price_cents==price_bounds(s,c.mapping("supply_chain"))[1] for s in high.suppliers)


def test_quotes_apply_next_round_without_repricing_settled_profit_and_replay():
    c=config();env=MarketEnv(c);state=env.reset(episode_id="supplier-replay",episode_seed=140011,max_rounds=10)
    manifest=EpisodeManifest.create(env,state);transitions=[]
    while not state.terminal:
        actions={i:build_rule_action(c,state,i) for i in state.company_ids}
        result=env.step(f"{state.episode_id}:{state.round}:{state.state_version}",actions)
        after=result.state_after
        for supplier in after.supply_chain.suppliers:
            old=state.supply_chain.supplier(supplier.supplier_id)
            assert supplier.last_settled_unit_price_cents==old.unit_price_cents
            assert supplier.round_profit_cents==supplier.round_sales_orders*(old.unit_price_cents-old.unit_cost_cents)
            if not after.terminal:
                assert supplier.quote_decision.observed_capacity_orders==old.available_capacity_orders
                assert supplier.quote_decision.applies_round==after.round
        assert MarketState.from_dict(after.to_dict())==after
        assert CODEC.decode(CODEC.encode(after))==after
        transitions.append(MarketTransition.create(state,actions,result));state=after
    assert verify_replay(MarketEnv(c),manifest,tuple(transitions))[-1].state_hash==state.state_hash


def test_quote_tampering_is_rejected_before_loading():
    c=config();env=MarketEnv(c);s=env.reset(episode_id="tamper")
    after=env.step("tamper:1:0",{i:build_rule_action(c,s,i) for i in s.company_ids}).state_after
    supplier=after.supply_chain.suppliers[0]
    forged=replace(supplier,quote_decision=replace(supplier.quote_decision,utilization_ppm=0))
    altered=replace(after,supply_chain=replace(after.supply_chain,suppliers=(forged,*after.supply_chain.suppliers[1:])),state_hash="")
    from game_theory_agent.market.protocols import state_hash
    altered=replace(altered,state_hash=state_hash(altered.to_dict()))
    with pytest.raises(StateInvariantError,match="quote audit"):
        MarketEnv(c).load_state(altered)


@pytest.mark.parametrize("field,value",[("step_ppm",True),("step_ppm",300000),("lower_threshold_ppm",900000),("min_markup_ppm",1000000),("policy_version","unknown")])
def test_invalid_supplier_policy_is_rejected(field,value):
    data=config().to_dict();data["supply_chain"]["autonomous_pricing"][field]=value
    with pytest.raises(ConfigError): MarketConfig.from_mapping(data)


def test_legacy_supply_serialization_has_no_autonomous_fields():
    c=load_market_config(ROOT/"configs/market_v10_multi_objective.yaml")
    env=MarketEnv(c);s=env.reset(episode_id="legacy")
    s=env.step("legacy:1:0",{i:build_rule_action(c,s,i) for i in s.company_ids}).state_after
    for supplier in s.supply_chain.suppliers:
        assert "quote_decision" not in supplier.to_dict()
        assert "last_settled_unit_price_cents" not in supplier.to_dict()
        assert supplier.unit_price_cents==c.mapping("supply_chain")["suppliers"][supplier.supplier_id]["unit_price_cents"]


def test_sqlite_restart_preserves_quote_and_next_settlement(tmp_path):
    from game_theory_agent.api import EpisodeSession
    from game_theory_agent.persistence import restore
    c=config();env=MarketEnv(c);state=env.reset(episode_id="quote-checkpoint",episode_seed=140031)
    session=EpisodeSession(env=env,manifest=EpisodeManifest.create(env,state))
    for _ in range(3):
        state=env.step(f"{state.episode_id}:{state.round}:{state.state_version}",{i:build_rule_action(c,state,i) for i in state.company_ids}).state_after
    path=tmp_path/"supplier.sqlite3"
    SessionStore(path).save(session)
    restored=restore(SessionStore(path).load(state.episode_id),c,EpisodeSession)
    assert restored.env.get_state()==state
    actions={i:build_rule_action(c,state,i) for i in state.company_ids}
    step_id=f"{state.episode_id}:{state.round}:{state.state_version}"
    expected=env.step(step_id,actions)
    actual=restored.env.step(step_id,actions)
    assert actual.state_after.state_hash==expected.state_after.state_hash
    assert restored.env.step(step_id,actions)==actual


def test_public_forecast_accepts_supplier_quotes_without_private_buyer_state():
    from game_theory_agent.agents.personas import PersonaRegistry
    from game_theory_agent.belief import BeliefLedger
    from game_theory_agent.opponent import OpponentModelLedger
    from game_theory_agent.experiments.advisor_coverage_v11 import observe
    from game_theory_agent.strategic_reliability.public_rollout import build_public_forecast_state
    from game_theory_agent.strategic_reliability import PublicMarketRolloutAdvisor
    c=config();env=MarketEnv(c);state=env.reset(episode_id="quote-advisor",episode_seed=140011,max_rounds=20)
    beliefs=BeliefLedger(episode_id=state.episode_id,company_ids=state.company_ids)
    opponents=OpponentModelLedger(episode_id=state.episode_id,company_ids=state.company_ids)
    for _ in range(7):
        actions={i:build_rule_action(c,state,i) for i in state.company_ids}
        after=env.step(f"{state.episode_id}:{state.round}:{state.state_version}",actions).state_after
        beliefs.update_after_settlement(state,actions);opponents.update_after_settlement(state,after,actions);state=after
    observation,b,o=observe(c,state,beliefs,opponents)
    profile=PersonaRegistry.from_market_config(c).get("balanced_v1")
    forecast,_=build_public_forecast_state(config=c,observation=observation,company_id="company_A",persona_profile=profile,belief_state=b,opponent_model=o)
    MarketEnv(c).load_state(forecast)
    assert forecast.supply_chain==state.supply_chain
    advice=PublicMarketRolloutAdvisor(c).advise(observation=observation,company_id="company_A",persona_profile=profile,belief_state=b,opponent_model=o,horizon_rounds=3,scenario_count=5,advisor_mode="strategic_market_v9")
    assert advice.final_market_gate_policy["policy_version"]=="paired-baseline-dominance-v1.0.0"
    assert advice.execution_disposition in {"recommend","defer_to_agent"}


def test_sampled_opponent_response_never_revives_exited_company():
    from game_theory_agent.market.models import CompanyOperatingStatus
    from game_theory_agent.strategic_reliability.rollout import _opponent_response_action, generate_candidate_actions
    c=config();state=MarketEnv(c).reset(episode_id="absorbing-exit")
    candidate=generate_candidate_actions(c,state,"company_A")[0]
    lifecycle=tuple(replace(item,status=CompanyOperatingStatus.EXITED) if item.company_id=="company_B" else item for item in state.strategic_market.company_lifecycle)
    state=replace(state,strategic_market=replace(state.strategic_market,company_lifecycle=lifecycle))
    frozen=build_rule_action(c,state,"company_B")
    for scenario in range(20):
        action=_opponent_response_action(config=c,state=state,focal_company_id="company_A",opponent_id="company_B",focal_candidate=candidate,scenario_index=scenario,opponent_model=None,belief_state=None)
        assert action==frozen
