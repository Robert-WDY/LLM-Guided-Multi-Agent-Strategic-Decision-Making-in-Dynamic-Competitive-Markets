from dataclasses import replace
from pathlib import Path
import pytest

from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import MarketEnv, MarketState, load_market_config
from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.exceptions import ConfigError
from game_theory_agent.market.autonomous_market import government_decision, investment_decision
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.persistence import CODEC
from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.strategic_reliability import build_public_forecast_state, generate_public_overlay_candidates

ROOT=Path(__file__).resolve().parents[1]


def setup():
    data=load_market_config(ROOT/"configs/market_v13_autonomous.yaml").to_dict()
    return data


def run(data, seed=180100, rounds=20):
    config=MarketConfig.from_mapping(data);env=MarketEnv(config)
    state=env.reset(episode_id=f"autonomous-{seed}",episode_seed=seed,max_rounds=rounds,cooperation_mode="combined_v1")
    states=[]
    while not state.terminal:
        actions={i:build_rule_action(config,state,i) for i in state.company_ids}
        state=env.step(f"{state.episode_id}:{state.round}:{state.state_version}",actions).state_after
        states.append(state)
    return config,states


def test_four_actor_settlement_and_full_state_codec():
    data=setup();data["autonomous_market"]["supplier_investment"]["utilization_threshold_ppm"]=300000
    config,states=run(data)
    assert any(s.investment_decision.added_capacity_orders>0 for st in states for s in st.supply_chain.suppliers)
    assert all(s.investment_decision.investment_cents==0 for s in states[-1].supply_chain.suppliers)
    assert any(sum(v for _,v in st.government.last_decision.support_by_company_cents)>0 for st in states)
    for state in states:
        assert MarketState.from_dict(state.to_dict())==state
        restored=CODEC.decode(CODEC.encode(state));MarketEnv(config).load_state(restored)


def test_government_spends_only_own_cash_on_public_pressure():
    policy=setup()["autonomous_market"]["government"]
    for cash in (0,99999,200001,50000000):
        d=government_decision(cash=cash,observed_round=7,stockout=1000,demand=1000,hhi=800000,active_ids=("B","A"),policy=policy)
        assert d.inspection_cost_cents+sum(v for _,v in d.support_by_company_cents)<=cash
        assert d.active_company_ids==("A","B") and d.applies_round==8


def test_supplier_cash_reserve_capacity_bound_and_last_round():
    p=setup()["autonomous_market"]["supplier_investment"]
    args=dict(sales=100,available=100,base_capacity=29950,cash=50000000,observed_round=3,terminal=False,policy=p)
    d=investment_decision(**args)
    assert d.added_capacity_orders==50 and d.applies_round==4
    assert investment_decision(**{**args,"terminal":True}).investment_cents==0
    assert investment_decision(**{**args,"cash":p["reserve_cash_cents"]}).investment_cents==0
    assert investment_decision(**{**args,"available":0,"sales":0}).investment_cents==0


def test_quantity_survives_intent_resolver_and_validation():
    config=MarketConfig.from_mapping(setup());env=MarketEnv(config)
    state=env.reset(episode_id="quantity",episode_seed=180099)
    cid=state.company_ids[0];action=replace(build_rule_action(config,state,cid),procurement_quantity_orders=0)
    resolved=resolve_action_request(config,state,cid,action.to_dict(),source="test",action_id=action.action_id)
    assert resolved.action.procurement_quantity_orders==0
    bad=replace(action,procurement_quantity_orders=True)
    assert not env.validator.validate(bad,state=state,company_id=cid).valid


def test_consumer_budget_rejects_every_unaffordable_offer():
    data=setup();data["action"]["bounds"]["price_cents"]["max"]=40000
    config=MarketConfig.from_mapping(data);env=MarketEnv(config)
    state=env.reset(episode_id="consumer-budget",episode_seed=180098)
    actions={i:replace(build_rule_action(config,state,i),price_cents=config.mapping("action","bounds","price_cents")["max"]) for i in state.company_ids}
    after=env.step(f"{state.episode_id}:1:0",actions).state_after
    assert all(d.spending_cents==0 for d in after.consumer_decisions)
    assert after.market.no_purchase_orders==after.market.realized_demand_orders


def test_budgeted_regulator_inspects_and_fine_transfers_close():
    data=setup();data["autonomous_market"]["government"].update(concentration_threshold_ppm=0,detection_boost_ppm=1000000)
    config=MarketConfig.from_mapping(data);env=MarketEnv(config)
    state=env.reset(episode_id="regulator",episode_seed=180097,cooperation_mode="combined_v1")
    actions={i:build_rule_action(config,state,i) for i in state.company_ids}
    a,b=state.company_ids[:2]
    for cid,partner in ((a,b),(b,a)):
        actions[cid]=replace(actions[cid],price_cents=12000,price_coordination_partner_company_id=partner,price_coordination_target_cents=12000)
    after=env.step(f"{state.episode_id}:1:0",actions).state_after
    assert after.government.round_fines_cents>0
    assert after.government.cash_cents==state.government.cash_cents+after.government.round_fines_cents-after.government.last_decision.inspection_cost_cents


@pytest.mark.parametrize("section,field,value",[("government","initial_cash_cents",-1),("supplier_investment","unit_capacity_cost_cents",0),("government","enabled",1)])
def test_invalid_actor_budgets_rejected(section,field,value):
    data=setup();data["autonomous_market"][section][field]=value
    with pytest.raises(ConfigError):MarketConfig.from_mapping(data)


def test_public_forecast_uses_legal_accounting_priors_and_legal_actions():
    data=setup();data["autonomous_market"]["supplier_investment"]["utilization_threshold_ppm"]=300000
    config,states=run(data,seed=180096)
    profile=PersonaRegistry.from_market_config(config).get("balanced_v1")
    for state in (states[0],states[6],states[12],states[18]):
        cid=state.company_ids[0];observation=ObservationBuilder().build(state,cid,"public")
        assert all("investment_decision" not in s for s in observation["market"]["supply_chain"]["suppliers"].values())
        forecast,record=build_public_forecast_state(config=config,observation=observation,company_id=cid,persona_profile=profile)
        env=MarketEnv(config);env.load_state(forecast)
        assert record.uses_authoritative_hidden_market_state is False
        candidates=generate_public_overlay_candidates(config,forecast,cid)
        assert candidates
        actions={i:build_rule_action(config,forecast,i) for i in forecast.company_ids}
        env.step(f"{forecast.episode_id}:{forecast.round}:{forecast.state_version}",actions)


def test_payback_guard_rejects_disruption_no_excess_and_short_horizon():
    p=load_market_config(ROOT/"configs/market_v13_1_local.yaml").data["autonomous_market"]["supplier_investment"]
    args=dict(sales=10000,available=10000,base_capacity=10000,cash=50000000,observed_round=3,terminal=False,
              policy=p,requested=12000,unit_margin=2000,remaining_rounds=60)
    assert investment_decision(**args).added_capacity_orders>0
    for changes in ({"available":1000,"sales":1000},{"requested":10000},{"remaining_rounds":5},{"unit_margin":0}):
        assert investment_decision(**{**args,**changes}).added_capacity_orders==0


def test_guarded_supplier_long_run_and_public_forecast_restore():
    data=load_market_config(ROOT/"configs/market_v13_1_local.yaml").to_dict()
    for supplier in data["supply_chain"]["suppliers"].values():
        supplier["base_capacity_orders"]=5000
        supplier["reliability_ppm"]=1000000
    config,states=run(data,seed=180095,rounds=60)
    assert any(s.investment_decision.investment_cents for st in states for s in st.supply_chain.suppliers)
    state=states[20];cid=state.company_ids[0]
    observation=ObservationBuilder().build(state,cid,"public")
    assert all("round_requested_orders" not in s for s in observation["market"]["supply_chain"]["suppliers"].values())
    forecast,_=build_public_forecast_state(config=config,observation=observation,company_id=cid,
        persona_profile=PersonaRegistry.from_market_config(config).get("balanced_v1"))
    env=MarketEnv(config);env.load_state(forecast)
    actions={i:build_rule_action(config,forecast,i) for i in forecast.company_ids}
    env.step(f"{forecast.episode_id}:{forecast.round}:{forecast.state_version}",actions)
