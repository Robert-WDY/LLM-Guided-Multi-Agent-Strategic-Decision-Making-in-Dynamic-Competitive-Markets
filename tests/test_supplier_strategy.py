from dataclasses import replace
from pathlib import Path
import pytest
from game_theory_agent.market import MarketEnv,MarketConfig,MarketState,load_market_config
from game_theory_agent.market.supplier_strategy import decode,encode,supplier_failures,settle_supplier,negotiated_book
from game_theory_agent.market.supply_chain import _allocate_integer
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market.protocols import state_hash

CONFIG=load_market_config(Path(__file__).parents[1]/"configs/market_v14_complete.yaml")

def test_inventory_contracts_replay_and_balances_across_20_rounds():
    env=MarketEnv(CONFIG);state=env.reset(episode_id="contracts",episode_seed=220101,max_rounds=20)
    inventories=[];improvements=[];agreements=[]
    for n in range(20):
        if n in (3,9):
            clone=MarketEnv(CONFIG);clone.load_state(MarketState.from_dict(state.to_dict()));env=clone
        actions={cid:build_rule_action(CONFIG,state,cid) for cid in state.company_ids}
        state=env.step(f"contracts:{state.round}:{state.state_version}",actions).state_after
        for s in state.supply_chain.suppliers:
            d=decode(s);inventories.append(d["inventory_orders"]);improvements.append(d["audit"]["reliability_gain_ppm"])
            agreements.extend(x for x in d["audit"]["negotiations"] if x["accepted"])
            assert not supplier_failures(s,CONFIG.mapping("supply_chain","strategic_policy"),20000000)
    assert max(inventories)>0 and max(improvements)>0 and agreements
    assert all(decode(s)["inventory_orders"]==decode(s)["debt_cents"]==0 for s in state.supply_chain.suppliers)

def test_bargaining_rejects_low_bid_and_buyer_underorder_terminates():
    env=MarketEnv(CONFIG);state=env.reset(episode_id="bid",max_rounds=5)
    s=state.supply_chain.suppliers[0];policy=CONFIG.mapping("supply_chain","strategic_policy")
    a=replace(build_rule_action(CONFIG,state,"company_A"),primary_supplier_id=s.supplier_id,contract_quantity_orders=1000,contract_duration_rounds=3,contract_bid_cents=1)
    contracts,prices,events=negotiated_book(s,{"company_A":a},1,5,policy)
    assert not contracts and events[0]["counter_offer_cents"]>1
    a=replace(a,contract_bid_cents=s.unit_price_cents)
    contracts,prices,events=negotiated_book(s,{"company_A":a},1,5,policy)
    assert contracts and events[0]["accepted"]
    result,_=settle_supplier(s,{"company_A":500},prices,contracts,events,policy,1,5,_allocate_integer)
    assert decode(result)["contracts"][0]["end_round"]==1
    assert decode(result)["audit"]["contract_outcomes"][0]["buyer_shortfall"]

def test_finite_credit_and_maturity_bankruptcy_close_both_sides():
    env=MarketEnv(CONFIG);state=env.reset(episode_id="credit",max_rounds=5)
    s=state.supply_chain.suppliers[0];policy=dict(CONFIG.mapping("supply_chain","strategic_policy"))
    # A small, consistent initial supplier endowment stresses financing.
    s=replace(s,account=replace(s.account,opening_cash_cents=0,cash_cents=0),round_requested_orders=6000,round_sales_orders=6000)
    result,_=settle_supplier(s,{"company_A":5000},{"company_A":1800},[],[],policy,1,5,_allocate_integer)
    d=decode(result);assert d["audit"]["borrowed_cents"]>0
    assert not supplier_failures(result,policy,0)
    # External shock ends demand; force the published maturity on the next round.
    d["debt_due_round"]=2
    result=replace(result,strategic_ledger=encode(d))
    policy["fixed_overhead_cents"]=10000000
    result,_=settle_supplier(result,{}, {},[],[],policy,2,1,_allocate_integer)
    d=decode(result)
    assert d["debt_cents"]==0 and d["inventory_orders"]==0
    assert not supplier_failures(result,policy,0)
    assert d["audit"]["repaid_cents"]+d["audit"]["defaulted_cents"]>0
    assert d["bankrupt"] and d["audit"]["defaulted_cents"]>0

def test_bankruptcy_without_credit_never_allocates_again():
    env=MarketEnv(CONFIG);state=env.reset(episode_id="exit",max_rounds=5)
    s=state.supply_chain.suppliers[0];p=dict(CONFIG.mapping("supply_chain","strategic_policy"))
    s=replace(s,account=replace(s.account,opening_cash_cents=0,cash_cents=0))
    r,_=settle_supplier(s,{}, {},[],[],p,1,5,_allocate_integer)
    assert decode(r)["bankrupt"] and not supplier_failures(r,p,0)
    r,allocation=settle_supplier(r,{"company_A":1000},{"company_A":1800},[],[],p,2,4,_allocate_integer)
    assert allocation["company_A"]==0 and r.round_sales_orders==0

def test_public_supplier_reports_support_honest_forecast_and_hide_cash_bids():
    from game_theory_agent.agents.observation import ObservationBuilder
    from game_theory_agent.agents.personas import PersonaRegistry
    from game_theory_agent.strategic_reliability import build_public_forecast_state
    env=MarketEnv(CONFIG);state=env.reset(episode_id="forecast22",episode_seed=220111,max_rounds=20,cooperation_mode="combined_v1")
    for _ in range(12):
        state=env.step(f"{state.episode_id}:{state.round}:{state.state_version}",{cid:build_rule_action(CONFIG,state,cid) for cid in state.company_ids}).state_after
        observation=ObservationBuilder().build(state,"company_A","public")
        for supplier in observation["market"]["supply_chain"]["suppliers"].values():
            assert "account" not in supplier and "strategic_ledger" not in supplier
            assert "opening_cash_cents" not in supplier["strategic_public"]["audit"]
            assert "negotiations" not in supplier["strategic_public"]["audit"]
        forecast,record=build_public_forecast_state(config=CONFIG,observation=observation,company_id="company_A",persona_profile=PersonaRegistry.from_market_config(CONFIG).get("balanced_v1"))
        assert not record.uses_authoritative_hidden_market_state
        shadow=MarketEnv(CONFIG);shadow.load_state(forecast)
        shadow.step(f"{forecast.episode_id}:{forecast.round}:{forecast.state_version}",{cid:build_rule_action(CONFIG,forecast,cid) for cid in forecast.company_ids})
