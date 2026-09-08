from dataclasses import replace
from pathlib import Path

import pytest

from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.exceptions import ConfigError
from game_theory_agent.market.procurement_policy import PublicSupplierQuote, ProcurementView, select_procurement, procurement_view, apply_procurement, VERSION
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.persistence import CODEC

ROOT=Path(__file__).resolve().parents[1]


def view(first=11000,second=11000,prices=(1800,2400),previous=500000):
    return ProcurementView(1,3500,4,(PublicSupplierQuote("a",prices[0],first),PublicSupplierQuote("b",prices[1],second)),previous)


def test_capacity_aware_buyer_does_not_pile_into_cheapest_limited_supplier():
    plan=select_procurement(view())
    assert plan.expected_fulfilled_orders==3500
    assert 500000 < plan.primary_share_ppm < 1000000
    assert plan.expected_fulfilled_orders>11000//4


def test_current_outage_moves_orders_to_available_supplier():
    plan=select_procurement(view(first=0,second=20000))
    assert plan.primary_share_ppm==0 and plan.expected_fulfilled_orders==3500
    gradual=select_procurement(view(first=0,second=20000),"gradual")
    assert gradual.primary_share_ppm==250000
    assert gradual.expected_fulfilled_orders<plan.expected_fulfilled_orders


def test_hysteresis_and_zero_supply_keep_previous_allocation():
    assert select_procurement(view(first=0,second=0,previous=350000)).primary_share_ppm==350000
    assert select_procurement(view(prices=(1800,1800),previous=500000)).retained_previous


def test_price_reversal_changes_allocation_without_private_costs():
    a=select_procurement(view(prices=(1800,2400)))
    b=select_procurement(view(prices=(2600,1600)))
    assert b.primary_share_ppm<a.primary_share_ppm


@pytest.mark.parametrize("field,value",[("active_buyer_count",0),("own_requested_orders",-1),("previous_primary_share_ppm",1000001)])
def test_invalid_observation_fails_closed(field,value):
    with pytest.raises(ValueError):select_procurement(replace(view(),**{field:value}))


def test_other_buyers_private_cash_and_supplier_cost_never_enter_view():
    c=load_market_config(ROOT/"configs/market_v11_supplier.yaml");s=MarketEnv(c).reset(episode_id="public-only")
    original=procurement_view(s,"company_A")
    companies=tuple(replace(x,financial=replace(x.financial,cash_balance_cents=1)) if x.company_id!="company_A" else x for x in s.companies)
    supply=replace(s.supply_chain,suppliers=tuple(replace(x,unit_cost_cents=999999,reliability_ppm=1) for x in s.supply_chain.suppliers))
    assert procurement_view(replace(s,companies=companies,supply_chain=supply),"company_A")==original


def test_opt_in_rule_policy_and_restart_use_same_procurement():
    data=load_market_config(ROOT/"configs/market_v11_supplier.yaml").to_dict()
    data["rule_procurement"]={"enabled":True,"mode":"capacity","policy_version":VERSION}
    c=MarketConfig.from_mapping(data);env=MarketEnv(c);s=env.reset(episode_id="procurement-save",max_rounds=5)
    for _ in range(3):
        actions={i:build_rule_action(c,s,i) for i in s.company_ids}
        for i in s.strategic_market.active_company_ids:
            assert actions[i].primary_supplier_share_ppm==select_procurement(procurement_view(s,i)).primary_share_ppm
        s=env.step(f"{s.episode_id}:{s.round}:{s.state_version}",actions).state_after
    restored=CODEC.decode(CODEC.encode(s))
    assert build_rule_action(c,s,"company_A")==build_rule_action(c,restored,"company_A")
    assert s.supply_chain.last_procurement_outcomes


def test_config_rejects_unknown_policy():
    data=load_market_config(ROOT/"configs/market_v11_supplier.yaml").to_dict()
    data["rule_procurement"]={"enabled":True,"mode":"clairvoyant","policy_version":VERSION}
    with pytest.raises(ConfigError):MarketConfig.from_mapping(data)
