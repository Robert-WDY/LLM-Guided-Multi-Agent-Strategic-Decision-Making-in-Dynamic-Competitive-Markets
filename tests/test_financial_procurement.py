from dataclasses import replace

from game_theory_agent.market.financial_procurement import ProcurementFinance, guarded_procurement
from game_theory_agent.market.procurement_policy import ProcurementView, PublicSupplierQuote


def inputs():
    view=ProcurementView(5,3500,4,(PublicSupplierQuote("a",1800,11000),PublicSupplierQuote("b",2400,11000)),500000)
    finance=ProcurementFinance(30000000,11000,6000,2100,2100,400000,1000000,3500000,1200,3500,50000,(2700,3600))
    return view,finance


def test_positive_margin_cash_and_stress_can_release():
    view,finance=inputs();plan,audit=guarded_procurement(view,finance)
    assert audit["released"] and plan.primary_share_ppm!=500000
    assert all(p["profit_cents"]>=0 for p in audit["projections"])


def test_high_input_cost_rejects_volume_only_proposal():
    view,finance=inputs();finance=replace(finance,last_actual_unit_cost_cents=10500)
    plan,audit=guarded_procurement(view,finance)
    assert not audit["released"] and plan.primary_share_ppm==500000
    assert "nonpositive_unit_contribution" in audit["reasons"]


def test_cash_reserve_and_fixed_costs_are_independent_guards():
    view,finance=inputs()
    plan,audit=guarded_procurement(view,replace(finance,cash_cents=1000000))
    assert not audit["released"] and "cash_reserve_insufficient" in audit["reasons"]
    plan,audit=guarded_procurement(view,replace(finance,fixed_spend_cents=25000000))
    assert not audit["released"] and "fixed_cost_coverage_insufficient" in audit["reasons"]


def test_quote_feedback_stress_and_finance_are_bound_into_audit():
    view,finance=inputs();_,a=guarded_procurement(view,finance)
    _,b=guarded_procurement(view,replace(finance,quote_step_ppm=250000))
    assert b["projections"][1]["weighted_input_price_cents"]>a["projections"][1]["weighted_input_price_cents"]
    assert a["audit_hash"]!=b["audit_hash"]
