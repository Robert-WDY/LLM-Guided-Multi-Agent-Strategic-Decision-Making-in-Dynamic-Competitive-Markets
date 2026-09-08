"""Feasible four-role policies and information boundaries for joint experiments."""
from dataclasses import replace
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.decisioning import resolve_action_request
from .government_strategy import OPTIONS as GOVERNMENT_OPTIONS
from .supplier_strategy import decode

COMPANY_OPTIONS=("balanced","value","margin","resilience")
CONSUMER_OPTIONS=("balanced","value","service","cautious")
SUPPLIER_OPTIONS=("balanced","inventory","reliability","no_credit")


def actor_ids(state):
    return (*state.company_ids,*state.supply_chain.supplier_ids,"consumers","government")


def options(actor):
    return COMPANY_OPTIONS if actor.startswith("company_") else CONSUMER_OPTIONS if actor=="consumers" else GOVERNMENT_OPTIONS if actor=="government" else SUPPLIER_OPTIONS


def company_action(config,state,cid,option):
    if option not in COMPANY_OPTIONS:raise ValueError("unknown company portfolio")
    base=build_rule_action(config,state,cid);request=base.to_dict()
    if cid not in state.strategic_market.active_company_ids:
        return base
    if option=="value":request["price_cents"]=base.price_cents*950000//1000000
    elif option=="margin":request["price_cents"]=base.price_cents*1050000//1000000
    elif option=="resilience":
        request["resilience_budget_cents"]=min(200000,max(0,state.company(cid).financial.cash_balance_cents//20)) if state.rounds_remaining>1 else 0
    return resolve_action_request(config,state,cid,request,source="four-actor-portfolio",
                                  action_id=f"portfolio:{state.round}:{cid}:{option}").action


def actions_for(config,state,choices):
    return {cid:company_action(config,state,cid,choices.get(cid,"balanced")) for cid in state.company_ids}


def observation(state,actor):
    # Explicit allowlist: each prompt receives only its own financial state and public settled market data.
    common=dict(round=state.round,remaining=state.rounds_remaining,demand=state.market.realized_demand_orders,
                stockout=state.market.lost_after_stockout_orders,prices={c.company_id:c.commercial.price_cents for c in state.companies})
    if actor in state.company_ids:
        c=state.company(actor)
        common.update(cash=c.financial.cash_balance_cents,profit=c.financial.round_profit_cents,
                      sales=c.commercial.sales_orders,capacity=c.operations.base_capacity_orders,resilience=c.risk.resilience_ppm)
    elif actor in state.supply_chain.supplier_ids:
        s=state.supply_chain.supplier(actor);d=decode(s)
        common.update(cash=s.account.cash_cents,profit=s.round_profit_cents,sales=s.round_sales_orders,
                      orders=s.round_requested_orders,inventory=d["inventory_orders"],debt=d["debt_cents"],reliability=s.reliability_ppm)
    elif actor=="government":
        common.update(cash=state.government.cash_cents,hhi=state.strategic_market.hhi_ppm,
                      resilience={c.company_id:c.risk.resilience_ppm for c in state.companies})
    elif actor=="consumers":
        common.update(previous_spending=sum(d.spending_cents for d in state.consumer_decisions),
                      previous_refunds=sum(d.refund_cents for d in state.consumer_decisions),
                      outside=state.market.no_purchase_orders,price_anchor=state.market.price_anchor_cents)
    else:raise ValueError("unknown actor")
    return common


def descriptions(actor):
    descriptions={"balanced":"baseline portfolio","value":"prioritize lower prices","margin":"raise own sale price 5%",
        "resilience":"invest in resilience (company) or reimburse actual investment (government)",
        "service":"prioritize service quality","cautious":"increase outside-option preference; budget unchanged",
        "inventory":"double target inventory","reliability":"double conditional reliability budget",
        "no_credit":"refuse new borrowing","reserve":"preserve fiscal budget","enforce":"fund inspections and double assessed fines",
        "consumer":"post-purchase rebates prioritized by lower budgets"}
    return [dict(id=o,effect=descriptions[o]) for o in options(actor)]


def objective(actor):
    if actor.startswith("company_"):return "Maximize remaining own profit while staying solvent."
    if actor=="government":return "Maximize remaining total welfare within own budget; transfers alone create no welfare."
    if actor=="consumers":return "Choose a shopping policy to maximize consumer surplus within fixed cohort budgets."
    return "Maximize remaining own operating profit while staying solvent and fulfilling contracts."


def reward(state,actor):
    if actor in state.company_ids:return state.company(actor).financial.round_profit_cents
    if actor in state.supply_chain.supplier_ids:return state.supply_chain.supplier(actor).round_profit_cents
    if actor=="consumers":return state.welfare_accounting.round_consumer_surplus_cents
    return state.welfare_accounting.round_total_economic_welfare_cents


def consumer_coefficients(coefficients,option):
    c=dict(coefficients)
    if option=="value":c["price"]=c["price"]*3//2
    if option=="service":c["service"]=c["service"]*3//2
    return c
