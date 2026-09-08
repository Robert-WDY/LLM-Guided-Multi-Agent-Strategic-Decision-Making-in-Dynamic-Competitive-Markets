"""Explicit legal-input reconstruction, never reads the authoritative state."""
from copy import deepcopy
from dataclasses import replace
from game_theory_agent.market.models import SupplyChainState, SupplierAccount, MaterialSettlement
from game_theory_agent.market.autonomous_market import investment_decision
from game_theory_agent.market.supplier_strategy import initial as initial_supplier_ledger
from game_theory_agent.market.protocols import sha256_hash
import json


def reconstruct_strategic(s,sc,autonomous,state_version,terminal,remaining_rounds):
    report=deepcopy(s.pop("strategic_public",json.loads(initial_supplier_ledger(sc["strategic_policy"]))))
    s.pop("bankrupt",None)
    cost=sc["suppliers"][s["supplier_id"]]["unit_cost_cents"]
    s["unit_cost_cents"]=cost
    cash=sc["transaction_accounting"]["initial_supplier_cash_cents"]+s["cumulative_profit_cents"]+report["debt_cents"]-report["inventory_orders"]*cost
    a=report.get("audit")
    investment=0
    if a:
        a["unit_cost_cents"]=cost
        investment=a["receipts_cents"]-a["sold_orders"]*cost-a["spoilage_orders"]*cost-a["overhead_cents"]-a["reliability_spend_cents"]-a["interest_cents"]-a["liquidation_loss_cents"]+a["defaulted_cents"]-s["round_profit_cents"]
        a["opening_cash_cents"]=cash-a["receipts_cents"]+a["production_cost_cents"]+a["overhead_cents"]+a["reliability_spend_cents"]+a["interest_cents"]-a["borrowed_cents"]+a["repaid_cents"]-a["liquidation_receipts_cents"]+investment
        a["opening_lender_cash_cents"]=report["lender_cash_cents"]+a["borrowed_cents"]-a["repaid_cents"]-a["interest_cents"]
        a["negotiations"]=[]  # rejected/offered bids are unobserved
        a["hash"]=sha256_hash(a)
    s["account"]=SupplierAccount(a["opening_cash_cents"] if a else cash,cash,a["receipts_cents"] if a else 0,a["production_cost_cents"] if a else 0,s.get("last_settled_unit_price_cents",s["unit_price_cents"]) if a else 0,investment if a else None).to_dict()
    s["strategic_ledger"]=report
    s["round_requested_orders"]=0
    if state_version:
        policy=autonomous["supplier_investment"];price=s["account"]["settlement_price_cents"]
        available=(s.get("quote_decision") or {}).get("observed_capacity_orders",s["available_capacity_orders"]) if not terminal else s["available_capacity_orders"]
        old_capacity=s["base_capacity_orders"]-investment//policy["unit_capacity_cost_cents"]
        requested=old_capacity+investment//policy["unit_capacity_cost_cents"]
        s["round_requested_orders"]=requested
        d=investment_decision(sales=s["round_sales_orders"],available=available,base_capacity=old_capacity,cash=cash+investment,
                             observed_round=state_version,terminal=terminal,policy=policy,requested=requested,unit_margin=price-cost,remaining_rounds=remaining_rounds)
        if d.investment_cents!=investment:raise ValueError("public strategic supplier investment does not reconcile")
        s["investment_decision"]=d.to_dict()


def forecast_accounting(*, config, public_supply, companies, company_id, strategic, state_version, terminal, remaining_rounds):
    data=deepcopy(public_supply);sc=config.mapping("supply_chain")
    autonomous=config.data.get("autonomous_market")
    for sid,s in data["suppliers"].items():
        if sc.get("strategic_policy"):
            reconstruct_strategic(s,sc,autonomous,state_version,terminal,remaining_rounds)
            continue
        s["unit_cost_cents"]=sc["suppliers"][sid]["unit_cost_cents"]
        price=s.get("last_settled_unit_price_cents",s["unit_price_cents"])
        revenue=s["round_sales_orders"]*price
        production=s["round_sales_orders"]*s["unit_cost_cents"]
        investment=revenue-production-s["round_profit_cents"] if autonomous and state_version else None
        cash=sc["transaction_accounting"]["initial_supplier_cash_cents"]+s["cumulative_profit_cents"]
        s["account"]=SupplierAccount(cash-s["round_profit_cents"],cash,revenue,production,price,investment).to_dict()
        if sc.get("track_supplier_demand"):s["round_requested_orders"]=0
        if autonomous and state_version:
            policy=autonomous["supplier_investment"]
            available=s.get("quote_decision",{}).get("observed_capacity_orders",s["available_capacity_orders"]) if not terminal else s["available_capacity_orders"]
            old_capacity=s["base_capacity_orders"]-investment//policy["unit_capacity_cost_cents"]
            requested=old_capacity+investment//policy["unit_capacity_cost_cents"] if policy.get("payback_guard_ppm") else None
            if requested is not None:s["round_requested_orders"]=requested
            d=investment_decision(sales=s["round_sales_orders"],available=available,
                base_capacity=old_capacity,
                cash=cash+investment,observed_round=state_version,terminal=terminal,policy=policy,
                requested=requested,unit_margin=price-s["unit_cost_cents"],remaining_rounds=remaining_rounds)
            if d.investment_cents!=investment:raise ValueError("public supplier investment cannot be reconciled with declared priors")
            s["investment_decision"]=d.to_dict()
    by_company={c.company_id:c for c in companies};estimated=[]
    for cid,c in by_company.items():
        outcome=data["last_procurement_outcomes"].get(cid)
        if not outcome:
            if cid != company_id:c=replace(c,financial=replace(c.financial,round_material_payment_cents=0))
            estimated.append(c);continue
        payments=tuple(sorted((sid,qty*(data["suppliers"][sid]["strategic_ledger"]["audit"]["settlement_prices_cents"][cid] if sc.get("strategic_policy") else data["suppliers"][sid]["account"]["settlement_price_cents"])) for sid,qty in outcome["supplier_allocation_orders"].items()))
        paid=sum(value for _,value in payments)
        used=c.commercial.sales_orders-(c.commercial.mutual_aid_fulfilled_orders or 0)+(c.commercial.mutual_aid_provided_orders or 0)
        outcome["material"]=MaterialSettlement(paid,paid,payments,used,outcome["fulfilled_orders"]-used).to_dict()
        if cid != company_id:
            processing=c.operations.base_unit_cost_cents*(1000000-sc["downstream_input_cost_share_ppm"])//1000000
            transfer=sum(t.total_transfer_fee_cents for t in strategic.last_mutual_aid_transfers if t.recipient_company_id==cid) if strategic else 0
            c=replace(c,operations=replace(c.operations,actual_unit_cost_cents=processing),
                financial=replace(c.financial,round_material_payment_cents=paid,round_variable_cost_cents=paid+used*processing+transfer))
        estimated.append(c)
    return SupplyChainState.from_dict(data),tuple(estimated)
