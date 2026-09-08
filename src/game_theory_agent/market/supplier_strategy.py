"""Inventory, bilateral supply reservations and finite-credit supplier accounts.

Contracts fix a unit price and priority capacity for a bounded term. Purchases
remain prepaid; a buyer ordering below its reservation terminates that contract.
Disruption shortfalls are recorded, never silently counted as fulfilled.
"""
import json
from dataclasses import replace
from .models import SupplierAccount
from .protocols import sha256_hash

PPM=1_000_000
VERSION="supplier-contract-inventory-credit-v1.0.0"

def decode(s):
    return json.loads(s.strategic_ledger) if s.strategic_ledger else {}

def encode(d):
    return json.dumps(d,ensure_ascii=False,sort_keys=True,separators=(",",":"))

def initial(policy):
    return encode(dict(version=VERSION,inventory_orders=0,debt_cents=0,debt_due_round=0,
        lender_cash_cents=policy["credit_limit_cents"],cumulative_lender_profit_cents=0,
        bankrupt=False,contracts=[],audit=None))

def negotiated_book(s, actions, round_number, rounds_remaining, policy):
    old=decode(s)
    contracts=[dict(c) for c in old.get("contracts",[]) if c["end_round"]>=round_number]
    events=[];prices={cid:s.unit_price_cents for cid in actions}
    for c in contracts: prices[c["company_id"]]=c["unit_price_cents"]
    available=max(0,s.base_capacity_orders-sum(c["quantity_orders"] for c in contracts))
    for cid,a in sorted(actions.items()):
        if old.get("bankrupt") or a.primary_supplier_id!=s.supplier_id or not a.contract_duration_rounds or any(c["company_id"]==cid for c in contracts):
            continue
        quantity=min(a.contract_quantity_orders or 0,available)
        ask=max((s.unit_cost_cents*(PPM+policy["minimum_contract_margin_ppm"])+PPM-1)//PPM,
                s.unit_price_cents*(PPM-policy["contract_discount_ppm"])//PPM)
        bid=a.contract_bid_cents or 0
        accepted=quantity>0 and bid>=ask and rounds_remaining>1
        event=dict(company_id=cid,bid_cents=bid,counter_offer_cents=ask,accepted=accepted,quantity_orders=quantity)
        if accepted:
            price=min(s.unit_price_cents,(ask+bid)//2)
            c=dict(company_id=cid,start_round=round_number,end_round=round_number+min(a.contract_duration_rounds,rounds_remaining)-1,
                   quantity_orders=quantity,unit_price_cents=price,terms="prepaid_priority_reservation_v1")
            contracts.append(c);available-=quantity;prices[cid]=price
            event["agreed_unit_price_cents"]=price
        events.append(event)
    return contracts,prices,events

def settle_supplier(s, demands, prices, contracts, negotiations, policy, round_number, rounds_remaining, allocate, option="balanced"):
    if option not in {"balanced","inventory","reliability","no_credit"}:raise ValueError("unknown supplier option")
    policy=dict(policy)
    if option=="inventory":policy["inventory_target_ppm"]=min(PPM,policy["inventory_target_ppm"]*2)
    if option=="reliability":policy["reliability_budget_cents"]*=2
    old=decode(s);cash0=s.account.cash_cents;cash=cash0;cost=s.unit_cost_cents
    inv0=old["inventory_orders"];debt0=old["debt_cents"];debt=debt0
    lender0=old["lender_cash_cents"];lender=lender0
    terminal=rounds_remaining<=1;bankrupt=old["bankrupt"]
    spoil=(inv0*policy["inventory_spoilage_ppm"]+PPM-1)//PPM if inv0 else 0
    inv=inv0-spoil;production_capacity=max(0,s.available_capacity_orders-inv0)
    supply=0 if bankrupt else production_capacity+inv
    priority={cid:min(q,next((c["quantity_orders"] for c in contracts if c["company_id"]==cid),0)) for cid,q in demands.items()}
    alloc=allocate(min(supply,sum(priority.values())),priority)
    remaining={cid:q-alloc.get(cid,0) for cid,q in demands.items()}
    spot=allocate(min(supply-sum(alloc.values()),sum(remaining.values())),remaining)
    alloc={cid:alloc.get(cid,0)+spot.get(cid,0) for cid in demands}
    sold=sum(alloc.values());receipts=sum(prices[cid]*q for cid,q in alloc.items());cash+=receipts
    # No forecast reads another actor's private state or future disturbances.
    target=0 if terminal or bankrupt else min(policy["max_inventory_orders"],(s.round_requested_orders or 0)*policy["inventory_target_ppm"]//PPM)
    production_need=max(0,sold-inv)
    intended=min(production_capacity,max(production_need,sold+target-inv))
    overhead=0 if bankrupt else policy["fixed_overhead_cents"]
    loan=0
    shortage=max(0,intended*cost+overhead+policy["reserve_cash_cents"]-cash)
    expected_margin=max(0,s.unit_price_cents-cost)*(s.round_sales_orders or sold)*max(0,rounds_remaining-1)
    if option!="no_credit" and not terminal and not bankrupt and expected_margin>=shortage*(PPM+policy["interest_ppm"]*policy["loan_term_rounds"])//PPM:
        loan=min(shortage,max(0,policy["credit_limit_cents"]-debt),lender)
    cash+=loan;debt+=loan;lender-=loan
    due=old["debt_due_round"] if debt0 else (round_number+policy["loan_term_rounds"] if loan else 0)
    produced=max(production_need,min(intended,max(0,cash-overhead)//cost))
    production=produced*cost;cash-=production
    if cash<0: raise ValueError("prepaid production lacks cash")
    overhead_paid=min(cash,overhead);cash-=overhead_paid
    inv=inv+produced-sold
    improvement=0;reliability_spend=0
    lost_margin=(s.round_requested_orders or 0)*max(0,s.unit_price_cents-cost)*max(0,rounds_remaining-1)
    if not terminal and not bankrupt:
        proposed=min(policy["reliability_budget_cents"],max(0,cash-policy["reserve_cash_cents"]))
        gain=min(policy["max_reliability_ppm"]-s.reliability_ppm,proposed//policy["reliability_cost_per_ppm"])
        if gain>0 and lost_margin*gain//PPM>=gain*policy["reliability_cost_per_ppm"]:
            improvement=gain;reliability_spend=gain*policy["reliability_cost_per_ppm"];cash-=reliability_spend
    interest_due=debt*policy["interest_ppm"]//PPM
    interest=min(cash,interest_due);cash-=interest;lender+=interest
    mature=bool(debt) and (terminal or round_number>=due)
    repay=min(debt,cash if mature else max(0,cash-policy["reserve_cash_cents"])//2)
    debt-=repay;cash-=repay;lender+=repay
    failed=overhead_paid<overhead or interest<interest_due or (mature and debt>0)
    liquidated=liquidation_receipts=liquidation_loss=default=0
    if terminal or failed:
        liquidated=inv;liquidation_receipts=inv*cost*policy["liquidation_recovery_ppm"]//PPM
        liquidation_loss=inv*cost-liquidation_receipts;cash+=liquidation_receipts;inv=0
        more=min(cash,debt);repay+=more;debt-=more;cash-=more;lender+=more
        if failed or debt:
            bankrupt=True;default=debt;debt=0
    lender_profit=interest-default
    profit=receipts-sold*cost-spoil*cost-overhead_paid-reliability_spend-interest-liquidation_loss+default
    outcomes=[]
    for c in contracts:
        cid=c["company_id"];requested=demands.get(cid,0);delivered=alloc.get(cid,0)
        breach=requested<c["quantity_orders"]
        outcomes.append(dict(company_id=cid,reserved_orders=c["quantity_orders"],requested_orders=requested,delivered_orders=delivered,
                             buyer_shortfall=breach,supplier_shortfall_orders=max(0,min(requested,c["quantity_orders"])-delivered)))
        if breach or bankrupt:c["end_round"]=round_number
    audit=dict(round=round_number,opening_cash_cents=cash0,opening_inventory_orders=inv0,opening_debt_cents=debt0,
        opening_lender_cash_cents=lender0,production_capacity_orders=production_capacity,spoilage_orders=spoil,produced_orders=produced,
        sold_orders=sold,unit_cost_cents=cost,settlement_prices_cents=prices,receipts_cents=receipts,production_cost_cents=production,
        overhead_cents=overhead_paid,reliability_spend_cents=reliability_spend,reliability_gain_ppm=improvement,borrowed_cents=loan,
        interest_cents=interest,repaid_cents=repay,defaulted_cents=default,liquidated_orders=liquidated,
        liquidation_receipts_cents=liquidation_receipts,liquidation_loss_cents=liquidation_loss,lender_profit_cents=lender_profit,
        negotiations=negotiations,contract_outcomes=outcomes,option=option,policy="reserve_inventory_improve_reliability_bounded_credit")
    audit["hash"]=sha256_hash(audit)
    ledger=dict(version=VERSION,inventory_orders=inv,debt_cents=debt,debt_due_round=due if debt else 0,lender_cash_cents=lender,
        cumulative_lender_profit_cents=old["cumulative_lender_profit_cents"]+lender_profit,bankrupt=bankrupt,contracts=contracts,audit=audit)
    return replace(s,strategic_ledger=encode(ledger),reliability_ppm=s.reliability_ppm+improvement,
        round_sales_orders=sold,round_requested_orders=sum(demands.values()),round_profit_cents=profit,
        cumulative_profit_cents=s.cumulative_profit_cents+profit,last_settled_unit_price_cents=s.unit_price_cents,
        account=SupplierAccount(opening_cash_cents=cash0,cash_cents=cash,receipts_cents=receipts,production_cost_cents=production,settlement_price_cents=s.unit_price_cents)),alloc

def supplier_failures(s, policy, initial_cash):
    d=decode(s);a=d.get("audit");fail=[]
    if min(d["inventory_orders"],d["debt_cents"],d["lender_cash_cents"],s.account.cash_cents)<0:fail.append("supplier negative strategic asset")
    if s.account.cash_cents != initial_cash+s.cumulative_profit_cents+d["debt_cents"]-d["inventory_orders"]*s.unit_cost_cents:
        fail.append("supplier assets liability and cumulative profit disagree")
    if d["lender_cash_cents"]+d["debt_cents"]!=policy["credit_limit_cents"]+d["cumulative_lender_profit_cents"]:
        fail.append("finite creditor balance does not close")
    if not a:return fail
    expected_profit=a["receipts_cents"]-a["sold_orders"]*a["unit_cost_cents"]-a["spoilage_orders"]*a["unit_cost_cents"]-a["overhead_cents"]-a["reliability_spend_cents"]-a["interest_cents"]-a["liquidation_loss_cents"]+a["defaulted_cents"]-(s.account.investment_cents or 0)
    if expected_profit!=s.round_profit_cents:fail.append("strategic supplier realized profit does not close")
    if a["production_cost_cents"]!=a["produced_orders"]*s.unit_cost_cents or not 0<=a["produced_orders"]<=a["production_capacity_orders"]:fail.append("supplier production budget or capacity mismatch")
    if a["opening_lender_cash_cents"]-a["borrowed_cents"]+a["repaid_cents"]+a["interest_cents"]!=d["lender_cash_cents"]:fail.append("creditor round cash does not close")
    if a["lender_profit_cents"]!=a["interest_cents"]-a["defaulted_cents"]:fail.append("creditor profit misattributed")
    if a["hash"]!=sha256_hash({k:v for k,v in a.items() if k!="hash"}):fail.append("supplier strategy audit hash differs")
    expected=a["opening_cash_cents"]+a["receipts_cents"]-a["production_cost_cents"]-a["overhead_cents"]-a["reliability_spend_cents"]-a["interest_cents"]+a["borrowed_cents"]-a["repaid_cents"]+a["liquidation_receipts_cents"]-(s.account.investment_cents or 0)
    if expected!=s.account.cash_cents:fail.append("supplier strategic cash flow does not close")
    if a["opening_inventory_orders"]-a["spoilage_orders"]+a["produced_orders"]-a["sold_orders"]-a["liquidated_orders"]!=d["inventory_orders"]:
        fail.append("supplier physical inventory does not close")
    if a["opening_debt_cents"]+a["borrowed_cents"]-a["repaid_cents"]-a["defaulted_cents"]!=d["debt_cents"]:
        fail.append("supplier debt does not close")
    if s.account.receipts_cents!=a["receipts_cents"] or s.account.production_cost_cents!=a["production_cost_cents"]:fail.append("supplier strategic account attribution differs")
    return fail
