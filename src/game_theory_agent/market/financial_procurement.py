"""Own-finance gate around capacity procurement; fallback is explicit 50:50."""
from dataclasses import asdict, dataclass, replace

from .models import CompanyAction, MarketState
from .procurement_policy import ProcurementPlan, ProcurementView, procurement_view, select_procurement
from .protocols import sha256_hash

PPM=1_000_000
VERSION="financial-procurement-guard-v1.0.0"


@dataclass(frozen=True)
class ProcurementFinance:
    cash_cents: int
    current_price_cents: int
    last_actual_unit_cost_cents: int
    last_input_price_cents: int
    baseline_input_price_cents: int
    input_cost_share_ppm: int
    fixed_spend_cents: int
    overhead_cents: int
    fulfillment_unit_cost_cents: int
    sales_ceiling_orders: int
    quote_step_ppm: int
    quote_caps_cents: tuple[int,int]
    reserve_rounds: int=3
    cost_stress_ppm: int=1050000


def finance_view(config,state:MarketState,company_id:str,action:CompanyAction)->ProcurementFinance:
    own=state.company(company_id);sc=config.mapping("supply_chain")
    pricing=sc.get("autonomous_pricing",{})
    quotes=sorted(state.supply_chain.suppliers,key=lambda s:s.supplier_id)
    return ProcurementFinance(
        cash_cents=own.financial.cash_balance_cents,current_price_cents=action.price_cents,
        last_actual_unit_cost_cents=own.operations.actual_unit_cost_cents,
        last_input_price_cents=own.operations.procurement_unit_input_price_cents or sc["baseline_input_price_cents"],
        baseline_input_price_cents=sc["baseline_input_price_cents"],input_cost_share_ppm=sc["downstream_input_cost_share_ppm"],
        fixed_spend_cents=action.fixed_spend_cents,overhead_cents=config.integer("operating_costs","fixed_overhead_cents"),
        fulfillment_unit_cost_cents=config.integer("operating_costs","fulfillment_cost_per_order_cents"),
        sales_ceiling_orders=min(own.operations.base_capacity_orders,own.commercial.potential_demand_orders) if state.state_version else own.operations.base_capacity_orders,
        quote_step_ppm=pricing.get("step_ppm",0) if pricing.get("enabled") else 0,
        quote_caps_cents=tuple(sc["suppliers"][s.supplier_id]["unit_price_cents"]*pricing.get("max_base_price_ppm",PPM)//PPM if pricing.get("enabled") else s.unit_price_cents for s in quotes),
    )


def project(view:ProcurementView,finance:ProcurementFinance,share:int,stress:bool):
    requested=(view.own_requested_orders*share+PPM//2)//PPM
    quantities=[min(requested,view.quotes[0].available_capacity_orders//view.active_buyer_count),min(view.own_requested_orders-requested,view.quotes[1].available_capacity_orders//view.active_buyer_count)]
    filled=sum(quantities)
    prices=[min(finance.quote_caps_cents[i],(q.unit_price_cents*(PPM+finance.quote_step_ppm)+PPM-1)//PPM) if stress else q.unit_price_cents for i,q in enumerate(view.quotes)]
    input_cost=sum(q*p for q,p in zip(quantities,prices))
    weighted=(input_cost+filled-1)//filled if filled else finance.baseline_input_price_cents
    old_multiplier=PPM-finance.input_cost_share_ppm+finance.input_cost_share_ppm*finance.last_input_price_cents//finance.baseline_input_price_cents
    new_multiplier=PPM-finance.input_cost_share_ppm+finance.input_cost_share_ppm*weighted//finance.baseline_input_price_cents
    unit_cost=(finance.last_actual_unit_cost_cents*new_multiplier+old_multiplier-1)//old_multiplier
    if stress:unit_cost=(unit_cost*finance.cost_stress_ppm+PPM-1)//PPM
    margin=finance.current_price_cents-unit_cost-finance.fulfillment_unit_cost_cents
    sold=min(filled,finance.sales_ceiling_orders)
    profit=sold*margin-finance.fixed_spend_cents-finance.overhead_cents
    return {"fulfilled_orders":filled,"sales_orders":sold,"weighted_input_price_cents":weighted,"unit_contribution_cents":margin,"profit_cents":profit,"cash_after_cents":finance.cash_cents+profit,"input_cost_cents":input_cost}


def guarded_procurement(view:ProcurementView,finance:ProcurementFinance):
    proposal=select_procurement(view,"capacity")
    proposed=[project(view,finance,proposal.primary_share_ppm,stress) for stress in (False,True)]
    baseline=[project(view,finance,500000,stress) for stress in (False,True)]
    reasons=[]
    if proposal.primary_share_ppm==500000:reasons.append("proposal_equals_fallback")
    if any(p["unit_contribution_cents"]<=0 for p in proposed):reasons.append("nonpositive_unit_contribution")
    if any(p["profit_cents"]<0 for p in proposed):reasons.append("fixed_cost_coverage_insufficient")
    if any(p["cash_after_cents"]<finance.reserve_rounds*finance.overhead_cents for p in proposed):reasons.append("cash_reserve_insufficient")
    if any(p["profit_cents"]<b["profit_cents"] for p,b in zip(proposed,baseline)):reasons.append("profit_below_diverse_fallback")
    released=not reasons
    selected=proposed[0] if released else baseline[0]
    plan=replace(proposal,policy_version=VERSION,mode="financial_guarded",primary_share_ppm=proposal.primary_share_ppm if released else 500000,expected_fulfilled_orders=selected["fulfilled_orders"],expected_input_cost_cents=selected["input_cost_cents"],retained_previous=False)
    audit={"policy_version":VERSION,"procurement_view":asdict(view),"own_finance":asdict(finance),"released":released,"reasons":reasons,"proposed_share_ppm":proposal.primary_share_ppm,"selected_share_ppm":plan.primary_share_ppm,"projections":proposed,"fallback_projections":baseline,"limitations":"Flat own demand and current capacity estimates; stress assumes one quote increase and 5% unit-cost shock, not a calibrated probability or solvency guarantee."}
    audit["audit_hash"]=sha256_hash(audit)
    return plan,audit
