"""Budgeted independent rule decisions for the local four-actor market."""
from dataclasses import replace
from .models import GovernmentDecision, GovernmentState, SupplierInvestmentDecision

PPM = 1_000_000


def government_decision(*, cash, observed_round, stockout, demand, hhi, active_ids, policy):
    active_ids = tuple(sorted(active_ids))
    enabled = policy["enabled"] and bool(active_ids)
    stressed = stockout * PPM >= max(1, demand) * policy["stockout_threshold_ppm"]
    cases = min(policy["max_inspection_cases"], len(active_ids)//2,
                cash//policy["inspection_cost_per_case_cents"]) if enabled and hhi >= policy["concentration_threshold_ppm"] else 0
    inspection = cases * policy["inspection_cost_per_case_cents"]
    support = min(policy["max_support_cents"], (cash-inspection)*policy["support_cash_share_ppm"]//PPM) if enabled and stressed else 0
    split = tuple((cid, support//len(active_ids)+(index < support%len(active_ids))) for index,cid in enumerate(active_ids))
    return GovernmentDecision(observed_round, observed_round+1, cash, stockout, demand, hhi,
        active_ids, cases, inspection, policy["detection_boost_ppm"] if cases else 0, split,
        "market_stress_support" if support else "concentration_inspection" if cases else "preserve_budget")


def decide_government(state, policy, option=None):
    base = government_decision(cash=state.government.cash_cents, observed_round=state.state_version,
        stockout=state.market.lost_after_stockout_orders, demand=state.market.realized_demand_orders,
        hhi=state.strategic_market.hhi_ppm, active_ids=state.strategic_market.active_company_ids, policy=policy)
    if policy.get("strategic_policy"):
        from .government_strategy import plan, unpack
        return plan(base, policy, {c.company_id:c.risk.resilience_ppm for c in state.companies if c.company_id in base.active_company_ids},
                    unpack(state.government.policy_memory), state.rounds_remaining, option)
    return base


def settle_government(previous, decision, fines, support=None, rebates=0):
    actual = sum(v for _,v in decision.support_by_company_cents) if support is None else support
    change = fines - decision.inspection_cost_cents - actual - rebates
    return GovernmentState(previous.cash_cents+change, previous.cumulative_net_cents+change, fines, decision,
                           previous.policy_memory, rebates if decision.strategic_policy else None,
                           actual if decision.strategic_policy else None)


def investment_decision(*, sales, available, base_capacity, cash, observed_round, terminal, policy, requested=None, unit_margin=None, remaining_rounds=None):
    addition = 0
    reason = "preserve_capacity"
    if terminal: reason = "terminal_no_investment"
    elif not policy["enabled"]: reason = "disabled"
    elif available and sales * PPM >= available * policy["utilization_threshold_ppm"]:
        addition = max(0, min(policy["max_addition_orders"], policy["max_base_capacity_orders"]-base_capacity,
                             (cash-policy["reserve_cash_cents"])//policy["unit_capacity_cost_cents"]))
        reason = "expand_high_utilization" if addition else "cash_or_capacity_limit"
    guarded=policy.get("payback_guard_ppm") is not None
    if guarded:
        if requested is None or unit_margin is None or remaining_rounds is None:
            raise ValueError("payback guard requires own demand, margin and horizon")
        if addition:
            if available < base_capacity:
                addition=0;reason="temporary_disruption_not_capacity_shortage"
            elif requested <= base_capacity:
                addition=0;reason="no_excess_orders"
            elif max(0,unit_margin)*policy["margin_realization_ppm"]*remaining_rounds < policy["unit_capacity_cost_cents"]*policy["payback_guard_ppm"]:
                addition=0;reason="payback_horizon_insufficient"
            else:
                addition=min(addition,requested-base_capacity);reason="funded_excess_demand_payback"
    return SupplierInvestmentDecision(observed_round, observed_round+1, sales, available, cash,
        base_capacity, addition, addition*policy["unit_capacity_cost_cents"], reason,
        policy_version="demand-payback-investment-v1.0.0" if guarded else "cash-bounded-investment-v1.0.0",
        observed_requested_orders=requested if guarded else None,observed_unit_margin_cents=unit_margin if guarded else None,
        remaining_rounds=remaining_rounds if guarded else None)


def advance_investment(supply, policy, observed_round, terminal, remaining_rounds=None):
    suppliers = []; total_cost = 0
    for s in supply.suppliers:
        decision = investment_decision(sales=s.round_sales_orders, available=s.available_capacity_orders,
            base_capacity=s.base_capacity_orders, cash=s.account.cash_cents, observed_round=observed_round,
            terminal=terminal, policy=policy,requested=s.round_requested_orders,
            unit_margin=s.account.settlement_price_cents-s.unit_cost_cents,remaining_rounds=remaining_rounds)
        cost = decision.investment_cents; total_cost += cost
        suppliers.append(replace(s, base_capacity_orders=s.base_capacity_orders+decision.added_capacity_orders,
            account=replace(s.account, cash_cents=s.account.cash_cents-cost, investment_cents=cost),
            round_profit_cents=s.round_profit_cents-cost, cumulative_profit_cents=s.cumulative_profit_cents-cost,
            investment_decision=decision))
    return replace(supply, suppliers=tuple(suppliers),
        round_upstream_producer_surplus_cents=supply.round_upstream_producer_surplus_cents-total_cost,
        cumulative_upstream_producer_surplus_cents=supply.cumulative_upstream_producer_surplus_cents-total_cost)


def company_procurement(config, state, company_id, action):
    """Maximise conservative one-round surplus over public share/own order grid."""
    own = state.company(company_id); sc = config.mapping("supply_chain")
    suppliers = sorted(state.supply_chain.suppliers, key=lambda s:s.supplier_id)
    if len(suppliers) != 2: raise ValueError("local procurement requires two suppliers")
    a,b = suppliers; buyers = max(1,len(state.strategic_market.active_company_ids))
    cash = max(0,own.financial.cash_balance_cents-action.fixed_spend_cents-config.integer("operating_costs","fixed_overhead_cents"))
    unit_cost = (own.operations.actual_unit_cost_cents if state.state_version else
        own.operations.base_unit_cost_cents*(PPM-sc["downstream_input_cost_share_ppm"])//PPM)
    demand = min(own.operations.base_capacity_orders, own.commercial.potential_demand_orders if state.state_version else own.operations.base_capacity_orders)
    target = min(demand, cash//max(1,a.unit_price_cents,b.unit_price_cents))
    pricing = sc.get("autonomous_pricing", {})
    price_stress = pricing.get("step_ppm",0) if pricing.get("enabled") else 0
    processing_stress = (unit_cost*1050000+PPM-1)//PPM + config.integer("operating_costs","fulfillment_cost_per_order_cents")
    choices = []
    for quantity in sorted({0,target//2,target*3//4,target}):
        for share in range(0,PPM+1,50000):
            requested_a = (quantity*share+PPM//2)//PPM
            qa = min(requested_a,a.available_capacity_orders//buyers)
            qb = min(quantity-requested_a,b.available_capacity_orders//buyers)
            invoice = (qa*a.unit_price_cents+qb*b.unit_price_cents)*(PPM+price_stress)//PPM
            profit = (qa+qb)*(action.price_cents-processing_stress)-invoice
            choices.append((profit, -(abs(share-500000)), -quantity, share, quantity))
    best = max(choices)
    contract={}
    if sc.get("strategic_policy") and state.rounds_remaining>1:
        contract=dict(contract_quantity_orders=(best[4]*best[3])//PPM,contract_duration_rounds=min(3,state.rounds_remaining),
                      contract_bid_cents=a.unit_price_cents*970000//PPM)
    return replace(action, **contract, primary_supplier_id=a.supplier_id, backup_supplier_id=b.supplier_id,
                   primary_supplier_share_ppm=best[3], procurement_quantity_orders=best[4],
                   strategy_summary=(action.strategy_summary+f"; cash-material-procurement-v1: quantity={best[4]}, share={best[3]}, stress_contribution={best[0]}")[:500])


def actor_failures(state, config):
    failures = []; policy = config["autonomous_market"]; govt = state.government
    if govt is None: return ["government state missing"]
    gp = policy["government"]
    if govt.cash_cents < 0 or govt.cash_cents != gp["initial_cash_cents"]+govt.cumulative_net_cents:
        failures.append("government cumulative cash does not close")
    decision = govt.last_decision
    if state.state_version:
        if decision is None: return failures+["government audit missing"]
        expected = government_decision(cash=decision.opening_cash_cents, observed_round=decision.observed_round,
            stockout=decision.observed_stockout_orders,demand=decision.observed_demand_orders,hhi=decision.observed_hhi_ppm,
            active_ids=decision.active_company_ids,policy=gp)
        support = sum(v for _,v in decision.support_by_company_cents)
        consumer_rebate = 0
        if decision.strategic_policy:
            from .government_strategy import plan, unpack, matched_support, rebates
            audit = unpack(decision.strategic_policy)
            expected = plan(expected, gp, audit["observed_resilience"], audit["opening_memory"],
                            audit["remaining_rounds"], audit["option"] if audit["selection"]=="external" else None)
            supports = {cid:min(cap, govt.round_matched_support_cents or 0) for cid,cap in decision.support_by_company_cents}
            if state.last_joint_action and matched_support(decision, {a.agent_id:a for a in state.last_joint_action}) != supports:
                failures.append("government resilience grant exceeds verified expense")
            support = sum(supports.values())
            consumer_rebate = sum(d.government_rebate_cents or 0 for d in state.consumer_decisions)
            if rebates(decision, state.consumer_decisions) != state.consumer_decisions or consumer_rebate != govt.round_consumer_rebate_cents or support != govt.round_matched_support_cents:
                failures.append("government program settlement conflicts")
        if expected != decision or decision.applies_round != state.state_version:
            failures.append("government decision audit conflicts")
        if govt.cash_cents != decision.opening_cash_cents+govt.round_fines_cents-decision.inspection_cost_cents-support-consumer_rebate:
            failures.append("government round cash does not close")
        if sum(c.financial.round_government_support_cents or 0 for c in state.companies) != support:
            failures.append("government support transfer does not close")
        if not decision.strategic_policy: supports=dict(decision.support_by_company_cents)
        if any((c.financial.round_government_support_cents or 0) != supports.get(c.company_id,0) for c in state.companies):
            failures.append("government support recipient attribution conflicts")
        if sum(c.financial.round_regulatory_fine_cents or 0 for c in state.companies) != govt.round_fines_cents:
            failures.append("government fine receipts do not close")
        w = state.welfare_accounting
        if w.round_government_net_budget_cents != govt.round_fines_cents-decision.inspection_cost_cents-support-consumer_rebate:
            failures.append("government welfare accounting conflicts")
        if decision.strategic_policy:
            from .government_strategy import learn, pack
            before = replace(govt, policy_memory=pack(audit["opening_memory"]))
            if learn(before, w.round_total_economic_welfare_cents).policy_memory != govt.policy_memory:
                failures.append("government learning memory conflicts with settled welfare")
    for s in state.supply_chain.suppliers:
        d = s.investment_decision
        if not state.state_version: continue
        if d is None: failures.append("supplier investment audit missing"); continue
        expected = investment_decision(sales=d.observed_sales_orders,available=d.observed_capacity_orders,
            base_capacity=d.previous_base_capacity_orders,cash=d.observed_cash_cents,observed_round=d.observed_round,
            terminal=state.terminal,policy=policy["supplier_investment"],requested=d.observed_requested_orders,
            unit_margin=d.observed_unit_margin_cents,remaining_rounds=d.remaining_rounds)
        if (d != expected or d.observed_round != state.state_version or d.observed_sales_orders != s.round_sales_orders
            or s.base_capacity_orders != d.previous_base_capacity_orders+d.added_capacity_orders
            or s.account.cash_cents != d.observed_cash_cents-d.investment_cents
            or s.account.investment_cents != d.investment_cents):
            failures.append("supplier investment cash/timing audit conflicts")
        if policy["supplier_investment"].get("payback_guard_ppm") and (
            d.remaining_rounds!=state.rounds_remaining or d.observed_requested_orders!=s.round_requested_orders
            or d.observed_unit_margin_cents!=s.account.settlement_price_cents-s.unit_cost_cents):
            failures.append("supplier payback inputs conflict with own settlement")
    if state.state_version:
        if not state.consumer_decisions: failures.append("consumer choice audits missing")
        purchased = {cid:0 for cid in state.company_ids}; voluntary = stockout = demand = 0
        for d in state.consumer_decisions:
            prices = dict(d.posted_prices_cents); quantities = dict(d.purchases_by_company)
            if (d.settled_round != state.state_version or min(d.unit_budget_cents,d.voluntary_no_purchase_orders,d.stockout_orders) < 0
                or any(q < 0 or cid not in purchased or (q>0 and prices[cid]>d.unit_budget_cents) for cid,q in quantities.items())
                or sum(quantities.values())+d.voluntary_no_purchase_orders+d.stockout_orders != d.demand_orders
                or d.spending_cents != sum(prices[cid]*q for cid,q in quantities.items())):
                failures.append("consumer choice budget or demand does not close")
            if not 0 <= d.refund_cents <= d.spending_cents:
                failures.append("consumer refund outside payment bounds")
            for cid,q in quantities.items():
                if cid in purchased: purchased[cid] += q
            voluntary += d.voluntary_no_purchase_orders; stockout += d.stockout_orders; demand += d.demand_orders
        if (voluntary != state.market.no_purchase_orders or stockout != state.market.lost_after_stockout_orders
            or demand != state.market.realized_demand_orders or any(purchased[c.company_id] != c.commercial.sales_orders for c in state.companies)):
            failures.append("consumer choices disagree with market settlement")
        if sum(d.refund_cents for d in state.consumer_decisions) != sum(c.financial.round_incident_cost_cents for c in state.companies):
            failures.append("consumer refunds do not match company payments")
        transfer_income=sum(t.total_transfer_fee_cents for t in state.strategic_market.last_mutual_aid_transfers)
        if sum(d.spending_cents for d in state.consumer_decisions) != sum(c.financial.round_revenue_cents for c in state.companies)-transfer_income:
            failures.append("consumer spending does not match company sales receipts")
    return failures
