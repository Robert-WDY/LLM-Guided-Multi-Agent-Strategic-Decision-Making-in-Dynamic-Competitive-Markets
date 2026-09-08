"""Prepaid physical inputs; internal transfers cancel in aggregate surplus."""
from dataclasses import replace
from typing import Mapping
from .supplier_strategy import decode, supplier_failures


VERSION = "cash-material-v1.0.0"


def enabled(config):
    return isinstance(config, Mapping) and config.get("transaction_accounting") is not None


def close_materials(supply, used_by_company):
    outcomes = []
    for outcome in supply.last_procurement_outcomes:
        used = used_by_company.get(outcome.company_id, 0)
        if not 0 <= used <= outcome.fulfilled_orders:
            raise ValueError("physical input consumption exceeds delivered quantity")
        outcomes.append(replace(outcome, material=replace(
            outcome.material, used_orders=used, wasted_orders=outcome.fulfilled_orders-used)))
    return replace(supply, last_procurement_outcomes=tuple(outcomes))


def accounting_failures(state, config):
    failures = []
    supply = state.supply_chain
    if any(s.account is None for s in supply.suppliers):
        return ["supplier cash account missing"]
    receipts = {s.supplier_id: 0 for s in supply.suppliers}
    by_company = {o.company_id: o for o in supply.last_procurement_outcomes}
    for outcome in supply.last_procurement_outcomes:
        m = outcome.material
        if m is None:
            failures.append("material settlement missing")
            continue
        if any(sid not in supply.supplier_ids for sid, _ in outcome.supplier_allocation_orders):
            failures.append("invoice supplier is unknown")
            continue
        expected = {sid: qty * (decode(supply.supplier(sid))["audit"]["settlement_prices_cents"][outcome.company_id] if supply.supplier(sid).strategic_ledger else supply.supplier(sid).account.settlement_price_cents)
                    for sid, qty in outcome.supplier_allocation_orders}
        if dict(m.payments_by_supplier) != expected or m.payment_cents != sum(expected.values()):
            failures.append("buyer invoice does not match supplier delivery")
        if not 0 <= m.payment_cents <= m.budget_cents:
            failures.append("buyer payment exceeds procurement budget")
        company = state.company(outcome.company_id)
        internal = (company.commercial.sales_orders - (company.commercial.mutual_aid_fulfilled_orders or 0)
                    + (company.commercial.mutual_aid_provided_orders or 0))
        if (m.used_orders != internal or m.wasted_orders < 0
                or m.used_orders + m.wasted_orders != outcome.fulfilled_orders):
            failures.append("material consumption and waste do not close")
        if company.financial.round_material_payment_cents != m.payment_cents:
            failures.append("company material expense differs from paid invoice")
        # Production costs and outsourcing are disjoint from prepaid materials.
        transfer = sum(t.total_transfer_fee_cents for t in state.strategic_market.last_mutual_aid_transfers
                       if t.recipient_company_id == outcome.company_id) if state.strategic_market else 0
        if company.financial.round_variable_cost_cents != m.payment_cents + internal * company.operations.actual_unit_cost_cents + transfer:
            failures.append("company variable cost double counts or omits inputs")
        for sid, value in expected.items():
            receipts[sid] += value
    for supplier in supply.suppliers:
        a = supplier.account
        if a is None:
            failures.append("supplier cash account missing")
            continue
        if supplier.strategic_ledger:
            if a.receipts_cents != receipts[supplier.supplier_id]:
                failures.append("strategic supplier receipts differ from paid invoices")
            failures.extend(supplier_failures(supplier,config["strategic_policy"],config["transaction_accounting"]["initial_supplier_cash_cents"]))
            continue
        if (a.cash_cents < 0 or a.opening_cash_cents < 0 or a.receipts_cents != receipts[supplier.supplier_id]
                or a.production_cost_cents != supplier.round_sales_orders * supplier.unit_cost_cents
                or a.cash_cents != a.opening_cash_cents + a.receipts_cents - a.production_cost_cents - (a.investment_cents or 0)
                or supplier.round_profit_cents != a.receipts_cents - a.production_cost_cents - (a.investment_cents or 0)):
            failures.append("supplier cash and trading accounts do not close")
        initial = config["transaction_accounting"]["initial_supplier_cash_cents"]
        if a.cash_cents != initial + supplier.cumulative_profit_cents:
            failures.append("supplier cumulative cash does not close")
    for company in state.companies:
        if company.company_id not in by_company and (company.financial.round_material_payment_cents or 0) != 0:
            failures.append("company without delivery has material expense")
    return failures
