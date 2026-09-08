"""Replayable upstream procurement for the v8 supply-chain market."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Mapping, Sequence
from game_theory_agent.market.supplier_policy import pricing_enabled
from game_theory_agent.market.transaction_accounting import enabled as cash_accounting
from game_theory_agent.market import supplier_strategy

from game_theory_agent.market.models import (
    CompanyAction,
    CompanyState,
    ProcurementOutcome,
    SupplierState,
    SupplyChainState,
    SupplierAccount,
    MaterialSettlement,
)


PPM = 1_000_000


def _allocate_integer(total: int, weights: Mapping[str, int]) -> dict[str, int]:
    positive = {str(key): max(0, int(value)) for key, value in weights.items()}
    denominator = sum(positive.values())
    if total <= 0 or denominator <= 0:
        return {key: 0 for key in positive}
    exact = {key: total * value / denominator for key, value in positive.items()}
    allocated = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(allocated.values())
    order = sorted(exact, key=lambda key: (-(exact[key] - allocated[key]), key))
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def initial_supply_chain_state(config: Mapping[str, object]) -> SupplyChainState:
    definitions = config["suppliers"]
    assert isinstance(definitions, Mapping)
    suppliers = tuple(
        SupplierState(
            supplier_id=str(supplier_id),
            label=str(payload["label"]),
            unit_price_cents=int(payload["unit_price_cents"]),
            unit_cost_cents=int(payload["unit_cost_cents"]),
            base_capacity_orders=int(payload["base_capacity_orders"]),
            available_capacity_orders=int(payload["base_capacity_orders"]),
            reliability_ppm=int(payload["reliability_ppm"]),
            round_requested_orders=0 if config.get("track_supplier_demand") else None,
            strategic_ledger=supplier_strategy.initial(config["strategic_policy"]) if config.get("strategic_policy") else None,
            account=(SupplierAccount(config["transaction_accounting"]["initial_supplier_cash_cents"],
                                     config["transaction_accounting"]["initial_supplier_cash_cents"])
                     if cash_accounting(config) else None),
        )
        for supplier_id, payload in sorted(definitions.items())
    )
    return SupplyChainState(
        protocol_version=str(config["protocol_version"]),
        suppliers=suppliers,
    )


def settle_procurement(
    *,
    previous: SupplyChainState,
    companies: Sequence[CompanyState],
    actions: Mapping[str, CompanyAction],
    config: Mapping[str, object],
    fixed_overhead_cents: int = 0,
    rounds_remaining: int | None = None,
    actor_choices: Mapping[str,str] | None = None,
) -> tuple[SupplyChainState, dict[str, int], dict[str, int]]:
    """Allocate scarce supplier capacity simultaneously and proportionally."""

    supplier_ids = set(previous.supplier_ids)
    default_supplier_id = str(config["default_supplier_id"])
    requests: dict[str, dict[str, int]] = {
        supplier_id: {} for supplier_id in previous.supplier_ids
    }
    requested_total: dict[str, int] = {}
    selections: dict[str, tuple[str, str | None, int]] = {}
    budgets: dict[str, int] = {}
    policy=config.get("strategic_policy")
    books={}
    round_number=next(iter(actions.values())).round if actions else 0
    if policy:
        if rounds_remaining is None: raise ValueError("strategic supplier requires the known horizon")
        books={s.supplier_id:supplier_strategy.negotiated_book(s,actions,round_number,rounds_remaining,policy) for s in previous.suppliers}
    def price(sid,cid):
        return books[sid][1][cid] if policy else previous.supplier(sid).unit_price_cents
    for company in companies:
        company_id = company.company_id
        action = actions[company_id]
        requested = max(0, company.operations.base_capacity_orders)
        if action.procurement_quantity_orders is not None:
            requested = min(requested, action.procurement_quantity_orders)
        primary = action.primary_supplier_id or default_supplier_id
        if primary not in supplier_ids:
            primary = default_supplier_id
        backup = action.backup_supplier_id
        if backup not in supplier_ids or backup == primary:
            backup = None
        share = (
            int(action.primary_supplier_share_ppm)
            if backup is not None and action.primary_supplier_share_ppm is not None
            else PPM
        )
        share = min(PPM, max(0, share))
        if cash_accounting(config):
            budgets[company_id] = max(0, company.financial.cash_balance_cents - action.fixed_spend_cents - fixed_overhead_cents)
            maximum_price = max(price(primary,company_id), price(backup,company_id) if backup else 0)
            if maximum_price:
                requested = min(requested, budgets[company_id] // maximum_price)
        primary_request = (requested * share + PPM // 2) // PPM
        backup_request = requested - primary_request
        requests[primary][company_id] = (
            requests[primary].get(company_id, 0) + primary_request
        )
        if backup is not None and backup_request:
            requests[backup][company_id] = (
                requests[backup].get(company_id, 0) + backup_request
            )
        requested_total[company_id] = requested
        selections[company_id] = (primary, backup, share)

    allocations_by_supplier: dict[str, dict[str, int]] = {}
    next_suppliers: list[SupplierState] = []
    round_upstream_surplus = 0
    for supplier in previous.suppliers:
        demand = requests[supplier.supplier_id]
        if policy:
            contracts,prices,negotiations=books[supplier.supplier_id]
            settled,allocation=supplier_strategy.settle_supplier(supplier,demand,prices,contracts,negotiations,policy,round_number,rounds_remaining,_allocate_integer,(actor_choices or {}).get(supplier.supplier_id,"balanced"))
            next_suppliers.append(settled)
            allocations_by_supplier[supplier.supplier_id]=allocation
            round_upstream_surplus+=settled.round_profit_cents+supplier_strategy.decode(settled)["audit"]["lender_profit_cents"]
            continue
        total_demand = sum(demand.values())
        sold = min(total_demand, supplier.available_capacity_orders)
        allocation = _allocate_integer(sold, demand)
        allocations_by_supplier[supplier.supplier_id] = allocation
        profit = sold * max(0, supplier.unit_price_cents - supplier.unit_cost_cents)
        round_upstream_surplus += profit
        next_suppliers.append(
            replace(
                supplier,
                round_sales_orders=sold,
                round_requested_orders=total_demand if config.get("track_supplier_demand") else None,
                round_profit_cents=profit,
                cumulative_profit_cents=supplier.cumulative_profit_cents + profit,
                last_settled_unit_price_cents=(supplier.unit_price_cents if pricing_enabled(config) else None),
                account=(SupplierAccount(
                    opening_cash_cents=supplier.account.cash_cents,
                    cash_cents=supplier.account.cash_cents + profit,
                    receipts_cents=sold * supplier.unit_price_cents,
                    production_cost_cents=sold * supplier.unit_cost_cents,
                    settlement_price_cents=supplier.unit_price_cents,
                ) if cash_accounting(config) else None),
            )
        )

    outcomes: list[ProcurementOutcome] = []
    capacity_by_company: dict[str, int] = {}
    unit_price_by_company: dict[str, int] = {}
    baseline_price = int(config["baseline_input_price_cents"])
    for company in companies:
        company_id = company.company_id
        primary, backup, share = selections[company_id]
        allocations = {
            supplier_id: allocation.get(company_id, 0)
            for supplier_id, allocation in allocations_by_supplier.items()
            if allocation.get(company_id, 0) > 0
        }
        fulfilled = sum(allocations.values())
        procurement_cost = sum(
            quantity * price(supplier_id,company_id)
            for supplier_id, quantity in allocations.items()
        )
        weighted_price = (
            (procurement_cost + fulfilled // 2) // fulfilled
            if fulfilled
            else baseline_price
        )
        requested = requested_total[company_id]
        outcomes.append(
            ProcurementOutcome(
                company_id=company_id,
                primary_supplier_id=primary,
                backup_supplier_id=backup,
                primary_supplier_share_ppm=share,
                requested_orders=requested,
                fulfilled_orders=fulfilled,
                unfulfilled_orders=max(0, requested - fulfilled),
                weighted_unit_input_price_cents=weighted_price,
                supplier_allocation_orders=tuple(sorted(allocations.items())),
                material=(MaterialSettlement(
                    budget_cents=budgets[company_id], payment_cents=procurement_cost,
                    payments_by_supplier=tuple(sorted((sid, qty * price(sid,company_id))
                                                      for sid, qty in allocations.items())),
                    wasted_orders=fulfilled,
                ) if cash_accounting(config) else None),
            )
        )
        capacity_by_company[company_id] = fulfilled
        unit_price_by_company[company_id] = weighted_price

    return (
        SupplyChainState(
            protocol_version=previous.protocol_version,
            suppliers=tuple(next_suppliers),
            last_procurement_outcomes=tuple(outcomes),
            round_upstream_producer_surplus_cents=round_upstream_surplus,
            cumulative_upstream_producer_surplus_cents=(
                previous.cumulative_upstream_producer_surplus_cents
                + round_upstream_surplus
            ),
        ),
        capacity_by_company,
        unit_price_by_company,
    )


def advance_supplier_availability(
    *,
    settled: SupplyChainState,
    uniform_draw_by_supplier: Mapping[str, float],
    config: Mapping[str, object],
) -> SupplyChainState:
    disruption_capacity_ppm = int(config["disruption_capacity_ppm"])
    suppliers = []
    for supplier in settled.suppliers:
        disrupted = (
            float(uniform_draw_by_supplier[supplier.supplier_id])
            >= supplier.reliability_ppm / PPM
        )
        available = supplier.base_capacity_orders
        if disrupted:
            available = (
                available * disruption_capacity_ppm + PPM // 2
            ) // PPM
        if supplier.strategic_ledger:
            ledger=supplier_strategy.decode(supplier)
            available=0 if ledger["bankrupt"] else available+ledger["inventory_orders"]
        suppliers.append(
            replace(
                supplier,
                available_capacity_orders=available,
                disrupted=disrupted,
            )
        )
    return replace(settled, suppliers=tuple(suppliers))
