"""Transparent welfare accounting for consumers, firms, suppliers and government."""

from __future__ import annotations

from typing import Mapping, Sequence

from game_theory_agent.market.models import (
    CompanyState,
    SupplyChainState,
    WelfareAccountingState,
)


def initial_welfare_state(config: Mapping[str, object]) -> WelfareAccountingState:
    return WelfareAccountingState(
        protocol_version=str(config["protocol_version"]),
        valuation_source=str(config["valuation_source"]),
    )


def advance_welfare_state(
    *,
    previous: WelfareAccountingState,
    companies: Sequence[CompanyState],
    supply_chain: SupplyChainState | None,
    consumer_surplus_cents: int,
    government_fine_revenue_cents: int,
    enforcement_case_count: int,
    lost_after_stockout_orders: int,
    voluntary_no_purchase_orders: int,
    newly_exited_company_count: int,
    config: Mapping[str, object],
    actual_enforcement_cost_cents: int | None = None,
    government_support_cents: int = 0,
) -> WelfareAccountingState:
    downstream = sum(item.financial.round_profit_cents for item in companies)
    upstream = (
        supply_chain.round_upstream_producer_surplus_cents
        if supply_chain is not None
        else 0
    )
    enforcement_cost = enforcement_case_count * int(
        config["government_enforcement_cost_per_case_cents"]
    )
    if actual_enforcement_cost_cents is not None:
        enforcement_cost = actual_enforcement_cost_cents
    government_net = government_fine_revenue_cents - enforcement_cost - government_support_cents
    stockout_externality = lost_after_stockout_orders * int(
        config["stockout_externality_per_order_cents"]
    )
    exit_externality = newly_exited_company_count * int(
        config["business_exit_externality_per_company_cents"]
    )
    externality = stockout_externality + exit_externality
    total = (
        consumer_surplus_cents
        + downstream
        + upstream
        + government_net
        - externality
    )
    return WelfareAccountingState(
        protocol_version=previous.protocol_version,
        valuation_source=previous.valuation_source,
        round_consumer_surplus_cents=consumer_surplus_cents,
        round_downstream_producer_surplus_cents=downstream,
        round_upstream_producer_surplus_cents=upstream,
        round_government_fine_revenue_cents=government_fine_revenue_cents,
        round_government_enforcement_cost_cents=enforcement_cost,
        round_government_net_budget_cents=government_net,
        round_stockout_externality_cents=stockout_externality,
        round_business_exit_externality_cents=exit_externality,
        round_total_economic_welfare_cents=total,
        round_service_continuity_orders=sum(
            item.commercial.sales_orders for item in companies
        ),
        round_voluntary_no_purchase_orders=voluntary_no_purchase_orders,
        cumulative_consumer_surplus_cents=(
            previous.cumulative_consumer_surplus_cents
            + consumer_surplus_cents
        ),
        cumulative_downstream_producer_surplus_cents=(
            previous.cumulative_downstream_producer_surplus_cents + downstream
        ),
        cumulative_upstream_producer_surplus_cents=(
            previous.cumulative_upstream_producer_surplus_cents + upstream
        ),
        cumulative_government_net_budget_cents=(
            previous.cumulative_government_net_budget_cents + government_net
        ),
        cumulative_externality_cost_cents=(
            previous.cumulative_externality_cost_cents + externality
        ),
        cumulative_total_economic_welfare_cents=(
            previous.cumulative_total_economic_welfare_cents + total
        ),
    )
