"""Observed facts and expectation gaps, without unsupported causal claims."""

from __future__ import annotations

from typing import Any

from game_theory_agent.agents.contracts import (
    CompanyOutcomeSnapshot,
    ExpectationAssessment,
    ExpectedOutcome,
    GoalAssessment,
    MarketChanges,
    MarketOutcomeSnapshot,
    MetricChanges,
    ObservedOutcome,
    ResultAnalysis,
    SuccessCriteria,
)
from game_theory_agent.market.models import CompanyState, MarketState


def _direction(delta: int, tolerance: int = 0) -> str:
    if delta > tolerance:
        return "up"
    if delta < -tolerance:
        return "down"
    return "stable"


def _company_snapshot(company: CompanyState) -> CompanyOutcomeSnapshot:
    return CompanyOutcomeSnapshot(
        cash_balance_cents=company.financial.cash_balance_cents,
        round_profit_cents=company.financial.round_profit_cents,
        market_share_ppm=company.commercial.market_share_ppm,
        sales_orders=company.commercial.sales_orders,
        price_cents=company.commercial.price_cents,
        base_capacity_orders=company.operations.base_capacity_orders,
        capacity_utilization_ppm=company.operations.capacity_utilization_ppm,
        stockout_rate_ppm=company.brand.last_attempted_unfulfilled_rate_ppm,
        reputation_ppm=company.brand.reputation_ppm,
        resilience_ppm=company.risk.resilience_ppm,
        active_incident=(
            company.risk.active_incident.to_dict()
            if company.risk.active_incident
            else None
        ),
    )


def _market_snapshot(state: MarketState) -> MarketOutcomeSnapshot:
    market = state.market
    return MarketOutcomeSnapshot(
        realized_demand_orders=market.realized_demand_orders,
        average_paid_price_cents=market.average_paid_price_cents,
        no_purchase_orders=market.no_purchase_orders,
        lost_after_stockout_orders=market.lost_after_stockout_orders,
    )


def _exogenous_events(
    before: MarketState,
    after: MarketState,
    company_id: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {
            "type": "market_event_active_during_round",
            "event_id": event.event_id,
            "event_type": event.event_type,
            "severity": event.severity,
        }
        for event in before.active_market_events
    ]
    before_event_ids = {item.event_id for item in before.active_market_events}
    events.extend(
        {
            "type": "market_event_started_after_round",
            "event_id": event.event_id,
            "event_type": event.event_type,
            "severity": event.severity,
        }
        for event in after.active_market_events
        if event.event_id not in before_event_ids
    )
    incident_before = before.company(company_id).risk.active_incident
    incident_after = after.company(company_id).risk.active_incident
    before_id = incident_before.incident_id if incident_before else None
    after_id = incident_after.incident_id if incident_after else None
    if after_id and after_id != before_id:
        events.append(
            {
                "type": "company_incident_started",
                "incident_id": after_id,
                "incident_type": incident_after.incident_type,
                "severity": incident_after.severity,
            }
        )
    if before_id and after_id != before_id:
        events.append(
            {
                "type": "company_incident_ended",
                "incident_id": before_id,
                "incident_type": incident_before.incident_type,
                "severity": incident_before.severity,
            }
        )
    return events


class ResultAnalyzer:
    def analyze(
        self,
        state_before: MarketState,
        state_after: MarketState,
        company_id: str,
        expected: ExpectedOutcome | None,
        resolution_adjustments: list[dict[str, Any]],
        success_criteria: SuccessCriteria | None = None,
        counterfactual_analysis: dict[str, Any] | None = None,
    ) -> ResultAnalysis:
        before = state_before.company(company_id)
        after = state_after.company(company_id)
        before_company = _company_snapshot(before)
        after_company = _company_snapshot(after)
        before_market = _market_snapshot(state_before)
        after_market = _market_snapshot(state_after)
        has_profit_baseline = bool(before.history.recent_profit_cents)
        previous_round_profit = (
            before.financial.round_profit_cents if has_profit_baseline else None
        )
        capacity_investment_cents = (
            after.history.last_action.capacity_investment_cents
            if after.history.last_action is not None
            else 0
        )
        changes = MetricChanges(
            cash_delta_cents=(
                after.financial.cash_balance_cents
                - before.financial.cash_balance_cents
            ),
            previous_round_profit_cents=previous_round_profit,
            profit_change_vs_previous_round_cents=(
                after.financial.round_profit_cents - previous_round_profit
                if previous_round_profit is not None
                else None
            ),
            market_share_delta_ppm=(
                after.commercial.market_share_ppm
                - before.commercial.market_share_ppm
            ),
            sales_delta_orders=(
                after.commercial.sales_orders - before.commercial.sales_orders
            ),
            base_capacity_delta_orders=(
                after.operations.base_capacity_orders
                - before.operations.base_capacity_orders
            ),
            capacity_investment_cents=capacity_investment_cents,
            capacity_utilization_delta_ppm=(
                after.operations.capacity_utilization_ppm
                - before.operations.capacity_utilization_ppm
            ),
            awareness_delta_ppm=(
                after.brand.brand_awareness_ppm - before.brand.brand_awareness_ppm
            ),
            service_delta_ppm=(
                after.brand.service_quality_ppm - before.brand.service_quality_ppm
            ),
            reputation_delta_ppm=(
                after.brand.reputation_ppm - before.brand.reputation_ppm
            ),
            resilience_delta_ppm=(
                after.risk.resilience_ppm - before.risk.resilience_ppm
            ),
            stockout_orders=after.commercial.attempted_unfulfilled_orders,
            stockout_rate_delta_ppm=(
                after.brand.last_attempted_unfulfilled_rate_ppm
                - before.brand.last_attempted_unfulfilled_rate_ppm
            ),
        )
        market_changes = MarketChanges(
            realized_demand_delta_orders=(
                after_market.realized_demand_orders
                - before_market.realized_demand_orders
            ),
            average_paid_price_delta_cents=(
                after_market.average_paid_price_cents
                - before_market.average_paid_price_cents
            ),
            no_purchase_delta_orders=(
                after_market.no_purchase_orders - before_market.no_purchase_orders
            ),
            lost_after_stockout_delta_orders=(
                after_market.lost_after_stockout_orders
                - before_market.lost_after_stockout_orders
            ),
        )

        observed = {
            "profit": (
                _direction(changes.profit_change_vs_previous_round_cents, 100_000)
                if changes.profit_change_vs_previous_round_cents is not None
                else "baseline_unavailable"
            ),
            "market_share": _direction(changes.market_share_delta_ppm, 2_000),
            "capacity": _direction(changes.base_capacity_delta_orders),
            "risk_exposure": _direction(-changes.resilience_delta_ppm, 2_000),
        }
        actual = {
            **observed,
            # Expected capacity describes the company's deliberate capacity
            # choice. Physical capacity remains separately observable and may
            # decline because the market applies natural depreciation.
            "capacity": "up" if capacity_investment_cents > 0 else "stable",
        }
        comparison_basis = {
            "profit": (
                "previous_settled_round"
                if has_profit_baseline
                else "baseline_unavailable"
            ),
            "market_share": "previous_state",
            "capacity": "capacity_investment_action",
            "risk_exposure": "observed_resilience_change",
        }
        expected_values = expected.model_dump() if expected is not None else {}
        matches = {
            key: (
                None
                if expected is None or actual[key] == "baseline_unavailable"
                else actual[key] == expected_values[key]
            )
            for key in actual
        }
        mismatches = [
            f"{key} moved {actual[key]} instead of {expected_value}"
            for key, expected_value in expected_values.items()
            if actual[key] != "baseline_unavailable"
            and actual[key] != expected_value
        ]
        successful: list[str] = []
        attention: list[str] = []
        if changes.market_share_delta_ppm > 2_000:
            successful.append("market share increased")
        if changes.base_capacity_delta_orders > 0:
            successful.append("future base capacity increased")
        if changes.reputation_delta_ppm > 2_000:
            successful.append("reputation increased")
        if changes.stockout_orders > 0:
            attention.append("stockout and fulfillment capacity")
        if after.financial.cash_balance_cents < 8_000_000:
            attention.append("cash reserve")
        if after.operations.capacity_utilization_ppm > 900_000:
            attention.append("high capacity utilization")
        if resolution_adjustments:
            attention.append("requested action was adjusted by policy")

        criteria = success_criteria or SuccessCriteria()
        executed_action = after.history.last_action
        fixed_spend = executed_action.fixed_spend_cents if executed_action else 0
        goal_checks: dict[str, bool | None] = {
            "minimum_round_profit": (
                after.financial.round_profit_cents
                >= criteria.minimum_round_profit_cents
            ),
            "minimum_cash_reserve": (
                after.financial.cash_balance_cents
                >= criteria.minimum_cash_reserve_cents
            ),
            "maximum_fixed_spend": (
                fixed_spend <= criteria.maximum_fixed_spend_cents
                if criteria.maximum_fixed_spend_cents is not None
                else None
            ),
            "minimum_market_share": (
                after.commercial.market_share_ppm
                >= criteria.minimum_market_share_ppm
                if criteria.minimum_market_share_ppm is not None
                else None
            ),
        }
        goal_violations = [
            key for key, passed in goal_checks.items() if passed is False
        ]
        goal_achieved = not goal_violations

        return ResultAnalysis(
            company_id=company_id,
            settled_round=state_before.round,
            observed_outcome=ObservedOutcome(
                company_before=before_company,
                company_after=after_company,
                company_changes=changes,
                market_before=before_market,
                market_after=after_market,
                market_changes=market_changes,
                round_market_conditions={
                    "market_sentiment_ppm": (
                        state_before.market.market_sentiment_ppm
                    ),
                    "base_supply_cost_index_ppm": (
                        state_before.market.base_supply_cost_index_ppm
                    ),
                    "actual_supply_cost_index_ppm": (
                        state_before.market.actual_supply_cost_index_ppm
                    ),
                },
                next_round_market_conditions={
                    "market_sentiment_ppm": state_after.market.market_sentiment_ppm,
                    "base_supply_cost_index_ppm": (
                        state_after.market.base_supply_cost_index_ppm
                    ),
                    "actual_supply_cost_index_ppm": (
                        state_after.market.actual_supply_cost_index_ppm
                    ),
                },
                exogenous_events=_exogenous_events(
                    state_before, state_after, company_id
                ),
            ),
            expectation_assessment=ExpectationAssessment(
                actual_directions=actual,
                observed_directions=observed,
                comparison_basis=comparison_basis,
                matches=matches,
                mismatches=mismatches,
                causal_claim=(
                    "controlled_same_seed_counterfactual"
                    if counterfactual_analysis is not None
                    else "unavailable_without_counterfactual"
                ),
            ),
            goal_assessment=GoalAssessment(
                criteria=criteria,
                checks=goal_checks,
                achieved=goal_achieved,
                violations=goal_violations,
            ),
            counterfactual_analysis=counterfactual_analysis,
            resolution_adjustments=resolution_adjustments,
            successful_effects=successful,
            next_round_attention=attention,
            summary=(
                f"observed cash {changes.cash_delta_cents:+d}; "
                f"share {changes.market_share_delta_ppm:+d} ppm; "
                f"current round profit {after.financial.round_profit_cents}; "
                f"profit comparison {actual['profit']}; "
                f"physical capacity {changes.base_capacity_delta_orders:+d} orders "
                f"with investment {capacity_investment_cents}; "
                "causal attribution requires a counterfactual"
            ),
        )
