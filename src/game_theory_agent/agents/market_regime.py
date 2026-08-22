"""Deterministic strategic summary of a market snapshot for Agent context."""

from __future__ import annotations

from typing import Any

from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.models import MarketState


class MarketRegimeEvaluator:
    """Classify observable state without changing any market equation."""

    def __init__(self, config: MarketConfig) -> None:
        self.thresholds = config.mapping("agent_context", "regime_thresholds")

    def evaluate(
        self,
        state: MarketState,
        *,
        information_mode: str = "perfect",
    ) -> dict[str, Any]:
        if information_mode not in {"perfect", "public"}:
            raise ValueError("unsupported information mode")
        ppm = 1_000_000
        companies = state.companies
        count = len(companies)
        average_posted_price = round(
            sum(item.commercial.price_cents for item in companies) / count
        )
        average_utilization = (
            round(
                sum(
                    item.operations.capacity_utilization_ppm
                    for item in companies
                )
                / count
            )
            if information_mode == "perfect"
            else None
        )
        hhi_ppm = round(
            sum(item.commercial.market_share_ppm**2 for item in companies) / ppm
        )
        outside_option_ppm = (
            round(state.market.no_purchase_orders * ppm / state.market.base_demand_orders)
            if state.market.base_demand_orders
            else 0
        )

        anchor = (
            state.market.price_anchor_cents
            if information_mode == "perfect"
            else (
                state.market.average_paid_price_cents
                or average_posted_price
            )
        )
        discount_threshold = round(
            anchor * int(self.thresholds["price_war_discount_ppm"]) / ppm
        )
        discounted_companies = sum(
            item.commercial.price_cents <= discount_threshold for item in companies
        )
        competition = (
            "price_war"
            if (
                discounted_companies
                >= int(self.thresholds["price_war_min_companies"])
                and average_posted_price <= discount_threshold
            )
            else "normal_competition"
        )

        if state.state_version == 0 or state.market.realized_demand_orders == 0:
            demand = "unobserved"
            demand_ratio_ppm = ppm
        else:
            demand_ratio_ppm = round(
                state.market.realized_demand_orders
                * ppm
                / state.market.base_demand_orders
            )
            if demand_ratio_ppm >= int(self.thresholds["high_demand_ratio_ppm"]):
                demand = "high"
            elif demand_ratio_ppm <= int(
                self.thresholds["low_demand_ratio_ppm"]
            ):
                demand = "low"
            else:
                demand = "normal"

        if average_utilization is None:
            capacity = "unknown"
        elif average_utilization >= int(
            self.thresholds["capacity_constrained_ppm"]
        ):
            capacity = "constrained"
        elif average_utilization <= int(
            self.thresholds["capacity_slack_ppm"]
        ):
            capacity = "slack"
        else:
            capacity = "balanced"

        supply = state.market.actual_supply_cost_index_ppm
        if supply >= int(self.thresholds["supply_crisis_ppm"]):
            cost = "crisis"
        elif supply >= int(self.thresholds["supply_high_ppm"]):
            cost = "high"
        elif supply <= int(self.thresholds["supply_low_ppm"]):
            cost = "low"
        else:
            cost = "normal"

        if hhi_ppm >= int(self.thresholds["hhi_concentrated_ppm"]):
            concentration = "concentrated"
        elif hhi_ppm >= int(self.thresholds["hhi_moderate_ppm"]):
            concentration = "moderate"
        else:
            concentration = "competitive"

        privately_observed_incident = (
            information_mode == "perfect"
            and any(
                item.risk.active_incident is not None for item in companies
            )
        )
        if state.active_market_events or privately_observed_incident:
            risk = "active_disruption"
        elif any(
            signal.estimated_probability_ppm
            >= int(self.thresholds["risk_warning_probability_ppm"])
            for signal in state.risk_signals
        ):
            risk = "warning"
        else:
            risk = "normal"

        if risk == "active_disruption":
            primary = "market_disruption"
        elif cost == "crisis":
            primary = "supply_cost_crisis"
        elif competition == "price_war":
            primary = "price_war"
        elif capacity == "constrained" and demand == "high":
            primary = "high_demand_capacity_constrained"
        elif risk == "warning":
            primary = "disaster_warning"
        elif demand == "low" and capacity == "slack":
            primary = "weak_demand"
        else:
            primary = "normal_competition"

        metrics = {
            "average_posted_price_cents": average_posted_price,
            "discounted_company_count": discounted_companies,
            "demand_ratio_ppm": demand_ratio_ppm,
            "hhi_ppm": hhi_ppm,
            "outside_option_ppm": outside_option_ppm,
        }
        if information_mode == "perfect":
            metrics.update(
                {
                    "price_anchor_cents": anchor,
                    "average_capacity_utilization_ppm": average_utilization,
                }
            )
        return {
            "regime_schema_version": "market-regime-v1.0.0",
            "primary": primary,
            "competition": competition,
            "demand": demand,
            "capacity": capacity,
            "cost": cost,
            "concentration": concentration,
            "risk": risk,
            "metrics": metrics,
        }
