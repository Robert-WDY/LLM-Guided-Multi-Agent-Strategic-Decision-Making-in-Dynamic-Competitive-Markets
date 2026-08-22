"""Research metrics for multi-Agent market episodes.

These metrics describe outcomes; they are deliberately excluded from hard
engineering acceptance so experiments remain free to produce negative results.
"""

from __future__ import annotations

import math
from itertools import combinations
from statistics import mean, pstdev
from typing import Any, Iterable

from game_theory_agent.market import MarketConfig
from game_theory_agent.orchestration.round_event import RoundEvent


ACTION_FIELDS = (
    "price_cents",
    "advertising_budget_cents",
    "service_budget_cents",
    "capacity_investment_cents",
    "resilience_budget_cents",
)


def _entropy(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    counts = {value: values.count(value) for value in set(values)}
    raw = -sum((count / len(values)) * math.log(count / len(values)) for count in counts.values())
    return raw / math.log(len(values))


def _action_distance(
    first: dict[str, Any], second: dict[str, Any], config: MarketConfig
) -> float:
    bounds = config.mapping("action", "bounds")
    distances: list[float] = []
    for field in ACTION_FIELDS:
        width = max(1, int(bounds[field]["max"]) - int(bounds[field]["min"]))
        distances.append(abs(int(first[field]) - int(second[field])) / width)
    first_incident = first["incident_response"]
    second_incident = second["incident_response"]
    distances.append(float(first_incident["mode"] != second_incident["mode"]))
    repair_width = max(
        1,
        int(bounds["repair_budget_cents"]["max"])
        - int(bounds["repair_budget_cents"]["min"]),
    )
    distances.append(
        abs(
            int(first_incident["repair_budget_cents"])
            - int(second_incident["repair_budget_cents"])
        )
        / repair_width
    )
    return mean(distances)


def compute_research_metrics(
    events: Iterable[RoundEvent], config: MarketConfig
) -> dict[str, Any]:
    episodes = list(events)
    if not episodes:
        return {"metrics_schema_version": "multi-agent-research-v1.0.0"}

    price_dispersions: list[float] = []
    price_entropies: list[float] = []
    full_action_distances: list[float] = []
    capacity_hhi: list[float] = []
    capacity_investments: list[int] = []
    capacity_gaps: list[int] = []
    positive_return_gap_no_investment = 0
    material_risk_no_investment = 0
    diagnostic_counts: dict[str, int] = {}
    growth_efficiency_values: list[int] = []
    all_resilience: list[int] = []

    for event in episodes:
        actions = [trace.final_action for trace in event.traces]
        prices = [int(action["price_cents"]) for action in actions]
        price_dispersions.append(pstdev(prices))
        price_entropies.append(_entropy(prices))
        full_action_distances.extend(
            _action_distance(first, second, config)
            for first, second in combinations(actions, 2)
        )
        round_capacity = [
            int(action["capacity_investment_cents"]) for action in actions
        ]
        capacity_investments.extend(round_capacity)
        total_capacity = sum(round_capacity)
        capacity_hhi.append(
            sum((value / total_capacity) ** 2 for value in round_capacity)
            if total_capacity > 0
            else 0.0
        )

        for trace in event.traces:
            company = event.state_after["companies"][trace.company_id]
            capacity_gaps.append(
                abs(
                    int(company["commercial"]["potential_demand_orders"])
                    - int(company["operations"]["effective_capacity_orders"])
                )
            )
            all_resilience.append(int(company["risk"]["resilience_ppm"]))
            context = trace.decision_context or {}
            support = context.get("decision_support", {})
            efficiency = support.get("growth_spend_efficiency_ppm")
            if efficiency is not None:
                growth_efficiency_values.append(int(efficiency))
            flags = context.get("diagnostic_flags", [])
            for flag in flags:
                flag_type = str(flag.get("type", "unknown"))
                diagnostic_counts[flag_type] = diagnostic_counts.get(flag_type, 0) + 1
                if (
                    flag_type == "positive_return_capacity_gap"
                    and int(trace.final_action["capacity_investment_cents"]) == 0
                ):
                    positive_return_gap_no_investment += 1
                if (
                    flag_type == "material_uncovered_risk"
                    and int(trace.final_action["resilience_budget_cents"]) == 0
                ):
                    material_risk_no_investment += 1

    states = [event.state_after for event in episodes]
    final_companies = states[-1]["companies"]
    market_profit = sum(
        int(company["financial"]["cumulative_profit_cents"])
        for company in final_companies.values()
    )
    unserved = sum(
        int(state["market"]["lost_after_stockout_orders"]) for state in states
    )
    outside_option = sum(
        int(state["market"]["no_purchase_orders"]) for state in states
    )
    return {
        "metrics_schema_version": "multi-agent-research-v1.0.0",
        "episode_rounds": len(episodes),
        "market_total_profit_cents": market_profit,
        "cumulative_unserved_demand_orders": unserved,
        "cumulative_outside_option_orders": outside_option,
        "strategy_diversity": {
            "mean_price_dispersion_cents": round(mean(price_dispersions), 3),
            "mean_normalized_price_entropy": round(mean(price_entropies), 6),
            "mean_normalized_full_action_distance": round(
                mean(full_action_distances), 6
            ),
        },
        "resource_allocation": {
            "capacity_investment_std_cents": round(
                pstdev(capacity_investments), 3
            ),
            "mean_round_capacity_investment_hhi": round(mean(capacity_hhi), 6),
            "mean_demand_capacity_absolute_gap_orders": round(
                mean(capacity_gaps), 3
            ),
            "minimum_resilience_ppm": min(all_resilience),
            "positive_return_capacity_gap_no_investment_rounds": (
                positive_return_gap_no_investment
            ),
            "material_uncovered_risk_no_investment_rounds": (
                material_risk_no_investment
            ),
        },
        "decision_quality_proxies": {
            "mean_historical_growth_spend_efficiency_ppm": (
                round(mean(growth_efficiency_values), 3)
                if growth_efficiency_values
                else None
            ),
            "growth_efficiency_method": (
                "historical_proxy_not_causal_counterfactual"
            ),
            "diagnostic_flag_counts": diagnostic_counts,
        },
        "metric_classification": {
            "hard_engineering_acceptance": [],
            "soft_market_health": [
                "cumulative_unserved_demand_orders",
                "cumulative_outside_option_orders",
                "resource_allocation",
            ],
            "research_no_required_direction": [
                "market_total_profit_cents",
                "strategy_diversity",
                "decision_quality_proxies",
            ],
        },
    }
