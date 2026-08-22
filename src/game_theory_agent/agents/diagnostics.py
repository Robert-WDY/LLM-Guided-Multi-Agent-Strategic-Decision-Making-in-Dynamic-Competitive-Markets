"""Optional, non-enforcing diagnostics for reflective Agent experiments."""

from __future__ import annotations

from typing import Any

from game_theory_agent.agents.personas import PersonaProfile


PPM = 1_000_000


def build_diagnostic_flags(
    *,
    decision_support: dict[str, Any],
    rolling_summary: dict[str, Any],
    persona_profile: PersonaProfile,
    rounds_remaining: int,
) -> list[dict[str, Any]]:
    """Detect questionable trade-offs without prescribing or changing actions."""

    flags: list[dict[str, Any]] = []
    action_frequency = rolling_summary.get("action_frequency", {})
    efficiency = decision_support.get("growth_spend_efficiency_ppm")
    high_growth_rounds = int(action_frequency.get("high_advertising_rounds", 0))
    if (
        efficiency is not None
        and int(efficiency) < 0
        and high_growth_rounds >= 2
    ):
        flags.append(
            {
                "type": "low_growth_efficiency",
                "severity": "high",
                "evidence": {
                    "high_growth_spend_rounds": high_growth_rounds,
                    "growth_spend_efficiency_ppm": int(efficiency),
                    **decision_support.get("growth_efficiency_evidence", {}),
                },
                "required_response": (
                    "说明为什么继续扩张优于恢复单位利润，或说明将如何调整。"
                ),
                "enforcement": "none",
            }
        )

    gap = decision_support.get("forecast_capacity_gap_orders", {})
    demand = decision_support.get("expected_demand_orders", {})
    payback = decision_support.get("capacity_investment_payback_rounds", {})
    expected_gap = int(gap.get("expected", 0) or 0)
    expected_demand = int(demand.get("expected", 0) or 0)
    expected_payback = payback.get("expected")
    affordable = int(decision_support["safe_discretionary_budget_cents"]) > 0
    if (
        expected_gap > max(100, expected_demand // 10)
        and expected_payback is not None
        and int(expected_payback) <= rounds_remaining
        and affordable
    ):
        flags.append(
            {
                "type": "positive_return_capacity_gap",
                "severity": "medium",
                "evidence": {
                    "forecast_capacity_gap_orders": gap,
                    "capacity_investment_payback_rounds": payback,
                    "rounds_remaining": rounds_remaining,
                    "safe_discretionary_budget_cents": int(
                        decision_support["safe_discretionary_budget_cents"]
                    ),
                },
                "required_response": (
                    "说明是否投资产能；若不投资，说明现金、回收期或需求风险依据。"
                ),
                "enforcement": "none",
            }
        )

    expected_loss = int(
        decision_support.get("expected_incident_loss_cents", {}).get("mean", 0)
        or 0
    )
    reference_cost = 1_000_000
    risk_tolerance_multiplier_ppm = max(
        750_000,
        1_500_000 - persona_profile.traits_ppm.risk_aversion // 2,
    )
    tolerated_cost = reference_cost * risk_tolerance_multiplier_ppm // PPM
    marginal_reduction = int(
        decision_support.get(
            "resilience_marginal_loss_reduction_cents_per_1000000", 0
        )
        or 0
    )
    if expected_loss > tolerated_cost and marginal_reduction > 0 and affordable:
        flags.append(
            {
                "type": "material_uncovered_risk",
                "severity": "medium",
                "evidence": {
                    "expected_incident_loss_cents": decision_support[
                        "expected_incident_loss_cents"
                    ],
                    "current_resilience_coverage_ppm": int(
                        decision_support["current_resilience_coverage_ppm"]
                    ),
                    "resilience_marginal_loss_reduction_cents_per_1000000": (
                        marginal_reduction
                    ),
                    "risk_tolerance_multiplier_ppm": risk_tolerance_multiplier_ppm,
                },
                "required_response": (
                    "比较韧性投入成本与预期未覆盖损失，并说明接受或拒绝该风险。"
                ),
                "enforcement": "none",
            }
        )
    return flags
