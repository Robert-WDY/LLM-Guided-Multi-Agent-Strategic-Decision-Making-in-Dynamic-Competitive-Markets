"""Bounded strategic memory: recent detail, trends, and selected events."""

from __future__ import annotations

from collections import deque
from statistics import mean
from typing import TYPE_CHECKING, Any

from game_theory_agent.agents.contracts import AgentDecision, ResultAnalysis

if TYPE_CHECKING:
    from game_theory_agent.agents.personas import PersonaUtilityAssessment


def _trend(values: list[int], tolerance: int) -> str:
    if len(values) < 2:
        return "insufficient_history"
    change = values[-1] - values[0]
    if change > tolerance:
        return "rising"
    if change < -tolerance:
        return "declining"
    return "stable"


class EpisodeMemory:
    def __init__(
        self,
        max_recent_rounds: int = 3,
        trend_window_rounds: int = 5,
        max_critical_events: int = 10,
    ) -> None:
        if max_recent_rounds < 1 or trend_window_rounds < max_recent_rounds:
            raise ValueError("memory windows must be positive and trend >= recent")
        self._recent: deque[dict[str, Any]] = deque(maxlen=max_recent_rounds)
        self._trend: deque[dict[str, Any]] = deque(maxlen=trend_window_rounds)
        self._critical_events: deque[dict[str, Any]] = deque(
            maxlen=max_critical_events
        )
        self._trend_window_rounds = trend_window_rounds
        self.fallback_count = 0
        self.current_plan: dict[str, Any] | None = None

    def record(
        self,
        decision: AgentDecision,
        final_action: dict[str, Any],
        analysis: ResultAnalysis,
        persona_utility: "PersonaUtilityAssessment | None" = None,
    ) -> None:
        self._record_round(
            decision,
            final_action,
            analysis,
            "submitted",
            None,
            persona_utility,
        )

    def record_fallback_outcome(
        self,
        decision: AgentDecision | None,
        final_action: dict[str, Any],
        analysis: ResultAnalysis,
        error_code: str | None,
        persona_utility: "PersonaUtilityAssessment | None" = None,
    ) -> None:
        self._record_round(
            decision,
            final_action,
            analysis,
            "rule_fallback",
            error_code,
            persona_utility,
        )

    def _record_round(
        self,
        decision: AgentDecision | None,
        final_action: dict[str, Any],
        analysis: ResultAnalysis,
        decision_status: str,
        error_code: str | None,
        persona_utility: "PersonaUtilityAssessment | None",
    ) -> None:
        outcome = analysis.observed_outcome
        changes = outcome.company_changes
        expectation = analysis.expectation_assessment
        round_record = {
            "round": analysis.settled_round,
            "decision_status": decision_status,
            "error_code": error_code,
            "plan": decision.plan.model_dump(mode="json") if decision else None,
            "requested_action": (
                decision.requested_action.model_dump(mode="json")
                if decision
                else None
            ),
            "resolved_action": final_action,
            "company_changes": changes.model_dump(mode="json"),
            "market_changes": outcome.market_changes.model_dump(mode="json"),
            "exogenous_events": outcome.exogenous_events,
            "expectation_result": expectation.model_dump(mode="json"),
            "goal_result": analysis.goal_assessment.model_dump(mode="json"),
            "counterfactual_analysis": analysis.counterfactual_analysis,
            "persona_utility": (
                persona_utility.model_dump(mode="json")
                if persona_utility is not None
                else None
            ),
        }
        self._recent.append(round_record)
        self._trend.append(
            {
                "round": analysis.settled_round,
                "company_before": outcome.company_before.model_dump(mode="json"),
                "company_after": outcome.company_after.model_dump(mode="json"),
                "market_after": outcome.market_after.model_dump(mode="json"),
                "company_changes": changes.model_dump(mode="json"),
                "final_action": final_action,
            }
        )
        self._record_critical_events(analysis)

    def _record_critical_events(self, analysis: ResultAnalysis) -> None:
        outcome = analysis.observed_outcome
        changes = outcome.company_changes
        after = outcome.company_after
        round_number = analysis.settled_round

        def add(
            event_type: str,
            summary: str,
            affected_fields: list[str],
            importance_ppm: int,
        ) -> None:
            self._critical_events.append(
                {
                    "event_id": f"{analysis.company_id}:critical:{round_number}:{event_type}",
                    "type": event_type,
                    "started_round": round_number,
                    "resolved_round": None,
                    "summary": summary,
                    "affected_fields": affected_fields,
                    "evidence_rounds": [round_number],
                    "status": "active",
                    "importance_ppm": importance_ppm,
                }
            )

        for event in outcome.exogenous_events:
            add(
                str(event["type"]),
                f"External event observed: {event.get('event_type', event['type'])}",
                ["market", "risk"],
                1_000_000,
            )
        if after.capacity_utilization_ppm > 950_000:
            add(
                "capacity_threshold",
                "Capacity utilization exceeded 95%.",
                ["capacity_utilization", "stockout"],
                800_000,
            )
        if after.stockout_rate_ppm > 150_000:
            add(
                "stockout_threshold",
                "Stockout rate exceeded 15%.",
                ["stockout", "reputation"],
                850_000,
            )
        if abs(changes.market_share_delta_ppm) > 50_000:
            add(
                "market_share_shift",
                "Market share changed by more than 5 percentage points.",
                ["market_share"],
                750_000,
            )
        if abs(changes.reputation_delta_ppm) > 50_000:
            add(
                "reputation_shift",
                "Reputation changed by more than 5 percentage points.",
                ["reputation"],
                750_000,
            )
        if analysis.expectation_assessment.mismatches:
            add(
                "expectation_mismatch",
                "; ".join(analysis.expectation_assessment.mismatches),
                list(analysis.expectation_assessment.actual_directions),
                700_000,
            )
        if not analysis.goal_assessment.achieved:
            add(
                "goal_violation",
                "Goal criteria failed: "
                + ", ".join(analysis.goal_assessment.violations),
                list(analysis.goal_assessment.violations),
                900_000,
            )

    def record_fallback(self) -> None:
        self.fallback_count += 1

    def set_current_plan(self, plan: dict[str, Any]) -> None:
        self.current_plan = dict(plan)

    def snapshot(self) -> dict[str, Any]:
        records = list(self._trend)
        profits = [item["company_after"]["round_profit_cents"] for item in records]
        shares = [item["company_after"]["market_share_ppm"] for item in records]
        cash = [item["company_after"]["cash_balance_cents"] for item in records]
        consecutive_losses = 0
        consecutive_profits = 0
        for profit in reversed(profits):
            if profit < 0:
                consecutive_losses += 1
            else:
                break
        for profit in reversed(profits):
            if profit > 0:
                consecutive_profits += 1
            else:
                break
        starting_cash = (
            records[0]["company_before"]["cash_balance_cents"] if records else 0
        )
        cash_drawdown_ppm = (
            max(0, starting_cash - cash[-1]) * 1_000_000 // starting_cash
            if starting_cash > 0 and cash
            else 0
        )
        utilization = [
            item["company_after"]["capacity_utilization_ppm"] for item in records
        ]
        stockout_rates = [
            item["company_after"]["stockout_rate_ppm"] for item in records
        ]
        price_cut_rounds = sum(
            item["final_action"]["price_cents"]
            < item["company_before"]["price_cents"]
            for item in records
        )
        capacity_rounds = sum(
            item["final_action"].get("capacity_investment_cents", 0) > 0
            for item in records
        )
        resilience_rounds = sum(
            item["final_action"].get("resilience_budget_cents", 0) > 0
            for item in records
        )
        high_advertising_rounds = sum(
            item["final_action"].get("advertising_budget_cents", 0) >= 1_500_000
            for item in records
        )
        cash_trend = _trend(cash, 200_000)
        if (
            len(cash) >= 2
            and cash[0] > 0
            and cash[-1] - cash[0] <= -round(cash[0] * 0.2)
        ):
            cash_trend = "declining_fast"
        average_utilization = round(mean(utilization)) if utilization else 0
        average_stockout = round(mean(stockout_rates)) if stockout_rates else 0
        if (
            price_cut_rounds >= 2
            and average_utilization >= 900_000
            and average_stockout >= 100_000
        ):
            main_pattern = (
                "repeated price cuts under capacity pressure are associated with stockouts"
            )
        elif consecutive_losses >= 2:
            main_pattern = "consecutive losses require profit recovery"
        elif cash_trend in {"declining", "declining_fast"}:
            main_pattern = "cash is declining while discretionary spending continues"
        else:
            main_pattern = "no dominant multi-round pattern yet"

        return {
            "memory_schema_version": "episode-memory-v2.0.0",
            "recent_rounds": list(self._recent),
            "rolling_summary": {
                "window_rounds": min(len(records), self._trend_window_rounds),
                "profit_trend": _trend(profits, 200_000),
                "market_share_trend": _trend(shares, 5_000),
                "cash_trend": cash_trend,
                "consecutive_loss_rounds": consecutive_losses,
                "consecutive_profitable_rounds": consecutive_profits,
                "recent_cumulative_profit_cents": sum(profits),
                "cash_drawdown_ppm": cash_drawdown_ppm,
                "average_capacity_utilization_ppm": average_utilization,
                "average_stockout_rate_ppm": average_stockout,
                "action_frequency": {
                    "price_cut_rounds": price_cut_rounds,
                    "high_advertising_rounds": high_advertising_rounds,
                    "capacity_investment_rounds": capacity_rounds,
                    "resilience_investment_rounds": resilience_rounds,
                },
                "fallback_count": self.fallback_count,
                "main_pattern": main_pattern,
            },
            "critical_events": list(self._critical_events),
            "current_plan": self.current_plan,
        }
