"""Deterministic cross-round strategic phase and guardrail tracker."""

from __future__ import annotations

from typing import Any


class PlanTracker:
    def evaluate(
        self,
        *,
        round_number: int,
        decision_support: dict[str, Any],
        rolling_summary: dict[str, Any],
        previous_plan: dict[str, Any] | None,
        critical_events: list[dict[str, Any]] | None = None,
        risk_signals: list[dict[str, Any]] | None = None,
        active_incident: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        loss_streak = int(decision_support["consecutive_loss_rounds"])
        drawdown = int(decision_support["cash_drawdown_ppm"])
        prior_phase = (previous_plan or {}).get("phase")
        phase = str(decision_support["strategic_phase"])
        if phase == "liquidity_crisis":
            trigger = "cash_runway_below_threshold"
        elif phase == "profit_recovery":
            trigger = (
                "consecutive_losses"
                if loss_streak > 0
                else "cash_drawdown"
            )
        else:
            phase = "growth"
            trigger = "healthy_unit_economics_and_liquidity"

        safe_spend = int(decision_support["maximum_discretionary_budget_cents"])

        important_types = {
            "company_incident_started",
            "company_incident_ended",
            "capacity_threshold",
            "stockout_threshold",
            "market_share_shift",
            "reputation_shift",
            "expectation_mismatch",
            "goal_violation",
        }
        handled = set((previous_plan or {}).get("handled_trigger_event_ids", []))
        current_events = [
            event
            for event in (critical_events or [])
            if event.get("type") in important_types
        ]
        event_ids = {str(event.get("event_id")) for event in current_events}
        signal_ids = {
            f"risk:{signal.get('signal_id', signal.get('event_type', 'unknown'))}"
            for signal in (risk_signals or [])
        }
        incident_ids = (
            {f"incident:{active_incident.get('incident_id', 'active')}"}
            if active_incident
            else set()
        )
        new_trigger_ids = sorted((event_ids | signal_ids | incident_ids) - handled)
        previous_expiry = int(
            (previous_plan or {}).get(
                "expires_round",
                int((previous_plan or {}).get("entered_round", round_number)) + 2,
            )
        )
        if previous_plan is None:
            replan_reason = "initial_plan"
        elif prior_phase != phase:
            replan_reason = f"strategic_phase_changed:{prior_phase}->{phase}"
        elif round_number > previous_expiry:
            replan_reason = "plan_expired"
        elif new_trigger_ids:
            replan_reason = "critical_event:" + ",".join(new_trigger_ids)
        else:
            replan_reason = None
        replanned = replan_reason is not None
        entered_round = round_number if replanned else int(
            previous_plan.get("entered_round", round_number)
        )
        priorities = {
            "growth": [
                "positive unit economics",
                "persona utility",
                "financial flexibility",
            ],
            "profit_recovery": ["restore profit", "preserve cash", "market share"],
            "liquidity_crisis": ["survival", "preserve cash", "avoid new investment"],
        }[phase]
        rules = [
            "price must not be below minimum_price_cents",
            "total discretionary spend must not exceed maximum_discretionary_spend_cents",
        ]
        if phase == "profit_recovery":
            rules.append("do not lower price while loss streak remains active")
        if phase == "liquidity_crisis":
            rules.extend(
                (
                    "do not make capacity or resilience investments",
                    "set nonessential advertising and service spend to zero",
                )
            )
        objectives = {
            "growth": "在安全现金约束内提高长期竞争力",
            "profit_recovery": "停止无效扩张并恢复稳定盈利",
            "liquidity_crisis": "优先保证公司存续和现金安全",
        }
        subgoals = {
            "growth": ["验证单位经济", "选择最有效的增长投入", "保留现金缓冲"],
            "profit_recovery": ["停止主动降价", "压缩低效投入", "连续两轮恢复盈利"],
            "liquidity_crisis": ["停止非必要投入", "避免新增长期投资", "恢复最低现金跑道"],
        }
        horizon = 3
        plan_id = (
            f"plan-{phase}-{round_number:03d}"
            if replanned
            else str(previous_plan["plan_id"])
        )
        return {
            "plan_schema_version": "strategic-plan-state-v1.1.0",
            "plan_id": plan_id,
            "created_round": (
                round_number
                if replanned
                else int(previous_plan.get("created_round", entered_round))
            ),
            "horizon": horizon,
            "expires_round": (
                round_number + horizon - 1 if replanned else previous_expiry
            ),
            "last_evaluated_round": round_number,
            "replanned": replanned,
            "replan_reason": replan_reason,
            "phase": phase,
            "trigger": trigger,
            "entered_round": entered_round,
            "consecutive_loss_rounds": loss_streak,
            "cash_drawdown_ppm": drawdown,
            "priorities": priorities,
            "objective": objectives[phase],
            "pending_subgoals": subgoals[phase],
            "replan_triggers": [
                "plan_expired",
                "company_incident",
                "risk_signal",
                "capacity_or_stockout_threshold",
                "major_market_or_reputation_shift",
                "expectation_or_goal_failure",
                "strategic_phase_change",
            ],
            "handled_trigger_event_ids": sorted(
                handled | event_ids | signal_ids | incident_ids
            ),
            "constraints": {
                "minimum_price_cents": int(
                    decision_support["minimum_safe_price_cents"]
                ),
                "minimum_cash_reserve_cents": int(
                    decision_support["minimum_cash_reserve_cents"]
                ),
                "maximum_discretionary_spend_cents": safe_spend,
            },
            "rules": rules,
            "exit_conditions": (
                ["two consecutive profitable rounds", "cash runway at least 3 rounds"]
                if phase != "growth"
                else []
            ),
        }
