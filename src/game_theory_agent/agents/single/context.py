"""Build bounded, visible decision context for the single-agent workflow."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .gateway import GatewaySnapshot
from .models import (
    DecisionContext,
    DecisionTrace,
    EpisodeMemoryView,
    RoundFeedback,
    SnapshotKey,
    StrategyReflection,
)


def build_decision_context(
    snapshot: GatewaySnapshot,
    prior_traces: list[DecisionTrace],
    *,
    history_limit: int = 2,
) -> DecisionContext:
    """Create the visible, bounded context passed to the decision provider."""

    bounded_limit = max(1, min(history_limit, 5))
    raw_history = snapshot.observation.get("public_history") or []
    recent_raw = raw_history[-bounded_limit:]
    recent_feedback = [
        RoundFeedback.model_validate(_round_feedback_payload(item))
        for item in recent_raw
        if isinstance(item, dict)
    ]
    diagnostic_codes: list[str] = []
    previous_selected_candidate_id: str | None = None
    previous_expected_outcome: str | None = None

    if recent_feedback:
        last_round = recent_feedback[-1].settled_round
        matching_traces = [
            trace
            for trace in prior_traces
            if trace.episode_id == snapshot.episode_id
            and trace.company_id == snapshot.company_id
            and trace.status == "accepted"
            and trace.round == last_round
        ]
        if matching_traces:
            previous_trace = matching_traces[-1]
            previous_selected_candidate_id = previous_trace.selected_candidate_id
            previous_expected_outcome = _selected_expected_outcome(previous_trace)
        elif prior_traces:
            diagnostic_codes.append("history_trace_mismatch")

    memory = EpisodeMemoryView(
        history_limit=bounded_limit,
        recent_feedback=recent_feedback,
        previous_selected_candidate_id=previous_selected_candidate_id,
        previous_expected_outcome=previous_expected_outcome,
        diagnostic_codes=diagnostic_codes,
    )
    reflection = build_deterministic_reflection(memory)

    return DecisionContext(
        snapshot_key=SnapshotKey(
            episode_id=snapshot.episode_id,
            company_id=snapshot.company_id,
            round=snapshot.round,
            state_version=snapshot.state_version,
            state_hash=snapshot.state_hash,
        ),
        observation=_bounded_observation(snapshot.observation),
        action_contract=deepcopy(snapshot.action_contract),
        memory=memory,
        reflection=reflection,
    )


def build_deterministic_reflection(memory: EpisodeMemoryView) -> StrategyReflection:
    """Summarize recent visible outcomes without an additional model call."""

    if not memory.recent_feedback:
        return StrategyReflection(
            source="deterministic",
            lesson_codes=["first_round_baseline"],
            adjustments=["首轮没有历史结果，基于当前公司状态与动作约束建立经营基线。"],
            evidence_paths=[
                "observation.own_company",
                "action_contract.bounds",
            ],
            summary="首轮暂无历史结果；本轮以当前公司状态和动作约束作为策略基线。",
        )

    latest = memory.recent_feedback[-1]
    previous = memory.recent_feedback[-2] if len(memory.recent_feedback) >= 2 else None
    lesson_codes: list[str] = []
    adjustments: list[str] = []
    evidence_paths: list[str] = []

    profit = int(latest.own_result.get("round_profit_cents", 0))
    if profit > 0:
        lesson_codes.append("profit_positive")
        adjustments.append("保持现金约束内的有效投入。")
        evidence_paths.append("memory.recent_feedback[-1].own_result.round_profit_cents")
    elif profit < 0:
        lesson_codes.append("profit_negative")
        adjustments.append("降低固定投入或提高单笔贡献。")
        evidence_paths.append("memory.recent_feedback[-1].own_result.round_profit_cents")

    latest_share = latest.own_result.get("market_share_ppm")
    previous_share = previous.own_result.get("market_share_ppm") if previous else None
    if isinstance(latest_share, int) and isinstance(previous_share, int):
        if latest_share > previous_share:
            lesson_codes.append("share_up")
            adjustments.append("延续带来份额提升的价格和服务组合。")
            evidence_paths.append("memory.recent_feedback[-1].own_result.market_share_ppm")
        elif latest_share < previous_share:
            lesson_codes.append("share_down")
            adjustments.append("检查价格、服务或广告是否削弱吸引力。")
            evidence_paths.append("memory.recent_feedback[-1].own_result.market_share_ppm")

    if int(latest.market.get("lost_after_stockout_orders", 0)) > 0:
        lesson_codes.append("stockout")
        adjustments.append("关注产能余量，避免缺货流失。")
        evidence_paths.append("memory.recent_feedback[-1].market.lost_after_stockout_orders")

    if int(latest.own_result.get("round_incident_cost_cents", 0)) > 0:
        lesson_codes.append("incident_cost")
        adjustments.append("事故成本出现时提高韧性或维修优先级。")
        evidence_paths.append("memory.recent_feedback[-1].own_result.round_incident_cost_cents")

    if "history_trace_mismatch" in memory.diagnostic_codes:
        lesson_codes.append("system_adjustment_unknown")
        adjustments.append("上一轮 trace 未能与公开历史对齐，本轮只信任 Gateway 事实。")
        evidence_paths.append("memory.diagnostic_codes")

    return StrategyReflection(
        source="deterministic",
        lesson_codes=lesson_codes,
        adjustments=adjustments,
        evidence_paths=evidence_paths,
        summary="；".join(adjustments[:3]),
    )


def _bounded_observation(observation: dict[str, Any]) -> dict[str, Any]:
    bounded = deepcopy(observation)
    bounded.pop("public_history", None)
    public_companies = bounded.get("public_companies")
    if isinstance(public_companies, list):
        bounded["public_companies"] = [
            _public_company_without_private_fields(company)
            for company in public_companies
            if isinstance(company, dict)
        ]
    return bounded


def _round_feedback_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "settled_round": item.get("settled_round"),
        "own_action": item.get("own_action") or {},
        "own_result": item.get("own_result") or {},
        "market": item.get("market") or {},
        "active_events_during_round": item.get("active_events_during_round") or [],
        "resolved_signal_outcomes": item.get("resolved_signal_outcomes") or [],
    }


def _public_company_without_private_fields(company: dict[str, Any]) -> dict[str, Any]:
    public_company = deepcopy(company)
    public_company.pop("financial", None)
    return public_company


def _selected_expected_outcome(trace: DecisionTrace) -> str | None:
    if not trace.selected_candidate_id:
        return None
    for candidate in trace.candidates:
        if candidate.candidate_id == trace.selected_candidate_id:
            return candidate.expected_outcome
    return None
