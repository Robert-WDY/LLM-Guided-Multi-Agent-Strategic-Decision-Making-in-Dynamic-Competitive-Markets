"""Canonical decision resolution shared by people, rules, and external agents.

Callers may request economic parameters, but only this module creates the
``CompanyAction`` that is allowed to reach ``MarketEnv.step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from game_theory_agent.economics import decision_support_metrics
from game_theory_agent.market import (
    CompanyAction,
    IncidentResponse,
    IncidentResponseMode,
    MarketConfig,
    MarketState,
)
from game_theory_agent.market.exceptions import ActionValidationError


POLICY_VERSION = "decision-policy-v1.1.0"
ECONOMIC_FIELDS = (
    "price_cents",
    "advertising_budget_cents",
    "service_budget_cents",
    "capacity_investment_cents",
    "resilience_budget_cents",
)


@dataclass(frozen=True, slots=True)
class DecisionAdjustment:
    field: str
    requested: Any
    resolved: Any
    reason_code: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "requested": self.requested,
            "resolved": self.resolved,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ResolvedDecision:
    action: CompanyAction
    source: str
    policy_version: str
    adjustments: tuple[DecisionAdjustment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "policy_version": self.policy_version,
            "action": self.action.to_dict(),
            "adjustments": [item.to_dict() for item in self.adjustments],
        }


def _integer(raw: Mapping[str, Any], field: str, default: int) -> int:
    value = raw.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActionValidationError(f"{field} must be an integer")
    return value


def _adjust(
    adjustments: list[DecisionAdjustment],
    field: str,
    requested: Any,
    resolved: Any,
    reason_code: str,
    reason: str,
) -> Any:
    if requested != resolved:
        adjustments.append(
            DecisionAdjustment(field, requested, resolved, reason_code, reason)
        )
    return resolved


def _clamp(
    adjustments: list[DecisionAdjustment],
    field: str,
    value: int,
    low: int,
    high: int,
) -> int:
    resolved = min(max(value, low), high)
    return _adjust(
        adjustments,
        field,
        value,
        resolved,
        "BOUND_CLAMPED",
        f"统一规则将数值限制在 [{low}, {high}]。",
    )


def resolve_action_request(
    config: MarketConfig,
    state: MarketState,
    company_id: str,
    request: Mapping[str, Any],
    *,
    source: str,
    action_id: str | None = None,
) -> ResolvedDecision:
    """Validate an intent and produce the only action form accepted by settlement."""

    if state.terminal:
        raise ActionValidationError("episode is already terminal")
    if company_id not in state.company_ids:
        raise ActionValidationError("unknown company_id")
    company = state.company(company_id)
    bounds = config.mapping("action", "bounds")
    adjustments: list[DecisionAdjustment] = []
    values: dict[str, int] = {}
    defaults = {
        "price_cents": company.commercial.price_cents,
        "advertising_budget_cents": 0,
        "service_budget_cents": 0,
        "capacity_investment_cents": 0,
        "resilience_budget_cents": 0,
    }
    for field in ECONOMIC_FIELDS:
        requested = _integer(request, field, defaults[field])
        values[field] = _clamp(
            adjustments,
            field,
            requested,
            int(bounds[field]["min"]),
            int(bounds[field]["max"]),
        )

    cooperation_enabled = state.shared_resilience is not None
    requested_shared_raw = request.get(
        "shared_resilience_contribution_cents", 0
    )
    requested_shared = (
        0
        if requested_shared_raw is None
        else _integer(
            {"shared_resilience_contribution_cents": requested_shared_raw},
            "shared_resilience_contribution_cents",
            0,
        )
    )
    if cooperation_enabled:
        shared_bounds = bounds["shared_resilience_contribution_cents"]
        values["shared_resilience_contribution_cents"] = _clamp(
            adjustments,
            "shared_resilience_contribution_cents",
            requested_shared,
            int(shared_bounds["min"]),
            int(shared_bounds["max"]),
        )
    else:
        _adjust(
            adjustments,
            "shared_resilience_contribution_cents",
            requested_shared,
            0,
            "COOPERATION_DISABLED",
            "当前 Episode 未启用共享韧性合作动作。",
        )

    economics = decision_support_metrics(config, state, company_id)
    enforceable_price_floor = min(
        int(bounds["price_cents"]["max"]),
        int(economics["minimum_safe_price_cents"]),
    )
    if economics["strategic_phase"] in {
        "profit_recovery",
        "liquidity_crisis",
    }:
        enforceable_price_floor = max(
            enforceable_price_floor,
            company.commercial.price_cents,
        )
    values["price_cents"] = _adjust(
        adjustments,
        "price_cents",
        values["price_cents"],
        max(values["price_cents"], enforceable_price_floor),
        (
            "RECOVERY_PRICE_FLOOR"
            if economics["strategic_phase"] != "growth"
            else "NEGATIVE_UNIT_MARGIN_PROTECTED"
        ),
        "价格不得低于单位经济安全线；恢复阶段不得继续主动降价。",
    )

    # Saturating assets should not absorb unlimited cash. These are execution
    # guardrails, not hidden strategy choices, and apply equally to every source.
    awareness = company.brand.brand_awareness_ppm
    if awareness >= 900_000:
        cap = 0
    elif awareness >= 850_000:
        cap = 600_000
    else:
        cap = values["advertising_budget_cents"]
    values["advertising_budget_cents"] = _adjust(
        adjustments,
        "advertising_budget_cents",
        values["advertising_budget_cents"],
        min(values["advertising_budget_cents"], cap),
        "AWARENESS_SATURATED",
        "知名度接近或达到 90%，停止或限制边际广告投入。",
    )

    service_quality = company.brand.service_quality_ppm
    reputation = company.brand.reputation_ppm
    if service_quality >= 900_000 and reputation >= 900_000:
        cap = 0
    elif service_quality >= 850_000 and reputation >= 850_000:
        cap = 600_000
    else:
        cap = values["service_budget_cents"]
    values["service_budget_cents"] = _adjust(
        adjustments,
        "service_budget_cents",
        values["service_budget_cents"],
        min(values["service_budget_cents"], cap),
        "SERVICE_SATURATED",
        "服务与声誉接近或达到 90%，停止或限制边际服务投入。",
    )

    if state.rounds_remaining <= 1:
        for field in ("capacity_investment_cents", "resilience_budget_cents"):
            values[field] = _adjust(
                adjustments,
                field,
                values[field],
                0,
                "LAST_ROUND_DISABLED",
                "最后一轮的长期投资无法形成后续经营收益。",
            )
        if cooperation_enabled:
            values["shared_resilience_contribution_cents"] = _adjust(
                adjustments,
                "shared_resilience_contribution_cents",
                values["shared_resilience_contribution_cents"],
                0,
                "LAST_ROUND_DISABLED",
                "最后一轮的共享韧性贡献无法形成下一轮公共收益。",
            )
    elif (
        company.operations.capacity_utilization_ppm < 750_000
        and company.brand.last_attempted_unfulfilled_rate_ppm == 0
    ):
        values["capacity_investment_cents"] = _adjust(
            adjustments,
            "capacity_investment_cents",
            values["capacity_investment_cents"],
            0,
            "CAPACITY_NOT_NEEDED",
            "利用率低于 75% 且上一轮无缺货，暂不追加产能。",
        )

    response_raw = request.get("incident_response", {})
    if response_raw is None:
        response_raw = {}
    if not isinstance(response_raw, Mapping):
        raise ActionValidationError("incident_response must be an object")
    requested_mode = str(response_raw.get("mode", "wait"))
    requested_repair = _integer(response_raw, "repair_budget_cents", 0)
    incident = company.risk.active_incident
    mode = IncidentResponseMode.WAIT
    repair = 0
    overhead = int(config.mapping("operating_costs")["fixed_overhead_cents"])
    repair_cash_limit = max(0, company.financial.cash_balance_cents - overhead)
    if incident is not None and requested_mode != "wait" and requested_repair > 0:
        useful = min(
            requested_repair,
            incident.remaining_repair_cents,
            int(bounds["repair_budget_cents"]["max"]),
            repair_cash_limit,
        )
        if useful >= incident.remaining_repair_cents:
            mode = IncidentResponseMode.FULL_REPAIR
            repair = incident.remaining_repair_cents
        elif useful > 0 and incident.remaining_repair_cents > 1:
            mode = IncidentResponseMode.PARTIAL_REPAIR
            repair = min(useful, incident.remaining_repair_cents - 1)
    resolved_response = {"mode": mode.value, "repair_budget_cents": repair}
    requested_response = {
        "mode": requested_mode,
        "repair_budget_cents": requested_repair,
    }
    _adjust(
        adjustments,
        "incident_response",
        requested_response,
        resolved_response,
        "INCIDENT_RESPONSE_NORMALIZED",
        "维修方式由事故状态、剩余维修额与可用现金统一确定。",
    )

    # Preserve current/future overhead before allowing discretionary spending.
    spend_fields = (
        "advertising_budget_cents",
        "service_budget_cents",
        "capacity_investment_cents",
        "resilience_budget_cents",
    ) + (
        ("shared_resilience_contribution_cents",)
        if cooperation_enabled
        else ()
    )
    available = max(
        0,
        company.financial.cash_balance_cents
        - repair
        - int(economics["minimum_cash_reserve_cents"]),
    )
    available = min(
        available, int(economics["maximum_discretionary_budget_cents"])
    )
    requested_operating = sum(values[field] for field in spend_fields)
    if requested_operating > available:
        original = {field: values[field] for field in spend_fields}
        active_fields = [field for field in spend_fields if original[field] > 0]
        allocated = 0
        for field in active_fields[:-1]:
            values[field] = original[field] * available // requested_operating
            allocated += values[field]
        if active_fields:
            values[active_fields[-1]] = available - allocated
        for field in spend_fields:
            if field not in active_fields:
                values[field] = 0
        for field in spend_fields:
            _adjust(
                adjustments,
                field,
                original[field],
                values[field],
                "LIQUIDITY_RESERVE_PROTECTED",
                "固定投入按比例缩放，并预留本轮及安全期固定运营成本。",
            )

    action = CompanyAction(
        action_id=action_id
        or f"resolved:{source}:{state.episode_id}:{state.round}:{company_id}",
        episode_id=state.episode_id,
        agent_id=company_id,
        round=state.round,
        state_version=state.state_version,
        incident_response=IncidentResponse(mode, repair),
        shared_resilience_contribution_cents=(
            values.pop("shared_resilience_contribution_cents")
            if cooperation_enabled
            else None
        ),
        strategy_summary=str(request.get("strategy_summary", source))[:500],
        **values,
    )
    # Defense in depth: the canonical action must satisfy the environment too.
    from game_theory_agent.market.validation import ActionValidator

    ActionValidator(config).validate(action, state=state, company_id=company_id).require_valid()
    return ResolvedDecision(action, source, POLICY_VERSION, tuple(adjustments))
