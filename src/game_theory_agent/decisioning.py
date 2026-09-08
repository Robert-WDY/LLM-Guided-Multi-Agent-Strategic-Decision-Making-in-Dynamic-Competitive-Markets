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
STRATEGIC_POLICY_VERSION = "decision-policy-v1.2.0"
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

    threshold_project = (
        state.strategic_market.threshold_project
        if state.strategic_market is not None
        else None
    )
    project_enabled = (
        threshold_project is not None
        and threshold_project.status.value == "active"
        and state.round <= threshold_project.deadline_round
    )
    requested_project_raw = request.get(
        "threshold_project_contribution_cents", 0
    )
    requested_project = (
        0
        if requested_project_raw is None
        else _integer(
            {"threshold_project_contribution_cents": requested_project_raw},
            "threshold_project_contribution_cents",
            0,
        )
    )
    if project_enabled:
        project_bounds = bounds["threshold_project_contribution_cents"]
        values["threshold_project_contribution_cents"] = _clamp(
            adjustments,
            "threshold_project_contribution_cents",
            requested_project,
            int(project_bounds["min"]),
            int(project_bounds["max"]),
        )
    else:
        _adjust(
            adjustments,
            "threshold_project_contribution_cents",
            requested_project,
            0,
            "THRESHOLD_PROJECT_DISABLED",
            "当前没有可接受贡献的阈值合作项目。",
        )

    mutual_aid_enabled = bool(
        state.strategic_market is not None
        and state.strategic_market.mutual_aid_enabled
    )
    requested_partner_raw = request.get("mutual_aid_partner_company_id")
    requested_partner = (
        str(requested_partner_raw).strip()
        if requested_partner_raw is not None
        else None
    )
    requested_offer_raw = request.get("mutual_aid_capacity_offer_orders", 0)
    requested_request_raw = request.get(
        "mutual_aid_capacity_request_orders", 0
    )
    requested_offer = (
        0
        if requested_offer_raw is None
        else _integer(
            {"mutual_aid_capacity_offer_orders": requested_offer_raw},
            "mutual_aid_capacity_offer_orders",
            0,
        )
    )
    requested_request = (
        0
        if requested_request_raw is None
        else _integer(
            {"mutual_aid_capacity_request_orders": requested_request_raw},
            "mutual_aid_capacity_request_orders",
            0,
        )
    )
    resolved_partner: str | None = None
    resolved_offer = 0
    resolved_request = 0
    if mutual_aid_enabled:
        for field, requested_value in (
            ("mutual_aid_capacity_offer_orders", requested_offer),
            ("mutual_aid_capacity_request_orders", requested_request),
        ):
            mutual_bounds = bounds[field]
            values[field] = _clamp(
                adjustments,
                field,
                requested_value,
                int(mutual_bounds["min"]),
                int(mutual_bounds["max"]),
            )
        resolved_offer = values.pop("mutual_aid_capacity_offer_orders")
        resolved_request = values.pop("mutual_aid_capacity_request_orders")
        valid_partners = set(state.strategic_market.active_company_ids) - {
            company_id
        }
        if (resolved_offer or resolved_request) and (
            requested_partner not in valid_partners
        ):
            _adjust(
                adjustments,
                "mutual_aid_partner_company_id",
                requested_partner,
                None,
                "MUTUAL_AID_PARTNER_INVALID",
                "互助对象必须是仍在经营的其他公司；无效互助意向已清零。",
            )
            resolved_offer = 0
            resolved_request = 0
        elif resolved_offer or resolved_request:
            resolved_partner = requested_partner
    else:
        for field, requested_value in (
            ("mutual_aid_capacity_offer_orders", requested_offer),
            ("mutual_aid_capacity_request_orders", requested_request),
        ):
            _adjust(
                adjustments,
                field,
                requested_value,
                0,
                "MUTUAL_AID_DISABLED",
                "当前 Episode 未启用公司间应急互助。",
            )
        _adjust(
            adjustments,
            "mutual_aid_partner_company_id",
            requested_partner,
            None,
            "MUTUAL_AID_DISABLED",
            "当前 Episode 未启用公司间应急互助。",
        )

    price_coordination_enabled = bool(
        state.strategic_market is not None
        and state.strategic_market.price_coordination_enabled
    )
    coordination_partner_raw = request.get(
        "price_coordination_partner_company_id"
    )
    coordination_partner = (
        str(coordination_partner_raw).strip()
        if coordination_partner_raw is not None
        else None
    )
    coordination_target_raw = request.get("price_coordination_target_cents")
    coordination_target = (
        _integer(
            {"price_coordination_target_cents": coordination_target_raw},
            "price_coordination_target_cents",
            company.commercial.price_cents,
        )
        if coordination_target_raw is not None
        else None
    )
    resolved_coordination_partner: str | None = None
    resolved_coordination_target: int | None = None
    if price_coordination_enabled and (
        coordination_partner is not None and coordination_target is not None
    ):
        valid_partners = set(state.strategic_market.active_company_ids) - {
            company_id
        }
        if coordination_partner not in valid_partners:
            _adjust(
                adjustments,
                "price_coordination_partner_company_id",
                coordination_partner,
                None,
                "PRICE_COORDINATION_PARTNER_INVALID",
                "价格协调对象必须是仍在经营的其他公司；无效声明已撤销。",
            )
        else:
            price_bounds = bounds["price_cents"]
            resolved_coordination_partner = coordination_partner
            resolved_coordination_target = _clamp(
                adjustments,
                "price_coordination_target_cents",
                coordination_target,
                int(price_bounds["min"]),
                int(price_bounds["max"]),
            )
    elif coordination_partner is not None or coordination_target is not None:
        reason_code = (
            "PRICE_COORDINATION_INCOMPLETE"
            if price_coordination_enabled
            else "PRICE_COORDINATION_DISABLED"
        )
        reason = (
            "价格协调对象与目标价必须同时提供；不完整声明已撤销。"
            if price_coordination_enabled
            else "当前 Episode 未启用价格协调研究机制。"
        )
        _adjust(
            adjustments,
            "price_coordination_partner_company_id",
            coordination_partner,
            None,
            reason_code,
            reason,
        )
        _adjust(
            adjustments,
            "price_coordination_target_cents",
            coordination_target,
            None,
            reason_code,
            reason,
        )

    company_is_active = not (
        state.strategic_market is not None
        and state.strategic_market.lifecycle(company_id).status.value == "exited"
    )
    supply_chain_enabled = state.supply_chain is not None and company_is_active
    requested_primary_raw = request.get("primary_supplier_id")
    requested_backup_raw = request.get("backup_supplier_id")
    requested_primary = (
        str(requested_primary_raw).strip()
        if requested_primary_raw is not None
        else None
    )
    requested_backup = (
        str(requested_backup_raw).strip()
        if requested_backup_raw is not None
        else None
    )
    requested_share_raw = request.get("primary_supplier_share_ppm")
    requested_share = (
        _integer(
            {"primary_supplier_share_ppm": requested_share_raw},
            "primary_supplier_share_ppm",
            1_000_000,
        )
        if requested_share_raw is not None
        else 1_000_000
    )
    resolved_primary: str | None = None
    resolved_backup: str | None = None
    resolved_primary_share: int | None = None
    if supply_chain_enabled:
        assert state.supply_chain is not None
        supply_cfg = config.mapping("supply_chain")
        supplier_ids = set(state.supply_chain.supplier_ids)
        default_supplier = str(supply_cfg["default_supplier_id"])
        if requested_primary not in supplier_ids:
            resolved_primary = _adjust(
                adjustments,
                "primary_supplier_id",
                requested_primary,
                default_supplier,
                "SUPPLIER_DEFAULTED",
                "主供应商无效或未填写，已使用市场默认供应商。",
            )
        else:
            resolved_primary = requested_primary
        if (
            requested_backup in supplier_ids
            and requested_backup != resolved_primary
        ):
            resolved_backup = requested_backup
            share_bounds = bounds["primary_supplier_share_ppm"]
            resolved_primary_share = _clamp(
                adjustments,
                "primary_supplier_share_ppm",
                requested_share,
                int(share_bounds["min"]),
                int(share_bounds["max"]),
            )
        else:
            if requested_backup is not None:
                _adjust(
                    adjustments,
                    "backup_supplier_id",
                    requested_backup,
                    None,
                    "BACKUP_SUPPLIER_INVALID",
                    "备用供应商必须存在且不同于主供应商。",
                )
            resolved_primary_share = _adjust(
                adjustments,
                "primary_supplier_share_ppm",
                requested_share,
                1_000_000,
                "SINGLE_SOURCE_NORMALIZED",
                "单一来源采购的主供应商份额固定为100%。",
            )
    else:
        for field, requested_value in (
            ("primary_supplier_id", requested_primary),
            ("backup_supplier_id", requested_backup),
            ("primary_supplier_share_ppm", requested_share_raw),
        ):
            _adjust(
                adjustments,
                field,
                requested_value,
                None,
                "SUPPLY_CHAIN_DISABLED",
                "当前市场或公司状态不允许供应链采购选择。",
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
        if project_enabled:
            values["threshold_project_contribution_cents"] = _adjust(
                adjustments,
                "threshold_project_contribution_cents",
                values["threshold_project_contribution_cents"],
                0,
                "LAST_ROUND_DISABLED",
                "最后一轮不能完成后续生效的阈值合作项目。",
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
    ) + (
        ("threshold_project_contribution_cents",)
        if project_enabled
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
        threshold_project_contribution_cents=(
            values.pop("threshold_project_contribution_cents")
            if project_enabled
            # Preserve the configured strategic action schema after the
            # project reaches a terminal status.  Market validation
            # canonicalizes a present project to an explicit zero; returning
            # None here made the controller receipt hash differ from the
            # MarketEnv-executed joint action from the first post-deadline
            # round onward.
            else 0
            if threshold_project is not None
            else None
        ),
        mutual_aid_partner_company_id=resolved_partner,
        mutual_aid_capacity_offer_orders=(
            resolved_offer if mutual_aid_enabled else None
        ),
        mutual_aid_capacity_request_orders=(
            resolved_request if mutual_aid_enabled else None
        ),
        price_coordination_partner_company_id=(
            resolved_coordination_partner
        ),
        price_coordination_target_cents=resolved_coordination_target,
        **{key:(_clamp(adjustments,key,_integer(request,key,0),0,maximum) if config.data.get("supply_chain",{}).get("strategic_policy") and request.get(key) is not None else None) for key,maximum in (("contract_quantity_orders",company.operations.base_capacity_orders),("contract_duration_rounds",5),("contract_bid_cents",100000))},
        primary_supplier_id=resolved_primary,
        backup_supplier_id=resolved_backup,
        primary_supplier_share_ppm=resolved_primary_share,
        procurement_quantity_orders=(
            _clamp(adjustments,"procurement_quantity_orders",_integer(request,"procurement_quantity_orders",0),0,company.operations.base_capacity_orders)
            if config.data.get("autonomous_market") and request.get("procurement_quantity_orders") is not None else None
        ),
        strategy_summary=str(request.get("strategy_summary", source))[:500],
        **values,
    )
    # Defense in depth: the canonical action must satisfy the environment too.
    from game_theory_agent.market.validation import ActionValidator

    ActionValidator(config).validate(action, state=state, company_id=company_id).require_valid()
    return ResolvedDecision(
        action,
        source,
        STRATEGIC_POLICY_VERSION if state.strategic_market is not None else POLICY_VERSION,
        tuple(adjustments),
    )
