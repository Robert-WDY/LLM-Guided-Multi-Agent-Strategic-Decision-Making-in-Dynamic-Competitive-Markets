"""Versioned deterministic strategy population for final-market research."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import CompanyAction, IncidentResponse, MarketConfig, MarketState
from game_theory_agent.market.models import CompanyOperatingStatus

from .contracts import StrategyVersion


STRATEGIES = (
    StrategyVersion.create(
        strategy_id="balanced_competitor",
        version="1.0.0",
        label="均衡竞争",
        family="baseline",
        operationally_allowed=True,
        parameters={"policy": "seeded_rule_baseline"},
    ),
    StrategyVersion.create(
        strategy_id="aggressive_growth",
        version="1.0.0",
        label="激进增长",
        family="competition",
        operationally_allowed=True,
        parameters={
            "price_discount_cents": 800,
            "advertising_budget_cents": 1_500_000,
            "capacity_investment_cents": 1_200_000,
        },
    ),
    StrategyVersion.create(
        strategy_id="premium_defender",
        version="1.0.0",
        label="高价防守",
        family="competition",
        operationally_allowed=True,
        parameters={
            "price_markup_cents": 1_200,
            "service_budget_cents": 900_000,
            "resilience_budget_cents": 500_000,
        },
    ),
    StrategyVersion.create(
        strategy_id="contextual_defender_v1",
        version="1.0.0",
        label="情境防守",
        family="competition",
        operationally_allowed=True,
        parameters={
            "development_parent": "premium_defender",
            "fallback_parent": "balanced_competitor",
            "fallback_market_model": "value_oriented",
            "fallback_minimum_base_capacity_orders": 4_000,
            "development_seeds": [98001, 98002, 98003],
            "uses_holdout_for_parameters": False,
        },
    ),
    StrategyVersion.create(
        strategy_id="resilience_cooperator",
        version="1.0.0",
        label="公共韧性合作者",
        family="cooperation",
        operationally_allowed=True,
        parameters={
            "round_one_threshold_contribution_cents": 2_000_000,
            "shared_contribution_cents": 250_000,
            "shared_contribution_through_round": 5,
        },
    ),
    StrategyVersion.create(
        strategy_id="free_rider",
        version="1.0.0",
        label="搭便车者",
        family="cooperation",
        operationally_allowed=True,
        parameters={
            "shared_contribution_cents": 0,
            "threshold_contribution_cents": 0,
            "price_discount_cents": 150,
        },
    ),
    StrategyVersion.create(
        strategy_id="mutual_aid_reciprocal",
        version="1.0.0",
        label="互惠应急互助",
        family="mutual_aid",
        operationally_allowed=True,
        parameters={"requested_or_offered_orders": 1_500},
    ),
    StrategyVersion.create(
        strategy_id="cartel_honorer",
        version="1.0.0",
        label="价格协调遵守者",
        family="coordination",
        operationally_allowed=False,
        research_only_reason=(
            "反竞争价格协调只用于检测背叛、消费者损害与监管机制，"
            "不得晋级为生产决策策略。"
        ),
        parameters={"target_markup_cents": 1_200, "undercut_cents": 0},
    ),
    StrategyVersion.create(
        strategy_id="cartel_undercutter",
        version="1.0.0",
        label="价格协调背叛者",
        family="coordination",
        operationally_allowed=False,
        research_only_reason=(
            "背叛价格协调是重复博弈研究对照，不得晋级为生产决策策略。"
        ),
        parameters={"target_markup_cents": 1_200, "undercut_cents": 900},
    ),
    StrategyVersion.create(
        strategy_id="predatory_price_stressor",
        version="1.0.0",
        label="掠夺性低价压力源",
        family="adversarial",
        operationally_allowed=False,
        research_only_reason=(
            "故意低于可持续成本的动作只作为压力测试对手，绕过本方"
            "单位经济安全线，不能进入可晋级策略。"
        ),
        parameters={
            "below_sustainable_cost_cents": 600,
            "advertising_cash_fraction_ppm": 180_000,
        },
    ),
)
STRATEGY_BY_ID = {item.strategy_id: item for item in STRATEGIES}


def strategy_versions() -> tuple[StrategyVersion, ...]:
    return STRATEGIES


def operational_strategy_ids() -> tuple[str, ...]:
    return tuple(
        item.strategy_id for item in STRATEGIES if item.operationally_allowed
    )


def paired_partner(state: MarketState, company_id: str) -> str | None:
    company_ids = tuple(sorted(state.company_ids))
    index = company_ids.index(company_id)
    partner_index = index + 1 if index % 2 == 0 else index - 1
    partner = company_ids[partner_index]
    return (
        partner
        if partner in set(state.strategic_market.active_company_ids)
        else None
    )


def _request(base: CompanyAction) -> dict[str, Any]:
    payload = base.to_dict()
    payload["strategy_summary"] = base.strategy_summary
    return payload


def _resolved(
    config: MarketConfig,
    state: MarketState,
    company_id: str,
    strategy_id: str,
    updates: dict[str, Any],
) -> CompanyAction:
    base = build_rule_action(config, state, company_id)
    request = _request(base)
    request.update(updates)
    request["strategy_summary"] = f"self-play:{strategy_id}:v1"
    return resolve_action_request(
        config,
        state,
        company_id,
        request,
        source=f"self-play:{strategy_id}",
    ).action


def _predatory_stressor(
    config: MarketConfig, state: MarketState, company_id: str
) -> CompanyAction:
    base = build_rule_action(config, state, company_id)
    if (
        state.strategic_market.lifecycle(company_id).status
        is CompanyOperatingStatus.EXITED
    ):
        return base
    company = state.company(company_id)
    bounds = config.mapping("action", "bounds")
    sustainable = company.operations.actual_unit_cost_cents + config.integer(
        "operating_costs", "fulfillment_cost_per_order_cents"
    )
    price = max(
        int(bounds["price_cents"]["min"]),
        sustainable - 600,
    )
    overhead = config.integer("operating_costs", "fixed_overhead_cents")
    available = max(0, company.financial.cash_balance_cents - overhead)
    advertising = min(
        int(bounds["advertising_budget_cents"]["max"]),
        available * 180_000 // 1_000_000,
    )
    return CompanyAction(
        action_id=(
            f"self-play:predatory:{state.episode_id}:{state.round}:{company_id}"
        ),
        episode_id=state.episode_id,
        agent_id=company_id,
        round=state.round,
        state_version=state.state_version,
        price_cents=price,
        advertising_budget_cents=advertising,
        shared_resilience_contribution_cents=0,
        threshold_project_contribution_cents=0,
        mutual_aid_capacity_offer_orders=0,
        mutual_aid_capacity_request_orders=0,
        incident_response=IncidentResponse(),
        strategy_summary=(
            "self-play:predatory_price_stressor:research-only adversarial action"
        ),
    )


def build_strategy_action(
    config: MarketConfig,
    state: MarketState,
    company_id: str,
    strategy_id: str,
) -> CompanyAction:
    """Build one legal action from a frozen strategy version."""

    if strategy_id not in STRATEGY_BY_ID:
        raise KeyError(strategy_id)
    if (
        state.strategic_market.lifecycle(company_id).status
        is CompanyOperatingStatus.EXITED
    ):
        return build_rule_action(config, state, company_id)
    if strategy_id == "balanced_competitor":
        return replace(
            build_rule_action(config, state, company_id),
            strategy_summary="self-play:balanced_competitor:v1",
        )
    if strategy_id == "predatory_price_stressor":
        return _predatory_stressor(config, state, company_id)

    if strategy_id == "contextual_defender_v1":
        company = state.company(company_id)
        parent = (
            "balanced_competitor"
            if state.market.market_model_id == "value_oriented"
            and company.operations.base_capacity_orders >= 4_000
            else "premium_defender"
        )
        return replace(
            build_strategy_action(config, state, company_id, parent),
            strategy_summary=(
                f"self-play:contextual_defender_v1:selected={parent}"
            ),
        )

    base = build_rule_action(config, state, company_id)
    price_bounds = config.mapping("action", "bounds")["price_cents"]
    if strategy_id == "aggressive_growth":
        return _resolved(
            config,
            state,
            company_id,
            strategy_id,
            {
                "price_cents": max(
                    int(price_bounds["min"]), base.price_cents - 800
                ),
                "advertising_budget_cents": 1_500_000,
                "capacity_investment_cents": (
                    1_200_000 if state.rounds_remaining > 2 else 0
                ),
            },
        )
    if strategy_id == "premium_defender":
        return _resolved(
            config,
            state,
            company_id,
            strategy_id,
            {
                "price_cents": min(
                    int(price_bounds["max"]),
                    state.market.price_anchor_cents + 1_200,
                ),
                "service_budget_cents": 900_000,
                "resilience_budget_cents": (
                    500_000 if state.risk_signals and state.rounds_remaining > 1 else 0
                ),
            },
        )
    if strategy_id == "resilience_cooperator":
        project = state.strategic_market.threshold_project
        project_open = bool(
            project is not None
            and project.status.value == "active"
            and state.round <= project.deadline_round
        )
        return _resolved(
            config,
            state,
            company_id,
            strategy_id,
            {
                "shared_resilience_contribution_cents": (
                    250_000
                    if state.round <= 5 and state.rounds_remaining > 1
                    else 0
                ),
                "threshold_project_contribution_cents": (
                    2_000_000 if project_open and state.round == 1 else 0
                ),
            },
        )
    if strategy_id == "free_rider":
        return _resolved(
            config,
            state,
            company_id,
            strategy_id,
            {
                "price_cents": max(
                    int(price_bounds["min"]), base.price_cents - 150
                ),
                "shared_resilience_contribution_cents": 0,
                "threshold_project_contribution_cents": 0,
            },
        )
    if strategy_id == "mutual_aid_reciprocal":
        partner = paired_partner(state, company_id)
        index = tuple(sorted(state.company_ids)).index(company_id)
        return _resolved(
            config,
            state,
            company_id,
            strategy_id,
            {
                "mutual_aid_partner_company_id": partner,
                "mutual_aid_capacity_request_orders": (
                    1_500 if partner is not None and index % 2 == 0 else 0
                ),
                "mutual_aid_capacity_offer_orders": (
                    1_500 if partner is not None and index % 2 == 1 else 0
                ),
            },
        )
    if strategy_id in {"cartel_honorer", "cartel_undercutter"}:
        partner = paired_partner(state, company_id)
        target = min(
            int(price_bounds["max"]), state.market.price_anchor_cents + 1_200
        )
        undercut = 900 if strategy_id == "cartel_undercutter" else 0
        return _resolved(
            config,
            state,
            company_id,
            strategy_id,
            {
                "price_cents": max(int(price_bounds["min"]), target - undercut),
                "price_coordination_partner_company_id": partner,
                "price_coordination_target_cents": target if partner else None,
            },
        )
    raise AssertionError(f"unhandled strategy: {strategy_id}")
