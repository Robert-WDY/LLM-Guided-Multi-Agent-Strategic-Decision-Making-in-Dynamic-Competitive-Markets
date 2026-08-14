"""Single-company gameplay, rule opponents, and evidence-based retrospectives."""

from __future__ import annotations

from statistics import mean
from typing import Any, Sequence

from game_theory_agent.market import (
    CompanyAction,
    IncidentResponse,
    IncidentResponseMode,
    MarketConfig,
    MarketState,
)
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition
from game_theory_agent.market.protocols import ComponentRng


PPM = 1_000_000


def _clip(value: int, low: int, high: int) -> int:
    return min(max(value, low), high)


def _status(value: float, warning: float, healthy: float) -> str:
    if value >= healthy:
        return "healthy"
    if value >= warning:
        return "watch"
    return "risk"


def settled_market_snapshot(transition: MarketTransition) -> dict[str, Any]:
    """Return one internally consistent view of the round that just settled.

    The state after a step intentionally contains next-round sentiment, supply
    pressure and events. Realized demand and paid price, however, describe the
    round that just settled. Public history must not combine those two clocks.
    """

    before = transition.state_before
    after = transition.state_after
    actual_supply = before.market.base_supply_cost_index_ppm
    for event in before.active_market_events:
        actual_supply = round(actual_supply * event.supply_cost_multiplier_ppm / PPM)
    return {
        "base_demand_orders": before.market.base_demand_orders,
        "realized_demand_orders": after.market.realized_demand_orders,
        "no_purchase_orders": after.market.no_purchase_orders,
        "lost_after_stockout_orders": after.market.lost_after_stockout_orders,
        "market_sentiment_ppm": before.market.market_sentiment_ppm,
        "base_supply_cost_index_ppm": before.market.base_supply_cost_index_ppm,
        "actual_supply_cost_index_ppm": actual_supply,
        "average_paid_price_cents": after.market.average_paid_price_cents,
        "market_model_id": before.market.market_model_id,
        "market_model_label": before.market.market_model_label,
        "market_model_description": before.market.market_model_description,
        "demand_bias_ppm": before.market.demand_bias_ppm,
        "price_anchor_cents": before.market.price_anchor_cents,
        "price_band_cents": before.market.price_band_cents,
        "utility_multipliers_ppm": {
            "price": before.market.utility_price_multiplier_ppm,
            "awareness": before.market.utility_awareness_multiplier_ppm,
            "service": before.market.utility_service_multiplier_ppm,
            "reputation": before.market.utility_reputation_multiplier_ppm,
            "prior_stockout": before.market.utility_prior_stockout_multiplier_ppm,
        },
    }


def build_company_analysis(
    state: MarketState,
    company_id: str,
    config: MarketConfig | None = None,
) -> dict[str, Any]:
    """Explain the decision-relevant company state without inventing causality."""

    company = state.company(company_id)
    competitors = [item for item in state.companies if item.company_id != company_id]
    average_share = PPM // len(state.companies)
    cash_ratio = company.financial.cash_balance_cents / 30_000_000
    capacity_buffer = max(
        0,
        company.operations.effective_capacity_orders - company.commercial.sales_orders,
    )
    capacity_buffer_ratio = capacity_buffer / max(
        1, company.operations.effective_capacity_orders
    )
    fulfillment_cost = (
        config.integer("operating_costs", "fulfillment_cost_per_order_cents")
        if config is not None
        else 0
    )
    margin_per_order = (
        company.commercial.price_cents
        - company.operations.actual_unit_cost_cents
        - fulfillment_cost
    )
    relative_price = company.commercial.price_cents - round(
        mean(item.commercial.price_cents for item in competitors)
    )
    factors = [
        {
            "key": "liquidity",
            "label": "现金安全",
            "value_ppm": _clip(round(cash_ratio * PPM), 0, 2 * PPM),
            "status": _status(cash_ratio, 0.45, 0.8),
            "summary": f"可用现金 {company.financial.cash_balance_cents / 100:,.0f} 元",
        },
        {
            "key": "market_position",
            "label": "市场位置",
            "value_ppm": company.commercial.market_share_ppm,
            "status": _status(
                company.commercial.market_share_ppm / max(1, average_share), 0.8, 1.05
            ),
            "summary": (
                f"份额 {company.commercial.market_share_ppm / 10_000:.1f}%"
                f"，均衡基准 {average_share / 10_000:.1f}%"
            ),
        },
        {
            "key": "capacity",
            "label": "履约余量",
            "value_ppm": round(capacity_buffer_ratio * PPM),
            "status": _status(capacity_buffer_ratio, 0.08, 0.2),
            "summary": (
                f"剩余 {capacity_buffer:,} 单，利用率 "
                f"{company.operations.capacity_utilization_ppm / 10_000:.0f}%"
            ),
        },
        {
            "key": "brand",
            "label": "品牌资产",
            "value_ppm": round(
                mean(
                    (
                        company.brand.brand_awareness_ppm,
                        company.brand.service_quality_ppm,
                        company.brand.reputation_ppm,
                    )
                )
            ),
            "status": _status(
                mean(
                    (
                        company.brand.brand_awareness_ppm,
                        company.brand.service_quality_ppm,
                        company.brand.reputation_ppm,
                    )
                )
                / PPM,
                0.45,
                0.65,
            ),
            "summary": (
                f"知名度 {company.brand.brand_awareness_ppm / 10_000:.0f}% · "
                f"服务 {company.brand.service_quality_ppm / 10_000:.0f}% · "
                f"声誉 {company.brand.reputation_ppm / 10_000:.0f}%"
            ),
        },
        {
            "key": "risk_readiness",
            "label": "风险准备",
            "value_ppm": company.risk.resilience_ppm,
            "status": _status(company.risk.resilience_ppm / PPM, 0.2, 0.45),
            "summary": (
                f"韧性 {company.risk.resilience_ppm / 10_000:.0f}% · "
                f"预警 {len(state.risk_signals)} 个"
            ),
        },
    ]
    score = round(
        mean(
            (
                min(cash_ratio, 1.0),
                min(company.commercial.market_share_ppm / max(1, average_share), 1.0),
                min(capacity_buffer_ratio / 0.25, 1.0),
                company.brand.reputation_ppm / PPM,
                company.risk.resilience_ppm / PPM,
            )
        )
        * 100
    )

    recommendations: list[dict[str, str]] = []
    incident = company.risk.active_incident
    if incident:
        recommendations.append(
            {
                "priority": "critical",
                "dimension": "repair",
                "title": "先评估事故维修",
                "rationale": (
                    f"{incident.incident_type} 仍持续 {incident.remaining_rounds} 轮，"
                    f"剩余维修成本 {(incident.remaining_repair_cents / 100):,.0f} 元。"
                ),
            }
        )
    if state.risk_signals and state.rounds_remaining > 1:
        strongest = max(
            state.risk_signals, key=lambda item: item.estimated_probability_ppm
        )
        recommendations.append(
            {
                "priority": "high",
                "dimension": "resilience",
                "title": "利用预警提前配置韧性",
                "rationale": (
                    f"{strongest.event_type} 指向 R{strongest.target_round}，"
                    f"估计发生概率 {strongest.estimated_probability_ppm / 10_000:.0f}%。"
                ),
            }
        )
    if company.operations.capacity_utilization_ppm >= 850_000:
        recommendations.append(
            {
                "priority": "high",
                "dimension": "capacity",
                "title": "产能接近瓶颈",
                "rationale": "高利用率会放大需求上升时的缺货和声誉损失；投资下一轮生效。",
            }
        )
    if company.brand.reputation_ppm < 550_000:
        recommendations.append(
            {
                "priority": "medium",
                "dimension": "service",
                "title": "修复服务与声誉",
                "rationale": "服务投入本轮影响消费者选择，并通过服务质量和声誉保留到后续轮次。",
            }
        )
    if relative_price > 1_000:
        recommendations.append(
            {
                "priority": "medium",
                "dimension": "price",
                "title": "价格明显高于竞争者",
                "rationale": f"当前报价比竞争者均价高 {relative_price / 100:.2f} 元，需确认高毛利是否足以补偿需求下降。",
            }
        )
    if margin_per_order < 1_500:
        recommendations.append(
            {
                "priority": "high",
                "dimension": "price",
                "title": "单位毛利空间偏薄",
                "rationale": f"报价扣除商品与逐单履约成本后仅贡献 {margin_per_order / 100:.2f} 元，固定运营、投入和供应冲击会进一步压缩利润。",
            }
        )
    if cash_ratio < 0.55:
        recommendations.append(
            {
                "priority": "high",
                "dimension": "cash",
                "title": "保留现金缓冲",
                "rationale": "现金下降会同时限制固定投入和负贡献订单的可履约数量。",
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "priority": "normal",
                "dimension": "balance",
                "title": "状态相对均衡",
                "rationale": "当前没有明显硬约束，可围绕份额增长与现金留存做主动取舍。",
            }
        )
    return {
        "company_id": company_id,
        "round": state.round,
        "health_score": score,
        "health_label": "稳健" if score >= 70 else "观察" if score >= 50 else "承压",
        "factors": factors,
        "recommendations": recommendations[:4],
        "decision_context": {
            "margin_per_order_cents": margin_per_order,
            "fulfillment_cost_per_order_cents": fulfillment_cost,
            "relative_price_cents": relative_price,
            "capacity_buffer_orders": capacity_buffer,
            "rounds_remaining": state.rounds_remaining,
        },
    }


def build_rule_action(
    config: MarketConfig, state: MarketState, company_id: str
) -> CompanyAction:
    """A seeded, varied rule policy; it is deliberately not an Agent."""

    company = state.company(company_id)
    bounds = config.mapping("action", "bounds")
    cash = company.financial.cash_balance_cents
    last_round = state.rounds_remaining <= 1
    warning = bool(state.risk_signals)
    weak_share = company.commercial.market_share_ppm < PPM // len(state.companies)
    supply_pressure = state.market.actual_supply_cost_index_ppm > 1_120_000

    # Each company receives a stable operating style for the episode, plus a
    # small reproducible round variation. Different Seeds create different
    # opponent mixes without introducing learning agents or hidden model calls.
    profile_rng = ComponentRng(
        config.rng_protocol_version,
        state.episode_seed,
        0,
        "rule_opponent_policy",
        company_id,
        0,
    )
    profile = profile_rng.weighted_choice(
        {"value": 270_000, "premium": 230_000, "growth": 260_000, "cautious": 240_000}
    )
    profiles = {
        "value": (-300, 1_100_000, 850_000, 840_000),
        "premium": (400, 850_000, 1_250_000, 850_000),
        "growth": (-100, 1_300_000, 1_100_000, 760_000),
        "cautious": (200, 700_000, 950_000, 880_000),
    }
    price_bias, ad_factor, service_factor, capacity_trigger = profiles[profile]
    round_rng = ComponentRng(
        config.rng_protocol_version,
        state.episode_seed,
        state.round,
        "rule_opponent_policy",
        company_id,
        1,
    )
    price_jitter = (round(round_rng.uniform() * 6) - 3) * 50
    budget_jitter_ppm = 900_000 + round(round_rng.uniform() * 200_000)
    high_utilization = (
        company.operations.capacity_utilization_ppm >= capacity_trigger
    )

    fulfillment_cost = config.integer(
        "operating_costs", "fulfillment_cost_per_order_cents"
    )
    cost_reference = (
        company.operations.actual_unit_cost_cents + fulfillment_cost + 1_800
    )
    paid_reference = (
        state.market.average_paid_price_cents
        if state.market.average_paid_price_cents > 0
        else state.market.price_anchor_cents
    )
    # Re-anchor every round to market willingness-to-pay and unit economics.
    # Using the company's previous quote as a hard floor created an artificial
    # one-way price ratchet even when costs and demand were mean reverting.
    target_price = round(
        state.market.price_anchor_cents * 0.45
        + paid_reference * 0.30
        + cost_reference * 0.25
    )
    if weak_share:
        target_price -= 200
    if high_utilization or supply_pressure:
        target_price += 250
    target_price += price_bias + price_jitter
    model_floor = max(
        int(bounds["price_cents"]["min"]),
        state.market.price_anchor_cents - state.market.price_band_cents,
        company.operations.actual_unit_cost_cents + fulfillment_cost + 600,
    )
    model_ceiling = min(
        int(bounds["price_cents"]["max"]),
        max(
            state.market.price_anchor_cents + state.market.price_band_cents,
            cost_reference + 700,
        ),
    )
    price = _clip(
        target_price,
        model_floor,
        model_ceiling,
    )

    incident = company.risk.active_incident
    response = IncidentResponse()
    repair = 0
    if incident:
        if incident.remaining_repair_cents <= cash // 4:
            repair = incident.remaining_repair_cents
            response = IncidentResponse(IncidentResponseMode.FULL_REPAIR, repair)
        elif incident.remaining_repair_cents > 1 and cash >= 2:
            repair = min(incident.remaining_repair_cents - 1, max(1, cash // 10))
            response = IncidentResponse(IncidentResponseMode.PARTIAL_REPAIR, repair)

    operating_overhead = config.integer("operating_costs", "fixed_overhead_cents")
    available = max(0, cash - repair - operating_overhead)
    operating_envelope = min(available, max(0, cash // 6))
    advertising = min(
        operating_envelope // 3,
        900_000 if company.brand.brand_awareness_ppm < 550_000 else 500_000,
    )
    advertising = _ppm_scale(advertising, ad_factor, budget_jitter_ppm)
    service = min(
        max(0, operating_envelope - advertising) // 2,
        900_000 if company.brand.reputation_ppm < 620_000 else 500_000,
    )
    service = _ppm_scale(service, service_factor, budget_jitter_ppm)
    if advertising + service > operating_envelope:
        service = max(0, operating_envelope - advertising)
    remaining = max(0, operating_envelope - advertising - service)
    resilience = 0
    if warning and not last_round:
        resilience = min(
            remaining, 900_000 if profile == "cautious" else 700_000
        )
    remaining -= resilience
    capacity = 0
    if high_utilization and not last_round:
        capacity = min(remaining, 800_000)

    return CompanyAction(
        action_id=f"rule:{state.episode_id}:{state.round}:{company_id}",
        episode_id=state.episode_id,
        agent_id=company_id,
        round=state.round,
        state_version=state.state_version,
        price_cents=price,
        advertising_budget_cents=advertising,
        service_budget_cents=service,
        capacity_investment_cents=capacity,
        resilience_budget_cents=resilience,
        incident_response=response,
        strategy_summary=(
            f"rule-opponent:{profile}: seeded variation; liquidity, margin, "
            "capacity and warning aware"
        ),
    )


def _ppm_scale(value: int, *factors: int) -> int:
    result = value
    for factor in factors:
        result = round(result * factor / PPM)
    return max(0, result)


def _round_reasons(
    transition: MarketTransition, player_company_id: str
) -> tuple[list[str], str]:
    before = transition.state_before
    after = transition.state_after
    before_company = before.company(player_company_id)
    after_company = after.company(player_company_id)
    actions = dict(transition.joint_action)
    action = actions[player_company_id]
    average_price = round(mean(item.price_cents for item in actions.values()))
    share_delta = (
        after_company.commercial.market_share_ppm
        - before_company.commercial.market_share_ppm
    )
    reasons: list[str] = []
    if action.price_cents < average_price - 300 and share_delta > 0:
        reasons.append("相对低价与份额上升同时出现，价格竞争可能是本轮增量来源。")
    elif action.price_cents > average_price + 300 and share_delta < 0:
        reasons.append("报价高于市场均值且份额下降，高价策略未被品牌或服务完全支撑。")
    elif action.price_cents > average_price + 300:
        reasons.append("高于均价仍维持份额，品牌、服务或产能可能支撑了溢价。")
    if after_company.commercial.attempted_unfulfilled_orders > 0:
        reasons.append(
            f"发生 {after_company.commercial.attempted_unfulfilled_orders} 单首次缺货，限制销量并损害声誉。"
        )
    if action.advertising_budget_cents > 0:
        awareness_delta = (
            after_company.brand.brand_awareness_ppm
            - before_company.brand.brand_awareness_ppm
        )
        reasons.append(
            f"广告投入 {action.advertising_budget_cents / 100:,.0f} 元，知名度变化 {awareness_delta / 10_000:+.1f} 个百分点。"
        )
    if action.service_budget_cents > 0:
        reputation_delta = (
            after_company.brand.reputation_ppm - before_company.brand.reputation_ppm
        )
        reasons.append(
            f"服务投入 {action.service_budget_cents / 100:,.0f} 元，声誉变化 {reputation_delta / 10_000:+.1f} 个百分点。"
        )
    if action.capacity_investment_cents > 0:
        reasons.append(
            f"产能投资 {action.capacity_investment_cents / 100:,.0f} 元从下一轮生效，当前利润先承担投入成本。"
        )
    if action.resilience_budget_cents > 0:
        reasons.append(
            f"韧性投入 {action.resilience_budget_cents / 100:,.0f} 元只保护后续回合，不抵消本轮已激活事件。"
        )
    if before.active_market_events:
        labels = "、".join(event.event_type for event in before.active_market_events)
        reasons.append(f"本轮受到市场事件 {labels} 的需求、成本或履约冲击。")
    if before_company.risk.active_incident:
        if action.incident_response.mode is IncidentResponseMode.WAIT:
            reasons.append("公司事故选择等待，节省现金但继续承受当轮运营损失。")
        else:
            reasons.append(
                f"事故采用 {action.incident_response.mode.value}，维修降低本轮影响；完全维修从下一轮清除事故。"
            )
    if after_company.financial.round_profit_cents < 0:
        verdict = "承压"
        reasons.append("本轮收入未覆盖商品、履约、固定运营与主动投入，现金价值下降。")
    elif share_delta >= 0:
        verdict = "有效增长"
    elif after_company.financial.round_profit_cents > 0:
        verdict = "盈利但失份额"
    else:
        verdict = "效果有限"
    return reasons[:5], verdict


def _value_breakdown(company: Any, config: MarketConfig) -> dict[str, int]:
    terminal_cfg = config.mapping("terminal")
    capacity_salvage = _ppm_scale(
        company.financial.capacity_book_value_cents,
        int(terminal_cfg["capacity_salvage_rate_ppm"]),
    )
    awareness_value = _ppm_scale(
        int(terminal_cfg["awareness_value_max_cents"]),
        company.brand.brand_awareness_ppm,
    )
    service_value = _ppm_scale(
        int(terminal_cfg["service_value_max_cents"]),
        company.brand.service_quality_ppm,
    )
    reputation_value = _ppm_scale(
        int(terminal_cfg["reputation_value_max_cents"]),
        company.brand.reputation_ppm,
    )
    resilience_value = _ppm_scale(
        int(terminal_cfg["resilience_value_max_cents"]),
        company.risk.resilience_ppm,
    )
    cash = company.financial.cash_balance_cents
    total_assets = cash + company.financial.capacity_book_value_cents
    enterprise_value = (
        cash
        + capacity_salvage
        + awareness_value
        + service_value
        + reputation_value
        + resilience_value
    )
    return {
        "cash_cents": cash,
        "capacity_book_value_cents": company.financial.capacity_book_value_cents,
        "total_assets_cents": total_assets,
        "capacity_salvage_cents": capacity_salvage,
        "awareness_value_cents": awareness_value,
        "service_value_cents": service_value,
        "reputation_value_cents": reputation_value,
        "resilience_value_cents": resilience_value,
        "enterprise_value_cents": enterprise_value,
    }


def build_terminal_rankings(state: MarketState, config: MarketConfig) -> dict[str, Any]:
    breakdowns = {
        company.company_id: _value_breakdown(company, config)
        for company in state.companies
    }

    def rows(value_key: str) -> list[dict[str, Any]]:
        order = sorted(
            state.company_ids,
            key=lambda company_id: breakdowns[company_id][value_key],
            reverse=True,
        )
        return [
            {
                "rank": index + 1,
                "company_id": company_id,
                "value_cents": breakdowns[company_id][value_key],
                "breakdown": breakdowns[company_id],
            }
            for index, company_id in enumerate(order)
        ]

    return {
        "composite": rows("enterprise_value_cents"),
        "total_assets": rows("total_assets_cents"),
        "methodology": {
            "composite": "综合价值 = 现金 + 产能残值 + 品牌知名度价值 + 服务价值 + 声誉价值 + 韧性价值。",
            "total_assets": "总资产 = 现金 + 产能账面价值，不包含品牌、服务、声誉与韧性估值。",
        },
    }


def build_retrospective(
    manifest: EpisodeManifest,
    transitions: Sequence[MarketTransition],
    player_company_id: str,
    config: MarketConfig,
) -> dict[str, Any]:
    initial_state = manifest.initial_state
    initial = initial_state.company(player_company_id)
    final_state = transitions[-1].state_after if transitions else initial_state
    final = final_state.company(player_company_id)
    breakdowns = {
        company.company_id: _value_breakdown(company, config)
        for company in final_state.companies
    }
    composite_ranking = sorted(
        final_state.company_ids,
        key=lambda company_id: breakdowns[company_id]["enterprise_value_cents"],
        reverse=True,
    )
    asset_ranking = sorted(
        final_state.company_ids,
        key=lambda company_id: breakdowns[company_id]["total_assets_cents"],
        reverse=True,
    )
    rank = composite_ranking.index(player_company_id) + 1
    asset_rank = asset_ranking.index(player_company_id) + 1

    def ranking_rows(order: list[str], value_key: str) -> list[dict[str, Any]]:
        return [
            {
                "rank": index + 1,
                "company_id": company_id,
                "value_cents": breakdowns[company_id][value_key],
                "breakdown": breakdowns[company_id],
            }
            for index, company_id in enumerate(order)
        ]

    rounds: list[dict[str, Any]] = []
    previous_market: dict[str, Any] | None = None
    for transition in transitions:
        before = transition.state_before
        after = transition.state_after
        before_company = before.company(player_company_id)
        after_company = after.company(player_company_id)
        action = dict(transition.joint_action)[player_company_id]
        reasons, verdict = _round_reasons(transition, player_company_id)
        market_result = settled_market_snapshot(transition)
        previous_demand = (
            previous_market["realized_demand_orders"]
            if previous_market
            else market_result["base_demand_orders"]
        )
        previous_supply = (
            previous_market["actual_supply_cost_index_ppm"]
            if previous_market
            else PPM
        )
        value_before = _value_breakdown(before_company, config)
        value_after = _value_breakdown(after_company, config)
        rounds.append(
            {
                "round": transition.step_result.settled_round,
                "verdict": verdict,
                "action": action.to_dict(),
                "profit_cents": after_company.financial.round_profit_cents,
                "operating_cost_cents": after_company.financial.round_operating_cost_cents,
                "cash_cents": after_company.financial.cash_balance_cents,
                "enterprise_value_cents": value_after["enterprise_value_cents"],
                "enterprise_value_delta_cents": (
                    value_after["enterprise_value_cents"]
                    - value_before["enterprise_value_cents"]
                ),
                "market_share_ppm": after_company.commercial.market_share_ppm,
                "share_delta_ppm": (
                    after_company.commercial.market_share_ppm
                    - before_company.commercial.market_share_ppm
                ),
                "sales_orders": after_company.commercial.sales_orders,
                "effective_capacity_orders": after_company.operations.effective_capacity_orders,
                "stockout_orders": after_company.commercial.attempted_unfulfilled_orders,
                "market_demand_orders": market_result["realized_demand_orders"],
                "market_demand_delta_orders": (
                    market_result["realized_demand_orders"] - previous_demand
                ),
                "supply_cost_index_ppm": market_result[
                    "actual_supply_cost_index_ppm"
                ],
                "supply_cost_delta_ppm": (
                    market_result["actual_supply_cost_index_ppm"] - previous_supply
                ),
                "outside_option_orders": market_result["no_purchase_orders"],
                "average_paid_price_cents": market_result[
                    "average_paid_price_cents"
                ],
                "awareness_ppm": after_company.brand.brand_awareness_ppm,
                "reputation_ppm": after_company.brand.reputation_ppm,
                "resilience_ppm": after_company.risk.resilience_ppm,
                "active_events": [
                    event.event_type for event in before.active_market_events
                ],
                "reasons": reasons,
                "state_after_hash": after.state_hash,
            }
        )
        previous_market = market_result

    states = [initial_state] + [transition.state_after for transition in transitions]
    trend_series = []
    for company_id in final_state.company_ids:
        points = []
        for index, state in enumerate(states):
            company = state.company(company_id)
            value = _value_breakdown(company, config)
            points.append(
                {
                    "round": index,
                    "enterprise_value_cents": value["enterprise_value_cents"],
                    "total_assets_cents": value["total_assets_cents"],
                    "cash_cents": value["cash_cents"],
                    "cumulative_profit_cents": company.financial.cumulative_profit_cents,
                    "market_share_ppm": company.commercial.market_share_ppm,
                }
            )
        trend_series.append({"company_id": company_id, "points": points})

    profitable_rounds = sum(item["profit_cents"] > 0 for item in rounds)
    stockout_rounds = sum(item["stockout_orders"] > 0 for item in rounds)
    success_reasons: list[str] = []
    failure_reasons: list[str] = []
    if profitable_rounds >= max(1, len(rounds) * 0.7):
        success_reasons.append(f"{profitable_rounds}/{len(rounds)} 个回合实现正利润。")
    else:
        failure_reasons.append(f"仅 {profitable_rounds}/{len(rounds)} 个回合实现正利润。")
    share_delta = final.commercial.market_share_ppm - initial.commercial.market_share_ppm
    if abs(share_delta) < 5_000:
        success_reasons.append(
            f"最终份额与初始基本持平（变化 {share_delta / 10_000:+.2f} 个百分点）。"
        )
    elif share_delta > 0:
        success_reasons.append(
            f"市场份额较初始提高 {share_delta / 10_000:.1f} 个百分点。"
        )
    else:
        failure_reasons.append(
            f"市场份额较初始下降 {-share_delta / 10_000:.1f} 个百分点。"
        )
    if final.financial.cash_balance_cents > initial.financial.cash_balance_cents:
        success_reasons.append("最终现金高于初始现金，经营产生了可留存价值。")
    else:
        failure_reasons.append("最终现金低于初始现金，投入尚未转化为足够经营回报。")
    if stockout_rounds:
        failure_reasons.append(f"共有 {stockout_rounds} 轮发生缺货，产能或现金约束损失了订单。")
    else:
        success_reasons.append("全程没有缺货，履约能力覆盖了获得的需求。")

    leader_id = composite_ranking[0]
    leader = breakdowns[leader_id]
    own = breakdowns[player_company_id]
    component_labels = {
        "cash_cents": "现金",
        "capacity_salvage_cents": "产能残值",
        "awareness_value_cents": "品牌知名度价值",
        "service_value_cents": "服务价值",
        "reputation_value_cents": "声誉价值",
        "resilience_value_cents": "韧性价值",
    }
    component_comparison = [
        {
            "key": key,
            "label": label,
            "own_value_cents": own[key],
            "leader_value_cents": leader[key],
            "gap_cents": leader[key] - own[key],
        }
        for key, label in component_labels.items()
    ]
    rank_explanation: list[str] = []
    if rank == 1:
        rank_explanation.append("综合价值位列第一，现金经营结果与长期资产价值形成了最高合计。")
    else:
        total_gap = leader["enterprise_value_cents"] - own["enterprise_value_cents"]
        rank_explanation.append(
            f"与冠军 {leader_id} 的综合价值差距为 {total_gap / 100:,.0f} 元。"
        )
        positive_gaps = sorted(
            (item for item in component_comparison if item["gap_cents"] > 0),
            key=lambda item: item["gap_cents"],
            reverse=True,
        )
        for item in positive_gaps[:2]:
            explanation = (
                f"{item['label']}少于冠军 {item['gap_cents'] / 100:,.0f} 元，"
                "是主要排名差距来源。"
            )
            rank_explanation.append(explanation)
            failure_reasons.append(explanation)
    rank_explanation.append(
        f"总资产榜排名第 {asset_rank}，说明现金与产能账面资产"
        + ("具备优势。" if asset_rank <= rank else "弱于综合价值排名。")
    )

    if rank == 1:
        outcome = "成功"
        headline = "你创造了市场中最高的综合价值。"
    elif rank == len(composite_ranking):
        outcome = "失败"
        headline = "公司仍在经营，但现金与长期价值合计落后于全部竞争者。"
    else:
        outcome = "部分成功"
        headline = "公司保持竞争力，终局差距可以由具体价值构成解释。"

    turning_points = sorted(
        rounds,
        key=lambda item: (
            abs(item["enterprise_value_delta_cents"])
            + abs(item["share_delta_ppm"]) * 20
            + (2_000_000 if item["active_events"] else 0)
        ),
        reverse=True,
    )[:3]
    return {
        "status": "complete" if final_state.terminal else "in_progress",
        "player_company_id": player_company_id,
        "outcome": outcome if final_state.terminal else "进行中",
        "headline": headline if final_state.terminal else "回溯将在 Episode 结束后给出最终判断。",
        "rank": rank,
        "asset_rank": asset_rank,
        "company_count": len(composite_ranking),
        "terminal_value_cents": own["enterprise_value_cents"],
        "market_model": {
            "id": initial_state.market.market_model_id,
            "label": initial_state.market.market_model_label,
            "description": initial_state.market.market_model_description,
            "demand_bias_ppm": initial_state.market.demand_bias_ppm,
            "price_anchor_cents": initial_state.market.price_anchor_cents,
            "price_band_cents": initial_state.market.price_band_cents,
            "utility_multipliers_ppm": {
                "price": initial_state.market.utility_price_multiplier_ppm,
                "awareness": initial_state.market.utility_awareness_multiplier_ppm,
                "service": initial_state.market.utility_service_multiplier_ppm,
                "reputation": initial_state.market.utility_reputation_multiplier_ppm,
                "prior_stockout": initial_state.market.utility_prior_stockout_multiplier_ppm,
            },
        },
        "rankings": {
            "composite": ranking_rows(composite_ranking, "enterprise_value_cents"),
            "total_assets": ranking_rows(asset_ranking, "total_assets_cents"),
        },
        "ranking_methodology": {
            "composite": "综合价值 = 现金 + 产能残值 + 品牌知名度价值 + 服务价值 + 声誉价值 + 韧性价值。",
            "total_assets": "总资产 = 现金 + 产能账面价值，不包含品牌、服务、声誉与韧性估值。",
        },
        "component_comparison": component_comparison,
        "rank_explanation": rank_explanation,
        "trend_series": trend_series,
        "summary": {
            "initial_cash_cents": initial.financial.cash_balance_cents,
            "final_cash_cents": final.financial.cash_balance_cents,
            "cumulative_profit_cents": final.financial.cumulative_profit_cents,
            "initial_share_ppm": initial.commercial.market_share_ppm,
            "final_share_ppm": final.commercial.market_share_ppm,
            "final_reputation_ppm": final.brand.reputation_ppm,
            "final_resilience_ppm": final.risk.resilience_ppm,
            "profitable_rounds": profitable_rounds,
            "stockout_rounds": stockout_rounds,
            "cumulative_operating_cost_cents": sum(item["operating_cost_cents"] for item in rounds),
        },
        "success_reasons": success_reasons,
        "failure_reasons": failure_reasons,
        "turning_point_rounds": [item["round"] for item in turning_points],
        "rounds": rounds,
        "methodology": (
            "归因使用同轮动作、同轮市场条件、前后状态差和终局价值构成；"
            "价格、投入与结果同时变化时只描述有规则证据支持的机制，不把相关性伪装成严格因果。"
        ),
    }
