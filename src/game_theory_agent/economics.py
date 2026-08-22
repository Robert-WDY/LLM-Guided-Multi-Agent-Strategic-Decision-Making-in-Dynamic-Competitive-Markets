"""Deterministic unit-economics and liquidity calculations."""

from __future__ import annotations

from typing import Any

from game_theory_agent.market import MarketConfig, MarketState


PPM = 1_000_000


def _round_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return round(numerator / denominator)


def _ceil_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-numerator // denominator)


def _ppm_mul(*values: int) -> int:
    result = PPM
    for value in values:
        result = _round_ratio(result * value, PPM)
    return result


def _sat_ppm(budget_cents: int, scale_cents: int) -> int:
    if budget_cents <= 0:
        return 0
    return _round_ratio(budget_cents * PPM, budget_cents + scale_cents)


def _expected_incident_loss(
    config: MarketConfig,
    *,
    resilience_ppm: int,
    expected_orders: int,
    contribution_cents: int,
) -> tuple[int, int, int, int]:
    """Decision-time incident expectation using only public config and state.

    Returns mean, low, high loss and incident probability. It intentionally
    never inspects future RNG draws or episode seeds.
    """

    incident_cfg = config.mapping("incidents")
    terminal_cfg = config.mapping("terminal")
    probability = _ppm_mul(
        int(incident_cfg["base_probability_ppm"]),
        PPM
        - _ppm_mul(
            int(incident_cfg["probability_reduction_ppm"]), resilience_ppm
        ),
    )
    impact = PPM - _ppm_mul(
        int(incident_cfg["severity_reduction_ppm"]), resilience_ppm
    )
    conditional_losses: list[tuple[int, int]] = []
    for incident_type, type_weight in incident_cfg["type_weights_ppm"].items():
        severity_table = incident_cfg["definitions"][incident_type]["severity"]
        for severity, severity_weight in incident_cfg["severity_weights_ppm"].items():
            definition = severity_table[severity]
            capacity_loss_ppm = _ppm_mul(
                PPM - int(definition["capacity_multiplier_ppm"]), impact
            )
            lost_contribution = _ppm_mul(
                max(0, contribution_cents) * max(1, expected_orders),
                capacity_loss_ppm,
            ) * int(definition["duration_rounds"])
            reputation_loss = _ppm_mul(
                int(terminal_cfg["reputation_value_max_cents"]),
                _ppm_mul(int(definition["reputation_penalty_ppm"]), impact),
            )
            gross_loss = (
                int(definition["repair_required_cents"])
                + lost_contribution
                + reputation_loss
            )
            weight = _ppm_mul(int(type_weight), int(severity_weight))
            conditional_losses.append((gross_loss, weight))
    conditional_mean = sum(loss * weight for loss, weight in conditional_losses) // PPM
    low = min(loss for loss, _ in conditional_losses)
    high = max(loss for loss, _ in conditional_losses)
    return (
        _ppm_mul(conditional_mean, probability),
        _ppm_mul(low, probability),
        _ppm_mul(high, probability),
        probability,
    )


def decision_support_metrics(
    config: MarketConfig, state: MarketState, company_id: str
) -> dict[str, Any]:
    """Return action-independent metrics safe to expose to the acting company."""

    company = state.company(company_id)
    operating = config.mapping("operating_costs")
    policy = config.mapping("decision_policy")
    overhead = int(operating["fixed_overhead_cents"])
    fulfillment = int(operating["fulfillment_cost_per_order_cents"])
    minimum_contribution = int(policy["minimum_unit_contribution_cents"])
    actual_unit_cost = _round_ratio(
        company.operations.base_unit_cost_cents
        * state.market.actual_supply_cost_index_ppm,
        PPM,
    )
    incident = company.risk.active_incident
    refund_rate = incident.refund_rate_ppm if incident else 0
    price_denominator = max(1, PPM - refund_rate)
    minimum_safe_price = _ceil_ratio(
        (actual_unit_cost + fulfillment + minimum_contribution) * PPM,
        price_denominator,
    )
    current_refund = _round_ratio(company.commercial.price_cents * refund_rate, PPM)
    current_contribution = (
        company.commercial.price_cents
        - actual_unit_cost
        - fulfillment
        - current_refund
    )
    expected_orders = max(
        company.commercial.potential_demand_orders,
        company.commercial.sales_orders,
        max(1, state.market.base_demand_orders // len(state.company_ids)),
    )
    break_even_orders = (
        _ceil_ratio(overhead, current_contribution)
        if current_contribution > 0
        else None
    )
    break_even_price = _ceil_ratio(
        (
            actual_unit_cost
            + fulfillment
            + _ceil_ratio(overhead, expected_orders)
        )
        * PPM,
        price_denominator,
    )
    future_reserve = min(
        int(policy["future_overhead_reserve_rounds"]),
        max(0, state.rounds_remaining - 1),
    )
    protected_overhead_rounds = 1 + future_reserve
    minimum_cash_reserve = overhead * protected_overhead_rounds
    safe_discretionary = max(
        0, company.financial.cash_balance_cents - minimum_cash_reserve
    )
    recent_profits = list(company.history.recent_profit_cents)
    consecutive_losses = 0
    consecutive_profits = 0
    for profit in reversed(recent_profits):
        if profit < 0:
            consecutive_losses += 1
        else:
            break
    for profit in reversed(recent_profits):
        if profit > 0:
            consecutive_profits += 1
        else:
            break
    initial_cash = int(config.mapping("company_initial")["cash_balance_cents"])
    cash_drawdown = (
        max(0, initial_cash - company.financial.cash_balance_cents)
        * PPM
        // max(1, initial_cash)
    )
    runway = company.financial.cash_balance_cents * 1000 // max(1, overhead)
    recovering_after_loss_streak = (
        consecutive_profits == 1
        and len(recent_profits) >= 3
        and recent_profits[-2] < 0
        and recent_profits[-3] < 0
    )
    if runway < int(policy["liquidity_crisis_runway_milli_rounds"]):
        phase = "liquidity_crisis"
        spend_cap_ppm = int(policy["crisis_spend_cap_ppm"])
    elif (
        consecutive_losses >= int(policy["recovery_loss_streak"])
        or recovering_after_loss_streak
        or (
            cash_drawdown >= int(policy["recovery_cash_drawdown_ppm"])
            and consecutive_profits < 2
        )
    ):
        phase = "profit_recovery"
        spend_cap_ppm = int(policy["recovery_spend_cap_ppm"])
    else:
        phase = "growth"
        spend_cap_ppm = PPM
    maximum_discretionary = safe_discretionary * spend_cap_ppm // PPM
    demand_uncertainty_ppm = int(
        config.mapping("market_models")["random_demand_jitter_ppm"]
    )
    demand_low = max(0, _ppm_mul(expected_orders, PPM - demand_uncertainty_ppm))
    demand_high = _ppm_mul(expected_orders, PPM + demand_uncertainty_ppm)
    effective_capacity = company.operations.effective_capacity_orders
    capacity_gap = max(0, expected_orders - effective_capacity)
    capacity_gap_low = max(0, demand_low - effective_capacity)
    capacity_gap_high = max(0, demand_high - effective_capacity)
    capacity_unit_cost = int(config.mapping("capacity")["investment_unit_cost_cents"])

    def payback(gap: int, demand: int) -> int | None:
        if current_contribution <= 0 or gap <= 0 or demand <= 0:
            return None
        utilization_probability_ppm = min(PPM, gap * PPM // demand)
        expected_unit_return = _ppm_mul(
            current_contribution, utilization_probability_ppm
        )
        return (
            _ceil_ratio(capacity_unit_cost, expected_unit_return)
            if expected_unit_return > 0
            else None
        )

    payback_expected = payback(capacity_gap, expected_orders)
    payback_low = payback(capacity_gap_high, demand_high)
    payback_high = payback(capacity_gap_low, max(1, demand_low))
    capacity_marginal_value = (
        max(0, current_contribution)
        * capacity_gap
        * max(0, state.rounds_remaining - 1)
    )

    incident_mean, incident_low, incident_high, incident_probability = (
        _expected_incident_loss(
            config,
            resilience_ppm=company.risk.resilience_ppm,
            expected_orders=expected_orders,
            contribution_cents=current_contribution,
        )
    )
    active_incident = company.risk.active_incident
    if active_incident is not None:
        incident_mean = active_incident.remaining_repair_cents
        incident_low = incident_mean
        incident_high = incident_mean
        incident_probability = PPM
    zero_resilience_loss = _expected_incident_loss(
        config,
        resilience_ppm=0,
        expected_orders=expected_orders,
        contribution_cents=current_contribution,
    )[0]
    risk_coverage_ppm = (
        max(0, zero_resilience_loss - incident_mean) * PPM
        // max(1, zero_resilience_loss)
    )
    update_cfg = config.mapping("state_updates")
    resilience_scale = int(
        config.mapping("action", "saturation_scales_cents")["resilience"]
    )
    next_resilience_with_reference_budget = min(
        PPM,
        _ppm_mul(
            int(update_cfg["resilience_retention_ppm"]),
            company.risk.resilience_ppm,
        )
        + _ppm_mul(
            int(update_cfg["resilience_input_weight_ppm"]),
            _sat_ppm(1_000_000, resilience_scale),
        ),
    )
    next_loss = (
        incident_mean
        if active_incident is not None
        else _expected_incident_loss(
            config,
            resilience_ppm=next_resilience_with_reference_budget,
            expected_orders=expected_orders,
            contribution_cents=current_contribution,
        )[0]
    )
    resilience_marginal_loss_reduction = max(0, incident_mean - next_loss)

    last_action = company.history.last_action
    last_growth_spend = (
        last_action.advertising_budget_cents
        + last_action.capacity_investment_cents
        if last_action is not None
        else 0
    )
    share_history = company.history.recent_market_share_ppm
    profit_history = company.history.recent_profit_cents
    recent_share_change = (
        share_history[-1] - share_history[-2] if len(share_history) >= 2 else 0
    )
    recent_profit_change = (
        profit_history[-1] - profit_history[-2] if len(profit_history) >= 2 else 0
    )
    growth_efficiency_proxy = (
        max(-PPM, min(PPM, recent_profit_change * PPM // last_growth_spend))
        if last_growth_spend > 0
        else None
    )
    return {
        "metrics_schema_version": "decision-support-v1.1.0",
        "actual_unit_cost_cents": actual_unit_cost,
        "fulfillment_cost_per_order_cents": fulfillment,
        "refund_rate_ppm": refund_rate,
        "current_unit_contribution_cents": current_contribution,
        "minimum_unit_contribution_cents": minimum_contribution,
        "minimum_safe_price_cents": minimum_safe_price,
        "estimated_break_even_price_cents": break_even_price,
        "break_even_orders_at_current_price": break_even_orders,
        "reference_orders": expected_orders,
        "fixed_overhead_cents": overhead,
        "protected_overhead_rounds": protected_overhead_rounds,
        "minimum_cash_reserve_cents": minimum_cash_reserve,
        "safe_discretionary_budget_cents": safe_discretionary,
        "maximum_discretionary_budget_cents": maximum_discretionary,
        "cash_runway_milli_rounds": runway,
        "consecutive_loss_rounds": consecutive_losses,
        "consecutive_profitable_rounds": consecutive_profits,
        "cash_drawdown_ppm": cash_drawdown,
        "strategic_phase": phase,
        "expected_demand_orders": {
            "expected": expected_orders,
            "low": demand_low,
            "high": demand_high,
            "method": "current_visible_demand_with_configured_jitter",
        },
        "forecast_capacity_gap_orders": {
            "expected": capacity_gap,
            "low": capacity_gap_low,
            "high": capacity_gap_high,
        },
        "capacity_investment_payback_rounds": {
            "expected": payback_expected,
            "low": payback_low,
            "high": payback_high,
            "method": "unit_cost_divided_by_expected_used_unit_contribution",
        },
        "capacity_marginal_value_cents": capacity_marginal_value,
        "expected_incident_loss_cents": {
            "mean": incident_mean,
            "low": incident_low,
            "high": incident_high,
            "confidence_ppm": PPM if active_incident is not None else 650_000,
            "incident_probability_ppm": incident_probability,
            "method": "current_state_and_public_config_expectation_no_future_rng",
        },
        "current_resilience_coverage_ppm": risk_coverage_ppm,
        "resilience_marginal_loss_reduction_cents_per_1000000": (
            resilience_marginal_loss_reduction
        ),
        "growth_spend_efficiency_ppm": growth_efficiency_proxy,
        "growth_efficiency_evidence": {
            "last_growth_spend_cents": last_growth_spend,
            "recent_share_change_ppm": recent_share_change,
            "recent_profit_change_cents": recent_profit_change,
            "method": "historical_proxy_not_causal_counterfactual",
        },
        "price_contribution_margin_ppm": (
            current_contribution * PPM // max(1, company.commercial.price_cents)
        ),
        "information_boundary": {
            "uses_future_rng": False,
            "uses_episode_seed": False,
            "uses_only_current_state_and_public_config": True,
        },
        "price_below_variable_cost_floor": (
            company.commercial.price_cents < minimum_safe_price
        ),
        "plan_thresholds": {
            "recovery_loss_streak": int(policy["recovery_loss_streak"]),
            "recovery_cash_drawdown_ppm": int(
                policy["recovery_cash_drawdown_ppm"]
            ),
            "liquidity_crisis_runway_milli_rounds": int(
                policy["liquidity_crisis_runway_milli_rounds"]
            ),
            "recovery_spend_cap_ppm": int(policy["recovery_spend_cap_ppm"]),
            "crisis_spend_cap_ppm": int(policy["crisis_spend_cap_ppm"]),
        },
    }
