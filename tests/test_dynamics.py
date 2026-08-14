from dataclasses import replace

import pytest

from game_theory_agent.market import (
    CompanyIncident,
    IncidentResponse,
    IncidentResponseMode,
    MarketEnv,
    MarketEvent,
    RiskSignal,
)
from game_theory_agent.market.exceptions import ActionValidationError
from game_theory_agent.market.protocols import state_hash


def _rehash(state):
    state = replace(state, state_hash="")
    return replace(state, state_hash=state_hash(state.to_dict()))


def _run(config, state, actions):
    env = MarketEnv(config)
    env.load_state(state)
    return env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions
    ).state_after


def test_financial_capacity_prevents_negative_cash(config, initial_state, make_actions):
    company = initial_state.company("company_A")
    company = replace(
        company,
        financial=replace(company.financial, cash_balance_cents=100_000),
        operations=replace(company.operations, base_unit_cost_cents=20_000),
    )
    forced = _rehash(
        replace(initial_state, companies=(company,) + initial_state.companies[1:])
    )
    actions = make_actions(
        forced,
        overrides={
            company_id: {
                "advertising_budget_cents": 0,
                "service_budget_cents": 0,
            }
            for company_id in forced.company_ids
        },
    )
    actions["company_A"] = replace(actions["company_A"], price_cents=7500)
    result = _run(config, forced, actions)
    company_after = result.company("company_A")
    assert company_after.financial.cash_balance_cents >= 0
    assert (
        company_after.commercial.sales_orders
        <= company_after.operations.financial_capacity_orders
    )


def test_fixed_spend_cannot_exceed_starting_cash(config, initial_state, make_actions):
    company = initial_state.company("company_A")
    company = replace(
        company, financial=replace(company.financial, cash_balance_cents=1000)
    )
    forced = _rehash(
        replace(initial_state, companies=(company,) + initial_state.companies[1:])
    )
    actions = make_actions(forced)
    with pytest.raises(ActionValidationError, match="BUDGET_EXCEEDED"):
        _run(config, forced, actions)


def test_full_repair_removes_incident_before_current_sales(
    config, initial_state, make_actions
):
    incident = CompanyIncident(
        incident_id="forced-incident",
        incident_type="warehouse_equipment_failure",
        severity="high",
        started_round=1,
        remaining_rounds=2,
        repair_required_cents=1_000_000,
        accumulated_repair_cents=0,
        capacity_multiplier_ppm=200_000,
        advertising_multiplier_ppm=1_000_000,
        service_penalty_ppm=100_000,
        reputation_penalty_ppm=50_000,
        refund_rate_ppm=20_000,
    )
    company = initial_state.company("company_A")
    company = replace(company, risk=replace(company.risk, active_incident=incident))
    forced = _rehash(
        replace(initial_state, companies=(company,) + initial_state.companies[1:])
    )
    wait = make_actions(forced, nonce="paired")
    repair = dict(wait)
    repair["company_A"] = replace(
        repair["company_A"],
        incident_response=IncidentResponse(IncidentResponseMode.FULL_REPAIR, 1_000_000),
    )
    wait_state = _run(config, forced, wait)
    repair_state = _run(config, forced, repair)
    no_incident_company = replace(
        company, risk=replace(company.risk, active_incident=None)
    )
    no_incident = _rehash(
        replace(
            forced,
            companies=(no_incident_company,) + forced.companies[1:],
        )
    )
    no_incident_state = _run(
        config, no_incident, make_actions(no_incident, nonce="paired")
    )
    assert (
        repair_state.company("company_A").operations.effective_capacity_orders
        > wait_state.company("company_A").operations.effective_capacity_orders
    )
    assert repair_state.company("company_A").risk.active_incident is None
    assert (
        repair_state.company("company_A").operations.effective_capacity_orders
        < no_incident_state.company("company_A").operations.effective_capacity_orders
    )


def test_operating_costs_are_visible_and_reconcile_profit(
    config, initial_state, make_actions
):
    state = _run(config, initial_state, make_actions(initial_state))
    company = state.company("company_A")
    expected = config.integer("operating_costs", "fixed_overhead_cents") + (
        company.commercial.sales_orders
        * config.integer("operating_costs", "fulfillment_cost_per_order_cents")
    )
    assert company.financial.round_operating_cost_cents == expected
    assert company.financial.round_profit_cents == (
        company.financial.round_revenue_cents
        - company.financial.round_variable_cost_cents
        - company.financial.round_fixed_spend_cents
        - company.financial.round_incident_cost_cents
        - company.financial.round_operating_cost_cents
    )


def test_reputation_is_a_slow_moving_asset(config, initial_state, make_actions):
    state = _run(config, initial_state, make_actions(initial_state))
    delta = (
        state.company("company_A").brand.reputation_ppm
        - initial_state.company("company_A").brand.reputation_ppm
    )
    assert 0 < delta < 40_000


def test_new_resilience_does_not_protect_current_active_event(
    config, initial_state, make_actions
):
    event = MarketEvent(
        event_id="forced-weather",
        event_type="extreme_weather",
        severity="high",
        started_round=1,
        remaining_rounds=2,
        demand_multiplier_ppm=900_000,
        supply_cost_multiplier_ppm=1_300_000,
        capacity_multiplier_ppm=500_000,
        advertising_multiplier_ppm=600_000,
        service_penalty_ppm=150_000,
        reputation_penalty_ppm=50_000,
    )
    forced = _rehash(replace(initial_state, active_market_events=(event,)))
    zero = make_actions(forced, nonce="paired")
    invest = make_actions(
        forced,
        nonce="paired",
        overrides={"company_A": {"resilience_budget_cents": 4_000_000}},
    )
    zero_state = _run(config, forced, zero)
    invest_state = _run(config, forced, invest)
    assert (
        zero_state.company("company_A").operations.effective_capacity_orders
        == invest_state.company("company_A").operations.effective_capacity_orders
    )
    assert (
        invest_state.company("company_A").risk.resilience_ppm
        > zero_state.company("company_A").risk.resilience_ppm
    )


def test_due_risk_signal_realizes_into_next_state(config, initial_state, make_actions):
    signal = RiskSignal(
        signal_id="forced-signal",
        event_type="extreme_weather",
        target_round=2,
        estimated_probability_ppm=1_000_000,
        severity="medium",
        lead_time_rounds=1,
    )
    forced = _rehash(replace(initial_state, risk_signals=(signal,)))
    state = _run(config, forced, make_actions(forced))
    assert any(
        event.event_type == "extreme_weather" for event in state.active_market_events
    )
    assert all(item.signal_id != "forced-signal" for item in state.risk_signals)
