from dataclasses import replace

import pytest

from game_theory_agent.market import PresetResolver
from game_theory_agent.market.exceptions import (
    ActionValidationError,
    StateVersionConflictError,
)


def test_numeric_action_and_dynamic_constraints(env, initial_state, make_actions):
    action = make_actions(initial_state)["company_A"]
    result = env.validate_action(action, "company_A")
    constraints = env.get_action_constraints("company_A", initial_state.state_version)

    assert result.valid
    assert result.require_valid() == action
    assert constraints["cash_available_cents"] == 30_000_000
    assert constraints["capacity_investment_enabled"]


def test_stale_and_illegal_repair_actions_are_rejected(
    env, initial_state, make_actions
):
    action = make_actions(initial_state)["company_A"]
    stale = replace(action, state_version=99)
    expensive = replace(
        action,
        advertising_budget_cents=5_000_000,
        service_budget_cents=5_000_000,
        capacity_investment_cents=6_000_000,
        resilience_budget_cents=4_000_000,
    )
    expensive = replace(
        expensive,
        incident_response=replace(
            expensive.incident_response, repair_budget_cents=8_000_000
        ),
    )

    assert "STATE_VERSION_CONFLICT" in env.validate_action(stale, "company_A").errors
    assert not env.validate_action(expensive, "company_A").valid
    with pytest.raises(StateVersionConflictError):
        env.get_action_constraints("company_A", 99)


def test_preset_resolves_to_numeric_action(config, initial_state):
    action = PresetResolver(config).resolve(
        {
            "price": "low",
            "advertising": "high",
            "service": "medium",
            "capacity": "low",
            "resilience": "low",
        },
        action_id="preset-1",
        episode_id=initial_state.episode_id,
        agent_id="company_A",
        round_number=initial_state.round,
        state_version=initial_state.state_version,
    )
    assert action.price_cents == 10000
    assert action.advertising_budget_cents == 800_000
    assert action.capacity_investment_cents == 0


def test_raw_discrete_v2_action_is_rejected(env, initial_state):
    result = env.validate_action(
        {
            "price_level": "low",
            "advertising_level": "high",
            "service_level": "medium",
        },
        "company_A",
    )
    assert not result.valid
    with pytest.raises(ActionValidationError):
        result.require_valid()
