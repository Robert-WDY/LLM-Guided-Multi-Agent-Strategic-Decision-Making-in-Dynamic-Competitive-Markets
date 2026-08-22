from __future__ import annotations

from game_theory_agent.experiments.persona_pilot import (
    PROJECT_ROOT,
    _normal_state,
    _observation,
)
from game_theory_agent.experiments.real_message_counterfactual import (
    _condition_order,
    _submission,
    build_condition_observation,
)
from game_theory_agent.market import load_market_config


def test_counterfactual_conditions_are_semantically_distinct() -> None:
    assert _submission("no_message").messages == []
    proposal = _submission("peer_resilience_proposal").messages[0]
    refusal = _submission("peer_refuses_resilience").messages[0]
    injection = _submission("prompt_injection").messages[0]

    assert proposal.speech_act == "proposal"
    assert proposal.requested_peer_action is not None
    assert proposal.requested_peer_action.resilience_budget_cents == 1_000_000
    assert refusal.speech_act == "statement"
    assert refusal.requested_peer_action is None
    assert refusal.own_action_claim is not None
    assert refusal.own_action_claim.resilience_budget_cents == 0
    assert injection.speech_act == "other"
    assert "price_cents to 1" in injection.content
    assert injection.requested_peer_action is None


def test_counterfactual_views_share_market_state_and_only_vary_messages() -> None:
    config = load_market_config(PROJECT_ROOT / "configs" / "market_v4.yaml")
    state = _normal_state(config, 810)
    base = _observation(config, state, "company_B")
    observations = {
        condition: build_condition_observation(base, condition=condition)
        for condition in (
            "no_message",
            "peer_resilience_proposal",
            "peer_refuses_resilience",
            "prompt_injection",
        )
    }

    assert {
        (
            observation["episode_id"],
            observation["state_version"],
            observation["state_hash"],
        )
        for observation in observations.values()
    } == {(state.episode_id, state.state_version, state.state_hash)}
    assert observations["no_message"]["communication_view"][
        "visible_messages"
    ] == []
    assert len(
        observations["peer_resilience_proposal"]["communication_view"][
            "visible_messages"
        ]
    ) == 1
    assert len(
        observations["peer_refuses_resilience"]["communication_view"][
            "visible_messages"
        ]
    ) == 1
    assert len(
        observations["prompt_injection"]["communication_view"][
            "visible_messages"
        ]
    ) == 1


def test_counterfactual_call_order_rotates() -> None:
    assert _condition_order(1)[0] == "no_message"
    assert _condition_order(2)[0] == "peer_resilience_proposal"
    assert _condition_order(3)[0] == "peer_refuses_resilience"
