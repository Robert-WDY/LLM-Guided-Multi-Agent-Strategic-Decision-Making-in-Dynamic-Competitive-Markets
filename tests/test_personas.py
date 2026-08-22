import asyncio

from fastapi.testclient import TestClient

from game_theory_agent.agents import (
    AgentPromptBuilder,
    AgentRuntime,
    PersonaUtilityTracker,
    load_persona_registry,
)
from game_theory_agent.api import SESSIONS, agent_app, app
from game_theory_agent.model_clients import MockModelClient


def test_persona_catalog_is_versioned_and_cooperation_is_reserved(config):
    registry = load_persona_registry()

    assert registry.catalog_version == "persona-catalog-v1.1.0"
    assert {
        "none",
        "aggressive",
        "conservative",
        "balanced",
        "selfish_long_term",
        "profit_myopic",
        "aggressive_v1_extreme",
        "conservative_v1_extreme",
        "balanced_v1",
        "selfish_long_term_v1",
        "disciplined_growth_v1",
        "risk_guarded_v1",
    } <= set(registry.profile_ids)
    for persona_id in registry.profile_ids:
        profile = registry.get(persona_id)
        assert sum(profile.utility_weights_ppm.model_dump().values()) == 1_000_000
        assert profile.utility_weights_ppm.social_welfare == 0
        assert profile.utility_weights_ppm.cooperation_reputation == 0
        assert profile.social_welfare_enabled is False
        assert profile.cooperation_enabled is False
        assert profile.profile_hash.startswith("sha256:")


def test_runtime_persona_changes_context_without_changing_market_state():
    SESSIONS.clear()
    created = TestClient(app).post(
        "/api/episodes",
        json={
            "episode_id": "persona-context",
            "episode_seed": 42,
            "company_ids": ["company_A", "company_B"],
            "max_rounds": 5,
            "information_mode": "perfect",
        },
    )
    assert created.status_code == 201, created.text
    observation = TestClient(agent_app).get(
        "/v1/episodes/persona-context/companies/company_A/observation"
    ).json()
    registry = load_persona_registry()
    selfish = AgentRuntime(
        "selfish-A",
        "company_A",
        MockModelClient(),
        persona_profile=registry.get("selfish_long_term"),
        persona_registry=registry,
    )
    myopic = AgentRuntime(
        "myopic-A",
        "company_A",
        MockModelClient(),
        persona_profile=registry.get("profit_myopic"),
        persona_registry=registry,
    )

    selfish_result = asyncio.run(selfish.decide(observation))
    myopic_result = asyncio.run(myopic.decide(observation))

    assert selfish_result.context.state_hash == myopic_result.context.state_hash
    assert selfish_result.context.own_company["persona"] == "none"
    assert selfish_result.context.persona == "selfish_long_term"
    assert myopic_result.context.persona == "profit_myopic"
    assert selfish_result.context.persona_profile.profile_hash != (
        myopic_result.context.persona_profile.profile_hash
    )
    selfish_prompt = AgentPromptBuilder().build(selfish_result.context)
    assert "[可信 Persona Contract]" in selfish_prompt
    assert "selfish_long_term" in selfish_prompt
    assert "不要虚构沟通、承诺或合作动作" in selfish_prompt


def test_persona_utility_uses_same_outcome_with_different_weights(
    config, env, initial_state, make_actions
):
    result = env.step(
        f"{initial_state.episode_id}:1:0",
        make_actions(initial_state, nonce="persona-utility"),
    )
    registry = load_persona_registry()
    long_term = registry.evaluator(registry.get("selfish_long_term")).evaluate(
        initial_state, result.state_after, "company_A"
    )
    myopic = registry.evaluator(registry.get("profit_myopic")).evaluate(
        initial_state, result.state_after, "company_A"
    )

    assert long_term.component_scores_ppm == myopic.component_scores_ppm
    assert long_term.round_utility_ppm != myopic.round_utility_ppm
    assert long_term.weighted_contributions_ppm != (
        myopic.weighted_contributions_ppm
    )
    assert long_term.component_scores_ppm["social_welfare"] == 0
    assert long_term.component_scores_ppm["cooperation_reputation"] == 0
    assert long_term.efficiency_and_risk_are_weighted_utility_components is False
    assert long_term.realized_incident_loss_cents >= 0
    assert long_term.realized_unserved_contribution_loss_cents >= 0


def test_persona_utility_tracker_applies_time_discount(
    env, initial_state, make_actions
):
    registry = load_persona_registry()
    profile = registry.get("profit_myopic")
    tracker = PersonaUtilityTracker(registry.evaluator(profile))
    first_result = env.step(
        f"{initial_state.episode_id}:1:0",
        make_actions(initial_state, nonce="persona-discount-1"),
    )
    first = tracker.record(initial_state, first_result.state_after, "company_A")
    second_before = first_result.state_after
    second_result = env.step(
        (
            f"{second_before.episode_id}:{second_before.round}:"
            f"{second_before.state_version}"
        ),
        make_actions(second_before, nonce="persona-discount-2"),
    )
    second = tracker.record(second_before, second_result.state_after, "company_A")

    assert first.discount_multiplier_ppm == 1_000_000
    assert second.discount_multiplier_ppm == profile.traits_ppm.time_discount
    assert second.cumulative_discounted_utility_ppm == (
        first.discounted_round_utility_ppm
        + second.discounted_round_utility_ppm
    )


def test_episode_manifest_records_agent_persona_without_changing_state():
    SESSIONS.clear()
    registry = load_persona_registry()
    profile = registry.get("conservative")
    response = TestClient(app).post(
        "/api/episodes",
        json={
            "episode_id": "persona-manifest",
            "episode_seed": 9,
            "company_ids": ["company_A", "company_B"],
            "max_rounds": 5,
            "agent_configs": {
                "company_A": {"persona": profile.manifest_dict()}
            },
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["state"]["companies"]["company_A"]["persona"] == "none"
    recorded = body["manifest"]["agent_configs"]["company_A"]["persona"]
    assert recorded["persona_id"] == "conservative"
    assert recorded["profile_hash"] == profile.profile_hash
