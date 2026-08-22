from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from game_theory_agent.agents.context import DecisionContextBuilder
from game_theory_agent.agents.memory import EpisodeMemory
from game_theory_agent.agents.prompt_builder import AgentPromptBuilder
from game_theory_agent.api import CONFIG, SESSIONS, agent_app, app
from game_theory_agent.belief import (
    BELIEF_SCHEMA_VERSION,
    BeliefLedger,
    BeliefReplayMismatchError,
    compute_belief_calibration,
    verify_belief_replay,
)
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.information import ObservationSnapshot
from game_theory_agent.market import MarketEnv
from game_theory_agent.model_clients import MockModelClient
from game_theory_agent.information import seal_observation


def test_public_action_ledger_updates_only_after_settlement_and_is_idempotent():
    env = MarketEnv(CONFIG)
    state = env.reset(
        ["company_A", "company_B", "company_C", "company_D"],
        episode_id="belief-ledger",
        episode_seed=91,
        market_model="balanced",
        max_rounds=5,
    )
    ledger = BeliefLedger(
        episode_id=state.episode_id, company_ids=state.company_ids
    )
    prior, prior_hash = ledger.company_view(
        observer_company_id="company_A", round_number=1, state_version=0
    )
    b_prior = prior.opponent_beliefs["company_B"]
    assert b_prior.evidence_count == 0
    assert b_prior.next_price_direction.model_dump() == {
        "price_cut_ppm": 333_334,
        "maintain_ppm": 333_333,
        "price_raise_ppm": 333_333,
    }
    assert prior_hash.startswith("sha256:")

    actions = {
        company_id: build_rule_action(CONFIG, state, company_id)
        for company_id in state.company_ids
    }
    actions["company_B"] = replace(
        actions["company_B"],
        price_cents=state.company("company_B").commercial.price_cents - 100,
    )
    evidence = ledger.update_after_settlement(state, actions)
    assert len(evidence) == 4
    assert ledger.update_after_settlement(state, actions) == evidence

    posterior, _ = ledger.company_view(
        observer_company_id="company_A", round_number=2, state_version=1
    )
    b_posterior = posterior.opponent_beliefs["company_B"]
    assert b_posterior.latest_observed_direction == "price_cut"
    assert b_posterior.next_price_direction.price_cut_ppm == 500_000
    assert b_posterior.next_price_direction.maintain_ppm == 250_000
    assert "company_A" not in posterior.opponent_beliefs


def _create_belief_episode(monkeypatch: pytest.MonkeyPatch, episode_id: str):
    token = "belief-controller-token"
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", token)
    SESSIONS.clear()
    response = TestClient(app).post(
        "/api/episodes",
        headers={"X-Controller-Token": token},
        json={
            "episode_id": episode_id,
            "episode_seed": 1234,
            "company_ids": ["company_A", "company_B", "company_C", "company_D"],
            "max_rounds": 5,
            "market_model": "balanced",
            "information_mode": "public",
            "belief_mode": "public_action_v1",
        },
    )
    assert response.status_code == 201, response.text
    return response.json(), token


def _observations(episode_id: str, tokens: dict[str, str]):
    client = TestClient(agent_app)
    return {
        company_id: client.get(
            f"/v1/episodes/{episode_id}/companies/{company_id}/observation",
            headers={"X-Agent-Token": token},
        ).json()
        for company_id, token in tokens.items()
    }


def test_belief_enters_observation_context_prompt_and_replay(monkeypatch):
    created, _ = _create_belief_episode(monkeypatch, "belief-api")
    tokens = created["agent_tokens"]
    first = _observations("belief-api", tokens)
    observation = first["company_A"]
    assert observation["belief_schema_version"] == BELIEF_SCHEMA_VERSION
    assert (
        observation["visibility_policy"]["belief_schema_version"]
        == BELIEF_SCHEMA_VERSION
    )
    assert observation["belief_hash"].startswith("sha256:")
    assert observation["belief_state"]["public_evidence_through_round"] == 0
    serialized = str(observation["belief_state"])
    assert "cash_balance" not in serialized
    assert "persona" not in serialized

    context = DecisionContextBuilder().build(
        observation, "company_A", EpisodeMemory()
    )
    assert context.belief_state == observation["belief_state"]
    assert context.meta.belief_hash == observation["belief_hash"]
    prompt = AgentPromptBuilder().build(context)
    assert "仅根据已结算公开价格历史" in prompt
    assert "不得从信念反推出对手现金" in prompt

    session = SESSIONS["belief-api"]
    state_before = session.env.get_state()
    actions = {
        company_id: build_rule_action(CONFIG, state_before, company_id).to_dict()
        for company_id in state_before.company_ids
    }
    actions["company_B"]["price_cents"] = (
        state_before.company("company_B").commercial.price_cents - 100
    )
    response = TestClient(app).post(
        "/api/episodes/belief-api/steps",
        json={"step_id": "belief-api:1:0", "joint_action": actions},
    )
    assert response.status_code == 200, response.text
    second = _observations("belief-api", tokens)
    assert second["company_A"]["belief_state"]["public_evidence_through_round"] == 1

    # Same frozen state, same deterministic policy; only the Belief treatment changes.
    on_observation = second["company_A"]
    off_observation = dict(on_observation)
    off_observation["belief_schema_version"] = "none"
    off_observation["belief_hash"] = None
    off_observation["belief_state"] = None
    off_observation["episode_config"] = dict(off_observation["episode_config"])
    off_observation["episode_config"]["belief_mode"] = "off"
    off_observation = seal_observation(off_observation)
    builder = DecisionContextBuilder()
    on_context = builder.build(on_observation, "company_A", EpisodeMemory())
    off_context = builder.build(off_observation, "company_A", EpisodeMemory())
    belief_sensitive = MockModelClient(
        belief_price_response_cents=100,
        belief_response_threshold_ppm=450_000,
    )
    import asyncio

    on_decision = asyncio.run(belief_sensitive.generate_decision(on_context))
    off_decision = asyncio.run(belief_sensitive.generate_decision(off_context))
    assert (
        on_decision.parsed_output["requested_action"]["price_cents"]
        < off_decision.parsed_output["requested_action"]["price_cents"]
    )

    first_event = SimpleNamespace(
        event_schema_version="agent-round-event-v1.8.0",
        state_before=state_before.to_dict(),
        joint_action=response.json()["decision_resolutions"],
        traces=[],
        communication_phase=None,
    )
    # Replace resolution envelopes with their authoritative final actions.
    first_event.joint_action = {
        company_id: item["action"]
        for company_id, item in first_event.joint_action.items()
    }
    first_event.traces = [
        SimpleNamespace(
            company_id=company_id,
            information_snapshot=ObservationSnapshot.from_observation(
                first[company_id], company_id
            )
        )
        for company_id in state_before.company_ids
    ]
    manifest = session.manifest
    verified = verify_belief_replay([first_event], manifest)
    assert len(verified) == 4
    metrics = compute_belief_calibration([first_event])
    assert metrics["prediction_count"] == 12

    tampered = first_event.traces[0].information_snapshot.model_copy(deep=True)
    tampered.observation["belief_state"]["opponent_beliefs"]["company_B"][
        "evidence_count"
    ] = 99
    first_event.traces[0] = SimpleNamespace(information_snapshot=tampered)
    with pytest.raises((BeliefReplayMismatchError, ValueError)):
        verify_belief_replay([first_event], manifest)


def test_belief_off_remains_null_baseline(monkeypatch):
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", "unused")
    SESSIONS.clear()
    response = TestClient(app).post(
        "/api/episodes",
        json={"episode_id": "belief-off", "episode_seed": 5, "max_rounds": 5},
    )
    assert response.status_code == 201
    observation = TestClient(agent_app).get(
        "/v1/episodes/belief-off/companies/company_A/observation"
    ).json()
    assert observation["belief_schema_version"] == "none"
    assert observation["belief_hash"] is None
    assert observation["belief_state"] is None
