from copy import deepcopy
from types import SimpleNamespace

import asyncio
import pytest
from fastapi.testclient import TestClient

from game_theory_agent.agents import AgentRuntime
from game_theory_agent.api import SESSIONS, agent_app, app
from game_theory_agent.information import (
    InformationReplayMismatchError,
    ObservationSnapshot,
    seal_observation,
    verify_information_replay,
    verify_information_snapshot,
)
from game_theory_agent.model_clients import MockModelClient


controller = TestClient(app)
gateway = TestClient(agent_app)


def _create(episode_id: str, information_mode: str = "public") -> None:
    response = controller.post(
        "/api/episodes",
        json={
            "episode_id": episode_id,
            "episode_seed": 7001,
            "company_ids": ["company_A", "company_B", "company_C"],
            "market_model": "balanced",
            "max_rounds": 5,
            "information_mode": information_mode,
        },
    )
    assert response.status_code == 201, response.text


def _observation(episode_id: str, company_id: str) -> dict:
    response = gateway.get(
        f"/v1/episodes/{episode_id}/companies/{company_id}/observation"
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_public_policy_hides_financial_operations_persona_and_market_internals():
    SESSIONS.clear()
    _create("information-public")
    company_a = _observation("information-public", "company_A")
    company_b = _observation("information-public", "company_B")

    assert company_a["visibility_policy_version"] == (
        "visibility-public-v2.0.0"
    )
    assert company_a["public_state"] == company_b["public_state"]
    assert company_a["private_state"]["company_id"] == "company_A"
    assert company_b["private_state"]["company_id"] == "company_B"
    assert "financial" in company_a["private_state"]["company"]
    assert set(company_a["competitors"][0]) == {
        "company_id",
        "price_cents",
        "market_share_ppm",
        "sales_orders",
        "reputation_ppm",
    }
    assert "round_revenue_cents" not in company_a["public_companies"][0]
    assert "service_quality_ppm" not in company_a["public_companies"][0]
    assert "persona" not in company_a["competitors"][0]
    assert "financial" not in company_a["competitors"][0]
    assert "operations" not in company_a["competitors"][0]
    assert "risk" not in company_a["competitors"][0]
    assert "price_anchor_cents" not in company_a["market"]
    assert "utility_price_multiplier_ppm" not in company_a["market"]
    assert "average_capacity_utilization_ppm" not in company_a[
        "market_regime"
    ]["metrics"]
    assert company_a["belief_schema_version"] == "none"
    assert company_a["belief_state"] is None

    snapshot = ObservationSnapshot.from_observation(company_a, "company_A")
    state = SESSIONS["information-public"].env.get_state()
    assert verify_information_snapshot(state, snapshot) == snapshot


def test_information_replay_rejects_rehashed_visibility_injection():
    SESSIONS.clear()
    _create("information-tamper")
    original = _observation("information-tamper", "company_A")
    forged = deepcopy(original)
    forged["competitors"][0]["financial"] = {
        "cash_balance_cents": 99_000_000
    }
    forged = seal_observation(forged)
    forged_snapshot = ObservationSnapshot.from_observation(
        forged, "company_A"
    )

    with pytest.raises(
        InformationReplayMismatchError,
        match="competitors differs from visibility policy",
    ):
        verify_information_snapshot(
            SESSIONS["information-tamper"].env.get_state(),
            forged_snapshot,
        )


def test_information_replay_binds_exact_model_context_and_requires_snapshot():
    SESSIONS.clear()
    _create("information-context")
    observation = _observation("information-context", "company_A")
    result = asyncio.run(
        AgentRuntime(
            "information-mock", "company_A", MockModelClient()
        ).decide(observation)
    )
    snapshot = ObservationSnapshot.from_observation(
        observation, "company_A"
    )
    trace = SimpleNamespace(
        company_id="company_A",
        observation=observation,
        observation_hash=observation["observation_hash"],
        information_snapshot=snapshot,
        decision_context=result.context.model_dump(mode="json"),
    )
    event = SimpleNamespace(
        event_schema_version="agent-round-event-v1.7.0",
        state_before=SESSIONS[
            "information-context"
        ].env.get_state().to_dict(),
        traces=[trace],
    )
    assert verify_information_replay([event]) == (snapshot,)

    missing = SimpleNamespace(
        **{
            **trace.__dict__,
            "information_snapshot": None,
        }
    )
    with pytest.raises(
        InformationReplayMismatchError, match="missing snapshot"
    ):
        verify_information_replay(
            [
                SimpleNamespace(
                    event_schema_version="agent-round-event-v1.7.0",
                    state_before=event.state_before,
                    traces=[missing],
                )
            ]
        )


def test_perfect_and_public_are_views_of_the_same_unchanged_true_state():
    SESSIONS.clear()
    _create("information-view-source", "perfect")
    state = SESSIONS["information-view-source"].env.get_state()
    before_hash = state.state_hash
    from game_theory_agent.agents import ObservationBuilder

    perfect = ObservationBuilder().build(state, "company_A", "perfect")
    public = ObservationBuilder().build(state, "company_A", "public")

    assert perfect["public_state"] == public["public_state"]
    assert perfect["private_state"] == public["private_state"]
    assert "financial" in perfect["competitors"][0]
    assert "financial" not in public["competitors"][0]
    assert SESSIONS["information-view-source"].env.get_state().state_hash == (
        before_hash
    )


def test_one_privileged_observer_gets_perfect_view_in_public_market(
    monkeypatch,
):
    SESSIONS.clear()
    token = "privileged-observer-controller"
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", token)
    created = controller.post(
        "/api/episodes",
        headers={"X-Controller-Token": token},
        json={
            "episode_id": "information-privileged-observer",
            "episode_seed": 7002,
            "company_ids": ["company_A", "company_B", "company_C"],
            "market_model": "balanced",
            "max_rounds": 5,
            "information_mode": "public",
            "observer_information_modes": {"company_A": "perfect"},
        },
    )
    assert created.status_code == 201, created.text
    tokens = created.json()["agent_tokens"]

    anonymous = gateway.get(
        "/v1/episodes/information-privileged-observer/companies/company_A/observation"
    )
    assert anonymous.status_code == 401
    company_a = gateway.get(
        "/v1/episodes/information-privileged-observer/companies/company_A/observation",
        headers={"X-Agent-Token": tokens["company_A"]},
    ).json()
    company_b = gateway.get(
        "/v1/episodes/information-privileged-observer/companies/company_B/observation",
        headers={"X-Agent-Token": tokens["company_B"]},
    ).json()

    assert company_a["information_mode"] == "perfect"
    assert company_a["visibility_policy_version"] == (
        "visibility-perfect-v1.0.0"
    )
    assert "financial" in company_a["competitors"][0]
    assert company_b["information_mode"] == "public"
    assert company_b["visibility_policy_version"] == (
        "visibility-public-v2.0.0"
    )
    assert "financial" not in company_b["competitors"][0]
    assert company_a["public_state"] == company_b["public_state"]

    session = SESSIONS["information-privileged-observer"]
    assert session.manifest.information_mode == "public"
    assert session.manifest.information_mode_for("company_A") == "perfect"
    assert session.manifest.information_mode_for("company_B") == "public"

    snapshots = tuple(
        ObservationSnapshot.from_observation(observation, company_id)
        for company_id, observation in (
            ("company_A", company_a),
            ("company_B", company_b),
        )
    )
    traces = [
        SimpleNamespace(
            company_id=snapshot.company_id,
            observation=snapshot.observation,
            observation_hash=snapshot.observation_hash,
            information_snapshot=snapshot,
            decision_context=None,
        )
        for snapshot in snapshots
    ]
    event = SimpleNamespace(
        event_schema_version="agent-round-event-v1.7.0",
        state_before=session.env.get_state().to_dict(),
        traces=traces,
    )
    assert verify_information_replay([event], session.manifest) == snapshots
