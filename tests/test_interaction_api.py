from __future__ import annotations

from fastapi.testclient import TestClient

from game_theory_agent.api import SESSIONS, agent_app, app
from game_theory_agent.orchestration.round_event import CommunicationPhaseRecord


CONTROLLER_TOKEN = "interaction-controller-token"
COMPANIES = ["company_A", "company_B", "company_C", "company_D"]


def _create_interaction_episode(
    monkeypatch,
    *,
    episode_id: str = "interaction-api",
    mode: str = "public_private",
    cooperation_mode: str = "off",
) -> tuple[TestClient, TestClient, dict, dict[str, str]]:
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", CONTROLLER_TOKEN)
    controller = TestClient(app)
    gateway = TestClient(agent_app)
    response = controller.post(
        "/api/episodes",
        headers={"X-Controller-Token": CONTROLLER_TOKEN},
        json={
            "episode_id": episode_id,
            "episode_seed": 314159,
            "company_ids": COMPANIES,
            "max_rounds": 5,
            "communication_mode": mode,
            "cooperation_mode": cooperation_mode,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    return controller, gateway, body, body["agent_tokens"]


def test_shared_resilience_proposal_commitment_and_partial_betrayal(monkeypatch):
    SESSIONS.clear()
    controller, gateway, created, tokens = _create_interaction_episode(
        monkeypatch,
        episode_id="cooperation-api",
        mode="public_private",
        cooperation_mode="shared_resilience_v1",
    )
    episode_id = created["state"]["episode_id"]
    controller_headers = {"X-Controller-Token": CONTROLLER_TOKEN}

    proposal = gateway.post(
        f"/v1/episodes/{episode_id}/companies/company_A/communication/submissions",
        headers=_agent_headers(tokens, "company_A"),
        json=_message_request(
            created,
            [
                {
                    "channel": "private",
                    "recipients": ["company_B"],
                    "speech_act": "proposal",
                    "content": "请在第2轮贡献100万元行业韧性。",
                    "cooperation_proposal": {
                        "target_round": 2,
                        "requested_contribution_cents": 1_000_000,
                    },
                }
            ],
        ),
    )
    assert proposal.status_code == 202, proposal.text
    proposal_id = proposal.json()["messages"][0]["cooperation_proposal"][
        "proposal_id"
    ]
    round1_hash = created["state"]["state_hash"]
    close1 = controller.post(
        f"/api/v1/controller/episodes/{episode_id}/communication/close",
        headers=controller_headers,
        json=_binding(created),
    )
    assert close1.status_code == 200, close1.text
    assert close1.json()["cooperation_close"]["proposals_created"][0][
        "proposal_id"
    ] == proposal_id
    assert close1.json()["cooperation_close"]["commitments_created"] == []
    assert controller.get(f"/api/episodes/{episode_id}/state").json()["state"][
        "state_hash"
    ] == round1_hash

    views1 = {
        company_id: gateway.get(
            f"/v1/episodes/{episode_id}/companies/{company_id}/communication/view",
            headers=_agent_headers(tokens, company_id),
        ).json()
        for company_id in COMPANIES
    }
    proposal_message_id = proposal.json()["message_ids"][0]
    assert proposal_message_id in {
        item["message_id"] for item in views1["company_B"]["visible_messages"]
    }
    assert proposal_message_id not in {
        item["message_id"] for item in views1["company_C"]["visible_messages"]
    }

    settle1 = controller.post(
        f"/api/v1/controller/episodes/{episode_id}/settle-agent-round",
        headers=controller_headers,
        json={
            "step_id": f"{episode_id}:1:0",
            "intent_ids": {},
            "fallback": "rule",
        },
    )
    assert settle1.status_code == 200, settle1.text
    round2 = controller.get(f"/api/episodes/{episode_id}/state").json()
    pending_b = gateway.get(
        f"/v1/episodes/{episode_id}/companies/company_B/observation",
        headers=_agent_headers(tokens, "company_B"),
    ).json()["cooperation"]["pending_proposals_received"]
    assert [item["proposal_id"] for item in pending_b] == [proposal_id]
    hidden_c = gateway.get(
        f"/v1/episodes/{episode_id}/companies/company_C/observation",
        headers=_agent_headers(tokens, "company_C"),
    ).json()["cooperation"]
    assert hidden_c["proposals_received"] == []

    response = gateway.post(
        f"/v1/episodes/{episode_id}/companies/company_B/communication/submissions",
        headers=_agent_headers(tokens, "company_B"),
        json=_message_request(
            round2,
            [
                {
                    "channel": "private",
                    "recipients": ["company_A"],
                    "speech_act": "response",
                    "content": "接受该提议。",
                    "cooperation_response": {
                        "proposal_id": proposal_id,
                        "response": "accept",
                    },
                }
            ],
        ),
    )
    assert response.status_code == 202, response.text
    round2_hash = round2["state"]["state_hash"]
    close2 = controller.post(
        f"/api/v1/controller/episodes/{episode_id}/communication/close",
        headers=controller_headers,
        json=_binding(round2),
    )
    assert close2.status_code == 200, close2.text
    commitment = close2.json()["cooperation_close"]["commitments_created"][0]
    assert commitment["binding"] is False
    assert commitment["promised_contribution_cents"] == 1_000_000
    assert controller.get(f"/api/episodes/{episode_id}/state").json()["state"][
        "state_hash"
    ] == round2_hash

    b_observation = gateway.get(
        f"/v1/episodes/{episode_id}/companies/company_B/observation",
        headers=_agent_headers(tokens, "company_B"),
    ).json()
    b_intent = gateway.post(
        f"/v1/episodes/{episode_id}/intents",
        headers=_agent_headers(tokens, "company_B"),
        json={
            "agent_id": "company_B",
            "company_id": "company_B",
            "round": 2,
            "state_version": 1,
            "observation_hash": b_observation["observation_hash"],
            "communication_view_digest": b_observation["communication_view"][
                "view_digest"
            ],
            "requested_action": {
                "price_cents": 10_000,
                "shared_resilience_contribution_cents": 300_000,
            },
        },
    )
    assert b_intent.status_code == 202, b_intent.text
    settled = controller.post(
        f"/api/v1/controller/episodes/{episode_id}/settle-agent-round",
        headers=controller_headers,
        json={
            "step_id": f"{episode_id}:2:1",
            "intent_ids": {"company_B": b_intent.json()["intent_id"]},
            "fallback": "rule",
        },
    )
    assert settled.status_code == 200, settled.text
    record = settled.json()["cooperation_round"]
    verification = record["verifications"][0]
    assert verification["actual_contribution_cents"] == 300_000
    assert verification["fulfillment_ratio_ppm"] == 300_000
    assert verification["status"] == "partial_betrayal"
    assert record["credibility_after"]["company_B"]["credibility_ppm"] < 500_000
    assert settled.json()["state"]["shared_resilience"][
        "industry_resilience_ppm"
    ] > 0


def _binding(created: dict) -> dict:
    state = created["state"]
    return {
        "round": state["round"],
        "state_version": state["state_version"],
        "state_hash": state["state_hash"],
    }


def _message_request(created: dict, messages: list[dict]) -> dict:
    return {
        **_binding(created),
        "submission": {
            "schema_version": "communication-submission-v1.0.0",
            "messages": messages,
        },
    }


def _agent_headers(tokens: dict[str, str], company_id: str) -> dict[str, str]:
    return {"X-Agent-Token": tokens[company_id]}


def test_protected_creation_returns_tokens_once_and_session_keeps_only_hashes(
    monkeypatch,
):
    SESSIONS.clear()
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", CONTROLLER_TOKEN)
    controller = TestClient(app)

    unprotected = controller.post(
        "/api/episodes",
        json={
            "episode_id": "unprotected-communication",
            "company_ids": ["company_A", "company_B"],
            "communication_mode": "public_only",
        },
    )
    assert unprotected.status_code == 401

    controller, gateway, created, tokens = _create_interaction_episode(
        monkeypatch,
        episode_id="protected-communication",
        mode="public_only",
    )
    assert set(tokens) == set(COMPANIES)
    assert created["agent_token_header"] == "X-Agent-Token"
    assert created["agent_tokens_returned_once"] is True
    assert created["manifest"]["communication_mode"] == "public_only"

    session = SESSIONS["protected-communication"]
    assert set(session.agent_token_hashes) == set(COMPANIES)
    for company_id, raw_token in tokens.items():
        assert raw_token != session.agent_token_hashes[company_id]
        assert session.agent_token_hashes[company_id].startswith("sha256:")

    later_state = controller.get(
        "/api/episodes/protected-communication/state"
    ).json()
    assert "agent_tokens" not in later_state
    assert "agent_token_hashes" not in later_state

    no_token = gateway.get(
        "/v1/episodes/protected-communication/companies/company_A/observation"
    )
    cross_company = gateway.get(
        "/v1/episodes/protected-communication/companies/company_B/observation",
        headers=_agent_headers(tokens, "company_A"),
    )
    authorized = gateway.get(
        "/v1/episodes/protected-communication/companies/company_A/observation",
        headers=_agent_headers(tokens, "company_A"),
    )
    assert no_token.status_code == 401
    assert cross_company.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["communication_mode"] == "public_only"
    assert authorized.json()["communication_view"] is None


def test_shared_resilience_supports_off_baseline_but_rejects_public_only(
    monkeypatch,
):
    SESSIONS.clear()
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", CONTROLLER_TOKEN)
    controller = TestClient(app)
    off = controller.post(
        "/api/episodes",
        headers={"X-Controller-Token": CONTROLLER_TOKEN},
        json={
            "episode_id": "cooperation-off-baseline",
            "company_ids": COMPANIES,
            "max_rounds": 5,
            "communication_mode": "off",
            "cooperation_mode": "shared_resilience_v1",
        },
    )
    public_only = controller.post(
        "/api/episodes",
        headers={"X-Controller-Token": CONTROLLER_TOKEN},
        json={
            "episode_id": "cooperation-public-only-invalid",
            "company_ids": COMPANIES,
            "max_rounds": 5,
            "communication_mode": "public_only",
            "cooperation_mode": "shared_resilience_v1",
        },
    )
    assert off.status_code == 201, off.text
    assert off.json()["manifest"]["cooperation_mode"] == "shared_resilience_v1"
    assert off.json()["manifest"]["communication_mode"] == "off"
    assert set(off.json()["agent_tokens"]) == set(COMPANIES)
    assert public_only.status_code == 422


def test_legacy_direct_step_cannot_bypass_enabled_communication_barrier(
    monkeypatch,
):
    SESSIONS.clear()
    controller, _gateway, created, _tokens = _create_interaction_episode(
        monkeypatch,
        episode_id="interaction-direct-step-blocked",
        mode="public_only",
    )
    episode_id = created["state"]["episode_id"]

    response = controller.post(
        f"/api/episodes/{episode_id}/steps",
        json={"step_id": "bypass-attempt", "joint_action": {}},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == (
        "INTERACTION_REQUIRES_AGENT_BARRIER"
    )
    assert controller.get(f"/api/episodes/{episode_id}/state").json()[
        "state"
    ]["state_version"] == 0


def test_message_close_visibility_and_intent_are_bound_to_company_and_state(
    monkeypatch,
):
    SESSIONS.clear()
    controller, gateway, created, tokens = _create_interaction_episode(monkeypatch)
    episode_id = created["state"]["episode_id"]
    initial_hash = created["state"]["state_hash"]

    premature_intent = gateway.post(
        f"/v1/episodes/{episode_id}/intents",
        headers=_agent_headers(tokens, "company_A"),
        json={
            "agent_id": "planner-A",
            "company_id": "company_A",
            "round": 1,
            "state_version": 0,
            "observation_hash": "sha256:premature-observation",
            "requested_action": {"price_cents": 10_000},
        },
    )
    assert premature_intent.status_code == 409
    assert premature_intent.json()["detail"]["code"] == (
        "COMMUNICATION_NOT_CLOSED"
    )

    a_submission = _message_request(
        created,
        [
            {
                "channel": "public",
                "speech_act": "promise",
                "content": "I may keep price near 10000.",
                "own_action_claim": {"price_cents": 10_000},
            },
            {
                "channel": "private",
                "recipients": ["company_B"],
                "speech_act": "proposal",
                "content": "Consider 10500.",
                "requested_peer_action": {"price_cents": 10_500},
            },
        ],
    )
    accepted = gateway.post(
        f"/v1/episodes/{episode_id}/companies/company_A/communication/submissions",
        headers=_agent_headers(tokens, "company_A"),
        json=a_submission,
    )
    repeated = gateway.post(
        f"/v1/episodes/{episode_id}/companies/company_A/communication/submissions",
        headers=_agent_headers(tokens, "company_A"),
        json=a_submission,
    )
    impersonated = gateway.post(
        f"/v1/episodes/{episode_id}/companies/company_B/communication/submissions",
        headers=_agent_headers(tokens, "company_A"),
        json=_message_request(created, []),
    )
    stale = gateway.post(
        f"/v1/episodes/{episode_id}/companies/company_C/communication/submissions",
        headers=_agent_headers(tokens, "company_C"),
        json={
            **_message_request(created, []),
            "state_hash": "sha256:not-current",
        },
    )
    assert accepted.status_code == 202, accepted.text
    assert repeated.status_code == 202
    assert repeated.json()["message_ids"] == accepted.json()["message_ids"]
    assert accepted.json()["market_state_unchanged"] is True
    assert impersonated.status_code == 401
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "STALE_COMMUNICATION_STATE"
    assert controller.get(f"/api/episodes/{episode_id}/state").json()["state"][
        "state_hash"
    ] == initial_hash

    open_view = gateway.get(
        f"/v1/episodes/{episode_id}/companies/company_A/communication/view",
        headers=_agent_headers(tokens, "company_A"),
    )
    assert open_view.status_code == 409

    close_request = _binding(created)
    unauthorized_close = controller.post(
        f"/api/v1/controller/episodes/{episode_id}/communication/close",
        json=close_request,
    )
    closed = controller.post(
        f"/api/v1/controller/episodes/{episode_id}/communication/close",
        headers={"X-Controller-Token": CONTROLLER_TOKEN},
        json=close_request,
    )
    closed_again = controller.post(
        f"/api/v1/controller/episodes/{episode_id}/communication/close",
        headers={"X-Controller-Token": CONTROLLER_TOKEN},
        json=close_request,
    )
    assert unauthorized_close.status_code == 401
    assert closed.status_code == 200, closed.text
    assert closed_again.json() == closed.json()
    phase = closed.json()["communication_phase"]
    parsed_phase = CommunicationPhaseRecord.model_validate(phase)
    assert phase["status"] == "closed"
    assert parsed_phase.closure.transcript_hash == phase["closure"][
        "transcript_hash"
    ]
    assert len(phase["closure"]["all_messages"]) == 2
    assert closed.json()["market_state_unchanged"] is True
    assert set(phase["company_views"]) == set(COMPANIES)
    assert controller.get(f"/api/episodes/{episode_id}/state").json()["state"][
        "state_hash"
    ] == initial_hash

    views = {}
    for company_id in COMPANIES:
        response = gateway.get(
            f"/v1/episodes/{episode_id}/companies/{company_id}/communication/view",
            headers=_agent_headers(tokens, company_id),
        )
        assert response.status_code == 200, response.text
        views[company_id] = response.json()
    private_id = accepted.json()["message_ids"][1]
    assert private_id in {
        item["message_id"] for item in views["company_A"]["visible_messages"]
    }
    assert private_id in {
        item["message_id"] for item in views["company_B"]["visible_messages"]
    }
    assert private_id not in {
        item["message_id"] for item in views["company_C"]["visible_messages"]
    }
    assert private_id not in {
        item["message_id"] for item in views["company_D"]["visible_messages"]
    }
    leaked = gateway.get(
        f"/v1/episodes/{episode_id}/companies/company_B/communication/view",
        headers=_agent_headers(tokens, "company_C"),
    )
    assert leaked.status_code == 401

    observation = gateway.get(
        f"/v1/episodes/{episode_id}/companies/company_A/observation",
        headers=_agent_headers(tokens, "company_A"),
    ).json()
    assert observation["communication_view"] == views["company_A"]

    post_close = gateway.post(
        f"/v1/episodes/{episode_id}/companies/company_B/communication/submissions",
        headers=_agent_headers(tokens, "company_B"),
        json=_message_request(created, []),
    )
    assert post_close.status_code == 409

    missing_digest = gateway.post(
        f"/v1/episodes/{episode_id}/intents",
        headers=_agent_headers(tokens, "company_A"),
        json={
            "agent_id": "planner-A",
            "company_id": "company_A",
            "round": 1,
            "state_version": 0,
            "observation_hash": observation["observation_hash"],
            "requested_action": {"price_cents": 10_000},
        },
    )
    wrong_digest = gateway.post(
        f"/v1/episodes/{episode_id}/intents",
        headers=_agent_headers(tokens, "company_A"),
        json={
            "agent_id": "planner-A",
            "company_id": "company_A",
            "round": 1,
            "state_version": 0,
            "observation_hash": observation["observation_hash"],
            "communication_view_digest": "sha256:wrong-view",
            "requested_action": {"price_cents": 10_000},
        },
    )
    wrong_observation = gateway.post(
        f"/v1/episodes/{episode_id}/intents",
        headers=_agent_headers(tokens, "company_A"),
        json={
            "agent_id": "planner-A",
            "company_id": "company_A",
            "round": 1,
            "state_version": 0,
            "observation_hash": "sha256:wrong-observation",
            "communication_view_digest": views["company_A"]["view_digest"],
            "requested_action": {"price_cents": 10_000},
        },
    )
    correct_intent = gateway.post(
        f"/v1/episodes/{episode_id}/intents",
        headers=_agent_headers(tokens, "company_A"),
        json={
            "agent_id": "planner-A",
            "company_id": "company_A",
            "round": 1,
            "state_version": 0,
            "observation_hash": observation["observation_hash"],
            "communication_view_digest": views["company_A"]["view_digest"],
            "requested_action": {"price_cents": 10_000},
        },
    )
    assert missing_digest.status_code == 409
    assert wrong_digest.status_code == 409
    assert wrong_observation.status_code == 409
    assert wrong_observation.json()["detail"]["code"] == (
        "OBSERVATION_VIEW_MISMATCH"
    )
    assert correct_intent.status_code == 202, correct_intent.text
    assert correct_intent.json()["agent_id"] == "company_A"
    assert correct_intent.json()["communication_view_digest"] == views[
        "company_A"
    ]["view_digest"]
    assert correct_intent.json()["observation_hash"] == observation[
        "observation_hash"
    ]
    assert controller.get(f"/api/episodes/{episode_id}/state").json()["state"][
        "state_hash"
    ] == initial_hash

    settled = controller.post(
        f"/api/v1/controller/episodes/{episode_id}/settle-agent-round",
        headers={"X-Controller-Token": CONTROLLER_TOKEN},
        json={
            "step_id": f"{episode_id}:1:0",
            "intent_ids": {
                "company_A": correct_intent.json()["intent_id"]
            },
            "fallback": "rule",
        },
    )
    assert settled.status_code == 200, settled.text
    assert settled.json()["state"]["state_version"] == 1
    old_close_after_settlement = controller.post(
        f"/api/v1/controller/episodes/{episode_id}/communication/close",
        headers={"X-Controller-Token": CONTROLLER_TOKEN},
        json=close_request,
    )
    assert old_close_after_settlement.json() == closed.json()

    next_observation = gateway.get(
        f"/v1/episodes/{episode_id}/companies/company_A/observation",
        headers=_agent_headers(tokens, "company_A"),
    )
    assert next_observation.status_code == 200
    assert next_observation.json()["round"] == 2
    assert next_observation.json()["communication_view"] is None
    assert len(SESSIONS[episode_id].communication_ledgers) == 2


def test_public_only_rejects_private_and_off_mode_remains_legacy_compatible(
    monkeypatch,
):
    SESSIONS.clear()
    _, gateway, created, tokens = _create_interaction_episode(
        monkeypatch,
        episode_id="public-only-api",
        mode="public_only",
    )
    rejected = gateway.post(
        "/v1/episodes/public-only-api/companies/company_A/communication/submissions",
        headers=_agent_headers(tokens, "company_A"),
        json=_message_request(
            created,
            [
                {
                    "channel": "private",
                    "recipients": ["company_B"],
                    "content": "not allowed in this condition",
                }
            ],
        ),
    )
    assert rejected.status_code == 422

    legacy = TestClient(app).post(
        "/api/episodes",
        json={
            "episode_id": "communication-off-legacy",
            "company_ids": ["company_A", "company_B"],
        },
    )
    assert legacy.status_code == 201
    assert "agent_tokens" not in legacy.json()
    observation = gateway.get(
        "/v1/episodes/communication-off-legacy/companies/company_A/observation"
    )
    assert observation.status_code == 200
    view = observation.json()["communication_view"]
    assert view["mode"] == "off"
    assert view["status"] == "closed"
    assert view["visible_messages"] == []
