import json

import httpx
import pytest

from game_theory_agent.agents.single.gateway import (
    AgentGatewayClient,
    GatewayReadError,
    StaleObservationError,
    SubmissionUnknownError,
)
from game_theory_agent.agents.single.models import EconomicAction


def test_gateway_reads_matching_observation_and_action_contract():
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/observation"):
            return httpx.Response(
                200,
                json={
                    "episode_id": "episode-1",
                    "company_id": "company_A",
                    "round": 3,
                    "state_version": 2,
                    "state_hash": "abc",
                    "terminal": False,
                },
            )
        if request.url.path.endswith("/action-contract"):
            return httpx.Response(
                200,
                json={
                    "episode_id": "episode-1",
                    "company_id": "company_A",
                    "round": 3,
                    "state_version": 2,
                    "bounds": {"price_cents": {"min": 800, "max": 2000}},
                },
            )
        raise AssertionError(request.url)

    gateway = AgentGatewayClient(
        base_url="http://gateway.test",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )

    snapshot = gateway.load_snapshot("episode-1", "company_A")

    assert snapshot.round == 3
    assert snapshot.state_version == 2
    assert snapshot.state_hash == "abc"
    assert snapshot.observation["state_hash"] == "abc"
    assert snapshot.action_contract["bounds"]["price_cents"]["max"] == 2000


def test_gateway_requires_non_empty_observation_state_hash():
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/observation"):
            return httpx.Response(
                200,
                json={
                    "episode_id": "episode-1",
                    "company_id": "company_A",
                    "round": 3,
                    "state_version": 2,
                    "state_hash": "",
                    "terminal": False,
                },
            )
        if request.url.path.endswith("/action-contract"):
            return httpx.Response(
                200,
                json={
                    "episode_id": "episode-1",
                    "company_id": "company_A",
                    "round": 3,
                    "state_version": 2,
                    "bounds": {"price_cents": {"min": 800, "max": 2000}},
                },
            )
        raise AssertionError(request.url)

    gateway = AgentGatewayClient(
        base_url="http://gateway.test",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )

    with pytest.raises(GatewayReadError, match="invalid snapshot"):
        gateway.load_snapshot("episode-1", "company_A")


def test_gateway_retries_safe_reads_at_most_twice_after_initial_attempt():
    attempts = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"detail": "unavailable"})

    gateway = AgentGatewayClient(
        base_url="http://gateway.test",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
        read_retries=2,
        retry_delay_seconds=0,
    )

    with pytest.raises(GatewayReadError):
        gateway.load_snapshot("episode-1", "company_A")

    assert attempts == 3


def test_gateway_submits_intent_once_and_returns_receipt():
    attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        body = json.loads(request.content)
        assert body["round"] == 3
        assert body["state_version"] == 2
        assert body["requested_action"]["price_cents"] == 1180
        return httpx.Response(202, json={"intent_id": "intent-1", "status": "accepted"})

    gateway = AgentGatewayClient(
        base_url="http://gateway.test",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )

    receipt = gateway.submit_intent(
        episode_id="episode-1",
        company_id="company_A",
        agent_id="single-agent-company-A",
        round_number=3,
        state_version=2,
        observation_hash="sha256:state",
        action=EconomicAction(price_cents=1180),
        rationale="在现金约束内平衡增长。",
        expected_outcome="份额小幅上升。",
    )

    assert receipt["intent_id"] == "intent-1"
    assert attempts == 1


def test_gateway_never_retries_stale_or_unknown_submissions():
    stale_attempts = 0

    def stale(_request: httpx.Request) -> httpx.Response:
        nonlocal stale_attempts
        stale_attempts += 1
        return httpx.Response(409, json={"detail": "STALE_OBSERVATION"})

    stale_gateway = AgentGatewayClient(
        base_url="http://gateway.test",
        client=httpx.Client(transport=httpx.MockTransport(stale)),
    )
    kwargs = dict(
        episode_id="episode-1",
        company_id="company_A",
        agent_id="single-agent-company-A",
        round_number=3,
        state_version=2,
        observation_hash="sha256:state",
        action=EconomicAction(price_cents=1180),
        rationale="合法短理由",
        expected_outcome="份额稳定",
    )

    with pytest.raises(StaleObservationError):
        stale_gateway.submit_intent(**kwargs)
    assert stale_attempts == 1

    unknown_attempts = 0

    def unknown(request: httpx.Request) -> httpx.Response:
        nonlocal unknown_attempts
        unknown_attempts += 1
        raise httpx.ReadError("connection lost", request=request)

    unknown_gateway = AgentGatewayClient(
        base_url="http://gateway.test",
        client=httpx.Client(transport=httpx.MockTransport(unknown)),
    )
    with pytest.raises(SubmissionUnknownError):
        unknown_gateway.submit_intent(**kwargs)
    assert unknown_attempts == 1
