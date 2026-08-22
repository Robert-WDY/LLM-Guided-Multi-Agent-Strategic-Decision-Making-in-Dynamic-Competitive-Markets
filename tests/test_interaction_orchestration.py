import asyncio
import json
import pytest
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from game_theory_agent.agents import (
    AgentCommunicationResult,
    AgentDecisionResult,
    AgentRuntime,
    CommunicationContext,
    DecisionContext,
    DecisionContextBuilder,
)
from game_theory_agent.api import CONFIG, SESSIONS, agent_app, app
from game_theory_agent.interaction import (
    CommunicationSubmission,
    MessageDraft,
    PartialActionClaim,
)
from game_theory_agent.interaction.replay import verify_interaction_replay
from game_theory_agent.information import verify_information_replay
from game_theory_agent.cooperation.replay import verify_cooperation_replay
from game_theory_agent.cooperation.replay import CooperationReplayMismatchError
from game_theory_agent.experiments.interaction_metrics import (
    compute_interaction_metrics,
)
from game_theory_agent.experiments.cooperation_metrics import (
    compute_cooperation_metrics,
)
from game_theory_agent.market import MarketEnv
from game_theory_agent.market.replay import verify_replay
from game_theory_agent.model_clients import MockModelClient
from game_theory_agent.orchestration import JsonlRoundEventLogger, RoundCoordinator


COMPANIES = ("company_A", "company_B", "company_C", "company_D")


class TokenGatewayAdapter:
    def __init__(self, tokens: dict[str, str]) -> None:
        self.client = TestClient(agent_app)
        self.tokens = tokens

    def _headers(self, company_id: str) -> dict[str, str]:
        return {"X-Agent-Token": self.tokens[company_id]}

    async def get_observation(
        self, episode_id: str, company_id: str
    ) -> dict[str, Any]:
        response = self.client.get(
            f"/v1/episodes/{episode_id}/companies/{company_id}/observation",
            headers=self._headers(company_id),
        )
        assert response.status_code == 200, response.text
        return response.json()

    async def submit_communication(
        self, episode_id: str, result: AgentCommunicationResult
    ) -> dict[str, Any]:
        context = result.context
        response = self.client.post(
            (
                f"/v1/episodes/{episode_id}/companies/{result.company_id}/"
                "communication/submissions"
            ),
            headers=self._headers(result.company_id),
            json={
                "round": context.round,
                "state_version": context.state_version,
                "state_hash": context.state_hash,
                "submission": result.submission.model_dump(mode="json"),
            },
        )
        assert response.status_code == 202, response.text
        return response.json()

    async def submit_intent(
        self, episode_id: str, result: AgentDecisionResult
    ) -> dict[str, Any]:
        assert result.decision is not None
        view = result.context.communication_view
        response = self.client.post(
            f"/v1/episodes/{episode_id}/intents",
            headers=self._headers(result.company_id),
            json={
                "agent_id": result.agent_id,
                "company_id": result.company_id,
                "round": result.context.round,
                "state_version": result.context.state_version,
                "observation_hash": result.context.meta.observation_hash,
                "requested_action": result.decision.requested_action.model_dump(
                    mode="json"
                ),
                "rationale": result.decision.plan.situation_summary,
                "expected_outcome": json.dumps(
                    result.decision.plan.expected_outcome.model_dump(mode="json")
                ),
                "communication_view_digest": (
                    view.view_digest if view is not None else None
                ),
            },
        )
        assert response.status_code == 202, response.text
        return response.json()


class ResponseLossGatewayAdapter(TokenGatewayAdapter):
    def __init__(self, tokens: dict[str, str]) -> None:
        super().__init__(tokens)
        self.lost_for: set[tuple[str, int]] = set()

    async def submit_communication(
        self, episode_id: str, result: AgentCommunicationResult
    ) -> dict[str, Any]:
        accepted = await super().submit_communication(episode_id, result)
        key = (result.company_id, result.context.round)
        if result.company_id == "company_A" and key not in self.lost_for:
            self.lost_for.add(key)
            raise ConnectionError("simulated response loss after server acceptance")
        return accepted

class InteractionControllerAdapter:
    def __init__(self, token: str) -> None:
        self.client = TestClient(app)
        self.headers = {"X-Controller-Token": token}

    async def get_episode(self, episode_id: str) -> dict[str, Any]:
        response = self.client.get(f"/api/episodes/{episode_id}/state")
        assert response.status_code == 200, response.text
        return response.json()

    async def close_communication(
        self,
        episode_id: str,
        round_number: int,
        state_version: int,
        state_hash: str,
    ) -> dict[str, Any]:
        response = self.client.post(
            f"/api/v1/controller/episodes/{episode_id}/communication/close",
            headers=self.headers,
            json={
                "round": round_number,
                "state_version": state_version,
                "state_hash": state_hash,
            },
        )
        assert response.status_code == 200, response.text
        return response.json()

    async def settle_agent_round(
        self,
        episode_id: str,
        step_id: str,
        intent_ids: dict[str, str],
    ) -> dict[str, Any]:
        response = self.client.post(
            f"/api/v1/controller/episodes/{episode_id}/settle-agent-round",
            headers=self.headers,
            json={
                "step_id": step_id,
                "intent_ids": intent_ids,
                "fallback": "rule",
            },
        )
        assert response.status_code == 200, response.text
        return response.json()


def _create_interaction_episode(
    monkeypatch,
    *,
    episode_id: str,
    communication_mode: str,
    max_rounds: int,
    cooperation_mode: str = "off",
) -> tuple[str, dict[str, str]]:
    token = f"controller-{episode_id}"
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", token)
    response = TestClient(app).post(
        "/api/episodes",
        headers={"X-Controller-Token": token},
        json={
            "episode_id": episode_id,
            "episode_seed": 808,
            "company_ids": list(COMPANIES),
            "max_rounds": max_rounds,
            "market_model": "balanced",
            "information_mode": "perfect",
            "communication_mode": communication_mode,
            "cooperation_mode": cooperation_mode,
        },
    )
    assert response.status_code == 201, response.text
    return token, response.json()["agent_tokens"]


def test_cooperation_round_trip_replays_proposal_commitment_and_partial_fulfillment(
    monkeypatch,
):
    SESSIONS.clear()
    episode_id = "cooperation-orchestration"
    token, agent_tokens = _create_interaction_episode(
        monkeypatch,
        episode_id=episode_id,
        communication_mode="public_private",
        cooperation_mode="shared_resilience_v1",
        max_rounds=5,
    )
    runtimes = {
        "company_A": AgentRuntime(
            "mock-company_A",
            "company_A",
            MockModelClient(
                model_name="mock-company_A",
                cooperation_proposal_receiver="company_B",
                cooperation_proposal_round=1,
                cooperation_proposal_target_round=2,
                cooperation_proposal_amount_cents=1_000_000,
            ),
        ),
        "company_B": AgentRuntime(
            "mock-company_B",
            "company_B",
            MockModelClient(
                model_name="mock-company_B",
                cooperation_response="accept",
                shared_resilience_contribution_cents=300_000,
                shared_resilience_contribution_rounds=(2,),
            ),
            context_builder=DecisionContextBuilder(
                cooperation_history_mode="none"
            ),
        ),
        **{
            company_id: AgentRuntime(
                f"mock-{company_id}",
                company_id,
                MockModelClient(model_name=f"mock-{company_id}"),
            )
            for company_id in ("company_C", "company_D")
        },
    }
    coordinator = RoundCoordinator(
        InteractionControllerAdapter(token),
        TokenGatewayAdapter(agent_tokens),
        runtimes,
    )

    rounds = asyncio.run(coordinator.run_episode(episode_id, max_rounds=2))
    events = [item.event for item in rounds]

    assert len(events) == 2
    first = events[0]
    second = events[1]
    assert first.cooperation_round is not None
    assert len(first.cooperation_round.close.proposals_created) == 1
    proposal = first.cooperation_round.close.proposals_created[0]
    proposal_message_id = proposal.source_message_id
    first_traces = {item.company_id: item for item in first.traces}
    assert proposal_message_id in {
        item.message_id
        for item in first_traces["company_B"].communication_view.visible_messages
    }
    assert proposal_message_id not in {
        item.message_id
        for item in first_traces["company_C"].communication_view.visible_messages
    }
    assert first.cooperation_round.close.commitments_created == ()

    assert second.cooperation_round is not None
    b_trace = next(
        trace for trace in second.traces if trace.company_id == "company_B"
    )
    assert b_trace.decision_context["cooperation_history_mode"] == "none"
    assert b_trace.decision_context["cooperation"]["commitment_history"] == []
    assert len(second.cooperation_round.close.commitments_created) == 1
    commitment = second.cooperation_round.close.commitments_created[0]
    assert commitment.binding is False
    assert commitment.promised_contribution_cents == 1_000_000
    assert second.joint_action["company_B"][
        "shared_resilience_contribution_cents"
    ] == 300_000
    verification = second.cooperation_round.verifications[0]
    assert verification.fulfillment_ratio_ppm == 300_000
    assert verification.status == "partial_betrayal"
    assert "COMMITMENTS_GENERATED" in second.phases
    assert "SHARED_RESILIENCE_UPDATED" in second.phases
    assert "COMMITMENTS_VERIFIED" in second.phases

    assert len(verify_interaction_replay(events)) == 2
    assert len(
        verify_information_replay(events, SESSIONS[episode_id].manifest)
    ) == 16
    rebuilt = verify_cooperation_replay(events, MarketEnv(CONFIG))
    assert len(rebuilt) == 2
    assert rebuilt[-1] == second.cooperation_round
    metrics = compute_cooperation_metrics(events)
    assert metrics["proposal_count"] == 1
    assert metrics["acceptance_count"] == 1
    assert metrics["commitment_count"] == 1
    assert metrics["amount_weighted_fulfillment_ppm"] == 300_000
    assert metrics["fulfillment_status_counts"]["partial_betrayal"] == 1
    assert set(second.cooperation_round.benefit_attribution_by_company) == set(
        COMPANIES
    )

    base_context = next(
        trace for trace in second.traces if trace.company_id == "company_B"
    ).decision_context
    assert base_context is not None
    behavior_policy = MockModelClient(
        honor_shared_resilience_commitments=True,
        minimum_proposer_credibility_ppm=500_000,
    )

    def behavior_contribution(*, credibility_ppm: int, include_proposal: bool) -> int:
        raw = json.loads(json.dumps(base_context))
        if not include_proposal:
            raw["cooperation"]["proposals_received"] = []
            raw["cooperation"]["active_commitments"] = []
        else:
            raw["cooperation"]["public_credibility"]["company_A"][
                "credibility_ppm"
            ] = credibility_ppm
        context = DecisionContext.model_validate(raw)
        generated = asyncio.run(behavior_policy.generate_decision(context))
        return int(
            generated.parsed_output["requested_action"].get(
                "shared_resilience_contribution_cents", 0
            )
            or 0
        )

    assert behavior_contribution(
        credibility_ppm=900_000, include_proposal=False
    ) == 0
    assert behavior_contribution(
        credibility_ppm=900_000, include_proposal=True
    ) == 1_000_000
    assert behavior_contribution(
        credibility_ppm=100_000, include_proposal=True
    ) == 0

    communication_context = next(
        trace
        for trace in second.communication_phase.generation_traces
        if trace.company_id == "company_B"
    ).communication_context
    assert communication_context is not None
    credibility_policy = MockModelClient(
        cooperation_response="accept",
        minimum_proposer_credibility_ppm=500_000,
    )

    def response_for(credibility_ppm: int) -> str:
        raw = json.loads(json.dumps(communication_context))
        raw["cooperation"]["public_credibility"]["company_A"][
            "credibility_ppm"
        ] = credibility_ppm
        generated = asyncio.run(
            credibility_policy.generate_communication(
                CommunicationContext.model_validate(raw)
            )
        )
        return generated.parsed_output["messages"][0][
            "cooperation_response"
        ]["response"]

    assert response_for(900_000) == "accept"
    assert response_for(100_000) == "reject"
    forged_verification = verification.model_copy(update={"status": "fulfilled"})
    forged_record = second.cooperation_round.model_copy(
        update={"verifications": [forged_verification]}
    )
    forged_event = second.model_copy(update={"cooperation_round": forged_record})
    with pytest.raises(CooperationReplayMismatchError):
        verify_cooperation_replay([first, forged_event], MarketEnv(CONFIG))
    session = SESSIONS[episode_id]
    market_states = verify_replay(
        MarketEnv(CONFIG), session.manifest, session.transitions
    )
    assert market_states[-1].state_hash == session.env.get_state().state_hash


def test_no_communication_cooperation_action_is_a_valid_causal_baseline(monkeypatch):
    SESSIONS.clear()
    episode_id = "cooperation-no-message-control"
    token, agent_tokens = _create_interaction_episode(
        monkeypatch,
        episode_id=episode_id,
        communication_mode="off",
        cooperation_mode="shared_resilience_v1",
        max_rounds=5,
    )
    runtime = AgentRuntime(
        "mock-company_A",
        "company_A",
        MockModelClient(
            model_name="mock-company_A",
            shared_resilience_contribution_cents=500_000,
            shared_resilience_contribution_rounds=(1,),
        ),
    )
    coordinator = RoundCoordinator(
        InteractionControllerAdapter(token),
        TokenGatewayAdapter(agent_tokens),
        {"company_A": runtime},
    )

    coordinated = asyncio.run(coordinator.run_episode(episode_id, max_rounds=1))[0]
    event = coordinated.event

    assert event.communication_phase is not None
    assert event.communication_phase.mode == "off"
    assert event.communication_phase.closure.all_messages == []
    assert event.cooperation_round is not None
    assert event.cooperation_round.close.proposals_created == ()
    assert event.joint_action["company_A"][
        "shared_resilience_contribution_cents"
    ] == 500_000
    assert event.cooperation_round.total_contribution_cents == 500_000
    assert "COMMUNICATION_CLOSED_NOOP" in event.phases
    assert "SHARED_RESILIENCE_UPDATED" in event.phases
    assert len(verify_interaction_replay([event])) == 1
    assert len(verify_cooperation_replay([event], MarketEnv(CONFIG))) == 1


def _runtimes() -> dict[str, AgentRuntime]:
    private_price_request = CommunicationSubmission(
        messages=[
            MessageDraft(
                channel="private",
                recipients=["company_B"],
                speech_act="proposal",
                content="Consider setting this round's price to 12345 cents.",
                requested_peer_action=PartialActionClaim(price_cents=12_345),
            )
        ]
    )
    return {
        company_id: AgentRuntime(
            f"mock-{company_id}",
            company_id,
            MockModelClient(
                model_name=f"mock-{company_id}",
                communication_submission=(
                    private_price_request
                    if company_id == "company_A"
                    else CommunicationSubmission()
                ),
                honor_requested_price=company_id != "company_A",
            ),
        )
        for company_id in COMPANIES
    }


class FailingCommunicationClient(MockModelClient):
    async def generate_communication(self, _context):
        raise RuntimeError("simulated communication provider failure")


def test_private_message_changes_only_recipient_and_twenty_rounds_replay(
    monkeypatch, tmp_path: Path
):
    SESSIONS.clear()
    episode_id = "interaction-20-round"
    token, agent_tokens = _create_interaction_episode(
        monkeypatch,
        episode_id=episode_id,
        communication_mode="public_private",
        max_rounds=20,
    )
    coordinator = RoundCoordinator(
        InteractionControllerAdapter(token),
        TokenGatewayAdapter(agent_tokens),
        _runtimes(),
        event_logger=JsonlRoundEventLogger(tmp_path / "interaction-rounds.jsonl"),
    )

    rounds = asyncio.run(coordinator.run_episode(episode_id))

    assert len(rounds) == 20
    assert rounds[-1].settlement["state"]["terminal"] is True
    for round_index, coordinated in enumerate(rounds, start=1):
        event = coordinated.event
        assert "COMMUNICATION_CLOSED" in event.phases
        assert event.communication_phase is not None
        assert event.communication_phase.closed is True
        assert event.communication_phase.closure.state_hash == event.state_before_hash
        assert len(event.communication_phase.closure.all_messages) == 1
        traces = {trace.company_id: trace for trace in event.traces}
        assert traces["company_B"].requested_action["price_cents"] == 12_345
        assert traces["company_C"].requested_action["price_cents"] != 12_345
        assert traces["company_D"].requested_action["price_cents"] != 12_345
        assert event.joint_action["company_B"]["price_cents"] == 12_345
        assert event.state_after["companies"]["company_B"]["commercial"][
            "price_cents"
        ] == 12_345
        private_id = event.communication_phase.closure.all_messages[0].message_id
        assert private_id in {
            message.message_id
            for message in traces["company_B"].communication_view.visible_messages
        }
        assert private_id not in {
            message.message_id
            for message in traces["company_C"].communication_view.visible_messages
        }
        assert private_id not in {
            message.message_id
            for message in traces["company_D"].communication_view.visible_messages
        }
        expected_history_length = min(round_index - 1, 3)
        for company_id, trace in traces.items():
            history = trace.decision_context["recent_communication_views"]
            assert len(history) == expected_history_length
            if company_id not in {"company_A", "company_B"}:
                assert all(
                    not historical_view["visible_messages"]
                    for historical_view in history
                )

    interaction = verify_interaction_replay([item.event for item in rounds])
    assert len(interaction) == 20
    metrics = compute_interaction_metrics([item.event for item in rounds])
    assert metrics["message_count"] == 20
    assert metrics["message_disposition_counts"] == {"accepted": 20}
    assert metrics["requested_peer_claim_field_count"] == 20
    assert metrics["requested_peer_claim_alignment_ppm"] == 1_000_000
    session = SESSIONS[episode_id]
    market_states = verify_replay(
        MarketEnv(CONFIG), session.manifest, session.transitions
    )
    assert market_states[-1].state_hash == session.env.get_state().state_hash


def test_same_seed_off_mode_does_not_apply_hidden_communication(monkeypatch):
    SESSIONS.clear()
    episode_id = "interaction-off-control"
    token, agent_tokens = _create_interaction_episode(
        monkeypatch,
        episode_id=episode_id,
        communication_mode="off",
        max_rounds=5,
    )
    coordinator = RoundCoordinator(
        InteractionControllerAdapter(token),
        TokenGatewayAdapter(agent_tokens),
        _runtimes(),
    )

    coordinated = asyncio.run(coordinator.run_round(episode_id))
    traces = {trace.company_id: trace for trace in coordinated.event.traces}

    assert traces["company_B"].requested_action["price_cents"] == 10_000
    assert coordinated.event.communication_phase is not None
    assert coordinated.event.communication_phase.mode == "off"
    assert coordinated.event.communication_phase.closure.all_messages == []
    assert verify_interaction_replay(coordinated.event)


def test_same_frozen_state_direct_message_effect_is_recipient_scoped(monkeypatch):
    episode_id = "interaction-paired-direct-effect"

    SESSIONS.clear()
    off_token, off_agent_tokens = _create_interaction_episode(
        monkeypatch,
        episode_id=episode_id,
        communication_mode="off",
        max_rounds=5,
    )
    off_round = asyncio.run(
        RoundCoordinator(
            InteractionControllerAdapter(off_token),
            TokenGatewayAdapter(off_agent_tokens),
            _runtimes(),
        ).run_round(episode_id)
    )

    SESSIONS.clear()
    on_token, on_agent_tokens = _create_interaction_episode(
        monkeypatch,
        episode_id=episode_id,
        communication_mode="public_private",
        max_rounds=5,
    )
    on_round = asyncio.run(
        RoundCoordinator(
            InteractionControllerAdapter(on_token),
            TokenGatewayAdapter(on_agent_tokens),
            _runtimes(),
        ).run_round(episode_id)
    )

    off_traces = {trace.company_id: trace for trace in off_round.event.traces}
    on_traces = {trace.company_id: trace for trace in on_round.event.traces}
    assert off_round.event.state_before_hash == on_round.event.state_before_hash
    assert off_traces["company_B"].requested_action["price_cents"] == 10_000
    assert on_traces["company_B"].requested_action["price_cents"] == 12_345
    assert off_round.event.joint_action["company_B"]["price_cents"] == 10_000
    assert on_round.event.joint_action["company_B"]["price_cents"] == 12_345
    for company_id in ("company_C", "company_D"):
        assert off_traces[company_id].requested_action == on_traces[
            company_id
        ].requested_action
        assert off_round.event.joint_action[company_id] == on_round.event.joint_action[
            company_id
        ]
        off_context = dict(off_traces[company_id].decision_context)
        on_context = dict(on_traces[company_id].decision_context)
        for context in (off_context, on_context):
            context.pop("communication_view", None)
            context.pop("recent_communication_views", None)
            context["meta"].pop("observation_hash", None)
        assert off_context == on_context


def test_communication_failure_and_response_loss_do_not_block_settlement(
    monkeypatch,
):
    SESSIONS.clear()
    episode_id = "interaction-failure-resilience"
    token, agent_tokens = _create_interaction_episode(
        monkeypatch,
        episode_id=episode_id,
        communication_mode="public_private",
        max_rounds=5,
    )
    runtimes = _runtimes()
    runtimes["company_C"] = AgentRuntime(
        "failing-company_C",
        "company_C",
        FailingCommunicationClient(model_name="mock-failing-company_C"),
    )
    coordinator = RoundCoordinator(
        InteractionControllerAdapter(token),
        ResponseLossGatewayAdapter(agent_tokens),
        runtimes,
    )

    rounds = asyncio.run(coordinator.run_episode(episode_id))

    assert len(rounds) == 5
    assert rounds[-1].settlement["state"]["terminal"] is True
    for coordinated in rounds:
        phase = coordinated.event.communication_phase
        assert phase is not None
        generations = {
            trace.company_id: trace for trace in phase.generation_traces
        }
        assert generations["company_A"].generation_status == "submitted"
        assert any(
            "response loss" in item
            for item in generations["company_A"].validation_errors
        )
        assert generations["company_C"].generation_status == "fallback"
        assert generations["company_C"].is_silence is True
    assert len(verify_interaction_replay([item.event for item in rounds])) == 5


def test_state_only_interaction_replay_accepts_intentionally_empty_history(
    monkeypatch,
):
    SESSIONS.clear()
    episode_id = "interaction-state-only-history"
    token, agent_tokens = _create_interaction_episode(
        monkeypatch,
        episode_id=episode_id,
        communication_mode="public_private",
        max_rounds=5,
    )
    runtimes = _runtimes()
    for company_id, runtime in list(runtimes.items()):
        runtimes[company_id] = AgentRuntime(
            runtime.agent_id,
            company_id,
            runtime.model_client,
            context_builder=DecisionContextBuilder(context_mode="state_only"),
        )
    coordinator = RoundCoordinator(
        InteractionControllerAdapter(token),
        TokenGatewayAdapter(agent_tokens),
        runtimes,
    )

    rounds = [
        asyncio.run(coordinator.run_round(episode_id)),
        asyncio.run(coordinator.run_round(episode_id)),
    ]

    for trace in rounds[1].event.traces:
        assert trace.decision_context["context_mode"] == "state_only"
        assert trace.decision_context["recent_communication_views"] == []
    assert len(verify_interaction_replay([item.event for item in rounds])) == 2
