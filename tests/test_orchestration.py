import asyncio
import json
from typing import Any

from fastapi.testclient import TestClient

from game_theory_agent.agents import AgentDecisionResult, AgentRuntime
from game_theory_agent.api import CONFIG, SESSIONS, agent_app, app
from game_theory_agent.information import verify_information_replay
from game_theory_agent.market import MarketEnv
from game_theory_agent.market.replay import verify_replay
from game_theory_agent.model_clients import MockModelClient
from game_theory_agent.orchestration import JsonlRoundEventLogger, RoundCoordinator


class GatewayAdapter:
    def __init__(self) -> None:
        self.client = TestClient(agent_app)

    async def get_observation(
        self, episode_id: str, company_id: str
    ) -> dict[str, Any]:
        response = self.client.get(
            f"/v1/episodes/{episode_id}/companies/{company_id}/observation"
        )
        assert response.status_code == 200, response.text
        return response.json()

    async def submit_intent(
        self, episode_id: str, result: AgentDecisionResult
    ) -> dict[str, Any]:
        assert result.decision is not None
        response = self.client.post(
            f"/v1/episodes/{episode_id}/intents",
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
                    result.decision.plan.expected_outcome.model_dump()
                ),
            },
        )
        assert response.status_code == 202, response.text
        return response.json()


class ControllerAdapter:
    def __init__(self, token: str) -> None:
        self.client = TestClient(app)
        self.headers = {"X-Controller-Token": token}

    async def get_episode(self, episode_id: str) -> dict[str, Any]:
        response = self.client.get(f"/api/episodes/{episode_id}/state")
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


class FailingModelClient:
    async def generate_decision(self, _context):
        raise RuntimeError("synthetic provider outage")


def test_coordinator_runs_complete_episode_and_keeps_action_lock(
    monkeypatch, tmp_path
):
    token = "coordinator-test-token"
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", token)
    SESSIONS.clear()
    created = TestClient(app).post(
        "/api/episodes",
        json={
            "episode_id": "coordinated-episode",
            "episode_seed": 77,
            "company_ids": [
                "company_A",
                "company_B",
                "company_C",
                "company_D",
            ],
            "max_rounds": 5,
            "market_model": "balanced",
            "information_mode": "perfect",
        },
    )
    assert created.status_code == 201, created.text
    logger = JsonlRoundEventLogger(tmp_path / "agent-rounds.jsonl")
    runtime = AgentRuntime(
        "mock-planner-A", "company_A", MockModelClient()
    )
    coordinator = RoundCoordinator(
        ControllerAdapter(token),
        GatewayAdapter(),
        {"company_A": runtime},
        event_logger=logger,
    )

    rounds = asyncio.run(coordinator.run_episode("coordinated-episode"))

    assert len(rounds) == 5
    assert rounds[-1].settlement["state"]["terminal"] is True
    for index, coordinated in enumerate(rounds, start=1):
        assert coordinated.event.settled_round == index
        assert coordinated.event.joint_action_hash == coordinated.settlement[
            "step_result"
        ]["joint_action_hash"]
        assert coordinated.event.state_before["state_hash"] == (
            coordinated.event.state_before_hash
        )
        assert coordinated.event.state_after["state_hash"] == (
            coordinated.event.state_after_hash
        )
        assert set(coordinated.event.joint_action) == {
            "company_A",
            "company_B",
            "company_C",
            "company_D",
        }
        traces = {trace.company_id: trace for trace in coordinated.event.traces}
        assert traces["company_A"].decision_status == "submitted"
        assert traces["company_A"].agent_type == "mock"
        assert traces["company_A"].observation is not None
        assert traces["company_A"].decision_context is not None
        assert traces["company_A"].decision_context["context_mode"] == "full"
        assert traces["company_A"].retry_count == 0
        assert traces["company_A"].persona == "none"
        assert traces["company_A"].persona_profile_hash.startswith("sha256:")
        assert traces["company_A"].persona_utility is not None
        assert traces["company_A"].persona_utility.cooperation_available is False
        assert traces["company_A"].resolution_source.startswith("agent-intent:")
        analysis = traces["company_A"].result_analysis
        assert analysis.analysis_schema_version == "result-analysis-v1.3.0"
        assert (
            analysis.expectation_assessment.causal_claim
            == "controlled_same_seed_counterfactual"
        )
        assert analysis.counterfactual_analysis is not None
        assert len(analysis.counterfactual_analysis["alternatives"]) == 2
        assert analysis.goal_assessment.criteria.minimum_round_profit_cents == 0
        assert analysis.goal_assessment.criteria.minimum_cash_reserve_cents > 0
        assert (
            analysis.goal_assessment.criteria.maximum_fixed_spend_cents
            is not None
        )
        assert analysis.observed_outcome.company_after.round_profit_cents == (
            coordinated.settlement["state"]["companies"]["company_A"][
                "financial"
            ]["round_profit_cents"]
        )
        changes = analysis.observed_outcome.company_changes
        assessment = analysis.expectation_assessment
        if index == 1:
            assert changes.previous_round_profit_cents is None
            assert changes.profit_change_vs_previous_round_cents is None
            assert assessment.actual_directions["profit"] == "baseline_unavailable"
            assert assessment.matches["profit"] is None
            assert not any(item.startswith("profit ") for item in assessment.mismatches)
            assert changes.base_capacity_delta_orders < 0
            assert changes.capacity_investment_cents == 0
            assert assessment.observed_directions["capacity"] == "down"
            assert assessment.actual_directions["capacity"] == "stable"
            assert assessment.matches["capacity"] is True
        else:
            assert changes.previous_round_profit_cents is not None
            assert changes.profit_change_vs_previous_round_cents is not None
        assert all(
            traces[company_id].resolution_source == "controller-rule-fallback"
            for company_id in ("company_B", "company_C", "company_D")
        )
    assert len(logger.read_all()) == 5
    assert len(
        verify_information_replay(
            logger.read_all(), SESSIONS["coordinated-episode"].manifest
        )
    ) == 5
    memory = runtime.memory.snapshot()
    assert len(memory["recent_rounds"]) == 3
    assert memory["rolling_summary"]["window_rounds"] == 5
    assert "profit_trend" in memory["rolling_summary"]
    assert memory["recent_rounds"][-1]["persona_utility"] is not None

    first_event = rounds[0].event
    repeated = asyncio.run(
        ControllerAdapter(token).settle_agent_round(
            "coordinated-episode",
            "coordinated-episode:1:0",
            {"company_A": first_event.traces[0].intent_id},
        )
    )
    assert repeated["state"]["state_hash"] == first_event.state_after_hash

    session = SESSIONS["coordinated-episode"]
    replayed = verify_replay(MarketEnv(CONFIG), session.manifest, session.transitions)
    assert replayed[-1].state_hash == rounds[-1].settlement["state"]["state_hash"]


def test_failed_model_fallback_is_kept_in_strategic_memory(monkeypatch):
    token = "fallback-memory-token"
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", token)
    SESSIONS.clear()
    response = TestClient(app).post(
        "/api/episodes",
        json={
            "episode_id": "fallback-memory",
            "episode_seed": 91,
            "company_ids": ["company_A", "company_B"],
            "max_rounds": 5,
        },
    )
    assert response.status_code == 201
    runtime = AgentRuntime(
        "failing-A", "company_A", FailingModelClient()
    )
    coordinator = RoundCoordinator(
        ControllerAdapter(token),
        GatewayAdapter(),
        {"company_A": runtime},
    )

    result = asyncio.run(coordinator.run_round("fallback-memory"))

    trace = next(
        item for item in result.event.traces if item.company_id == "company_A"
    )
    assert trace.decision_status == "fallback"
    memory = runtime.memory.snapshot()
    assert memory["rolling_summary"]["fallback_count"] == 1
    assert memory["recent_rounds"][0]["decision_status"] == "rule_fallback"
    assert memory["recent_rounds"][0]["error_code"] == "MODEL_ERROR"
