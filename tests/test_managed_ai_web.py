"""WebUI 管理回合必须运行 PersonaAgent，并返回可展示的真实节点事件。"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from fastapi.testclient import TestClient

import game_theory_agent.api as api
from game_theory_agent.agents.contracts import AgentRequestedAction


def _create_episode(client: TestClient, episode_id: str, *, human: bool = False):
    agent_configs = {
        "company_A": {
            "agent_id": "human-company_A" if human else "single-agent-company_A",
            "agent_type": "human" if human else "model",
            "model": None if human else "nvidia/nemotron-3-super-120b-a12b:free",
            "persona_name": "balanced_v1",
        },
        "company_B": {
            "agent_id": "single-agent-company_B",
            "agent_type": "model",
            "model": "nvidia/nemotron-nano-9b-v2:free",
            "persona_name": "risk_guarded_v1",
        },
    }
    response = client.post(
        "/api/episodes",
        json={
            "episode_id": episode_id,
            "episode_seed": 31,
            "company_ids": ["company_A", "company_B"],
            "max_rounds": 5,
            "information_mode": "perfect",
            "communication_mode": "off",
            "agent_configs": agent_configs,
            "game_mode": "single_company" if human else "market",
            "player_company_id": "company_A" if human else None,
        },
    )
    assert response.status_code == 201, response.text


def _fake_model_execution(
    *, episode_id, company_id, model_id, persona_name, agent_token, progress_callback=None
):
    session = api.SESSIONS[episode_id]
    state = session.env.get_state()
    observation = api._agent_observation(session, company_id)
    rule_action = api.build_rule_action(api.CONFIG, state, company_id)
    requested = AgentRequestedAction.model_validate(
        {
            key: value
            for key, value in rule_action.to_dict().items()
            if key in AgentRequestedAction.model_fields
        }
    )
    receipt = api.submit_agent_intent(
        episode_id,
        api.SubmitAgentIntentRequest(
            agent_id=f"single-agent-{company_id}",
            company_id=company_id,
            round=state.round,
            state_version=state.state_version,
            observation_hash=observation["observation_hash"],
            requested_action=requested,
            rationale="测试中的真实意图回执",
            expected_outcome="测试结算",
        ),
        agent_token=agent_token,
    )
    trace = {
        "status": "accepted",
        "repair_attempts": 0,
        "provider_usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        "latency_ms": 90,
        "provider_finish_reason": "stop",
        "provider_error_category": None,
        "error_code": None,
        "selected_candidate_id": "candidate_1",
    }
    return receipt["intent_id"], {
        "company_id": company_id,
        "model_id": model_id,
        "fallback_used": False,
        "events": [
            {"stage": "load_snapshot", "details": {}},
            {"stage": "generate_candidates", "details": {}},
            {"stage": "provider_response", "details": {"total_tokens": 30, "latency_ms": 90}},
            {"stage": "finalize", "details": {"status": "accepted"}},
        ],
        "trace": trace,
    }


def test_managed_round_executes_selected_models_and_returns_real_topology(monkeypatch):
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", "managed-test-token")
    monkeypatch.setattr(api, "_run_managed_persona_agent", _fake_model_execution)
    api.SESSIONS.clear()
    client = TestClient(api.app)
    _create_episode(client, "managed-models")

    response = client.post("/api/episodes/managed-models/managed-rounds", json={})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["state"]["round"] == 2
    assert set(payload["executions"]) == {"company_A", "company_B"}
    assert payload["executions"]["company_A"]["model_id"] == "nvidia/nemotron-3-super-120b-a12b:free"
    assert payload["executions"]["company_A"]["trace"]["provider_usage"]["total_tokens"] == 30
    assert payload["executions"]["company_A"]["fallback_used"] is False
    assert set(payload["observations"]) == {"company_A", "company_B"}
    assert payload["observations"]["company_A"]["own_company"]["company_id"] == "company_A"
    assert len(payload["history"]) == 1
    assert payload["history"][0]["step_result"]["settled_round"] == 1


def test_created_episode_exposes_real_observations_and_empty_history():
    api.SESSIONS.clear()
    client = TestClient(api.app)
    _create_episode(client, "managed-initial")

    payload = client.get("/api/episodes/managed-initial/state").json()

    assert set(payload["observations"]) == {"company_A", "company_B"}
    assert payload["observations"]["company_A"]["round"] == 1
    assert payload["history"] == []


def test_managed_round_progress_is_visible_before_post_completes(monkeypatch):
    entered_provider = threading.Event()
    release_provider = threading.Event()

    def blocking_execution(**kwargs):
        callback = kwargs.pop("progress_callback")
        callback("load_snapshot", {})
        callback(
            "provider_request",
            {"attempt": 1, "repair": False, "secret": "must-not-leak"},
        )
        entered_provider.set()
        assert release_provider.wait(timeout=5)
        callback(
            "provider_response",
            {
                "attempt": 1,
                "finish_reason": "stop",
                "usage_available": True,
                "total_tokens": 30,
                "latency_ms": 90,
            },
        )
        return _fake_model_execution(**kwargs)

    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", "managed-test-token")
    monkeypatch.setattr(api, "_run_managed_persona_agent", blocking_execution)
    api.SESSIONS.clear()
    client = TestClient(api.app)
    _create_episode(client, "managed-live-progress")

    with ThreadPoolExecutor(max_workers=1) as pool:
        response_future = pool.submit(
            client.post,
            "/api/episodes/managed-live-progress/managed-rounds",
            json={},
        )
        assert entered_provider.wait(timeout=5)
        progress_response = client.get(
            "/api/episodes/managed-live-progress/managed-rounds/progress"
        )
        assert progress_response.status_code == 200
        progress = progress_response.json()
        assert progress["status"] == "running"
        company = progress["companies"]["company_A"]
        assert company["current_stage"] == "provider_request"
        assert company["provider_waiting"] is True
        assert company["provider_attempts"] == 1
        assert "secret" not in company["events"][-1]["details"]
        release_provider.set()
        response = response_future.result(timeout=10)

    assert response.status_code == 200, response.text
    completed = client.get(
        "/api/episodes/managed-live-progress/managed-rounds/progress"
    ).json()
    assert completed["status"] == "completed"
    assert completed["companies"]["company_A"]["total_tokens"] == 30
    assert completed["companies"]["company_A"]["provider_latency_ms"] == 90


def test_managed_round_progress_returns_idle_for_existing_episode():
    api.SESSIONS.clear()
    client = TestClient(api.app)
    _create_episode(client, "managed-idle-progress")

    response = client.get(
        "/api/episodes/managed-idle-progress/managed-rounds/progress"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "idle"


def test_managed_round_combines_human_action_with_model_intent(monkeypatch):
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", "managed-test-token")
    monkeypatch.setattr(api, "_run_managed_persona_agent", _fake_model_execution)
    api.SESSIONS.clear()
    client = TestClient(api.app)
    _create_episode(client, "managed-human", human=True)

    response = client.post(
        "/api/episodes/managed-human/managed-rounds",
        json={
            "human_actions": {
                "company_A": {
                    "price_cents": 8750,
                    "advertising_budget_cents": 300000,
                    "shared_resilience_contribution_cents": 100000,
                    "strategy_summary": "人类提交",
                }
            }
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["decision_resolutions"]["company_A"]["action"]["price_cents"] == 8750
    assert payload["decision_resolutions"]["company_A"]["source"] == "agent-intent:human-company_A"
    assert payload["executions"]["company_B"]["fallback_used"] is False


def test_managed_round_reports_missing_fixed_secret_as_diagnostic_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", "managed-test-token")
    monkeypatch.setattr(api, "MANAGED_OPENROUTER_SECRET", tmp_path / "open_router-api_key.env")
    api.SESSIONS.clear()
    client = TestClient(api.app)
    _create_episode(client, "managed-missing-secret")

    response = client.post("/api/episodes/managed-missing-secret/managed-rounds", json={})

    assert response.status_code == 503
    assert "OpenRouter" in response.text
    assert "open_router-api_key.env" in response.text


def test_managed_trace_payload_keeps_detailed_safe_node_data():
    trace = SimpleNamespace(
        status="accepted",
        repair_attempts=1,
        provider_usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        latency_ms=90,
        provider_finish_reason="stop",
        provider_error_category=None,
        provider_usage_available=True,
        error_code=None,
        selected_candidate_id="candidate_1",
        selection_reason_codes=["profit_guard"],
        validation_errors=["first_output_invalid"],
        memory_view={"history_limit": 2, "recent_feedback": []},
        strategy_reflection={"source": "deterministic", "summary": "保持价格纪律"},
        prompt_audit={"system_prompt": "你是风险审慎的市场公司。", "user_prompt": "当前需求 12480。"},
        candidates=[{"candidate_id": "candidate_1", "label": "稳健价格", "action": {"price_cents": 9800}}],
        prepared_intent={"agent_id": "single-agent-company_A", "action": {"price_cents": 9800}},
        intent_receipt={"intent_id": "intent-1", "accepted": True},
    )

    payload = api._managed_trace_payload(trace)

    assert payload["memory_view"]["history_limit"] == 2
    assert payload["prompt_audit"]["user_prompt"] == "当前需求 12480。"
    assert payload["strategy_reflection"]["summary"] == "保持价格纪律"
    assert payload["candidates"][0]["candidate_id"] == "candidate_1"
    assert payload["prepared_intent"]["action"]["price_cents"] == 9800
    assert payload["intent_receipt"]["intent_id"] == "intent-1"
    assert "raw_response" not in payload
