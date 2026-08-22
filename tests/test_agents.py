import asyncio
import json
from dataclasses import replace

import pytest

from fastapi.testclient import TestClient

from game_theory_agent.agents import (
    AgentRuntime,
    DecisionContextBuilder,
    EpisodeMemory,
    MarketRegimeEvaluator,
)
from game_theory_agent.api import CONFIG, SESSIONS, agent_app, app
from game_theory_agent.model_clients import (
    DeepSeekModelClient,
    DoubaoModelClient,
    MockModelClient,
)


controller_client = TestClient(app)
gateway_client = TestClient(agent_app)


def _create(episode_id: str, information_mode: str) -> dict:
    response = controller_client.post(
        "/api/episodes",
        json={
            "episode_id": episode_id,
            "episode_seed": 420,
            "company_ids": ["company_A", "company_B"],
            "max_rounds": 5,
            "information_mode": information_mode,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_observation_information_modes_are_episode_scoped():
    SESSIONS.clear()
    perfect = _create("perfect-info", "perfect")
    public = _create("public-info", "public")
    assert perfect["manifest"]["information_mode"] == "perfect"
    assert public["manifest"]["information_mode"] == "public"

    perfect_observation = gateway_client.get(
        "/v1/episodes/perfect-info/companies/company_A/observation"
    ).json()
    public_observation = gateway_client.get(
        "/v1/episodes/public-info/companies/company_A/observation"
    ).json()

    assert perfect_observation["information_mode"] == "perfect"
    assert "episode_seed" not in perfect_observation
    assert "episode_seed" not in perfect_observation["episode_config"]
    assert perfect_observation["market_regime"]["primary"]
    assert "financial" in perfect_observation["competitors"][0]
    assert public_observation["information_mode"] == "public"
    assert "financial" not in public_observation["competitors"][0]
    assert "financial" in public_observation["own_company"]
    assert "financial" not in public_observation["public_companies"][0]
    assert "round_revenue_cents" not in public_observation["public_companies"][0]
    assert "service_quality_ppm" not in public_observation["public_companies"][0]
    assert public_observation["observation_hash"].startswith("sha256:")
    assert public_observation["visibility_policy_version"] == (
        "visibility-public-v2.0.0"
    )


def test_market_regime_is_deterministic_and_does_not_change_state():
    SESSIONS.clear()
    _create("regime-test", "perfect")
    state = SESSIONS["regime-test"].env.get_state()
    stressed_companies = tuple(
        replace(
            company,
            commercial=replace(company.commercial, price_cents=8_000),
            operations=replace(
                company.operations, capacity_utilization_ppm=950_000
            ),
        )
        for company in state.companies
    )
    stressed = replace(
        state,
        state_version=1,
        market=replace(
            state.market,
            realized_demand_orders=14_000,
            actual_supply_cost_index_ppm=1_160_000,
        ),
        companies=stressed_companies,
    )

    regime = MarketRegimeEvaluator(CONFIG).evaluate(stressed)

    assert regime["primary"] == "supply_cost_crisis"
    assert regime["competition"] == "price_war"
    assert regime["demand"] == "high"
    assert regime["capacity"] == "constrained"
    assert regime["cost"] == "crisis"
    assert SESSIONS["regime-test"].env.get_state().state_hash == state.state_hash


def test_mock_agent_builds_a_valid_intent_from_gateway_observation():
    SESSIONS.clear()
    _create("mock-agent", "perfect")
    observation = gateway_client.get(
        "/v1/episodes/mock-agent/companies/company_A/observation"
    ).json()
    runtime = AgentRuntime(
        "mock-planner-A", "company_A", MockModelClient()
    )

    result = asyncio.run(runtime.decide(observation))

    assert result.success
    assert result.decision is not None
    assert result.context.state_hash == observation["state_hash"]
    assert result.context.meta.rounds_remaining == 5
    assert result.context.market_regime["primary"]
    assert result.context.recent_rounds == []
    assert "recent_history" not in result.context.model_dump()
    assert result.decision.requested_action.price_cents == 10_000
    assert result.fallback_required is False


def test_state_only_context_hides_cross_round_plan_and_history():
    SESSIONS.clear()
    _create("state-only-agent", "perfect")
    observation = gateway_client.get(
        "/v1/episodes/state-only-agent/companies/company_A/observation"
    ).json()
    memory = EpisodeMemory()
    memory.set_current_plan({"plan_id": "prior-plan", "phase": "growth"})
    runtime = AgentRuntime(
        "state-only-planner",
        "company_A",
        MockModelClient(),
        memory=memory,
        context_builder=DecisionContextBuilder(context_mode="state_only"),
    )

    result = asyncio.run(runtime.decide(observation))

    assert result.context.context_mode == "state_only"
    assert result.context.current_plan is None
    assert result.context.recent_rounds == []
    assert result.context.critical_events == []


class _FakeUsage:
    input_tokens = 100
    output_tokens = 50


class _FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.usage = _FakeUsage()


class _FakeResponses:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self.outputs.pop(0))


class _FakeArkClient:
    def __init__(self, outputs: list[str]) -> None:
        self.responses = _FakeResponses(outputs)


def test_doubao_client_repairs_json_and_returns_valid_decision():
    SESSIONS.clear()
    _create("doubao-agent", "perfect")
    observation = gateway_client.get(
        "/v1/episodes/doubao-agent/companies/company_A/observation"
    ).json()
    mock_runtime = AgentRuntime(
        "source", "company_A", MockModelClient()
    )
    valid = asyncio.run(mock_runtime.decide(observation))
    assert valid.decision is not None
    fake = _FakeArkClient(
        [
            "这不是 JSON",
            "```json\n"
            + json.dumps(valid.decision.model_dump(mode="json"), ensure_ascii=False)
            + "\n```",
        ]
    )
    runtime = AgentRuntime(
        "doubao-planner-A",
        "company_A",
        DoubaoModelClient(client=fake, model="doubao-test"),
    )

    result = asyncio.run(runtime.decide(observation))

    assert result.success
    assert result.model_name == "doubao-test"
    assert result.prompt_version == "market-planner-prompt-v1.14.0"
    assert result.input_tokens == 200
    assert result.output_tokens == 100
    assert result.retry_count == 1
    assert len(fake.responses.calls) == 2
    assert fake.responses.calls[0]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert "上一次输出无效" in fake.responses.calls[1]["input"]


def test_doubao_client_requires_api_key_without_injected_client(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ARK_API_KEY"):
        DoubaoModelClient()


class _FakeChatUsage:
    prompt_tokens = 80
    completion_tokens = 40


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeChatResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeChatUsage()


class _FakeCompletions:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeChatResponse(self.outputs.pop(0))


class _FakeChat:
    def __init__(self, outputs: list[str]) -> None:
        self.completions = _FakeCompletions(outputs)


class _FakeDeepSeekClient:
    def __init__(self, outputs: list[str]) -> None:
        self.chat = _FakeChat(outputs)


def test_deepseek_client_uses_json_mode_and_validates_decision():
    SESSIONS.clear()
    _create("deepseek-agent", "perfect")
    observation = gateway_client.get(
        "/v1/episodes/deepseek-agent/companies/company_A/observation"
    ).json()
    source = AgentRuntime("source", "company_A", MockModelClient())
    source_result = asyncio.run(source.decide(observation))
    assert source_result.decision is not None
    output = json.dumps(
        source_result.decision.model_dump(mode="json"), ensure_ascii=False
    )
    fake = _FakeDeepSeekClient([output])
    runtime = AgentRuntime(
        "deepseek-planner-A",
        "company_A",
        DeepSeekModelClient(client=fake, model="deepseek-test"),
    )

    result = asyncio.run(runtime.decide(observation))

    assert result.success
    assert result.model_name == "deepseek-test"
    assert result.input_tokens == 80
    assert result.output_tokens == 40
    call = fake.chat.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "DecisionContext" in call["messages"][1]["content"]


def test_deepseek_client_requires_api_key_without_injected_client(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        DeepSeekModelClient()
