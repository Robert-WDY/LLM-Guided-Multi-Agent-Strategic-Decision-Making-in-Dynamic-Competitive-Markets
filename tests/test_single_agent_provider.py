import json

import httpx
import pytest

from game_theory_agent.agents.single.models import (
    DecisionContext,
    EpisodeMemoryView,
    PromptTemplate,
    RoundFeedback,
    SnapshotKey,
    StrategyReflection,
)
from game_theory_agent.agents.single.provider import (
    ALLOWED_FREE_MODELS,
    ModelNotAllowedError,
    OpenRouterProvider,
    ProviderInvalidDecisionError,
    ProviderResponseError,
    SecretConfigurationError,
    load_openrouter_api_key,
    validate_model_id,
)


def proposal_payload():
    action = {
        "price_cents": 1180,
        "advertising_budget_cents": 240000,
        "service_budget_cents": 160000,
        "capacity_investment_cents": 0,
        "resilience_budget_cents": 80000,
        "incident_response": {"mode": "wait", "repair_budget_cents": 0},
        "strategy_summary": "平衡扩张",
    }
    candidate = {
        "candidate_id": "balanced",
        "label": "平衡扩张",
        "action": action,
        "evidence_paths": ["own_company.financial.cash_balance_cents"],
        "persona_influences": [
            {
                "trait_key": "risk_tolerance",
                "direction": "increase",
                "affected_choice": "advertising_budget_cents",
                "summary": "较高风险容忍支持适度扩张。",
            }
        ],
        "tradeoffs": ["增长与现金消耗"],
        "risk_flags": [],
        "expected_outcome": "市场份额小幅提高",
    }
    conservative = json.loads(json.dumps(candidate))
    conservative["candidate_id"] = "conservative"
    conservative["label"] = "保守方案"
    growth = json.loads(json.dumps(candidate))
    growth["candidate_id"] = "growth"
    growth["label"] = "增长方案"
    return {
        "candidates": [conservative, candidate, growth],
        "selected_candidate_id": "balanced",
        "selection_reason_codes": ["share_growth", "cash_protection"],
    }


def decision_context() -> DecisionContext:
    return DecisionContext(
        snapshot_key=SnapshotKey(
            episode_id="episode-1",
            company_id="company_A",
            round=2,
            state_version=1,
            state_hash="hash-2",
        ),
        observation={
            "episode_id": "episode-1",
            "company_id": "company_A",
            "round": 2,
            "own_company": {"financial": {"cash_balance_cents": 10_000}},
            "public_companies": [{"company_id": "company_B", "rank": 2}],
        },
        action_contract={"constraints": {"bounds": {"price_cents": {"min": 800, "max": 2_000}}}},
        memory=EpisodeMemoryView(
            recent_feedback=[
                RoundFeedback(
                    settled_round=1,
                    own_action={"price_cents": 1_180},
                    own_result={"round_profit_cents": 500, "market_share_ppm": 230_000},
                    market={"realized_demand_orders": 1_200},
                )
            ],
            previous_selected_candidate_id="balanced",
            previous_expected_outcome="份额小幅提高",
        ),
        reflection=StrategyReflection(
            source="deterministic",
            lesson_codes=["profit_positive"],
            adjustments=["保持现金约束内的有效投入。"],
            evidence_paths=["memory.recent_feedback[-1].own_result.round_profit_cents"],
            summary="保持现金约束内的有效投入。",
        ),
    )


def test_secret_loader_accepts_raw_and_env_file_formats(tmp_path):
    raw = tmp_path / "raw.env"
    raw.write_text("sk-or-v1-test-value\n", encoding="utf-8")
    assigned = tmp_path / "assigned.env"
    assigned.write_text("# local only\nOPENROUTER_API_KEY=sk-or-v1-assigned\n", encoding="utf-8")

    assert load_openrouter_api_key(raw) == "sk-or-v1-test-value"
    assert load_openrouter_api_key(assigned) == "sk-or-v1-assigned"


def test_secret_loader_error_never_contains_file_contents(tmp_path):
    invalid = tmp_path / "invalid.env"
    invalid.write_text("OPENROUTER_API_KEY=\nPRIVATE_MARKER=do-not-leak", encoding="utf-8")

    with pytest.raises(SecretConfigurationError) as exc_info:
        load_openrouter_api_key(invalid)

    assert "do-not-leak" not in str(exc_info.value)


@pytest.mark.parametrize(
    "model_id",
    ["openai/gpt-test", "anthropic/claude-test", "google/gemini-test", "google~gemini-test"],
)
def test_model_policy_rejects_blocked_provider_namespaces(model_id):
    with pytest.raises(ModelNotAllowedError):
        validate_model_id(model_id)


def test_model_policy_exposes_only_the_four_approved_free_models():
    default = validate_model_id("nvidia/nemotron-3-super-120b-a12b:free")
    assert default.structured_output
    assert default.output_mode == "json_object"
    assert default.usage_supported is True
    assert len(ALLOWED_FREE_MODELS) == 4
    assert all(model_id.endswith(":free") for model_id in ALLOWED_FREE_MODELS)


def test_openrouter_provider_returns_validated_proposal_and_usage():
    captured_payload = {}

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        assert request.url == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        request_body = json.loads(request.content)
        captured_payload = request_body
        assert request_body["model"] == "nvidia/nemotron-3-super-120b-a12b:free"
        assert request_body["response_format"]["type"] == "json_object"
        assert request_body["reasoning"] == {"enabled": False, "exclude": True}
        assert request_body["max_completion_tokens"] == 1800
        assert "hidden chain of thought" not in request_body["messages"][0]["content"].lower()
        return httpx.Response(
            200,
            json={
                "model": request_body["model"],
                "choices": [{"message": {"content": json.dumps(proposal_payload())}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    provider = OpenRouterProvider(api_key="test-key", client=client)

    result = provider.generate_decision(
        model_id="nvidia/nemotron-3-super-120b-a12b:free",
        context=decision_context(),
        repair_errors=["provider_output_invalid"],
    )

    assert result.proposal.selected_candidate_id == "balanced"
    assert result.usage.total_tokens == 200
    assert result.model_id == "nvidia/nemotron-3-super-120b-a12b:free"
    user_message = captured_payload["messages"][1]["content"]
    user_payload = json.loads(user_message[user_message.index("{"):])
    visible_context = user_payload["visible_context"]
    assert visible_context["snapshot_key"]["state_hash"] == "hash-2"
    assert visible_context["memory"]["recent_feedback"][0]["settled_round"] == 1
    assert visible_context["reflection"]["lesson_codes"] == ["profit_positive"]
    assert user_payload["repair_errors"] == ["provider_output_invalid"]
    assert "persona" not in user_payload
    assert "persona influence" not in captured_payload["messages"][0]["content"].lower()
    assert '"candidates"' in user_message
    assert "persona_influences" not in user_message
    assert "intent_receipt" not in json.dumps(visible_context)


def test_custom_prompt_template_is_rendered_into_exact_provider_messages():
    captured_payload = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(proposal_payload())}}]},
        )

    provider = OpenRouterProvider(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    provider.generate_decision(
        model_id="z-ai/glm-5.2:free",
        context=decision_context(),
        prompt_template=PromptTemplate(
            system_prompt="自定义系统提示词",
            user_prompt_template="本轮输入：{{decision_input}}\n只返回 JSON。",
        ),
    )

    messages = captured_payload["messages"]
    assert messages[0]["content"].startswith("自定义系统提示词")
    assert "简体中文" in messages[0]["content"]
    assert messages[1]["content"].startswith("本轮输入：{")
    assert '"visible_context"' in messages[1]["content"]
    assert messages[1]["content"].endswith("\n只返回 JSON。")
    assert provider.last_prompt_audit.system_prompt == messages[0]["content"]
    assert provider.last_prompt_audit.user_prompt == messages[1]["content"]


def test_openrouter_provider_rejects_english_readable_candidate_fields():
    payload = proposal_payload()
    payload["candidates"][0]["label"] = "Conservative plan"

    provider = OpenRouterProvider(
        api_key="test-key",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": json.dumps(payload)}}]},
                )
            )
        ),
    )

    with pytest.raises(ProviderInvalidDecisionError):
        provider.generate_decision(model_id="z-ai/glm-5.2:free", context=decision_context())


def test_user_prompt_template_requires_one_decision_input_slot():
    with pytest.raises(ValueError):
        PromptTemplate(user_prompt_template="没有动态输入")
    with pytest.raises(ValueError):
        PromptTemplate(user_prompt_template="{{decision_input}}{{decision_input}}")


def test_openrouter_provider_reports_malformed_model_output_without_leaking_it():
    marker = "PRIVATE-MODEL-OUTPUT"

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": marker}}]})

    provider = OpenRouterProvider(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )

    with pytest.raises(ProviderInvalidDecisionError) as exc_info:
        provider.generate_decision(
            model_id="z-ai/glm-5.2:free",
            context=decision_context(),
        )

    assert marker not in str(exc_info.value)
    assert exc_info.value.code == "json_invalid"
    assert exc_info.value.latency_ms >= 0
    assert issubclass(ProviderInvalidDecisionError, ProviderResponseError)


@pytest.mark.parametrize(
    ("content", "finish_reason", "expected_code"),
    [
        ("", "stop", "empty_response"),
        ('{"candidates": [', "length", "truncated"),
        ("{}", "stop", "required_field_missing"),
    ],
)
def test_openrouter_provider_classifies_invalid_outputs(content, finish_reason, expected_code):
    provider = OpenRouterProvider(
        api_key="test-key",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={
                        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                    },
                )
            )
        ),
    )

    with pytest.raises(ProviderInvalidDecisionError) as exc_info:
        provider.generate_decision(model_id="nvidia/nemotron-3-super-120b-a12b:free", context=decision_context())

    assert exc_info.value.code == expected_code
    assert exc_info.value.finish_reason == finish_reason
    assert exc_info.value.usage.total_tokens == 18


def test_openrouter_provider_classifies_schema_rejection_without_response_body_leak():
    provider = OpenRouterProvider(
        api_key="test-key",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(400, text="PRIVATE provider schema details")
            )
        ),
    )

    with pytest.raises(ProviderResponseError) as exc_info:
        provider.generate_decision(model_id="nvidia/nemotron-3-super-120b-a12b:free", context=decision_context())

    assert exc_info.value.code == "schema_rejected"
    assert "PRIVATE" not in str(exc_info.value)


@pytest.mark.parametrize(
    "content",
    [
        lambda payload: f"```json\n{payload}\n```",
        lambda payload: [{"type": "text", "text": payload}],
        lambda payload: f"Decision JSON:\n{payload}\nEnd",
    ],
)
def test_openrouter_provider_accepts_common_json_wrappers(content):
    encoded = json.dumps(proposal_payload())

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content(encoded)}}]},
        )

    provider = OpenRouterProvider(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(respond)),
    )

    result = provider.generate_decision(
        model_id="z-ai/glm-5.2:free",
        context=decision_context(),
    )

    assert result.proposal.selected_candidate_id == "balanced"
