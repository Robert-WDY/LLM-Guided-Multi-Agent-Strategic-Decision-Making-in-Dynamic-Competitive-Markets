"""单 Agent 提示词与输出预算回归测试。"""

from __future__ import annotations

import json

from game_theory_agent.agents.single.provider import (
    ALLOWED_FREE_MODELS,
    DECISION_WIRE_SCHEMA,
    _build_payload,
    build_decision_prompts,
)
from tests.test_single_agent_provider import decision_context


def test_structured_output_payload_does_not_repeat_schema_in_user_prompt():
    context = decision_context()
    structured = ALLOWED_FREE_MODELS["nvidia/nemotron-nano-9b-v2:free"]

    payload = _build_payload(structured, context, [])
    user_prompt = payload["messages"][1]["content"]

    assert '"output_schema"' not in user_prompt
    assert payload["response_format"]["json_schema"]["schema"] == DECISION_WIRE_SCHEMA


def test_non_structured_output_keeps_schema_in_user_prompt():
    context = decision_context()
    unstructured = ALLOWED_FREE_MODELS["z-ai/glm-5.2:free"]

    payload = _build_payload(unstructured, context, [])

    assert '"output_schema"' in payload["messages"][1]["content"]
    assert "response_format" not in payload


def test_completion_token_budget_defaults_to_1800_and_clamps_environment(monkeypatch):
    monkeypatch.delenv("MARKET_AGENTS_MAX_COMPLETION_TOKENS", raising=False)
    spec = ALLOWED_FREE_MODELS["nvidia/nemotron-3-super-120b-a12b:free"]
    assert _build_payload(spec, decision_context(), [])["max_completion_tokens"] == 1800

    monkeypatch.setenv("MARKET_AGENTS_MAX_COMPLETION_TOKENS", "200")
    assert _build_payload(spec, decision_context(), [])["max_completion_tokens"] == 1200

    monkeypatch.setenv("MARKET_AGENTS_MAX_COMPLETION_TOKENS", "9000")
    assert _build_payload(spec, decision_context(), [])["max_completion_tokens"] == 4000

    monkeypatch.setenv("MARKET_AGENTS_MAX_COMPLETION_TOKENS", "invalid")
    assert _build_payload(spec, decision_context(), [])["max_completion_tokens"] == 1800


def test_payload_sends_completion_token_budget(monkeypatch):
    monkeypatch.setenv("MARKET_AGENTS_MAX_COMPLETION_TOKENS", "1600")
    spec = ALLOWED_FREE_MODELS["nvidia/nemotron-3-super-120b-a12b:free"]

    payload = _build_payload(spec, decision_context(), [])

    assert payload["max_completion_tokens"] == 1600


def test_structured_prompt_is_at_least_25_percent_smaller_than_schema_prompt():
    context = decision_context()
    structured_prompt = build_decision_prompts(context, [], include_output_schema=False).user_prompt
    schema_prompt = build_decision_prompts(context, [], include_output_schema=True).user_prompt

    assert len(structured_prompt) <= len(schema_prompt) * 0.75
    assert json.loads(structured_prompt[structured_prompt.index("{"):])["visible_context"]
