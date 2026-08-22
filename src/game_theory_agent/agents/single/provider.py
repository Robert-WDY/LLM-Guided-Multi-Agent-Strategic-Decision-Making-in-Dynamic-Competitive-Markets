"""OpenRouter adapter with a project-owned model policy and safe errors."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import DecisionContext, DecisionProposal, PromptAudit, PromptTemplate


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
BLOCKED_NAMESPACE = re.compile(r"^(?:openai|anthropic|google)(?:/|~)", re.IGNORECASE)
CHINESE_TEXT = re.compile(r"[\u3400-\u9fff]")
CHINESE_OUTPUT_POLICY = (
    "\n\n不可覆盖的输出语言规则：只返回符合 Schema 的 JSON。JSON key、枚举值、"
    "candidate_id、selection_reason_codes 和 evidence_paths 保持英文；candidate label、"
    "strategy_summary、tradeoffs、risk_flags、expected_outcome 等面向用户的自然语言必须使用简体中文。"
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    label: str
    structured_output: bool
    description: str
    output_mode: str = "prompt_json"
    usage_supported: bool = True
    finish_reason_supported: bool = True
    max_output_tokens: int = 4000
    last_verified: str = "2026-08-22"
    gated_probe_successes: int = 0
    gated_probe_attempts: int = 0


ALLOWED_FREE_MODELS: dict[str, ModelSpec] = {
    "nvidia/nemotron-3-super-120b-a12b:free": ModelSpec(
        model_id="nvidia/nemotron-3-super-120b-a12b:free",
        label="Nemotron 3 Super",
        structured_output=True,
        description="结构化输出与工具能力兼顾的默认模型。",
        output_mode="json_object",
        gated_probe_successes=3,
        gated_probe_attempts=3,
    ),
    "nvidia/nemotron-3-ultra-550b-a55b:free": ModelSpec(
        model_id="nvidia/nemotron-3-ultra-550b-a55b:free",
        label="Nemotron 3 Ultra",
        structured_output=False,
        description="强推理备选，使用 JSON Schema 提示与本地校验。",
        output_mode="prompt_json",
    ),
    "z-ai/glm-5.2:free": ModelSpec(
        model_id="z-ai/glm-5.2:free",
        label="GLM 5.2",
        structured_output=False,
        description="Agentic 能力对照，使用 JSON 校验与一次修复。",
        output_mode="prompt_json",
    ),
    "nvidia/nemotron-nano-9b-v2:free": ModelSpec(
        model_id="nvidia/nemotron-nano-9b-v2:free",
        label="Nemotron Nano 9B V2",
        structured_output=True,
        description="适合重复 smoke test 的低延迟结构化基线。",
        output_mode="json_schema",
    ),
}


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ProviderError(RuntimeError):
    """Safe provider error whose text never contains credentials or raw output."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_error",
        latency_ms: int = 0,
        usage: TokenUsage | None = None,
        finish_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.latency_ms = max(0, latency_ms)
        self.usage = usage or TokenUsage()
        self.finish_reason = finish_reason


class ModelNotAllowedError(ProviderError):
    pass


class SecretConfigurationError(ProviderError):
    pass


class ProviderResponseError(ProviderError):
    pass


class ProviderInvalidDecisionError(ProviderResponseError):
    pass


class ProviderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal: DecisionProposal
    model_id: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = Field(ge=0)
    finish_reason: str | None = None
    usage_available: bool = False


def validate_model_id(model_id: str) -> ModelSpec:
    normalized = model_id.strip()
    if BLOCKED_NAMESPACE.match(normalized):
        raise ModelNotAllowedError("model provider namespace is blocked")
    try:
        return ALLOWED_FREE_MODELS[normalized]
    except KeyError as exc:
        raise ModelNotAllowedError("model is not in the project allowlist") from exc


def load_openrouter_api_key(path: str | Path) -> str:
    secret_path = Path(path)
    try:
        text = secret_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise SecretConfigurationError("OpenRouter API key file is unavailable") from exc

    assignments: dict[str, str] = {}
    meaningful_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        meaningful_lines.append(line)
        if "=" in line:
            name, value = line.split("=", 1)
            assignments[name.strip()] = value.strip().strip('"').strip("'")

    assigned = assignments.get("OPENROUTER_API_KEY", "")
    if assigned:
        return assigned
    if len(meaningful_lines) == 1 and "=" not in meaningful_lines[0]:
        return meaningful_lines[0]
    raise SecretConfigurationError("OpenRouter API key file has an unsupported format")


class OpenRouterProvider:
    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.Client | None = None,
        timeout_seconds: float = 45.0,
    ) -> None:
        if not api_key.strip():
            raise SecretConfigurationError("OpenRouter API key is empty")
        self._api_key = api_key.strip()
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self.last_prompt_audit: PromptAudit | None = None

    def generate_decision(
        self,
        *,
        model_id: str,
        context: DecisionContext,
        repair_errors: list[str] | None = None,
        prompt_template: PromptTemplate | None = None,
    ) -> ProviderResult:
        spec = validate_model_id(model_id)
        payload = _build_payload(spec, context, repair_errors or [], prompt_template)
        self.last_prompt_audit = PromptAudit(
            system_prompt=payload["messages"][0]["content"],
            user_prompt=payload["messages"][1]["content"],
        )
        started = time.monotonic()
        try:
            response = self._client.post(
                OPENROUTER_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "Fresh Market Lab",
                },
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            code = "schema_rejected" if exc.response.status_code == 400 and spec.output_mode in {"json_schema", "json_object"} else "provider_request_failed"
            raise ProviderResponseError(
                "OpenRouter request failed",
                code=code,
                latency_ms=max(0, round((time.monotonic() - started) * 1000)),
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderResponseError(
                "OpenRouter request failed",
                code="provider_request_failed",
                latency_ms=max(0, round((time.monotonic() - started) * 1000)),
            ) from exc

        latency_ms = max(0, round((time.monotonic() - started) * 1000))
        usage = TokenUsage.model_validate(body.get("usage") or {})
        usage_available = isinstance(body.get("usage"), dict)
        finish_reason = None
        try:
            choice = body["choices"][0]
            finish_reason = choice.get("finish_reason")
            content = choice["message"]["content"]
            if not isinstance(content, (str, list)) or not content:
                raise ProviderInvalidDecisionError(
                    "OpenRouter returned an invalid decision",
                    code="empty_response",
                    latency_ms=latency_ms,
                    usage=usage,
                    finish_reason=finish_reason,
                )
            try:
                parsed = _parse_json_content(content)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ProviderInvalidDecisionError(
                    "OpenRouter returned an invalid decision",
                    code="truncated" if finish_reason == "length" else "json_invalid",
                    latency_ms=latency_ms,
                    usage=usage,
                    finish_reason=finish_reason,
                ) from exc
            try:
                proposal = DecisionProposal.model_validate(parsed)
            except ValidationError as exc:
                missing = any(item.get("type") == "missing" for item in exc.errors())
                raise ProviderInvalidDecisionError(
                    "OpenRouter returned an invalid decision",
                    code="required_field_missing" if missing else "domain_validation_failed",
                    latency_ms=latency_ms,
                    usage=usage,
                    finish_reason=finish_reason,
                ) from exc
            _validate_chinese_readable_fields(proposal)
        except ProviderInvalidDecisionError as exc:
            if exc.latency_ms or exc.usage.total_tokens or exc.finish_reason:
                raise
            raise ProviderInvalidDecisionError(
                "OpenRouter returned an invalid decision",
                code=exc.code,
                latency_ms=latency_ms,
                usage=usage,
                finish_reason=finish_reason,
            ) from exc
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderInvalidDecisionError(
                "OpenRouter returned an invalid decision",
                code="empty_response",
                latency_ms=latency_ms,
                usage=usage,
                finish_reason=finish_reason,
            ) from exc

        return ProviderResult(
            proposal=proposal,
            model_id=str(body.get("model") or spec.model_id),
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            usage_available=usage_available,
        )


def _parse_json_content(content: Any) -> Any:
    """Accept plain, fenced, or text-part JSON without retaining raw model output."""

    if isinstance(content, list):
        content = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {None, "text"}
        )
    if not isinstance(content, str):
        raise TypeError("provider content is not text")
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        parsed, _end = json.JSONDecoder().raw_decode(text[start:])
        return parsed


def _validate_chinese_readable_fields(proposal: DecisionProposal) -> None:
    """拒绝新结果中的英文可读文案，让 runtime 进入既有的一次 repair 路径。"""

    readable: list[str] = []
    for candidate in proposal.candidates:
        readable.extend(
            [
                candidate.label,
                candidate.action.strategy_summary,
                *candidate.tradeoffs,
                *candidate.risk_flags,
                candidate.expected_outcome,
            ]
        )
    if any(text.strip() and not CHINESE_TEXT.search(text) for text in readable):
        raise ProviderInvalidDecisionError(
            "OpenRouter returned non-Chinese readable fields",
            code="domain_validation_failed",
        )


DECISION_WIRE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["candidates", "selected_candidate_id", "selection_reason_codes"],
    "properties": {
        "candidates": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {
                "type": "object",
                "required": ["candidate_id", "label", "action", "evidence_paths", "tradeoffs", "expected_outcome"],
                "properties": {
                    "candidate_id": {"type": "string"}, "label": {"type": "string"},
                    "action": {
                        "type": "object",
                        "required": ["price_cents", "advertising_budget_cents", "service_budget_cents", "strategy_summary"],
                        "properties": {
                            "price_cents": {"type": "integer"}, "advertising_budget_cents": {"type": "integer"},
                            "service_budget_cents": {"type": "integer"}, "capacity_investment_cents": {"type": "integer"},
                            "resilience_budget_cents": {"type": "integer"}, "shared_resilience_contribution_cents": {"type": "integer"},
                            "incident_response": {"type": "object"}, "strategy_summary": {"type": "string"},
                        },
                    },
                    "evidence_paths": {"type": "array", "items": {"type": "string"}},
                    "tradeoffs": {"type": "array", "items": {"type": "string"}},
                    "risk_flags": {"type": "array", "items": {"type": "string"}},
                    "expected_outcome": {"type": "string"},
                },
            },
        },
        "selected_candidate_id": {"type": "string"},
        "selection_reason_codes": {"type": "array", "items": {"type": "string"}},
    },
}

def _build_payload(
    spec: ModelSpec,
    context: DecisionContext,
    repair_errors: list[str],
    prompt_template: PromptTemplate | None = None,
) -> dict[str, Any]:
    schema = DECISION_WIRE_SCHEMA
    prompts = build_decision_prompts(
        context,
        repair_errors,
        prompt_template,
        include_output_schema=spec.output_mode != "json_schema",
    )
    payload: dict[str, Any] = {
        "model": spec.model_id,
        "messages": [
            {"role": "system", "content": prompts.system_prompt},
            {"role": "user", "content": prompts.user_prompt},
        ],
        "temperature": 0.2,
        "max_completion_tokens": resolve_completion_token_budget(),
        # 来源：https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
        # OpenRouter 公开模型元数据表明当前 allowlist 模型的 reasoning 均非 mandatory。
        # 决策 wire JSON 需要把输出预算留给可校验内容，避免默认 reasoning 吞尽预算。
        "reasoning": {"enabled": False, "exclude": True},
    }
    if spec.output_mode == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "single_agent_decision",
                "strict": True,
                "schema": schema,
            },
        }
    elif spec.output_mode == "json_object":
        payload["response_format"] = {"type": "json_object"}
    return payload


def build_decision_prompts(
    context: DecisionContext,
    repair_errors: list[str],
    prompt_template: PromptTemplate | None = None,
    *,
    include_output_schema: bool = True,
) -> PromptAudit:
    """Build the exact safe messages sent to the provider, excluding credentials."""

    template = prompt_template or PromptTemplate()
    user_payload = {
        "visible_context": context.model_dump(mode="json"),
        "repair_errors": repair_errors,
    }
    if include_output_schema:
        user_payload["output_schema"] = DECISION_WIRE_SCHEMA
    return PromptAudit(
        system_prompt=template.system_prompt + CHINESE_OUTPUT_POLICY,
        user_prompt=template.user_prompt_template.replace(
            "{{decision_input}}",
            json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
        ),
    )


def resolve_completion_token_budget() -> int:
    """读取有界输出预算，避免异常配置制造超长推理。"""

    raw_value = os.environ.get("MARKET_AGENTS_MAX_COMPLETION_TOKENS", "1800")
    try:
        value = int(raw_value)
    except ValueError:
        value = 1800
    return max(1200, min(value, 4000))
