"""Small audited JSON client used only by preregistered real-model evaluations."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from openai import AsyncOpenAI

from game_theory_agent.model_clients.json_output import extract_json_object


@dataclass(frozen=True, slots=True)
class RealJSONGeneration:
    provider: str
    model: str
    prompt: str
    raw_response: str
    parsed: dict[str, Any] | None
    parse_error: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    request_started_at: str
    response_received_at: str
    request_id: str | None
    response_model: str | None


class RealJSONClient:
    """One provider call, no tool use and no hidden retry calls."""

    def __init__(self, provider: str, *, temperature: float = 0.0) -> None:
        self.provider = provider
        self.temperature = temperature
        if provider == "doubao":
            key = os.getenv("ARK_API_KEY")
            self.model = os.getenv("ARK_MODEL", "doubao-seed-2-0-lite-260215")
            base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        elif provider == "deepseek":
            key = os.getenv("DEEPSEEK_API_KEY")
            self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        else:
            raise ValueError(f"unsupported provider: {provider}")
        if not key:
            raise ValueError(f"missing API key for {provider}")
        self._client = AsyncOpenAI(
            api_key=key,
            base_url=base_url,
            timeout=60.0,
            max_retries=1,
        )

    async def generate(self, prompt: str) -> RealJSONGeneration:
        started_at = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        if self.provider == "doubao":
            response = await self._client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=300,
                temperature=self.temperature,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = str(getattr(response, "output_text", "") or "")
            if not raw:
                for item in getattr(response, "output", ()):
                    for content in getattr(item, "content", ()):
                        text = getattr(content, "text", None)
                        if isinstance(text, str):
                            raw += text
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        else:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是受约束的实验决策器，只输出合法 JSON，不调用工具。",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=300,
                temperature=self.temperature,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = str(response.choices[0].message.content or "")
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        parsed: dict[str, Any] | None = None
        parse_error: str | None = None
        try:
            value = extract_json_object(raw)
            if not isinstance(value, dict):
                raise ValueError("model output must be a JSON object")
            parsed = value
        except Exception as exc:  # audit the provider response instead of hiding it
            parse_error = f"{type(exc).__name__}: {exc}"
        return RealJSONGeneration(
            provider=self.provider,
            model=self.model,
            prompt=prompt,
            raw_response=raw,
            parsed=parsed,
            parse_error=parse_error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round((time.perf_counter() - started) * 1000),
            request_started_at=started_at,
            response_received_at=datetime.now(UTC).isoformat(),
            request_id=str(getattr(response, "id", "") or "") or None,
            response_model=str(getattr(response, "model", "") or "") or None,
        )
