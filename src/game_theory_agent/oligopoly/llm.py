"""OpenAI-compatible chat transport with a content-addressed response cache.

Every request is keyed by (provider, model, thinking, messages, sample_key).
A cached key is never re-sent, so any archived run can be replayed offline and
an interrupted study can be resumed without paying twice. Token usage is kept
per call so cost can be reconciled against provider billing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

PROVIDERS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "doubao": ("https://ark.cn-beijing.volces.com/api/v3", "ARK_API_KEY"),
}


@dataclass
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    cached: bool
    latency_s: float


class ChatModel(Protocol):
    name: str

    async def chat(self, messages: list[dict[str, str]], sample_key: str) -> ChatResult: ...


class BudgetExceeded(RuntimeError):
    pass


class TokenBudget:
    """Process-wide hard cap on tokens actually sent to paid providers."""

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.used = 0

    def charge(self, tokens: int) -> None:
        self.used += tokens

    def check(self) -> None:
        if self.used >= self.max_tokens:
            raise BudgetExceeded(f"token budget {self.max_tokens} exhausted")


class ResponseCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict[str, Any]] = {}
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        self._mem[row["key"]] = row

    def get(self, key: str) -> dict[str, Any] | None:
        return self._mem.get(key)

    def put(self, row: dict[str, Any]) -> None:
        self._mem[row["key"]] = row
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def request_key(name: str, messages: list[dict[str, str]], sample_key: str) -> str:
    blob = json.dumps([name, messages, sample_key], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class OpenAICompatibleModel:
    """One provider/model pair; thinking disabled unless requested."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        cache: ResponseCache,
        budget: TokenBudget | None = None,
        thinking: bool = False,
        temperature: float | None = None,
        max_tokens: int = 600,
        concurrency: int = 16,
        offline: bool = False,
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider}")
        self.provider = provider
        self.model = model
        self.thinking = thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache
        self.budget = budget
        self.offline = offline
        self.name = f"{provider}/{model}" + ("+thinking" if thinking else "")
        self._sem = asyncio.Semaphore(concurrency)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            base_url, env = PROVIDERS[self.provider]
            key = os.environ.get(env)
            if not key:
                raise RuntimeError(f"{env} is not set")
            self._client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=90, max_retries=0)
        return self._client

    async def chat(self, messages: list[dict[str, str]], sample_key: str) -> ChatResult:
        key = request_key(self.name, messages, sample_key)
        hit = self.cache.get(key)
        if hit is not None:
            return ChatResult(hit["text"], hit["input_tokens"], hit["output_tokens"], True, 0.0)
        if self.offline:
            raise RuntimeError(f"offline replay missing cached response {key[:12]}")
        if self.budget is not None:
            self.budget.check()
        extra: dict[str, Any] = {"thinking": {"type": "enabled" if self.thinking else "disabled"}}
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens if not self.thinking else max(self.max_tokens, 6000),
            "extra_body": extra,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        async with self._sem:
            delay = 2.0
            for attempt in range(6):
                started = time.perf_counter()
                try:
                    resp = await self._get_client().chat.completions.create(**kwargs)
                    break
                except Exception as exc:  # transport, rate limit, 5xx
                    if attempt == 5:
                        raise
                    status = getattr(exc, "status_code", None)
                    if status is not None and status < 500 and status != 429:
                        raise
                    await asyncio.sleep(delay + random.random())
                    delay = min(delay * 2, 60)
        latency = time.perf_counter() - started
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        row = {
            "key": key,
            "model": self.name,
            "response_model": getattr(resp, "model", None),
            "request_id": getattr(resp, "id", None),
            "text": text,
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "latency_s": round(latency, 3),
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.cache.put(row)
        if self.budget is not None:
            self.budget.charge(row["input_tokens"] + row["output_tokens"])
        return ChatResult(text, row["input_tokens"], row["output_tokens"], False, latency)


class ScriptedModel:
    """Deterministic stand-in used by tests: returns callable(messages)."""

    def __init__(self, fn, name: str = "scripted") -> None:
        self.fn = fn
        self.name = name
        self.calls = 0

    async def chat(self, messages: list[dict[str, str]], sample_key: str) -> ChatResult:
        self.calls += 1
        return ChatResult(self.fn(messages), 10, 5, False, 0.0)
