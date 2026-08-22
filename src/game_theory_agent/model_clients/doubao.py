"""Volcengine Ark / Doubao implementation of the provider-neutral ModelClient."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from pydantic import ValidationError

from game_theory_agent.agents.contracts import (
    AgentDecision,
    CommunicationContext,
    DecisionContext,
    ModelGeneration,
)
from game_theory_agent.agents.prompt_builder import (
    AgentPromptBuilder,
    CommunicationPromptBuilder,
)
from game_theory_agent.interaction.contracts import CommunicationSubmission
from game_theory_agent.model_clients.json_output import extract_json_object


DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_ARK_MODEL = "doubao-seed-2-0-lite-260215"


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    for item in getattr(response, "output", ()):
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", ()):
            text = getattr(content, "text", None)
            if isinstance(text, str) and text.strip():
                return text
    raise ValueError("Doubao response did not contain output text")


class DoubaoModelClient:
    """One request plus one optional JSON repair; no tools or market authority."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 40.0,
        max_schema_attempts: int = 2,
        temperature: float | None = None,
        top_p: float | None = None,
        prompt_builder: AgentPromptBuilder | None = None,
        communication_prompt_builder: CommunicationPromptBuilder | None = None,
        client: Any | None = None,
    ) -> None:
        if max_schema_attempts not in {1, 2}:
            raise ValueError("max_schema_attempts must be 1 or 2")
        self.model = model or os.getenv("ARK_MODEL", DEFAULT_ARK_MODEL)
        self.base_url = base_url or os.getenv("ARK_BASE_URL", DEFAULT_ARK_BASE_URL)
        self.max_schema_attempts = max_schema_attempts
        self.temperature = temperature
        self.top_p = top_p
        self.prompt_builder = prompt_builder or AgentPromptBuilder()
        self.communication_prompt_builder = (
            communication_prompt_builder or CommunicationPromptBuilder()
        )
        if client is not None:
            self._client = client
            return
        resolved_key = api_key or os.getenv("ARK_API_KEY")
        if not resolved_key:
            raise ValueError("ARK_API_KEY is required for DoubaoModelClient")
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=self.base_url,
            timeout=timeout_seconds,
            max_retries=1,
        )

    async def generate_communication(
        self, context: CommunicationContext
    ) -> ModelGeneration:
        started = time.perf_counter()
        prompt = self.communication_prompt_builder.build(context)
        total_input_tokens = 0
        total_output_tokens = 0
        last_raw = ""
        last_error = ""

        for attempt in range(self.max_schema_attempts):
            sampling = {}
            if self.temperature is not None:
                sampling["temperature"] = self.temperature
            if self.top_p is not None:
                sampling["top_p"] = self.top_p
            response = await self._client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=2000,
                extra_body={"thinking": {"type": "disabled"}},
                **sampling,
            )
            last_raw = _response_text(response)
            usage = getattr(response, "usage", None)
            total_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            total_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            try:
                parsed = extract_json_object(last_raw)
                submission = CommunicationSubmission.model_validate(parsed)
                return ModelGeneration(
                    model_name=self.model,
                    prompt_version=self.communication_prompt_builder.prompt_version,
                    parsed_output=submission.model_dump(mode="json"),
                    raw_response=last_raw,
                    latency_ms=round((time.perf_counter() - started) * 1000),
                    input_tokens=total_input_tokens or None,
                    output_tokens=total_output_tokens or None,
                    retry_count=attempt,
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                if attempt + 1 < self.max_schema_attempts:
                    prompt = self.communication_prompt_builder.build_repair(
                        context, last_raw, last_error
                    )

        raise ValueError(
            "Doubao did not return a valid CommunicationSubmission after "
            f"{self.max_schema_attempts} attempt(s): {last_error}"
        )

    async def generate_decision(self, context: DecisionContext) -> ModelGeneration:
        started = time.perf_counter()
        prompt = self.prompt_builder.build(context)
        total_input_tokens = 0
        total_output_tokens = 0
        last_raw = ""
        last_error = ""

        for attempt in range(self.max_schema_attempts):
            sampling = {}
            if self.temperature is not None:
                sampling["temperature"] = self.temperature
            if self.top_p is not None:
                sampling["top_p"] = self.top_p
            response = await self._client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=4000,
                extra_body={"thinking": {"type": "disabled"}},
                **sampling,
            )
            last_raw = _response_text(response)
            usage = getattr(response, "usage", None)
            total_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            total_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            try:
                parsed = extract_json_object(last_raw)
                decision = AgentDecision.model_validate(parsed)
                return ModelGeneration(
                    model_name=self.model,
                    prompt_version=self.prompt_builder.prompt_version,
                    parsed_output=decision.model_dump(mode="json"),
                    raw_response=last_raw,
                    latency_ms=round((time.perf_counter() - started) * 1000),
                    input_tokens=total_input_tokens or None,
                    output_tokens=total_output_tokens or None,
                    retry_count=attempt,
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                if attempt + 1 < self.max_schema_attempts:
                    prompt = self.prompt_builder.build_repair(
                        context, last_raw, last_error
                    )

        raise ValueError(
            "Doubao did not return a valid AgentDecision after "
            f"{self.max_schema_attempts} attempt(s): {last_error}"
        )
