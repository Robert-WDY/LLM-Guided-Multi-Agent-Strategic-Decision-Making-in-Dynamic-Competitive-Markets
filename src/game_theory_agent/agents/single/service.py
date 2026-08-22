"""Production dependency assembly for the local single-agent lab."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .gateway import AgentGatewayClient
from .provider import OpenRouterProvider, load_openrouter_api_key
from .runtime import SingleAgentRuntime
from .trace import JsonlTraceStore, open_sqlite_checkpointer


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(slots=True)
class AgentLabRuntimeBundle:
    runtime: SingleAgentRuntime
    provider: OpenRouterProvider
    gateway: AgentGatewayClient
    trace_store: JsonlTraceStore
    checkpointer: object


def build_agent_lab_runtime(
    *,
    episode_id: str,
    secret_path: str | Path,
    output_root: str | Path,
    gateway_base_url: str = "http://127.0.0.1:8011",
    provider_timeout_seconds: float | None = None,
) -> AgentLabRuntimeBundle:
    safe_episode_id = _SAFE_NAME.sub("_", episode_id).strip("._") or "episode"
    run_root = Path(output_root) / safe_episode_id
    trace_store = JsonlTraceStore(run_root / "traces")
    checkpointer = open_sqlite_checkpointer(run_root / "checkpoints.sqlite")
    timeout_seconds = provider_timeout_seconds or _provider_timeout_seconds()
    provider = OpenRouterProvider(
        api_key=load_openrouter_api_key(secret_path),
        timeout_seconds=timeout_seconds,
    )
    gateway = AgentGatewayClient(base_url=gateway_base_url)
    runtime = SingleAgentRuntime(
        provider=provider,
        gateway=gateway,
        trace_store=trace_store,
        checkpointer=checkpointer,
        history_limit=2,
    )
    return AgentLabRuntimeBundle(
        runtime=runtime,
        provider=provider,
        gateway=gateway,
        trace_store=trace_store,
        checkpointer=checkpointer,
    )


def _provider_timeout_seconds() -> float:
    raw_value = os.environ.get("MARKET_AGENTS_OPENROUTER_TIMEOUT_SECONDS", "12")
    try:
        return max(1.0, float(raw_value))
    except ValueError:
        return 12.0
