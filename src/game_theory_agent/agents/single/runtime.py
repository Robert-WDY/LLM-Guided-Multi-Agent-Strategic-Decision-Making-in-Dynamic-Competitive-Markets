"""Public single-round entrypoint around the bounded LangGraph."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver

from .graph import GraphDependencies, build_single_agent_graph
from .models import PersonaTraceManifest, PromptTemplate, RoundDecisionResult


class SingleAgentRuntime:
    def __init__(
        self,
        *,
        provider: Any,
        gateway: Any,
        trace_store: Any,
        checkpointer: Any = None,
        history_limit: int = 2,
    ):
        self._progress = _ProgressBridge()
        self._graph = build_single_agent_graph(
            GraphDependencies(
                provider=provider,
                gateway=gateway,
                trace_store=trace_store,
                history_limit=history_limit,
                progress=self._progress,
            ),
            checkpointer=checkpointer or MemorySaver(),
        )

    def decide_round(
        self,
        *,
        episode_id: str,
        company_id: str,
        model_id: str,
        persona_manifest: PersonaTraceManifest | dict[str, Any] | None = None,
        prompt_template: PromptTemplate | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> RoundDecisionResult:
        validated_persona = (
            PersonaTraceManifest.model_validate(persona_manifest)
            if persona_manifest is not None
            else None
        )
        with self._progress.using(progress_callback, cancel_checker):
            state = self._graph.invoke(
                {
                    "episode_id": episode_id,
                    "company_id": company_id,
                    "model_id": model_id,
                    "persona_manifest": validated_persona,
                    "prompt_template": prompt_template or PromptTemplate(),
                    # 每轮显式清空临时节点状态，避免同一 LangGraph thread 的
                    # checkpoint 将上一轮意图回执或候选结果合并进本轮 trace。
                    "snapshot": None,
                    "decision_context": None,
                    "strategy_reflection": None,
                    "provider_result": None,
                    "proposal": None,
                    "validation_errors": [],
                    "repair_attempts": 0,
                    "prepared_intent": None,
                    "intent_receipt": None,
                    "status": "running",
                    "error_code": None,
                    "trace": None,
                    "prompt_audit": None,
                    "provider_attempt_usage": {},
                    "provider_attempt_latency_ms": 0,
                    "provider_finish_reason": None,
                    "provider_error_category": None,
                    "provider_usage_available": False,
                },
                config={
                    "configurable": {
                        "thread_id": f"{episode_id}:{company_id}",
                        "checkpoint_ns": model_id,
                    }
                },
            )
        trace = state["trace"]
        receipt = state.get("intent_receipt") or {}
        return RoundDecisionResult(
            status=state["status"],
            episode_id=episode_id,
            company_id=company_id,
            round=trace.round,
            state_version=trace.state_version,
            intent_id=receipt.get("intent_id"),
            trace=trace,
        )


class _ProgressBridge:
    """为并发 Runtime 调用隔离每个 Job 的节点回调。"""

    def __init__(self) -> None:
        self._local = threading.local()

    @contextmanager
    def using(
        self,
        callback: Callable[[str, dict[str, Any]], None] | None,
        cancel_checker: Callable[[], bool] | None,
    ):
        previous = getattr(self._local, "callback", None)
        previous_cancel = getattr(self._local, "cancel_checker", None)
        self._local.callback = callback
        self._local.cancel_checker = cancel_checker
        try:
            yield
        finally:
            self._local.callback = previous
            self._local.cancel_checker = previous_cancel

    def report(self, stage: str, details: dict[str, Any] | None = None) -> None:
        callback = getattr(self._local, "callback", None)
        if callback:
            callback(stage, dict(details or {}))

    def cancelled(self) -> bool:
        checker = getattr(self._local, "cancel_checker", None)
        return bool(checker and checker())
