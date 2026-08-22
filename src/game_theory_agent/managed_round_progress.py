"""管理回合的进程内实时进度；只保存允许进入 WebUI 的安全事件字段。"""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any


SAFE_EVENT_FIELDS = frozenset(
    {
        "attempt",
        "repair",
        "repair_attempts",
        "status",
        "finish_reason",
        "usage_available",
        "total_tokens",
        "latency_ms",
        "error_category",
    }
)


def sanitize_progress_details(details: dict[str, Any]) -> dict[str, Any]:
    """仅保留节点层明确允许公开的标量审计字段。"""

    return {key: details[key] for key in SAFE_EVENT_FIELDS if key in details}


class ManagedRoundProgressRegistry:
    """按 Episode 隔离管理回合进度，并为并发模型线程提供原子更新。"""

    def __init__(self, *, clock_ms: Callable[[], int] | None = None) -> None:
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._lock = threading.RLock()
        self._latest: dict[str, dict[str, Any]] = {}

    def start(
        self,
        episode_id: str,
        round_number: int,
        state_version: int,
        companies: Iterable[tuple[str, str]],
    ) -> None:
        now = self._clock_ms()
        with self._lock:
            self._latest[episode_id] = {
                "episode_id": episode_id,
                "round": round_number,
                "state_version": state_version,
                "status": "running",
                "started_at_ms": now,
                "updated_at_ms": now,
                "elapsed_ms": 0,
                "companies": {
                    company_id: {
                        "company_id": company_id,
                        "model_id": model_id,
                        "status": "queued",
                        "current_stage": None,
                        "events": [],
                        "started_at_ms": None,
                        "updated_at_ms": now,
                        "elapsed_ms": 0,
                        "provider_attempts": 0,
                        "provider_waiting": False,
                        "total_tokens": None,
                        "provider_latency_ms": None,
                        "finish_reason": None,
                        "fallback_used": None,
                        "error_category": None,
                    }
                    for company_id, model_id in companies
                },
                "error_category": None,
            }

    def record_event(
        self,
        episode_id: str,
        company_id: str,
        stage: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        now = self._clock_ms()
        safe_details = sanitize_progress_details(details or {})
        with self._lock:
            progress = self._latest[episode_id]
            company = progress["companies"][company_id]
            if company["started_at_ms"] is None:
                company["started_at_ms"] = now
            company["status"] = "running"
            company["current_stage"] = stage
            company["updated_at_ms"] = now
            company["events"].append(
                {"stage": stage, "details": safe_details, "occurred_at_ms": now}
            )
            if stage == "provider_request":
                company["provider_attempts"] += 1
                company["provider_waiting"] = True
            elif stage in {"provider_response", "provider_error"}:
                company["provider_waiting"] = False
                if "total_tokens" in safe_details:
                    company["total_tokens"] = safe_details["total_tokens"]
                if "latency_ms" in safe_details:
                    company["provider_latency_ms"] = safe_details["latency_ms"]
                if "finish_reason" in safe_details:
                    company["finish_reason"] = safe_details["finish_reason"]
                if "error_category" in safe_details:
                    company["error_category"] = safe_details["error_category"]
            progress["updated_at_ms"] = now

    def mark_company_finished(
        self,
        episode_id: str,
        company_id: str,
        *,
        fallback_used: bool,
        error_category: str | None = None,
    ) -> None:
        now = self._clock_ms()
        with self._lock:
            progress = self._latest[episode_id]
            company = progress["companies"][company_id]
            company["status"] = "fallback" if fallback_used else "completed"
            company["fallback_used"] = fallback_used
            company["provider_waiting"] = False
            company["error_category"] = error_category or company["error_category"]
            company["updated_at_ms"] = now
            progress["updated_at_ms"] = now

    def mark_settling(self, episode_id: str) -> None:
        self._mark_global(episode_id, "settling")

    def complete(self, episode_id: str) -> None:
        self._mark_global(episode_id, "completed")

    def fail(self, episode_id: str, error_category: str) -> None:
        self._mark_global(episode_id, "failed", error_category=error_category)

    def snapshot(self, episode_id: str) -> dict[str, Any]:
        now = self._clock_ms()
        with self._lock:
            if episode_id not in self._latest:
                return {
                    "episode_id": episode_id,
                    "round": None,
                    "state_version": None,
                    "status": "idle",
                    "started_at_ms": None,
                    "updated_at_ms": None,
                    "elapsed_ms": 0,
                    "companies": {},
                    "error_category": None,
                }
            result = copy.deepcopy(self._latest[episode_id])
        running = result["status"] in {"running", "settling"}
        if running:
            result["elapsed_ms"] = now - result["started_at_ms"]
        for company in result["companies"].values():
            started = company["started_at_ms"]
            if started is not None:
                end = now if company["status"] in {"queued", "running"} else company["updated_at_ms"]
                company["elapsed_ms"] = end - started
        return result

    def _mark_global(
        self,
        episode_id: str,
        status: str,
        *,
        error_category: str | None = None,
    ) -> None:
        now = self._clock_ms()
        with self._lock:
            progress = self._latest[episode_id]
            progress["status"] = status
            progress["updated_at_ms"] = now
            progress["elapsed_ms"] = now - progress["started_at_ms"]
            progress["error_category"] = error_category
