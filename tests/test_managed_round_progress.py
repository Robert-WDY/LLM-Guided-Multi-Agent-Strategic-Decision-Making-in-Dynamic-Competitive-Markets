"""管理回合实时进度注册表的线程安全与安全字段测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from game_theory_agent.managed_round_progress import ManagedRoundProgressRegistry


class FakeClock:
    def __init__(self, value: int = 1_000) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value

    def advance(self, milliseconds: int) -> None:
        self.value += milliseconds


def test_progress_registry_tracks_provider_waiting_and_usage() -> None:
    clock = FakeClock()
    registry = ManagedRoundProgressRegistry(clock_ms=clock)
    registry.start("ep-1", 1, 0, [("company_B", "model-b")])
    registry.record_event("ep-1", "company_B", "load_snapshot", {})
    clock.advance(250)
    registry.record_event(
        "ep-1",
        "company_B",
        "provider_request",
        {"attempt": 1, "repair": False, "secret": "must-not-leak"},
    )

    running = registry.snapshot("ep-1")
    company = running["companies"]["company_B"]
    assert running["status"] == "running"
    assert company["current_stage"] == "provider_request"
    assert company["provider_waiting"] is True
    assert company["provider_attempts"] == 1
    assert company["elapsed_ms"] == 250
    assert company["events"][-1]["details"] == {"attempt": 1, "repair": False}

    clock.advance(500)
    registry.record_event(
        "ep-1",
        "company_B",
        "provider_response",
        {
            "attempt": 1,
            "finish_reason": "stop",
            "usage_available": True,
            "total_tokens": 321,
            "latency_ms": 480,
            "raw_response": "must-not-leak",
        },
    )
    company = registry.snapshot("ep-1")["companies"]["company_B"]
    assert company["provider_waiting"] is False
    assert company["total_tokens"] == 321
    assert company["provider_latency_ms"] == 480
    assert company["finish_reason"] == "stop"
    assert "raw_response" not in company["events"][-1]["details"]


def test_progress_registry_keeps_company_updates_isolated() -> None:
    registry = ManagedRoundProgressRegistry(clock_ms=FakeClock())
    registry.start(
        "ep-parallel",
        2,
        1,
        [("company_A", "model-a"), ("company_B", "model-b")],
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                registry.record_event,
                "ep-parallel",
                company_id,
                "provider_request",
                {"attempt": 1},
            )
            for company_id in ("company_A", "company_B")
        ]
        for future in futures:
            future.result()

    snapshot = registry.snapshot("ep-parallel")
    assert set(snapshot["companies"]) == {"company_A", "company_B"}
    assert all(
        item["events"][0]["stage"] == "provider_request"
        for item in snapshot["companies"].values()
    )


def test_progress_registry_exposes_idle_and_terminal_states_as_copies() -> None:
    clock = FakeClock()
    registry = ManagedRoundProgressRegistry(clock_ms=clock)
    assert registry.snapshot("never-started") == {
        "episode_id": "never-started",
        "round": None,
        "state_version": None,
        "status": "idle",
        "started_at_ms": None,
        "updated_at_ms": None,
        "elapsed_ms": 0,
        "companies": {},
        "error_category": None,
    }

    registry.start("ep-terminal", 1, 0, [("company_A", "model-a")])
    registry.mark_company_finished(
        "ep-terminal", "company_A", fallback_used=True, error_category="invalid_output"
    )
    registry.mark_settling("ep-terminal")
    clock.advance(100)
    registry.complete("ep-terminal")

    completed = registry.snapshot("ep-terminal")
    assert completed["status"] == "completed"
    assert completed["companies"]["company_A"]["status"] == "fallback"
    assert completed["companies"]["company_A"]["error_category"] == "invalid_output"
    completed["companies"]["company_A"]["events"].append({"stage": "forged"})
    assert registry.snapshot("ep-terminal")["companies"]["company_A"]["events"] == []

    registry.start("ep-failed", 3, 2, [("company_C", "model-c")])
    registry.fail("ep-failed", "controller_failed")
    failed = registry.snapshot("ep-failed")
    assert failed["status"] == "failed"
    assert failed["error_category"] == "controller_failed"
