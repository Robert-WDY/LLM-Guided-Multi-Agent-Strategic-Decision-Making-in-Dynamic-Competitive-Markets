import json
from pathlib import Path


def test_archived_stage611_smoke_is_bounded_and_transparent() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "experiment-results" / "stage6.11-v7-real-llm-smoke"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    incident = json.loads((directory / "incident.json").read_text(encoding="utf-8"))
    rows = json.loads((directory / "rows.json").read_text(encoding="utf-8"))

    assert summary["planned_calls"] == 6
    assert summary["lost_provider_calls"] == 1
    assert summary["recorded_calls"] == 5
    assert summary["successful_calls"] == 5
    assert summary["paired_complete"] == 2
    assert summary["positive_zero_negative_pairs"] == {
        "positive": 2,
        "zero": 0,
        "negative": 0,
    }
    assert summary["usage"]["calls"] + incident["lost_provider_call_count"] == 6
    assert len(rows) == 5
    assert all(row["provider_audit"] for row in rows)
