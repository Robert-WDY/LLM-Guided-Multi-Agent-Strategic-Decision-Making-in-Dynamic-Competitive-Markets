from __future__ import annotations

import json
import hashlib
from pathlib import Path

from game_theory_agent.experiments.stage610_v7_zero_token_holdout import (
    PERSONA_IDS,
    SEEDS,
    build_preregistration,
    write_preregistration,
)
from game_theory_agent.market.protocols import sha256_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_stage610_preregistration_is_deterministic_and_cost_closed(tmp_path):
    first = build_preregistration()
    repeated = build_preregistration()

    assert first == repeated
    assert len(SEEDS) == 10
    assert len(set(SEEDS)) == 10
    assert len(PERSONA_IDS) == 3
    assert first["real_llm_gate"]["calls_during_this_holdout"] == 0
    assert first["real_llm_gate"]["maximum_calls_if_open"] == 6
    assert first["outcome_thresholds"][
        "released_recommendation_negative_window_count"
    ] == 0
    expected_hash = sha256_hash(
        {
            "hash_protocol_version": "stage6.10-preregistration-hash-v1.0.0",
            "preregistration": {
                key: value
                for key, value in first.items()
                if key != "preregistration_hash"
            },
        }
    )
    assert first["preregistration_hash"] == expected_hash

    path = tmp_path / "PREREGISTRATION.json"
    written = write_preregistration(path)
    assert json.loads(path.read_text(encoding="utf-8")) == written


def test_stage610_archived_result_is_intact_and_keeps_paid_gate_closed():
    result_dir = (
        PROJECT_ROOT
        / "experiment-results"
        / "stage6.10-v7-zero-token-holdout"
    )
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    windows_path = result_dir / "decision-windows.jsonl"
    windows = [
        json.loads(line)
        for line in windows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(windows) == 300
    assert summary["metrics"]["window_count"] == 300
    assert summary["metrics"]["recommendation_coverage_ppm"] == 33_333
    assert summary["metrics"][
        "released_recommendation_negative_window_count"
    ] == 0
    assert summary["engineering_passed"]
    assert not summary["outcome_passed"]
    assert not summary["real_llm_smoke_gate_open"]
    assert summary["new_real_model_calls"] == 0
    assert summary["new_prompt_tokens"] == 0
    assert summary["new_completion_tokens"] == 0
    assert summary["report_hash"] == sha256_hash(
        {key: value for key, value in summary.items() if key != "report_hash"}
    )
    assert hashlib.sha256(windows_path.read_bytes()).hexdigest().upper() == (
        "6DA67381B62972B5C5FE9375486BDE77BA4F893E88CE0DAED956D2D88CD2AEB6"
    )
