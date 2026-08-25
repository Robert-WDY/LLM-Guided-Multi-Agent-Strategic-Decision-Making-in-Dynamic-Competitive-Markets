from __future__ import annotations

import json

from game_theory_agent.experiments.stage610_v7_zero_token_holdout import (
    PERSONA_IDS,
    SEEDS,
    build_preregistration,
    write_preregistration,
)
from game_theory_agent.market.protocols import sha256_hash


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
