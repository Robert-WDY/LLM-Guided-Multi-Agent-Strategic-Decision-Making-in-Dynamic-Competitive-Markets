from __future__ import annotations

import pytest

from game_theory_agent.experiments.stage68_advisor_external_proof import (
    DEFAULT_OUTPUT as STAGE68_OUTPUT,
)
from game_theory_agent.experiments.stage69_abstention_repair import run
from game_theory_agent.market.protocols import sha256_hash


def test_stage69_five_known_negative_windows_fail_closed(tmp_path):
    if not (STAGE68_OUTPUT / "summary.json").is_file():
        pytest.skip("stage6.8 recorded artifacts are not present")
    summary = run(output=tmp_path)

    assert summary["negative_window_count"] == 5
    assert summary["v7_defer_count"] == 5
    assert summary["v7_executable_recommendation_count"] == 0
    assert summary["v7_abstention_negative_delta_count"] == 0
    assert summary["prevented_fallback_loss_cents"] == 2_230_133
    assert summary["engineering_passed"]
    assert summary["new_real_model_calls"] == 0
    assert summary["new_total_tokens"] == 0
    assert summary["report_hash"] == sha256_hash(
        {key: value for key, value in summary.items() if key != "report_hash"}
    )
