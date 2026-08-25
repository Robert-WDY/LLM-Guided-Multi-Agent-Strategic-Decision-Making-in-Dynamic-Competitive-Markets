from __future__ import annotations

import asyncio
from types import SimpleNamespace

from game_theory_agent.experiments.stage65_real_adoption import (
    CONDITIONS,
    run,
)


def test_stage65_mock_matrix_is_matched_traced_and_budgeted(tmp_path):
    summary = asyncio.run(
        run(
            SimpleNamespace(
                seeds="1",
                personas="aggressive_v1_extreme",
                rounds=10,
                paid_round=7,
                provider="mock",
                model=None,
                temperature=0.0,
                top_p=0.1,
                timeout=30.0,
                output=tmp_path,
                authorize_real_model=True,
                resume=False,
                quiet=True,
            )
        )
    )

    assert len(summary["rows"]) == len(CONDITIONS)
    assert summary["real_model_usage"]["calls"] == len(CONDITIONS)
    assert summary["real_model_usage"]["estimated_cost_cny"] == 0
    assert summary["engineering_checks"]["matched_pre_decision_states"]
    assert summary["engineering_checks"]["advisor_traces_complete"]
    assert not summary["engineering_checks"]["provider_usage_complete"]
    assert all(
        row["pre_decision_economic_state_hash"]
        == summary["rows"][0]["pre_decision_economic_state_hash"]
        for row in summary["rows"]
    )
    assert not summary["stage65_complete"]

    resumed = asyncio.run(
        run(
            SimpleNamespace(
                seeds="1",
                personas="aggressive_v1_extreme",
                rounds=10,
                paid_round=7,
                provider="mock",
                model=None,
                temperature=0.0,
                top_p=0.1,
                timeout=30.0,
                output=tmp_path,
                authorize_real_model=True,
                resume=True,
                quiet=True,
            )
        )
    )
    assert resumed["real_model_usage"] == summary["real_model_usage"]
    assert resumed["rows"] == summary["rows"]
