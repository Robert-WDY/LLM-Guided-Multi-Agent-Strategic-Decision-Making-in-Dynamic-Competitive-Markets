from __future__ import annotations

from game_theory_agent.experiments.stage63_objective_calibration import (
    CONDITIONS,
    run,
)


def test_stage63_small_matrix_is_zero_llm_matched_and_v2_frozen():
    summary = run(
        seeds=(1,),
        personas=("aggressive_v1_extreme",),
        pool_ids=("known", "holdout"),
        rounds=5,
        horizon_rounds=1,
        scenario_count=1,
    )

    assert len(summary["rows"]) == 2 * len(CONDITIONS)
    assert summary["engineering_passed"]
    assert summary["real_model_usage"] == {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "estimated_cost": 0,
        "currency": "CNY",
    }
    assert summary["engineering_checks"]["v2_frozen_and_not_executed"]
    assert all(row["all_actions_legal"] for row in summary["rows"])
    assert all(row["best_opponent_enterprise_value_cents"] > 0 for row in summary["rows"])

    rows = summary["rows"]
    for pool_id in ("known", "holdout"):
        belief = next(
            row
            for row in rows
            if row["opponent_pool"] == pool_id
            and row["condition"] == "belief_persona"
        )
        gated = next(
            row
            for row in rows
            if row["opponent_pool"] == pool_id
            and row["condition"] == "belief_v1_profit_gate"
        )
        # The v1 gate is closed for non-profit personas, so the complete path
        # must remain byte-for-byte equivalent to the Belief control.
        assert belief["final_state_hash"] == gated["final_state_hash"]
        assert belief["recommendations"] == gated["recommendations"]

        objective_rows = [
            row
            for row in rows
            if row["opponent_pool"] == pool_id
            and row["condition"]
            in {
                "absolute_value",
                "persona_aligned",
                "competitive_250k",
                "competitive_500k",
                "competitive_750k",
                "competitive_only",
            }
        ]
        assert len(
            {row["round_trace"][0]["forecast_seed"] for row in objective_rows}
        ) == 1

    calibration = summary["calibration_by_persona"][
        "aggressive_v1_extreme"
    ]
    assert calibration["development_pair_count"] == 1
    assert calibration["holdout_pair_count"] == 1
    assert summary["belief_path_audit"]["pair_count"] == 2
