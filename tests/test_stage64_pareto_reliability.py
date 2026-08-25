from __future__ import annotations

from game_theory_agent.experiments.stage64_pareto_reliability import (
    CONDITIONS,
    PARETO_SPECS,
    run,
)


def test_stage64_small_matrix_is_matched_audited_and_zero_llm():
    summary = run(
        seeds=(1, 2),
        personas=("aggressive_v1_extreme",),
        pool_ids=("known", "holdout"),
        rounds=5,
        horizon_rounds=1,
        scenario_count=1,
    )

    assert len(summary["rows"]) == 2 * 2 * len(CONDITIONS)
    assert summary["real_model_usage"] == {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "estimated_cost": 0,
        "currency": "CNY",
    }
    assert not summary["engineering_passed"]
    assert not summary["engineering_checks"]["at_least_ten_common_seeds"]
    assert all(
        passed
        for check, passed in summary["engineering_checks"].items()
        if check != "at_least_ten_common_seeds"
    )
    assert all(row["all_actions_legal"] for row in summary["rows"])

    pareto_rows = [
        row
        for row in summary["rows"]
        if row["condition"] in PARETO_SPECS
    ]
    assert pareto_rows
    assert all(
        trace["decision_hash"]
        and trace["pareto_situation"]
        in {"protect_lead", "catch_up_early", "catch_up_late"}
        and trace["eligible_candidate_count"] >= 1
        and trace["frontier_candidate_count"] >= 1
        for row in pareto_rows
        for trace in row["round_trace"]
    )

    # Every treatment sees the same public state and uses the same scenario
    # seed in the first round of a matched seed/persona/opponent cell.
    matched_rows = [
        row
        for row in summary["rows"]
        if row["seed"] == 1
        and row["opponent_pool"] == "known"
        and row["condition"] != "rule_baseline"
    ]
    assert len({row["round_trace"][0]["pre_state_hash"] for row in matched_rows}) == 1
    assert len({row["round_trace"][0]["forecast_seed"] for row in matched_rows}) == 1
