from game_theory_agent.experiments.stage66_failure_forensics import (
    DEFAULT_STAGE65_SUMMARY,
)
from game_theory_agent.experiments.stage67_marginal_candidate_repair import (
    DEFAULT_CORE_ROWS,
    run,
)


def test_v6_marginal_repair_passes_frozen_real_llm_controls_without_tokens(
    tmp_path,
):
    result = run(DEFAULT_STAGE65_SUMMARY, DEFAULT_CORE_ROWS, tmp_path)

    assert result["acceptance_passed"]
    assert result["aggregate"]["pair_count"] == 5
    assert result["aggregate"]["nonnegative_pair_count"] == 5
    assert result["aggregate"]["worst_delta_cents"] >= 0
    assert result["checks"]["deterministic_advice_5_of_5"]
    assert result["checks"]["deterministic_market_replay_5_of_5"]
    assert result["checks"]["public_only_5_of_5"]
    assert result["checks"]["new_real_model_calls"] == 0
    assert result["checks"]["new_total_tokens"] == 0
    diagnosis = result["market_calibration_diagnosis"]
    assert diagnosis["marginal_probe_count"] == 20
    assert diagnosis["deterministic_transition_and_replay_passed"]
    assert diagnosis["real_world_parameter_calibration_required"]
    assert (tmp_path / "summary.json").exists()
