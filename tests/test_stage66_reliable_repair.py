from game_theory_agent.experiments.stage66_failure_forensics import (
    DEFAULT_STAGE65_SUMMARY,
)
from game_theory_agent.experiments.stage66_reliable_repair import run


def test_reliable_repair_covers_all_known_negative_pairs_without_new_llm_calls(
    tmp_path,
):
    result = run(DEFAULT_STAGE65_SUMMARY, tmp_path)

    assert result["acceptance_passed"]
    assert result["aggregate"]["pair_count"] == 5
    assert result["aggregate"]["nonnegative_pair_count"] == 5
    assert result["aggregate"]["worst_delta_cents"] >= 0
    assert result["checks"]["abstention_triggered_5_of_5"]
    assert result["checks"]["deterministic_advice_rebuild_5_of_5"]
    assert result["checks"]["public_only_gate_5_of_5"]
    assert result["checks"]["new_real_model_calls"] == 0
    assert result["checks"]["new_total_tokens"] == 0
    assert (tmp_path / "summary.json").exists()
