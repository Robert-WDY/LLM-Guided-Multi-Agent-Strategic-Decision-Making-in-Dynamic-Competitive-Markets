from game_theory_agent.experiments.midterm_market_validation import (
    validate_market_and_consumers,
    validate_native_cooperation_incentive,
)
from game_theory_agent.market import load_market_config


def test_market_and_consumer_validation_is_directionally_consistent() -> None:
    config = load_market_config("configs/market_v5_cooperation.yaml")
    result = validate_market_and_consumers(config)

    assert result["cell_count"] == 80
    assert result["all_paired_checks_passed"] is True
    assert result["dynamic_invariant_pass_count"] == 200
    assert result["same_direction_share_violation_count"] == 0


def test_native_cooperation_paid_gate_closes_without_incentive_separation() -> None:
    config = load_market_config("configs/market_v5_cooperation.yaml")
    result = validate_native_cooperation_incentive(config)

    assert result["economic_direction_separation_gate_passed"] is False
    assert result["real_llm_incentive_experiment_gate_open"] is False
    assert all(
        cell["positive_zero_negative"]["positive"] == 0
        for cell in result["cells"].values()
    )
