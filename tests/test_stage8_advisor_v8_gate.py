from game_theory_agent.experiments.stage8_advisor_v8_gate import (
    _policy_applies,
    evaluate_policy,
)


def _row(delta: int, *, horizon: int = 3, confidence: int = 700_000) -> dict:
    return {
        "seed": 1,
        "persona_id": "balanced_v1",
        "horizon_rounds": horizon,
        "action_changed": True,
        "opponent_confidence_ppm": confidence,
        "top_gap_ratio_ppm": 600_000,
        "predicted_gain_over_maintain_cents": 200_000,
        "passes_fallback_value_floor": True,
        "passes_fallback_worst_floor": True,
        "passes_fallback_persona_floor": True,
        "planner_minus_baseline_ev_cents": delta,
        "uses_hidden_state": False,
    }


POLICY = {
    "minimum_horizon_rounds": 3,
    "minimum_opponent_confidence_ppm": 650_000,
    "minimum_top_gap_ratio_ppm": 500_000,
    "minimum_predicted_gain_cents": 100_000,
}


def test_v8_gate_requires_mature_evidence_and_action_change() -> None:
    assert _policy_applies(_row(1), POLICY)
    assert not _policy_applies(_row(1, horizon=2), POLICY)
    assert not _policy_applies(_row(1, confidence=600_000), POLICY)
    unchanged = _row(0)
    unchanged["action_changed"] = False
    assert not _policy_applies(unchanged, POLICY)


def test_v8_metrics_count_only_released_actions() -> None:
    rows = [_row(100), _row(-50, confidence=600_000)]
    metrics = evaluate_policy(rows, POLICY)

    assert metrics["released_count"] == 1
    assert metrics["positive_zero_negative_released"] == {
        "positive": 1,
        "zero": 0,
        "negative": 0,
    }
    assert metrics["mean_all_window_ev_delta_cents"] == 50
