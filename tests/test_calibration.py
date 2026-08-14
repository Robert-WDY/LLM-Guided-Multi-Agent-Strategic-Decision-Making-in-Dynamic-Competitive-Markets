from game_theory_agent.api import CONFIG
from game_theory_agent.calibration import evaluate_presets


def test_all_investment_anchors_pass_200_seed_extreme_rank_threshold():
    result = evaluate_presets(CONFIG, seed_start=0, seed_count=200)

    assert result["passed"] is True
    assert result["seed_count"] == 200
    for preset in result["presets"].values():
        assert sum(preset["rank_counts"].values()) == 200
        assert preset["first_place_ratio"] <= 0.70
        assert preset["last_place_ratio"] <= 0.70
