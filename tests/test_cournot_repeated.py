from __future__ import annotations

from game_theory_agent.benchmark_games.repeated import (
    forecast_opponent_belief,
    rule_opponent_quantity,
)
from game_theory_agent.experiments.cournot_repeated_real import _decision_prompt


def test_rule_opponents_use_only_prior_focal_actions() -> None:
    assert rule_opponent_quantity("fixed_nash", focal_history=[]) == 8
    assert rule_opponent_quantity("fixed_nash", focal_history=[20]) == 8
    assert rule_opponent_quantity("adaptive_best_response", focal_history=[]) == 8
    assert rule_opponent_quantity("adaptive_best_response", focal_history=[12]) == 6


def test_llm_belief_is_recent_empirical_and_exact_ppm() -> None:
    belief = forecast_opponent_belief(
        "deepseek_llm",
        focal_history=[8, 8],
        opponent_history=[6, 8, 8],
    )
    assert belief == {6: 333_334, 8: 666_666}
    assert sum(belief.values()) == 1_000_000


def test_repeated_prompt_never_exposes_current_opponent_action() -> None:
    prompt = _decision_prompt(
        role="company_A",
        round_number=2,
        focal_history=[8],
        opponent_history=[9],
        belief_ppm={9: 1_000_000},
        advice=None,
    )
    assert '"current_opponent_action_visible": false' in prompt
    assert "两家公司同时行动" in prompt
