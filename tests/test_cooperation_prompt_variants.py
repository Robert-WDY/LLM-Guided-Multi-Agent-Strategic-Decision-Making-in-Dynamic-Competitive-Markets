from __future__ import annotations

from game_theory_agent.agents.personas import load_persona_registry
from game_theory_agent.cooperation.prompt_variants import (
    build_cooperation_prompt,
    cooperation_payoff,
    economic_only_persona_view,
    private_optimal_contribution,
)


def test_economic_only_view_masks_cooperation_language() -> None:
    profile = load_persona_registry().get("balanced")
    view = economic_only_persona_view(profile)
    assert "commitment_honesty" not in view["traits_ppm"]
    assert "opportunism" not in view["traits_ppm"]
    assert "cooperation_reputation" not in view["utility_weights_ppm"]
    assert view["source_profile_hash"] == profile.profile_hash


def test_incentives_have_opposite_private_optima() -> None:
    assert private_optimal_contribution("high_deviation_incentive") == 0
    assert private_optimal_contribution("low_deviation_incentive") == 10
    assert cooperation_payoff(0, "high_deviation_incentive")["private_value_tenths"] > cooperation_payoff(10, "high_deviation_incentive")["private_value_tenths"]
    assert cooperation_payoff(10, "low_deviation_incentive")["private_value_tenths"] > cooperation_payoff(0, "low_deviation_incentive")["private_value_tenths"]


def test_neutral_prompt_does_not_name_betrayal_options() -> None:
    view = economic_only_persona_view(load_persona_registry().get("balanced"))
    neutral, neutral_hash = build_cooperation_prompt(variant="neutral_numeric_v1", incentive="high_deviation_incentive", persona_view=view, repetition=1)
    explicit, explicit_hash = build_cooperation_prompt(variant="explicit_options_v1", incentive="high_deviation_incentive", persona_view=view, repetition=1)
    assert "背叛" not in neutral
    assert "完全履约" not in neutral
    assert "部分履约" not in neutral
    assert "背叛" in explicit
    assert neutral_hash != explicit_hash


def test_terminal_variant_forbids_invented_reputation_payoff() -> None:
    view = economic_only_persona_view(load_persona_registry().get("balanced"))
    prompt, _ = build_cooperation_prompt(
        variant="neutral_numeric_v1",
        incentive="high_deviation_incentive",
        persona_view=view,
        repetition=1,
        terminal_payoff_only=True,
    )
    assert '"rounds_total": 1' in prompt
    assert "不存在后续轮次、声誉变化、可信度变化" in prompt
    assert "唯一评价目标" in prompt
