from argparse import Namespace
from pathlib import Path

from game_theory_agent.experiments.privileged_information_persona import (
    CONDITIONS,
    _episode_args,
    aggregate,
    build_plan,
)


def _row(persona: str, seed: int, condition: str, rank: int) -> dict:
    return {
        "persona": persona,
        "seed": seed,
        "condition": condition,
        "passed": True,
        "observation_mode_matches": True,
        "private_visibility_check": True,
        "target_composite_rank": rank,
        "target_won_composite": rank == 1,
        "target_enterprise_value_cents": 100_000_000 - rank,
        "target_cumulative_profit_cents": 10_000_000 - rank,
        "target_market_share_ppm": 250_000 - rank,
        "token_usage": {
            "model_call_count": 5,
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "calls_without_token_usage": 0,
        },
    }


def test_plan_pairs_every_persona_seed_and_balances_call_order():
    plan = build_plan((1, 2), ("balanced_v1", "profit_myopic"))

    assert len(plan) == 8
    assert {
        (row["persona"], row["seed"], row["condition"])
        for row in plan
    } == {
        (persona, seed, condition)
        for persona in ("balanced_v1", "profit_myopic")
        for seed in (1, 2)
        for condition in CONDITIONS
    }
    assert [row["condition_call_order"] for row in plan] == [
        1,
        2,
        1,
        2,
        1,
        2,
        1,
        2,
    ]


def test_paired_conditions_reuse_episode_id_and_only_treatment_is_privileged():
    args = Namespace(
        rounds=10,
        market_model="balanced",
        provider="mock",
        model=None,
        temperature=0.0,
        top_p=1.0,
        timeout=60.0,
    )
    base = {"persona": "balanced_v1", "seed": 7}
    control = _episode_args(
        args, {**base, "condition": "public_control"}, Path("control")
    )
    treatment = _episode_args(
        args,
        {**base, "condition": "privileged_perfect"},
        Path("treatment"),
    )

    assert control.episode_id == treatment.episode_id
    assert control.information_mode == treatment.information_mode == "public"
    assert control.privileged_observer_company_id is None
    assert treatment.privileged_observer_company_id == "company_A"


def test_aggregate_does_not_turn_research_failure_into_engineering_failure():
    rows = [
        _row("balanced_v1", 1, "public_control", 3),
        _row("balanced_v1", 1, "privileged_perfect", 2),
        _row("profit_myopic", 1, "public_control", 4),
        _row("profit_myopic", 1, "privileged_perfect", 1),
    ]

    summary = aggregate(rows, (1,), ("balanced_v1", "profit_myopic"))

    assert summary["engineering_passed"] is True
    assert summary["research_hypothesis_supported"] is False
    assert summary["all_privileged_persona_seed_cases_first"] is False
    assert summary["privileged_first_place_rate"] == 0.5
    assert summary["mean_paired_effect"]["rank_improvement"] == 2
    assert summary["token_usage"]["model_call_count"] == 20
