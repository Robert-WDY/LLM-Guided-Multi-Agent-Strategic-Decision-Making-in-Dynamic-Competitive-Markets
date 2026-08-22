from game_theory_agent.experiments.belief_real_paired import (
    _action_metrics,
    _belief_reference,
    build_plan,
    parse_seeds,
)


def test_belief_real_plan_pairs_same_episode_and_alternates_order():
    plan = build_plan((701, 702))
    assert [item["condition"] for item in plan] == [
        "belief_off",
        "belief_on",
        "belief_on",
        "belief_off",
    ]
    assert plan[0]["episode_id"] == plan[1]["episode_id"]
    assert plan[2]["episode_id"] == plan[3]["episode_id"]
    assert parse_seeds("701, 702") == (701, 702)


def test_belief_reference_and_action_metrics_are_deterministic():
    trace = type(
        "Trace",
        (),
        {"planner_output": {"plan": {"key_factors": ["对手降价概率"]}}},
    )()
    assert _belief_reference(trace)["referenced"] is True
    generic_direction = type(
        "Trace",
        (),
        {"planner_output": {"plan": {"key_factors": ["对手最近降价"]}}},
    )()
    assert _belief_reference(generic_direction)["referenced"] is False
    metrics = _action_metrics(
        [
            {"price_cents": 100, "advertising_budget_cents": 10},
            {"price_cents": 200, "advertising_budget_cents": 30},
        ]
    )
    assert metrics["mean"]["price_cents"] == 150
    assert metrics["total"]["advertising_budget_cents"] == 40
