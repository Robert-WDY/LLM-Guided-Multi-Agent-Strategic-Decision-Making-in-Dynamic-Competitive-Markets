from game_theory_agent.experiments.core_real_llm_validation import (
    COOPERATION_CONDITIONS,
    COOPERATION_PERSONAS,
    NEGATIVE_KEYS,
    STRATEGIC_CONDITIONS,
    _cooperation_summary,
    _strategic_summary,
    build_plan,
)


def test_core_real_plan_is_exactly_cost_bounded_and_balanced() -> None:
    plan = build_plan()
    strategic = [
        row for row in plan if row["experiment"] == "strategic_v5_failure_retest"
    ]
    cooperation = [
        row
        for row in plan
        if row["experiment"] == "cooperation_credibility_persona"
    ]

    assert len(plan) == 22
    assert len(strategic) == 10
    assert len(cooperation) == 12
    assert {
        (row["seed"], row["persona_id"]) for row in strategic
    } == set(NEGATIVE_KEYS)
    for key in NEGATIVE_KEYS:
        assert {
            row["condition"]
            for row in strategic
            if (row["seed"], row["persona_id"]) == key
        } == set(STRATEGIC_CONDITIONS)
    for persona_id in COOPERATION_PERSONAS:
        for condition in COOPERATION_CONDITIONS:
            assert sum(
                row["persona_id"] == persona_id
                and row["condition"] == condition
                for row in cooperation
            ) == 2


def test_core_real_summaries_keep_pair_and_causal_directions() -> None:
    strategic_rows = []
    for index, (seed, persona_id) in enumerate(NEGATIVE_KEYS):
        for condition in STRATEGIC_CONDITIONS:
            strategic_rows.append(
                {
                    "experiment": "strategic_v5_failure_retest",
                    "seed": seed,
                    "persona_id": persona_id,
                    "condition": condition,
                    "success": True,
                    "final_action": {"price_cents": 100 + index + (condition == "pareto_reliable_v5")},
                    "terminal_enterprise_value_cents": (
                        1_000 + index + (10 if condition == "pareto_reliable_v5" else 0)
                    ),
                    "advisor_adoption": (
                        {"adoption_status": "exact_action", "target_alignment_ppm": 1_000_000}
                        if condition == "pareto_reliable_v5"
                        else None
                    ),
                }
            )
    strategic = _strategic_summary(strategic_rows)
    assert strategic["complete_pair_count"] == 5
    assert strategic["mean_ev_delta_cents"] == 10
    assert strategic["worst_ev_delta_cents"] == 10
    assert all(strategic["directional_gate"].values())

    cooperation_rows = []
    for persona_id in COOPERATION_PERSONAS:
        for condition in COOPERATION_CONDITIONS:
            for repetition in (1, 2):
                high = condition == "high_credibility_proposal"
                cooperation_rows.append(
                    {
                        "experiment": "cooperation_credibility_persona",
                        "persona_id": persona_id,
                        "condition": condition,
                        "repetition": repetition,
                        "success": True,
                        "proposal_accepted": high,
                        "contribution_cents": (
                            1_000_000
                            if high and persona_id == COOPERATION_PERSONAS[0]
                            else 500_000
                            if high
                            else 0
                        ),
                    }
                )
    cooperation = _cooperation_summary(cooperation_rows)
    assert cooperation["all_cells_complete"] is True
    assert cooperation["contrasts"][0]["high_minus_low_contribution_cents"] == 1_000_000
    assert cooperation["contrasts"][1]["high_minus_low_contribution_cents"] == 500_000
    assert all(cooperation["directional_gate"].values())
