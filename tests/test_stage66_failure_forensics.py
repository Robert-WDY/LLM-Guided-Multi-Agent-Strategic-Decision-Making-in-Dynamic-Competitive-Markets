from game_theory_agent.experiments.stage66_failure_forensics import (
    decompose_regret,
    economic_action,
    map_action_to_candidate,
)


def test_forensic_regret_decomposition_is_additive():
    result = decompose_regret(
        oracle_value=1_000,
        advisor_value=900,
        llm_value=850,
        final_value=825,
    )

    assert result == {
        "planner_regret_cents": 100,
        "adoption_regret_cents": 50,
        "execution_regret_cents": 25,
        "oracle_to_final_regret_cents": 175,
        "additive_identity_residual_cents": 0,
    }


def test_forensic_candidate_mapping_does_not_disguise_custom_action():
    candidates = {
        "maintain": economic_action({"price_cents": 10_000}),
        "price_cut_small": economic_action({"price_cents": 9_500}),
    }

    exact = map_action_to_candidate({"price_cents": 9_500}, candidates)
    custom = map_action_to_candidate(
        {"price_cents": 9_500, "advertising_budget_cents": 1}, candidates
    )

    assert exact["mapping_status"] == "exact_candidate"
    assert exact["candidate_id"] == "price_cut_small"
    assert custom["mapping_status"] == "custom_action"
    assert custom["candidate_id"] == "custom_action"
    assert custom["nearest_candidate_id"] == "price_cut_small"
    assert custom["different_field_count"] == 1
