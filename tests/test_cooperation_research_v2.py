from game_theory_agent.experiments.cooperation_research_v2_matrix import (
    run_matrix,
)


def test_free_rider_matrix_exposes_private_saving_and_public_cost() -> None:
    summary = run_matrix(
        seeds=(301, 302),
        rounds=5,
        contribution_cents=500_000,
    )

    assert summary["passed"]
    assert summary["checks"]["free_rider_saves_first_round_private_cost"]
    assert summary["checks"]["free_rider_receives_public_benefit"]
    assert summary["checks"]["one_free_rider_reduces_public_stock"]
    assert summary["checks"]["all_defect_reduces_public_stock"]
    assert summary["checks"]["all_defect_increases_stockout_loss"]
    assert summary["checks"]["all_defect_reduces_total_long_term_profit"]
