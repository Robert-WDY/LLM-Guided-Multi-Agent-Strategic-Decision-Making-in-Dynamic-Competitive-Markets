from __future__ import annotations

from game_theory_agent.experiments.stage62_strategic_ablation import (
    CONDITIONS,
    HOLDOUT_IDS,
    opponent_pools,
    run,
)
from game_theory_agent.opponent import (
    OPPONENT_MODEL_CANDIDATE_UPDATER_VERSION,
    OpponentModelLedger,
    PublicStrategyEvidence,
    compute_opponent_model_hash,
)
from game_theory_agent.strategic_reliability import (
    build_calibrated_opponent_state_v2,
)


def test_stage62_v2_state_uses_only_public_evidence(config):
    ledger = OpponentModelLedger(
        episode_id="stage62-v2-test",
        company_ids=("company_A", "company_B", "company_C", "company_D"),
    )
    v1, _ = ledger.company_view(
        observer_company_id="company_A",
        round_number=2,
        state_version=1,
    )
    evidence = PublicStrategyEvidence(
        evidence_id="stage62-v2-test:1:company_B",
        episode_id="stage62-v2-test",
        settled_round=1,
        target_company_id="company_B",
        previous_price_cents=10_000,
        settled_price_cents=9_400,
        price_direction="price_cut",
        market_share_delta_ppm=20_000,
        public_sales_orders=3_600,
        reputation_delta_ppm=-3_000,
        public_shared_resilience_contribution_cents=0,
    )

    first = build_calibrated_opponent_state_v2(v1, (evidence,))
    second = build_calibrated_opponent_state_v2(v1, (evidence,))

    assert first == second
    assert first.updater_version == OPPONENT_MODEL_CANDIDATE_UPDATER_VERSION
    assert not first.uses_hidden_cash
    assert not first.uses_hidden_cost
    assert not first.uses_hidden_persona
    assert first.opponent_models["company_B"].public_evidence_ids == [
        evidence.evidence_id
    ]
    assert compute_opponent_model_hash(first) == compute_opponent_model_hash(second)


def test_stage62_opponent_pools_are_deterministic_and_holdout_is_disjoint():
    first = opponent_pools()
    second = opponent_pools()

    assert first == second
    assert set(first) == {"known", "mixed", "holdout"}
    assert all(int(case_id.rsplit("_", 1)[-1]) >= 70 for case_id in HOLDOUT_IDS)
    for pool in first.values():
        assert set(pool) == {"company_B", "company_C", "company_D"}
        assert all(sum(item.model_dump().values()) == 1_000_000 for item in pool.values())


def test_stage62_small_ablation_is_matched_legal_and_zero_llm():
    summary = run(
        seeds=(1,),
        personas=("aggressive_v1_extreme", "risk_guarded_v1"),
        pool_ids=("known",),
        rounds=5,
        horizon_rounds=1,
        scenario_count=1,
    )

    assert len(summary["rows"]) == 2 * len(CONDITIONS)
    assert summary["real_model_usage"] == {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "estimated_cost": 0,
        "currency": "CNY",
    }
    assert summary["engineering_checks"]["matrix_complete"]
    assert summary["engineering_checks"]["all_actions_legal"]
    assert all(
        row["mean_true_one_step_regret_cents"] >= 0
        for row in summary["rows"]
    )
    assert set(summary["promotion_recommendations"]) == {
        "persona_planner",
        "belief",
        "opponent_v1",
        "opponent_v2_candidate",
    }
    assert summary["comparison_design"] == {
        "actual_market_seed_matched": True,
        "advisor_common_random_scenarios_on_same_observation": True,
        "opponent_policy_matched_but_state_responsive": True,
        "one_step_regret_uses_same_true_state_and_opponent_actions": True,
    }
    assert sum(
        sum(counts.values())
        for counts in summary["action_counts_by_condition"].values()
    ) == 2 * len(CONDITIONS) * 5
    assert set(summary["marginals_by_persona"]["persona_planner_vs_rule"]) == {
        "aggressive_v1_extreme",
        "risk_guarded_v1",
    }

    baselines = [
        row for row in summary["rows"] if row["condition"] == "rule_baseline"
    ]
    assert len({row["final_state_hash"] for row in baselines}) == 1
    assert len({row["enterprise_value_cents"] for row in baselines}) == 1
