from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.experiments.stage6_strategic_reliability import run
from game_theory_agent.gameplay import build_terminal_rankings
from game_theory_agent.market import MarketEnv
from game_theory_agent.strategic_reliability import (
    AuthoritativeMarketRolloutEvaluator,
    ObjectiveCalibrationDecision,
    ObjectiveCalibrationSpec,
    ParetoPlannerDecision,
    ParetoPlannerSpec,
    RealModelBudget,
    RealModelBudgetExceeded,
    RealModelCostGuard,
    StrategicReliabilityPlan,
    calibrate_objective_decision,
    compute_reliability_plan_hash,
    deterministic_dirichlet_weights,
    generate_candidate_actions,
    run_opponent_benchmark,
    select_pareto_decision,
)


def _state(config, *, seed: int = 42, episode_id: str = "stage6-test"):
    return MarketEnv(config).reset(
        ["company_A", "company_B", "company_C", "company_D"],
        episode_id=episode_id,
        episode_seed=seed,
        market_model="balanced",
        max_rounds=10,
    )


def test_candidate_set_is_bounded_distinct_and_market_legal(config):
    state = _state(config)
    candidates = generate_candidate_actions(config, state, "company_A")
    assert 7 <= len(candidates) <= 10
    assert candidates[0].candidate_id == "maintain"
    assert {
        "price_cut_small",
        "price_cut_large",
        "price_increase",
        "increase_service",
        "increase_capacity",
        "increase_resilience",
    }.issubset({item.candidate_id for item in candidates})
    assert len({item.action.model_dump_json() for item in candidates}) == len(
        candidates
    )


def test_authoritative_rollout_is_deterministic_and_not_public_agent_advice(config):
    state = _state(config)
    profile = PersonaRegistry.from_market_config(config).get("balanced_v1")
    evaluator = AuthoritativeMarketRolloutEvaluator(config)
    first = evaluator.evaluate(
        state=state,
        company_id="company_A",
        persona_profile=profile,
        horizon_rounds=3,
        scenario_count=3,
    )
    second = evaluator.evaluate(
        state=state,
        company_id="company_A",
        persona_profile=profile,
        horizon_rounds=3,
        scenario_count=3,
    )
    assert first == second
    assert first.plan_hash == second.plan_hash
    assert first.uses_authoritative_hidden_market_state
    assert not first.allowed_in_public_agent_context
    assert not first.claims_nash_equilibrium
    scenario_seeds = {
        tuple(item.scenario_seed for item in evaluation.outcomes)
        for evaluation in first.evaluations
    }
    assert len(scenario_seeds) == 1
    for evaluation in first.evaluations:
        assert evaluation.expected_competitive_margin_cents is not None
        assert evaluation.worst_case_competitive_margin_cents is not None
        assert evaluation.expected_final_rank_milli is not None
        for outcome in evaluation.outcomes:
            assert outcome.best_opponent_enterprise_value_cents is not None
            assert outcome.competitive_margin_cents == (
                outcome.enterprise_value_cents
                - outcome.best_opponent_enterprise_value_cents
            )
            assert outcome.final_rank is not None


def test_objective_calibration_is_tamper_evident_and_persona_compatible(config):
    state = _state(config, seed=3, episode_id="objective-calibration")
    profile = PersonaRegistry.from_market_config(config).get(
        "aggressive_v1_extreme"
    )
    plan = AuthoritativeMarketRolloutEvaluator(config).evaluate(
        state=state,
        company_id="company_A",
        persona_profile=profile,
        horizon_rounds=2,
        scenario_count=2,
    )
    persona = calibrate_objective_decision(
        plan,
        ObjectiveCalibrationSpec(
            mode="persona_aligned",
            risk_aversion_ppm=profile.traits_ppm.risk_aversion,
        ),
    )
    hybrid = calibrate_objective_decision(
        plan,
        ObjectiveCalibrationSpec(
            mode="competitive_hybrid",
            competitive_weight_ppm=500_000,
            risk_aversion_ppm=profile.traits_ppm.risk_aversion,
        ),
    )
    assert persona.recommended_candidate_id == plan.recommended_candidate_id
    assert hybrid.experimental_only
    assert not hybrid.allowed_in_agent_context
    forged = deepcopy(hybrid.model_dump(mode="json"))
    forged["candidate_scores"][0]["objective_score_cents"] += 1
    with pytest.raises(ValidationError):
        ObjectiveCalibrationDecision.model_validate(forged)


def test_competitive_labels_keep_pre_stage63_plan_hash_compatible(config):
    state = _state(config, seed=8, episode_id="stage62-plan-compatibility")
    profile = PersonaRegistry.from_market_config(config).get("balanced_v1")
    plan = AuthoritativeMarketRolloutEvaluator(config).evaluate(
        state=state,
        company_id="company_A",
        persona_profile=profile,
        horizon_rounds=1,
        scenario_count=1,
    )
    old_payload = deepcopy(plan.model_dump(mode="json"))
    for evaluation in old_payload["evaluations"]:
        evaluation.pop("expected_competitive_margin_cents")
        evaluation.pop("worst_case_competitive_margin_cents")
        evaluation.pop("expected_final_rank_milli")
        for outcome in evaluation["outcomes"]:
            outcome.pop("best_opponent_enterprise_value_cents")
            outcome.pop("competitive_margin_cents")
            outcome.pop("final_rank")
    old_payload["plan_hash"] = compute_reliability_plan_hash(old_payload)

    restored = StrategicReliabilityPlan.model_validate(old_payload)

    assert restored.plan_hash == old_payload["plan_hash"]
    assert all(
        item.expected_competitive_margin_cents is None
        for item in restored.evaluations
    )


def test_pareto_planner_filters_candidates_and_rejects_tampering(config):
    state = _state(config, seed=13, episode_id="pareto-planner")
    profile = PersonaRegistry.from_market_config(config).get("balanced_v1")
    plan = AuthoritativeMarketRolloutEvaluator(config).evaluate(
        state=state,
        company_id="company_A",
        persona_profile=profile,
        horizon_rounds=2,
        scenario_count=2,
    )
    rankings = build_terminal_rankings(state, config)["composite"]
    focal = next(item for item in rankings if item["company_id"] == "company_A")
    best_opponent = max(
        int(item["value_cents"])
        for item in rankings
        if item["company_id"] != "company_A"
    )
    first = select_pareto_decision(
        plan,
        ParetoPlannerSpec(spec_id="strict_test"),
        current_rank=int(focal["rank"]),
        current_competitive_margin_cents=(
            int(focal["value_cents"]) - best_opponent
        ),
        rounds_remaining=state.rounds_remaining,
    )
    second = select_pareto_decision(
        plan,
        ParetoPlannerSpec(spec_id="strict_test"),
        current_rank=int(focal["rank"]),
        current_competitive_margin_cents=(
            int(focal["value_cents"]) - best_opponent
        ),
        rounds_remaining=state.rounds_remaining,
    )

    assert first == second
    assert first.experimental_only
    assert not first.allowed_in_agent_context
    assert any(
        item.candidate_id == "maintain" and item.eligible
        for item in first.candidate_assessments
    )
    selected = next(
        item
        for item in first.candidate_assessments
        if item.candidate_id == first.recommended_candidate_id
    )
    assert selected.pareto_frontier
    forged = deepcopy(first.model_dump(mode="json"))
    forged["candidate_assessments"][0]["eligible"] = not forged[
        "candidate_assessments"
    ][0]["eligible"]
    with pytest.raises(ValidationError):
        ParetoPlannerDecision.model_validate(forged)


def test_rollout_rejects_tampered_result(config):
    state = _state(config)
    profile = PersonaRegistry.from_market_config(config).get("balanced_v1")
    plan = AuthoritativeMarketRolloutEvaluator(config).evaluate(
        state=state,
        company_id="company_A",
        persona_profile=profile,
        horizon_rounds=2,
        scenario_count=2,
    )
    forged = deepcopy(plan.model_dump(mode="json"))
    forged["expected_gain_over_baseline_cents"] += 1
    with pytest.raises(ValidationError):
        StrategicReliabilityPlan.model_validate(forged)


def test_persona_planning_layer_changes_ranking_on_same_market(config):
    state = _state(config, seed=3, episode_id="balanced-3")
    registry = PersonaRegistry.from_market_config(config)
    evaluator = AuthoritativeMarketRolloutEvaluator(config)
    aggressive = evaluator.evaluate(
        state=state,
        company_id="company_A",
        persona_profile=registry.get("aggressive_v1_extreme"),
        horizon_rounds=3,
        scenario_count=3,
    )
    guarded = evaluator.evaluate(
        state=state,
        company_id="company_A",
        persona_profile=registry.get("risk_guarded_v1"),
        horizon_rounds=3,
        scenario_count=3,
    )
    assert aggressive.state_hash == guarded.state_hash
    assert aggressive.recommended_candidate_id != guarded.recommended_candidate_id
    assert aggressive.recommended_candidate_id == "increase_service"
    assert guarded.recommended_candidate_id == "price_increase"


def test_dirichlet_holdout_weights_are_deterministic_and_normalized():
    first = deterministic_dirichlet_weights(20260824, "unknown_099")
    second = deterministic_dirichlet_weights(20260824, "unknown_099")
    assert first == second
    assert sum(first.model_dump().values()) == 1_000_000


def test_opponent_benchmark_has_disjoint_holdout_and_calibration_gate():
    summary = run_opponent_benchmark(random_count=100, evidence_rounds=12)
    assert not (
        set(summary["development_case_ids"])
        & set(summary["holdout_case_ids"])
    )
    assert len(summary["holdout_case_ids"]) == 30
    v1 = summary["holdout_scores"]["v1"]
    v2 = summary["holdout_scores"]["v2"]
    assert v2["mean_distribution_brier"] < v1["mean_distribution_brier"]
    assert v2["mean_cross_entropy"] < v1["mean_cross_entropy"]
    # Calibration improved, but top-type accuracy did not.  This prevents an
    # unjustified automatic replacement of the current online v1 model.
    assert v2["top_strategy_accuracy_ppm"] <= v1["top_strategy_accuracy_ppm"]


def test_stage6_p0_acceptance_uses_no_real_model_calls():
    summary = run(seeds=(1, 2), horizon_rounds=3, scenario_count=3)
    assert summary["real_model_usage"] == {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "estimated_cost": 0,
        "currency": "CNY",
    }
    assert summary["passed"]


def test_real_model_cost_guard_is_fail_closed_and_enforces_reservations():
    budget = RealModelBudget(
        max_calls=2,
        max_prompt_tokens=2_000,
        max_completion_tokens=400,
        max_estimated_cost_microunits=50_000,
    )
    blocked = RealModelCostGuard(budget)
    with pytest.raises(RealModelBudgetExceeded, match="explicit"):
        blocked.reserve(
            prompt_tokens=500,
            completion_tokens=100,
            estimated_cost_microunits=10_000,
        )

    guard = RealModelCostGuard(budget, explicitly_authorized=True)
    guard.reserve(
        prompt_tokens=500,
        completion_tokens=100,
        estimated_cost_microunits=10_000,
    )
    guard.record_actual(
        prompt_tokens=450,
        completion_tokens=80,
        estimated_cost_microunits=9_000,
    )
    with pytest.raises(RealModelBudgetExceeded, match="reserved"):
        guard.record_actual(
            prompt_tokens=100,
            completion_tokens=30,
            estimated_cost_microunits=2_000,
        )
    with pytest.raises(RealModelBudgetExceeded, match="prompt_tokens"):
        guard.reserve(
            prompt_tokens=1_600,
            completion_tokens=100,
            estimated_cost_microunits=10_000,
        )
