"""Stage 6.4 constrained Pareto Planner and ten-seed reliability gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.agents.personas import PersonaRegistry, PersonaUtilityTracker
from game_theory_agent.gameplay import build_terminal_rankings
from game_theory_agent.market import CompanyAction, MarketEnv, load_market_config
from game_theory_agent.strategic_reliability import (
    AuthoritativeMarketRolloutEvaluator,
    ObjectiveCalibrationSpec,
    PROMOTED_PARETO_SPEC,
    ParetoPlannerSpec,
    StrategicReliabilityPlan,
    build_public_forecast_state,
    calibrate_objective_decision,
    generate_public_overlay_candidates,
    select_pareto_decision,
)

from .stage62_strategic_ablation import (
    COMPANIES,
    PERSONAS,
    _candidate_to_action,
    _opponent_action,
    opponent_pools,
    parse_csv,
    parse_csv_ints,
)
from .stage63_objective_calibration import (
    _aggregate,
    _competitive_result,
    _paired,
    _selected_evaluation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v4.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "stage6.4-pareto" / "summary.json"
PARETO_SPECS = {
    "pareto_strict_value": PROMOTED_PARETO_SPEC,
    "pareto_strict_competition": ParetoPlannerSpec(
        spec_id="strict_competition",
        early_trailing_priority="competition",
    ),
    "pareto_late_5k": ParetoPlannerSpec(
        spec_id="late_5k",
        early_trailing_priority="value",
        late_expected_value_sacrifice_ppm=5_000,
    ),
    "pareto_catchup_5k": ParetoPlannerSpec(
        spec_id="catchup_5k",
        early_trailing_priority="competition",
        late_expected_value_sacrifice_ppm=5_000,
    ),
    "pareto_catchup_10k": ParetoPlannerSpec(
        spec_id="catchup_10k",
        early_trailing_priority="competition",
        late_expected_value_sacrifice_ppm=10_000,
    ),
}
CONDITIONS = (
    "rule_baseline",
    "persona_aligned",
    "competitive_250k",
    *PARETO_SPECS,
)


def _available_candidate_regrets(
    config,
    state,
    *,
    persona_profile,
    opponent_actions: dict[str, CompanyAction],
    candidates,
    chosen_candidate_id: str,
    seed: int,
    pool_id: str,
    pool,
    horizon_rounds: int,
) -> tuple[int, int]:
    """Oracle regrets over the exact public candidate set offered this round.

    The one-step diagnostic exposes immediate opportunity cost.  The horizon
    metric evaluates the same current choices followed by the public Rule
    policy and is the reliability gate, because Stage 6 explicitly targets
    long-term rather than myopic best response.
    """

    registry = PersonaRegistry.from_market_config(config)
    evaluator = registry.evaluator(persona_profile)
    one_step_values: dict[str, int] = {}
    horizon_values: dict[str, int] = {}
    for candidate in candidates:
        env = MarketEnv(config)
        env.load_state(state)
        simulated = state
        accumulated_adjustment = 0
        for offset in range(min(horizon_rounds, state.rounds_remaining)):
            if offset == 0:
                simulated_opponents = opponent_actions
                focal = _candidate_to_action(candidate, simulated)
            else:
                simulated_opponents = {
                    company_id: _opponent_action(
                        config,
                        simulated,
                        company_id,
                        seed=seed,
                        pool_id=pool_id,
                        distribution=pool[company_id],
                    )[0]
                    for company_id in simulated.company_ids[1:]
                }
                observation = ObservationBuilder().build(
                    simulated,
                    "company_A",
                    "public",
                    belief_schema_version="none",
                )
                forecast, _record = build_public_forecast_state(
                    config=config,
                    observation=observation,
                    company_id="company_A",
                    persona_profile=persona_profile,
                )
                baseline = next(
                    item
                    for item in generate_public_overlay_candidates(
                        config, forecast, "company_A"
                    )
                    if item.candidate_id == "maintain"
                )
                focal = _candidate_to_action(baseline, simulated)
            actions = {"company_A": focal, **simulated_opponents}
            before = simulated
            result = env.step(
                f"{before.episode_id}:{before.round}:{before.state_version}",
                actions,
            )
            assessment = evaluator.evaluate(
                before, result.state_after, "company_A"
            )
            loss = (
                assessment.realized_incident_loss_cents
                + assessment.realized_unserved_contribution_loss_cents
            )
            accumulated_adjustment += (
                assessment.round_utility_ppm
                * registry.profit_scale_cents
                // 1_000_000
            ) - loss
            simulated = result.state_after
            enterprise_value = next(
                int(item["value_cents"])
                for item in build_terminal_rankings(
                    simulated, config
                )["composite"]
                if item["company_id"] == "company_A"
            )
            if offset == 0:
                one_step_values[candidate.candidate_id] = (
                    enterprise_value + accumulated_adjustment
                )
        horizon_values[candidate.candidate_id] = (
            enterprise_value + accumulated_adjustment
        )
    if chosen_candidate_id not in one_step_values:
        raise ValueError(f"chosen candidate is missing: {chosen_candidate_id}")
    return (
        max(one_step_values.values()) - one_step_values[chosen_candidate_id],
        max(horizon_values.values()) - horizon_values[chosen_candidate_id],
    )


def run_episode(
    config,
    *,
    seed: int,
    persona_id: str,
    pool_id: str,
    pool,
    condition: str,
    rounds: int,
    horizon_rounds: int,
    scenario_count: int,
    plan_cache: dict[tuple[Any, ...], StrategicReliabilityPlan],
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown Stage 6.4 condition: {condition}")
    episode_id = f"stage6.4-{pool_id}-seed-{seed}"
    env = MarketEnv(config)
    state = env.reset(
        COMPANIES,
        episode_id=episode_id,
        episode_seed=seed,
        market_model="balanced",
        max_rounds=rounds,
        cooperation_mode="shared_resilience_v1",
    )
    registry = PersonaRegistry.from_market_config(config)
    profile = registry.get(persona_id)
    tracker = PersonaUtilityTracker(registry.evaluator(profile))
    oracle = AuthoritativeMarketRolloutEvaluator(config)
    recommendations: list[str] = []
    one_step_regrets: list[int] = []
    horizon_regrets: list[int] = []
    traces: list[dict[str, Any]] = []
    realized_loss = 0
    all_actions_legal = True

    while not state.terminal:
        opponent_actions: dict[str, CompanyAction] = {}
        for company_id in state.company_ids[1:]:
            action, _strategy = _opponent_action(
                config,
                state,
                company_id,
                seed=seed,
                pool_id=pool_id,
                distribution=pool[company_id],
            )
            opponent_actions[company_id] = action

        observation = ObservationBuilder().build(
            state,
            "company_A",
            "public",
            belief_schema_version="none",
        )
        forecast, forecast_record = build_public_forecast_state(
            config=config,
            observation=observation,
            company_id="company_A",
            persona_profile=profile,
        )
        candidates = generate_public_overlay_candidates(
            config, forecast, "company_A"
        )
        forecast_seed = forecast_record.forecast_seed
        decision_hash: str | None = None
        decision_situation: str | None = None
        eligible_count: int | None = None
        frontier_count: int | None = None
        if condition == "rule_baseline":
            chosen_id = "maintain"
            baseline = next(
                item for item in candidates if item.candidate_id == "maintain"
            )
            focal_action = _candidate_to_action(baseline, state)
        else:
            horizon = min(horizon_rounds, state.rounds_remaining)
            cache_key = (
                forecast.state_hash,
                profile.profile_hash,
                horizon,
                scenario_count,
            )
            plan = plan_cache.get(cache_key)
            if plan is None:
                plan = oracle.evaluate(
                    state=forecast,
                    company_id="company_A",
                    persona_profile=profile,
                    horizon_rounds=horizon,
                    scenario_count=scenario_count,
                    candidates=candidates,
                )
                plan_cache[cache_key] = plan
            if condition == "persona_aligned":
                decision = calibrate_objective_decision(
                    plan,
                    ObjectiveCalibrationSpec(
                        mode="persona_aligned",
                        risk_aversion_ppm=profile.traits_ppm.risk_aversion,
                    ),
                )
                chosen_id = decision.recommended_candidate_id
                decision_hash = decision.decision_hash
            elif condition == "competitive_250k":
                decision = calibrate_objective_decision(
                    plan,
                    ObjectiveCalibrationSpec(
                        mode="competitive_hybrid",
                        competitive_weight_ppm=250_000,
                        risk_aversion_ppm=profile.traits_ppm.risk_aversion,
                    ),
                )
                chosen_id = decision.recommended_candidate_id
                decision_hash = decision.decision_hash
            else:
                current_margin, _best_value, _best_id, current_rank = (
                    _competitive_result(forecast, config)
                )
                pareto = select_pareto_decision(
                    plan,
                    PARETO_SPECS[condition],
                    current_rank=current_rank,
                    current_competitive_margin_cents=current_margin,
                    rounds_remaining=state.rounds_remaining,
                )
                chosen_id = pareto.recommended_candidate_id
                decision_hash = pareto.decision_hash
                decision_situation = pareto.situation.situation
                eligible_count = sum(
                    item.eligible for item in pareto.candidate_assessments
                )
                frontier_count = sum(
                    item.pareto_frontier for item in pareto.candidate_assessments
                )
            evaluation = _selected_evaluation(plan, chosen_id)
            focal_action = _candidate_to_action(evaluation.candidate, state)

        one_step_regret, horizon_regret = _available_candidate_regrets(
            config,
            state,
            persona_profile=profile,
            opponent_actions=opponent_actions,
            candidates=candidates,
            chosen_candidate_id=chosen_id,
            seed=seed,
            pool_id=pool_id,
            pool=pool,
            horizon_rounds=horizon_rounds,
        )
        recommendations.append(chosen_id)
        one_step_regrets.append(one_step_regret)
        horizon_regrets.append(horizon_regret)
        validation = env.validate_action(focal_action, "company_A")
        all_actions_legal = all_actions_legal and validation.valid
        if not validation.valid or validation.action is None:
            raise ValueError(f"illegal Stage 6.4 action: {validation.errors}")
        actions = {"company_A": validation.action, **opponent_actions}
        before = state
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", actions
        )
        assessment = tracker.record(before, result.state_after, "company_A")
        realized_loss += (
            assessment.realized_incident_loss_cents
            + assessment.realized_unserved_contribution_loss_cents
        )
        post_margin, _value, _company, post_rank = _competitive_result(
            result.state_after, config
        )
        traces.append(
            {
                "round": before.round,
                "pre_state_hash": before.state_hash,
                "chosen_candidate_id": chosen_id,
                "true_one_step_regret_cents": one_step_regret,
                "true_horizon_regret_cents": horizon_regret,
                "forecast_seed": forecast_seed,
                "decision_hash": decision_hash,
                "pareto_situation": decision_situation,
                "eligible_candidate_count": eligible_count,
                "frontier_candidate_count": frontier_count,
                "realized_post_round_competitive_margin_cents": post_margin,
                "realized_post_round_rank": post_rank,
                "post_state_hash": result.state_after.state_hash,
            }
        )
        state = result.state_after

    margin, best_value, best_company, rank = _competitive_result(state, config)
    company = state.company("company_A")
    enterprise_value = next(
        int(item["value_cents"])
        for item in build_terminal_rankings(state, config)["composite"]
        if item["company_id"] == "company_A"
    )
    return {
        "seed": seed,
        "persona_id": persona_id,
        "opponent_pool": pool_id,
        "condition": condition,
        "enterprise_value_cents": enterprise_value,
        "cumulative_profit_cents": company.financial.cumulative_profit_cents,
        "final_market_share_ppm": company.commercial.market_share_ppm,
        "final_competitive_margin_cents": margin,
        "best_opponent_enterprise_value_cents": best_value,
        "best_opponent_company_id": best_company,
        "final_rank": rank,
        "first_place": rank == 1,
        "realized_loss_cents": realized_loss,
        "mean_true_one_step_regret_cents": round(mean(one_step_regrets)),
        "mean_true_horizon_regret_cents": round(mean(horizon_regrets)),
        "recommendations": recommendations,
        "non_baseline_action_count": sum(
            item != "maintain" for item in recommendations
        ),
        "round_trace": traces,
        "all_actions_legal": all_actions_legal,
        "final_state_hash": state.state_hash,
    }


def _paired_with_activity(
    rows: list[dict[str, Any]], treatment: str, control: str = "rule_baseline"
) -> dict[str, Any]:
    result = _paired(rows, treatment, control)
    controls = {
        (item["seed"], item["persona_id"], item["opponent_pool"]): item
        for item in rows
        if item["condition"] == control
    }
    treated = [item for item in rows if item["condition"] == treatment]
    changed_actions = sum(item["non_baseline_action_count"] for item in treated)
    total_actions = sum(len(item["recommendations"]) for item in treated)
    horizon_regret_deltas = [
        item["mean_true_horizon_regret_cents"]
        - controls[
            (item["seed"], item["persona_id"], item["opponent_pool"])
        ]["mean_true_horizon_regret_cents"]
        for item in treated
    ]
    result.update(
        {
            "changed_episode_count": sum(
                item["non_baseline_action_count"] > 0 for item in treated
            ),
            "non_baseline_action_count": changed_actions,
            "non_baseline_action_rate_ppm": (
                changed_actions * 1_000_000 // total_actions
            ),
            "mean_true_horizon_regret_delta_cents": round(
                mean(horizon_regret_deltas)
            ),
        }
    )
    return result


def _reliability_gate(item: dict[str, Any]) -> dict[str, bool]:
    strict_benefit = any(
        (
            item["mean_enterprise_value_delta_cents"] > 0,
            item["mean_competitive_margin_delta_cents"] > 0,
            item["mean_true_horizon_regret_delta_cents"] < 0,
            item["first_place_rate_delta_ppm"] > 0,
        )
    )
    return {
        "mean_enterprise_value_non_negative": (
            item["mean_enterprise_value_delta_cents"] >= 0
        ),
        "worst_enterprise_value_non_negative": (
            item["worst_enterprise_value_delta_cents"] >= 0
        ),
        "mean_competitive_margin_non_negative": (
            item["mean_competitive_margin_delta_cents"] >= 0
        ),
        "first_place_rate_not_lower": item["first_place_rate_delta_ppm"] >= 0,
        "mean_horizon_regret_not_higher": (
            item["mean_true_horizon_regret_delta_cents"] <= 0
        ),
        "non_trivial_action_change": item["non_baseline_action_rate_ppm"] >= 50_000,
        "strict_benefit_present": strict_benefit,
    }


def run(
    *,
    seeds: tuple[int, ...] = tuple(range(1, 11)),
    personas: tuple[str, ...] = PERSONAS,
    pool_ids: tuple[str, ...] = ("known", "mixed", "holdout"),
    rounds: int = 5,
    horizon_rounds: int = 2,
    scenario_count: int = 2,
) -> dict[str, Any]:
    if len(seeds) < 2:
        raise ValueError("Stage 6.4 requires at least two seeds")
    config = load_market_config(CONFIG_PATH)
    registry = PersonaRegistry.from_market_config(config)
    for persona_id in personas:
        registry.get(persona_id)
    pools = opponent_pools()
    if set(pool_ids) - set(pools):
        raise ValueError("unknown Stage 6.4 opponent pool")
    plan_cache: dict[tuple[Any, ...], StrategicReliabilityPlan] = {}
    rows = [
        run_episode(
            config,
            seed=seed,
            persona_id=persona_id,
            pool_id=pool_id,
            pool=pools[pool_id],
            condition=condition,
            rounds=rounds,
            horizon_rounds=horizon_rounds,
            scenario_count=scenario_count,
            plan_cache=plan_cache,
        )
        for seed in seeds
        for pool_id in pool_ids
        for persona_id in personas
        for condition in CONDITIONS
    ]
    by_condition = {
        condition: _aggregate([item for item in rows if item["condition"] == condition])
        for condition in CONDITIONS
    }
    all_comparisons = {
        condition: _paired_with_activity(rows, condition)
        for condition in CONDITIONS
        if condition != "rule_baseline"
    }
    all_gates = {
        condition: _reliability_gate(item)
        for condition, item in all_comparisons.items()
    }

    ordered_seeds = tuple(sorted(seeds))
    midpoint = len(ordered_seeds) // 2
    development_seeds = set(ordered_seeds[:midpoint])
    holdout_seeds = set(ordered_seeds[midpoint:])
    development_rows = [
        item
        for item in rows
        if item["seed"] in development_seeds
        and item["opponent_pool"] != "holdout"
    ]
    holdout_rows = [item for item in rows if item not in development_rows]
    calibration_by_persona: dict[str, Any] = {}
    for persona_id in personas:
        persona_development = [
            item for item in development_rows if item["persona_id"] == persona_id
        ]
        persona_holdout = [
            item for item in holdout_rows if item["persona_id"] == persona_id
        ]
        development_scores = {
            condition: _paired_with_activity(persona_development, condition)
            for condition in PARETO_SPECS
        }
        development_gates = {
            condition: _reliability_gate(score)
            for condition, score in development_scores.items()
        }
        selected = next(
            (
                condition
                for condition in PARETO_SPECS
                if all(development_gates[condition].values())
            ),
            None,
        )
        holdout_score = (
            _paired_with_activity(persona_holdout, selected)
            if selected is not None
            else None
        )
        holdout_gate = (
            _reliability_gate(holdout_score)
            if holdout_score is not None
            else None
        )
        calibration_by_persona[persona_id] = {
            "development_pair_count": len(persona_development) // len(CONDITIONS),
            "holdout_pair_count": len(persona_holdout) // len(CONDITIONS),
            "development_scores": development_scores,
            "development_gates": development_gates,
            "selected_condition": selected,
            "holdout_score": holdout_score,
            "holdout_gate": holdout_gate,
            "promotion_recommended": bool(
                selected is not None
                and holdout_gate is not None
                and all(holdout_gate.values())
            ),
        }

    situation_counts = {
        condition: dict(
            sorted(
                Counter(
                    trace["pareto_situation"]
                    for item in rows
                    if item["condition"] == condition
                    for trace in item["round_trace"]
                    if trace["pareto_situation"] is not None
                ).items()
            )
        )
        for condition in PARETO_SPECS
    }
    selected_conditions = {
        item["selected_condition"]
        for item in calibration_by_persona.values()
        if item["selected_condition"] is not None
    }
    final_stage_complete = bool(
        selected_conditions
        and all(
            item["promotion_recommended"]
            for item in calibration_by_persona.values()
        )
    )
    engineering_checks = {
        "real_llm_calls_zero": True,
        "real_llm_tokens_zero": True,
        "real_llm_cost_zero": True,
        "at_least_ten_common_seeds": len(seeds) >= 10,
        "matrix_complete": len(rows)
        == len(seeds) * len(personas) * len(pool_ids) * len(CONDITIONS),
        "all_actions_legal": all(item["all_actions_legal"] for item in rows),
        "development_holdout_seed_disjoint": not (
            development_seeds & holdout_seeds
        ),
        "pareto_decisions_audited": all(
            trace["decision_hash"] is not None
            for item in rows
            if item["condition"] in PARETO_SPECS
            for trace in item["round_trace"]
        ),
        "paid_model_calls_disabled": True,
    }
    return {
        "experiment_schema_version": "stage6.4-pareto-reliability-v1.0.0",
        "evidence_level": "deterministic_ten_seed_zero_llm",
        "seeds": list(seeds),
        "development_seeds": sorted(development_seeds),
        "holdout_seeds": sorted(holdout_seeds),
        "personas": list(personas),
        "opponent_pools": list(pool_ids),
        "conditions": list(CONDITIONS),
        "rounds": rounds,
        "horizon_rounds": horizon_rounds,
        "scenario_count": scenario_count,
        "real_model_usage": {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost": 0,
            "currency": "CNY",
        },
        "pareto_specs": {
            key: value.model_dump(mode="json")
            for key, value in PARETO_SPECS.items()
        },
        "rows": rows,
        "by_condition": by_condition,
        "all_comparisons_vs_rule": all_comparisons,
        "all_reliability_gates": all_gates,
        "calibration_by_persona": calibration_by_persona,
        "pareto_situation_counts": situation_counts,
        "plan_cache_entry_count": len(plan_cache),
        "engineering_checks": engineering_checks,
        "engineering_passed": all(engineering_checks.values()),
        "strategic_reliability_stage_complete": final_stage_complete,
        "completion_rule": (
            "every Persona must select a Pareto spec on development and pass all "
            "enterprise-value, competition, regret, activity and holdout gates"
        ),
        "conclusion_limits": [
            "ten deterministic seeds are a reliability gate, not real-world calibration",
            "Pareto Planner remains experimental until every Persona passes holdout",
            "no paid LLM call was authorized or executed",
            "opponent v2 remains frozen",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(i) for i in range(1, 11)))
    parser.add_argument("--personas", default=",".join(PERSONAS))
    parser.add_argument("--opponent-pools", default="known,mixed,holdout")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--horizon-rounds", type=int, default=2)
    parser.add_argument("--scenario-count", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(
        seeds=parse_csv_ints(args.seeds),
        personas=parse_csv(args.personas),
        pool_ids=parse_csv(args.opponent_pools),
        rounds=args.rounds,
        horizon_rounds=args.horizon_rounds,
        scenario_count=args.scenario_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "engineering_passed": summary["engineering_passed"],
                "strategic_reliability_stage_complete": summary[
                    "strategic_reliability_stage_complete"
                ],
                "real_model_usage": summary["real_model_usage"],
                "calibration_by_persona": summary["calibration_by_persona"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
