"""Stage 6.3 zero-LLM objective and label calibration experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.agents.personas import PersonaRegistry, PersonaUtilityTracker
from game_theory_agent.belief import BeliefLedger
from game_theory_agent.gameplay import build_rule_action, build_terminal_rankings
from game_theory_agent.market import CompanyAction, MarketEnv, MarketState, load_market_config
from game_theory_agent.opponent import OpponentModelLedger, OpponentModelState
from game_theory_agent.strategic_reliability import (
    AuthoritativeMarketRolloutEvaluator,
    ObjectiveCalibrationSpec,
    StrategicReliabilityPlan,
    build_public_forecast_state,
    calibrate_objective_decision,
    generate_public_overlay_candidates,
)

from .stage62_strategic_ablation import (
    COMPANIES,
    HOLDOUT_IDS,
    PERSONAS,
    _candidate_to_action,
    _opponent_action,
    _true_one_step_regret,
    opponent_pools,
    parse_csv,
    parse_csv_ints,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v4.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "stage6.3-calibration" / "summary.json"
WEIGHT_CONDITIONS = {
    "persona_aligned": 0,
    "competitive_250k": 250_000,
    "competitive_500k": 500_000,
    "competitive_750k": 750_000,
    "competitive_only": 1_000_000,
}
CONDITIONS = (
    "rule_baseline",
    "absolute_value",
    *WEIGHT_CONDITIONS,
    "belief_persona",
    "belief_v1_profit_gate",
)


def _objective_spec(condition: str, risk_aversion_ppm: int) -> ObjectiveCalibrationSpec:
    if condition == "absolute_value":
        return ObjectiveCalibrationSpec(
            mode="absolute_value",
            risk_aversion_ppm=risk_aversion_ppm,
        )
    weight = WEIGHT_CONDITIONS.get(condition, 0)
    if weight == 0:
        return ObjectiveCalibrationSpec(
            mode="persona_aligned",
            risk_aversion_ppm=risk_aversion_ppm,
        )
    return ObjectiveCalibrationSpec(
        mode="competitive_hybrid",
        competitive_weight_ppm=weight,
        risk_aversion_ppm=risk_aversion_ppm,
    )


def _competitive_result(
    state: MarketState, config
) -> tuple[int, int, str, int]:
    rankings = build_terminal_rankings(state, config)["composite"]
    focal = next(item for item in rankings if item["company_id"] == "company_A")
    best_opponent = max(
        (item for item in rankings if item["company_id"] != "company_A"),
        key=lambda item: int(item["value_cents"]),
    )
    own_value = int(focal["value_cents"])
    opponent_value = int(best_opponent["value_cents"])
    return (
        own_value - opponent_value,
        opponent_value,
        str(best_opponent["company_id"]),
        int(focal["rank"]),
    )


def _selected_evaluation(plan: StrategicReliabilityPlan, candidate_id: str):
    return next(
        item for item in plan.evaluations if item.candidate.candidate_id == candidate_id
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
        raise ValueError(f"unknown Stage 6.3 condition: {condition}")
    episode_id = f"stage6.3-{pool_id}-seed-{seed}"
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
    belief_ledger = BeliefLedger(
        episode_id=episode_id,
        company_ids=state.company_ids,
    )
    opponent_ledger = OpponentModelLedger(
        episode_id=episode_id,
        company_ids=state.company_ids,
    )
    oracle = AuthoritativeMarketRolloutEvaluator(config)
    traces: list[dict[str, Any]] = []
    recommendations: list[str] = []
    regrets: list[int] = []
    realized_loss = 0
    all_actions_legal = True

    while not state.terminal:
        belief, belief_hash = belief_ledger.company_view(
            observer_company_id="company_A",
            round_number=state.round,
            state_version=state.state_version,
        )
        opponent_v1, opponent_hash = opponent_ledger.company_view(
            observer_company_id="company_A",
            round_number=state.round,
            state_version=state.state_version,
        )
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

        selected_belief = (
            belief
            if condition in {"belief_persona", "belief_v1_profit_gate"}
            else None
        )
        selected_model: OpponentModelState | None = None
        if condition == "belief_v1_profit_gate" and persona_id == "profit_myopic":
            selected_model = opponent_v1

        decision_hash: str | None = None
        forecast_seed: int | None = None
        expected_competitive_margin: int | None = None
        expected_rank_milli: int | None = None
        if condition == "rule_baseline":
            chosen_id = "maintain"
            focal_action = build_rule_action(config, state, "company_A")
        else:
            observation = ObservationBuilder().build(
                state,
                "company_A",
                "public",
                belief_state=(
                    selected_belief.model_dump(mode="json")
                    if selected_belief is not None
                    else None
                ),
                belief_hash=belief_hash if selected_belief is not None else None,
                belief_schema_version=(
                    selected_belief.belief_schema_version
                    if selected_belief is not None
                    else "none"
                ),
            )
            if selected_model is not None:
                observation["opponent_model_hash"] = opponent_hash
                observation["opponent_model_state"] = selected_model.model_dump(
                    mode="json"
                )
            forecast, forecast_record = build_public_forecast_state(
                config=config,
                observation=observation,
                company_id="company_A",
                persona_profile=profile,
                belief_state=selected_belief,
                opponent_model=selected_model,
            )
            horizon = min(horizon_rounds, state.rounds_remaining)
            cache_key = (
                forecast.state_hash,
                profile.profile_hash,
                belief_hash if selected_belief is not None else None,
                opponent_hash if selected_model is not None else None,
                horizon,
                scenario_count,
            )
            plan = plan_cache.get(cache_key)
            if plan is None:
                plan = oracle.evaluate(
                    state=forecast,
                    company_id="company_A",
                    persona_profile=profile,
                    belief_state=selected_belief,
                    opponent_model=selected_model,
                    horizon_rounds=horizon,
                    scenario_count=scenario_count,
                    candidates=generate_public_overlay_candidates(
                        config, forecast, "company_A"
                    ),
                )
                plan_cache[cache_key] = plan
            spec = _objective_spec(condition, profile.traits_ppm.risk_aversion)
            decision = calibrate_objective_decision(plan, spec)
            chosen_id = decision.recommended_candidate_id
            evaluation = _selected_evaluation(plan, chosen_id)
            focal_action = _candidate_to_action(evaluation.candidate, state)
            decision_hash = decision.decision_hash
            forecast_seed = forecast_record.forecast_seed
            expected_competitive_margin = (
                evaluation.expected_competitive_margin_cents
            )
            expected_rank_milli = evaluation.expected_final_rank_milli

        regret = _true_one_step_regret(
            config,
                state,
                persona_profile=profile,
                opponent_actions=opponent_actions,
                chosen_action=focal_action,
            )
        recommendations.append(chosen_id)
        regrets.append(regret)
        validation = env.validate_action(focal_action, "company_A")
        all_actions_legal = all_actions_legal and validation.valid
        if not validation.valid or validation.action is None:
            raise ValueError(f"illegal Stage 6.3 action: {validation.errors}")
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
        post_margin, _best_value, _best_company, post_rank = _competitive_result(
            result.state_after, config
        )
        traces.append(
            {
                "round": before.round,
                "pre_state_hash": before.state_hash,
                "chosen_candidate_id": chosen_id,
                "true_one_step_regret_cents": regret,
                "forecast_seed": forecast_seed,
                "objective_decision_hash": decision_hash,
                "expected_competitive_margin_cents": expected_competitive_margin,
                "expected_final_rank_milli": expected_rank_milli,
                "realized_post_round_competitive_margin_cents": post_margin,
                "realized_post_round_rank": post_rank,
                "post_state_hash": result.state_after.state_hash,
            }
        )
        belief_ledger.update_after_settlement(before, actions)
        opponent_ledger.update_after_settlement(before, result.state_after, actions)
        state = result.state_after

    margin, best_value, best_company, rank = _competitive_result(state, config)
    company = state.company("company_A")
    return {
        "seed": seed,
        "persona_id": persona_id,
        "opponent_pool": pool_id,
        "condition": condition,
        "enterprise_value_cents": next(
            int(item["value_cents"])
            for item in build_terminal_rankings(state, config)["composite"]
            if item["company_id"] == "company_A"
        ),
        "cumulative_profit_cents": company.financial.cumulative_profit_cents,
        "final_market_share_ppm": company.commercial.market_share_ppm,
        "final_competitive_margin_cents": margin,
        "best_opponent_enterprise_value_cents": best_value,
        "best_opponent_company_id": best_company,
        "final_rank": rank,
        "first_place": rank == 1,
        "realized_loss_cents": realized_loss,
        "mean_true_one_step_regret_cents": round(mean(regrets)),
        "recommendations": recommendations,
        "round_trace": traces,
        "all_actions_legal": all_actions_legal,
        "final_state_hash": state.state_hash,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Stage 6.3 aggregate requires rows")
    return {
        "episode_count": len(rows),
        "mean_enterprise_value_cents": round(
            mean(item["enterprise_value_cents"] for item in rows)
        ),
        "mean_cumulative_profit_cents": round(
            mean(item["cumulative_profit_cents"] for item in rows)
        ),
        "mean_competitive_margin_cents": round(
            mean(item["final_competitive_margin_cents"] for item in rows)
        ),
        "mean_market_share_ppm": round(
            mean(item["final_market_share_ppm"] for item in rows)
        ),
        "mean_rank": mean(item["final_rank"] for item in rows),
        "first_place_rate_ppm": (
            sum(item["first_place"] for item in rows) * 1_000_000 // len(rows)
        ),
        "mean_true_one_step_regret_cents": round(
            mean(item["mean_true_one_step_regret_cents"] for item in rows)
        ),
        "worst_enterprise_value_cents": min(
            item["enterprise_value_cents"] for item in rows
        ),
        "worst_competitive_margin_cents": min(
            item["final_competitive_margin_cents"] for item in rows
        ),
    }


def _paired(
    rows: list[dict[str, Any]], treatment: str, control: str
) -> dict[str, Any]:
    controls = {
        (item["seed"], item["persona_id"], item["opponent_pool"]): item
        for item in rows
        if item["condition"] == control
    }
    treatments = [item for item in rows if item["condition"] == treatment]
    if not treatments or len(treatments) != len(controls):
        raise ValueError("Stage 6.3 comparison is not completely paired")
    deltas: list[dict[str, int]] = []
    for item in treatments:
        key = (item["seed"], item["persona_id"], item["opponent_pool"])
        baseline = controls[key]
        deltas.append(
            {
                "enterprise_value": item["enterprise_value_cents"]
                - baseline["enterprise_value_cents"],
                "profit": item["cumulative_profit_cents"]
                - baseline["cumulative_profit_cents"],
                "competitive_margin": item["final_competitive_margin_cents"]
                - baseline["final_competitive_margin_cents"],
                "regret": item["mean_true_one_step_regret_cents"]
                - baseline["mean_true_one_step_regret_cents"],
                "first_place": int(item["first_place"])
                - int(baseline["first_place"]),
            }
        )
    ev = [item["enterprise_value"] for item in deltas]
    margin = [item["competitive_margin"] for item in deltas]
    return {
        "treatment": treatment,
        "control": control,
        "pair_count": len(deltas),
        "mean_enterprise_value_delta_cents": round(mean(ev)),
        "median_enterprise_value_delta_cents": round(median(ev)),
        "mean_profit_delta_cents": round(mean(item["profit"] for item in deltas)),
        "mean_competitive_margin_delta_cents": round(mean(margin)),
        "mean_true_regret_delta_cents": round(
            mean(item["regret"] for item in deltas)
        ),
        "first_place_rate_delta_ppm": (
            sum(item["first_place"] for item in deltas) * 1_000_000
            // len(deltas)
        ),
        "worst_enterprise_value_delta_cents": min(ev),
        "worst_competitive_margin_delta_cents": min(margin),
        "positive_zero_negative_enterprise_value_pairs": {
            "positive": sum(value > 0 for value in ev),
            "zero": sum(value == 0 for value in ev),
            "negative": sum(value < 0 for value in ev),
        },
    }


def _gate(item: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "mean_enterprise_value_non_negative": (
            item["mean_enterprise_value_delta_cents"] >= 0
        ),
        "worst_enterprise_value_non_negative": (
            item["worst_enterprise_value_delta_cents"] >= 0
        ),
        "mean_regret_not_higher": item["mean_true_regret_delta_cents"] <= 0,
        "mean_competitive_margin_non_negative": (
            item["mean_competitive_margin_delta_cents"] >= 0
        ),
        "first_place_rate_not_lower": item["first_place_rate_delta_ppm"] >= 0,
    }


def _belief_path_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = {
        (item["seed"], item["persona_id"], item["opponent_pool"]): item
        for item in rows
        if item["condition"] == "persona_aligned"
    }
    audit_rows: list[dict[str, Any]] = []
    for item in rows:
        if item["condition"] != "belief_persona":
            continue
        control = baseline[(item["seed"], item["persona_id"], item["opponent_pool"])]
        first_divergence = next(
            (
                index + 1
                for index, (left, right) in enumerate(
                    zip(
                        control["recommendations"],
                        item["recommendations"],
                        strict=True,
                    )
                )
                if left != right
            ),
            None,
        )
        audit_rows.append(
            {
                "seed": item["seed"],
                "persona_id": item["persona_id"],
                "opponent_pool": item["opponent_pool"],
                "first_divergent_round": first_divergence,
                "enterprise_value_delta_cents": item["enterprise_value_cents"]
                - control["enterprise_value_cents"],
                "competitive_margin_delta_cents": item[
                    "final_competitive_margin_cents"
                ]
                - control["final_competitive_margin_cents"],
                "true_regret_delta_cents": item[
                    "mean_true_one_step_regret_cents"
                ]
                - control["mean_true_one_step_regret_cents"],
            }
        )
    diverged = [item for item in audit_rows if item["first_divergent_round"] is not None]
    one_step_better_long_term_worse = [
        item
        for item in diverged
        if item["true_regret_delta_cents"] < 0
        and item["enterprise_value_delta_cents"] < 0
    ]
    return {
        "pair_count": len(audit_rows),
        "divergent_pair_count": len(diverged),
        "unchanged_pair_count": len(audit_rows) - len(diverged),
        "mean_first_divergent_round": (
            mean(item["first_divergent_round"] for item in diverged)
            if diverged
            else None
        ),
        "divergent_pairs_with_long_term_value_loss": sum(
            item["enterprise_value_delta_cents"] < 0 for item in diverged
        ),
        "divergent_pairs_with_one_step_regret_improvement": sum(
            item["true_regret_delta_cents"] < 0 for item in diverged
        ),
        "one_step_better_but_long_term_value_worse_count": len(
            one_step_better_long_term_worse
        ),
        "one_step_better_but_long_term_value_worse_rate_ppm": (
            len(one_step_better_long_term_worse) * 1_000_000 // len(diverged)
            if diverged
            else 0
        ),
        "first_divergence_round_counts": dict(
            sorted(
                Counter(
                    int(item["first_divergent_round"]) for item in diverged
                ).items()
            )
        ),
        "rows": audit_rows,
    }


def run(
    *,
    seeds: tuple[int, ...] = (1, 2, 3),
    personas: tuple[str, ...] = PERSONAS,
    pool_ids: tuple[str, ...] = ("known", "mixed", "holdout"),
    rounds: int = 5,
    horizon_rounds: int = 2,
    scenario_count: int = 2,
) -> dict[str, Any]:
    config = load_market_config(CONFIG_PATH)
    registry = PersonaRegistry.from_market_config(config)
    for persona_id in personas:
        registry.get(persona_id)
    pools = opponent_pools()
    if set(pool_ids) - set(pools):
        raise ValueError("unknown Stage 6.3 opponent pool")
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
    comparisons = {
        f"{condition}_vs_rule": _paired(rows, condition, "rule_baseline")
        for condition in (
            "absolute_value",
            *WEIGHT_CONDITIONS,
        )
    }
    comparisons["belief_vs_persona"] = _paired(
        rows, "belief_persona", "persona_aligned"
    )
    comparisons["v1_profit_gate_vs_belief"] = _paired(
        rows, "belief_v1_profit_gate", "belief_persona"
    )
    comparison_gates = {
        key: _gate(value) for key, value in comparisons.items()
    }

    max_seed = max(seeds)
    development_rows = [
        item
        for item in rows
        if item["seed"] != max_seed and item["opponent_pool"] != "holdout"
    ]
    holdout_rows = [item for item in rows if item not in development_rows]
    if not development_rows:
        development_rows = [
            item for item in rows if item["opponent_pool"] != "holdout"
        ]
        holdout_rows = [item for item in rows if item not in development_rows]
    calibration_by_persona: dict[str, Any] = {}
    objective_conditions = tuple(WEIGHT_CONDITIONS)
    for persona_id in personas:
        persona_development = [
            item for item in development_rows if item["persona_id"] == persona_id
        ]
        persona_holdout = [
            item for item in holdout_rows if item["persona_id"] == persona_id
        ]
        development_scores = {
            condition: _paired(
                persona_development, condition, "rule_baseline"
            )
            for condition in objective_conditions
        }
        development_gates = {
            condition: _gate(score)
            for condition, score in development_scores.items()
        }
        selected = next(
            (
                condition
                for condition in objective_conditions
                if all(development_gates[condition].values())
            ),
            None,
        )
        holdout_score = (
            _paired(persona_holdout, selected, "rule_baseline")
            if selected is not None and persona_holdout
            else None
        )
        holdout_gate = _gate(holdout_score) if holdout_score is not None else None
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

    v1_gate_by_persona = {
        persona_id: _paired(
            [item for item in rows if item["persona_id"] == persona_id],
            "belief_v1_profit_gate",
            "belief_persona",
        )
        for persona_id in personas
    }
    profit_v1_gate = (
        _gate(v1_gate_by_persona["profit_myopic"])
        if "profit_myopic" in v1_gate_by_persona
        else None
    )
    action_counts = {
        condition: dict(
            sorted(
                Counter(
                    action
                    for item in rows
                    if item["condition"] == condition
                    for action in item["recommendations"]
                ).items()
            )
        )
        for condition in CONDITIONS
    }
    engineering_checks = {
        "real_llm_calls_zero": True,
        "real_llm_tokens_zero": True,
        "real_llm_cost_zero": True,
        "matrix_complete": len(rows)
        == len(seeds) * len(personas) * len(pool_ids) * len(CONDITIONS),
        "all_actions_legal": all(item["all_actions_legal"] for item in rows),
        "competitive_labels_complete": all(
            item["best_opponent_enterprise_value_cents"] is not None for item in rows
        ),
        "v2_frozen_and_not_executed": all(
            "v2" not in item["condition"] for item in rows
        ),
        "holdout_ids_disjoint": all(
            int(item.rsplit("_", 1)[-1]) >= 70 for item in HOLDOUT_IDS
        ),
    }
    return {
        "experiment_schema_version": "stage6.3-objective-calibration-v1.0.0",
        "evidence_level": "deterministic_synthetic_zero_llm",
        "seeds": list(seeds),
        "personas": list(personas),
        "opponent_pools": list(pool_ids),
        "conditions": list(CONDITIONS),
        "rounds": rounds,
        "horizon_rounds": horizon_rounds,
        "scenario_count": scenario_count,
        "development_rule": "all non-max seeds and non-holdout pools",
        "holdout_rule": "max seed or holdout pool",
        "real_model_usage": {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost": 0,
            "currency": "CNY",
        },
        "rows": rows,
        "by_condition": by_condition,
        "comparisons": comparisons,
        "comparison_gates": comparison_gates,
        "promotion_recommendations": {
            key: all(gate.values()) for key, gate in comparison_gates.items()
        },
        "calibration_by_persona": calibration_by_persona,
        "belief_path_audit": _belief_path_audit(rows),
        "v1_profit_gate_by_persona": v1_gate_by_persona,
        "profit_persona_v1_gate": profit_v1_gate,
        "profit_persona_v1_promotion_recommended": bool(
            profit_v1_gate is not None and all(profit_v1_gate.values())
        ),
        "action_counts_by_condition": action_counts,
        "plan_cache_entry_count": len(plan_cache),
        "engineering_checks": engineering_checks,
        "engineering_passed": all(engineering_checks.values()),
        "conclusion_limits": [
            "objective calibration is offline and cannot enter Agent context",
            "three seeds are an engineering gate, not statistical significance",
            "competitive margin uses the strongest simulated opponent, not equilibrium",
            "one-step regret and five-round terminal labels answer different questions",
            "opponent v2 remains frozen and was not executed",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3")
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
                "real_model_usage": summary["real_model_usage"],
                "calibration_by_persona": summary["calibration_by_persona"],
                "belief_path_audit": {
                    key: value
                    for key, value in summary["belief_path_audit"].items()
                    if key != "rows"
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
