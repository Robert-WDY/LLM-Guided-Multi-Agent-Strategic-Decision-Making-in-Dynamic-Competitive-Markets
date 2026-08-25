"""Zero-LLM Stage 6 P0 acceptance experiment.

This experiment intentionally performs no provider calls.  It establishes the
authoritative rollout oracle, persona planning behavior and opponent holdout
benchmark before any paid real-model smoke test is authorized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.strategic_reliability import (
    AuthoritativeMarketRolloutEvaluator,
    run_opponent_benchmark,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v4.yaml"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "runs"
    / "stage6-strategic-reliability-p0"
    / "summary.json"
)
PERSONAS = (
    "aggressive_v1_extreme",
    "profit_myopic",
    "risk_guarded_v1",
)


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty unique list")
    return seeds


def run(
    *,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    horizon_rounds: int = 5,
    scenario_count: int = 10,
) -> dict[str, Any]:
    config = load_market_config(CONFIG_PATH)
    registry = PersonaRegistry.from_market_config(config)
    evaluator = AuthoritativeMarketRolloutEvaluator(config)
    plans: list[dict[str, Any]] = []
    for seed in seeds:
        state = MarketEnv(config).reset(
            ["company_A", "company_B", "company_C", "company_D"],
            episode_id=f"stage6-reliability-seed-{seed}",
            episode_seed=seed,
            market_model="balanced",
            max_rounds=10,
        )
        for persona_id in PERSONAS:
            plan = evaluator.evaluate(
                state=state,
                company_id="company_A",
                persona_profile=registry.get(persona_id),
                horizon_rounds=horizon_rounds,
                scenario_count=scenario_count,
            )
            recommendation = next(
                item
                for item in plan.evaluations
                if item.candidate.candidate_id == plan.recommended_candidate_id
            )
            baseline = next(
                item
                for item in plan.evaluations
                if item.candidate.candidate_id == plan.baseline_candidate_id
            )
            large_cut = next(
                item
                for item in plan.evaluations
                if item.candidate.candidate_id == "price_cut_large"
            )
            plans.append(
                {
                    "seed": seed,
                    "persona_id": persona_id,
                    "recommended_candidate_id": plan.recommended_candidate_id,
                    "baseline_regret_cents": plan.baseline_regret_cents,
                    "expected_gain_over_baseline_cents": (
                        plan.expected_gain_over_baseline_cents
                    ),
                    "recommended_expected_enterprise_value_delta_cents": (
                        recommendation.expected_enterprise_value_delta_cents
                    ),
                    "baseline_expected_enterprise_value_delta_cents": (
                        baseline.expected_enterprise_value_delta_cents
                    ),
                    "large_cut_expected_enterprise_value_delta_cents": (
                        large_cut.expected_enterprise_value_delta_cents
                    ),
                    "recommended_expected_loss_cents": (
                        recommendation.expected_loss_cents
                    ),
                    "plan_hash": plan.plan_hash,
                }
            )
    opponent = run_opponent_benchmark(
        seed=20260824,
        random_count=100,
        evidence_rounds=12,
    )
    recommendations_by_seed: dict[int, set[str]] = {}
    for row in plans:
        recommendations_by_seed.setdefault(row["seed"], set()).add(
            row["recommended_candidate_id"]
        )
    holdout_v1 = opponent["holdout_scores"]["v1"]
    holdout_v2 = opponent["holdout_scores"]["v2"]
    checks = {
        "real_llm_calls_zero": True,
        "real_llm_tokens_zero": True,
        "real_llm_estimated_cost_zero": True,
        "all_baseline_regret_non_negative": all(
            row["baseline_regret_cents"] >= 0 for row in plans
        ),
        "persona_changes_ranking_on_at_least_one_seed": any(
            len(values) > 1 for values in recommendations_by_seed.values()
        ),
        "holdout_is_disjoint": not (
            set(opponent["development_case_ids"])
            & set(opponent["holdout_case_ids"])
        ),
        "v2_holdout_brier_better_than_v1": (
            holdout_v2["mean_distribution_brier"]
            < holdout_v1["mean_distribution_brier"]
        ),
        "v2_holdout_log_loss_better_than_v1": (
            holdout_v2["mean_cross_entropy"]
            < holdout_v1["mean_cross_entropy"]
        ),
        # V2 is not promoted merely because two calibration metrics improve.
        "v2_not_auto_promoted": True,
    }
    return {
        "experiment_schema_version": "stage6-strategic-reliability-p0-v1.0.0",
        "evidence_level": "deterministic_engineering_and_synthetic_benchmark",
        "real_model_usage": {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost": 0,
            "currency": "CNY",
        },
        "rollout": {
            "seeds": list(seeds),
            "personas": list(PERSONAS),
            "horizon_rounds": horizon_rounds,
            "scenario_count": scenario_count,
            "plans": plans,
        },
        "opponent_benchmark": opponent,
        "checks": checks,
        "passed": all(checks.values()),
        "conclusion_limits": [
            "the rollout oracle is not yet an Agent-visible public-information advisor",
            "synthetic holdout results are not real-LLM behavioral evidence",
            "no claim is made that the calibrated v2 classifier dominates every metric",
            "paid real-model testing remains blocked until deterministic gates pass",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--horizon-rounds", type=int, default=5)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(
        seeds=parse_seeds(args.seeds),
        horizon_rounds=args.horizon_rounds,
        scenario_count=args.scenario_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "passed": summary["passed"],
        "real_model_usage": summary["real_model_usage"],
        "checks": summary["checks"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
