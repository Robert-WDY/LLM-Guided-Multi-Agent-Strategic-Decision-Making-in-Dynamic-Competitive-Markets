"""Deterministic Game Theory Enhancement engineering benchmark.

This benchmark is deliberately synthetic.  It proves contracts, replayable
calculations, prediction separation, finite-action regret calculation and
strategic adaptation.  It does not claim equilibrium discovery or real-LLM
behavioral improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from game_theory_agent.advisor import BayesianStrategyAdvisor
from game_theory_agent.belief import (
    BeliefState,
    OpponentPriceBelief,
    PriceDirectionDistribution,
)
from game_theory_agent.opponent import (
    OpponentModelState,
    PublicStrategyEvidence,
    build_strategy_model,
    compute_opponent_model_hash,
)
from game_theory_agent.repeated_game import RepeatedGameStrategist
from game_theory_agent.utility_inference import (
    OpponentUtilityInferer,
    strategy_utility_template,
)


STRATEGIES = ("growth", "profit", "defensive", "cooperative")


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(kind: str) -> list[PublicStrategyEvidence]:
    result = []
    for round_number in range(1, 11):
        if kind == "growth":
            values = ("price_cut", 9_000, 20_000, -1_000, 0, 4_000)
        elif kind == "profit":
            values = ("price_raise", 10_500, 0, 0, 0, 3_000)
        elif kind == "defensive":
            values = ("maintain", 10_000, -10_000, 10_000, 0, 3_000)
        else:
            values = ("maintain", 10_000, 0, 10_000, 100_000, 3_000)
        direction, price, share, reputation, contribution, sales = values
        result.append(
            PublicStrategyEvidence(
                evidence_id=f"benchmark:{kind}:{round_number}",
                episode_id=f"benchmark-{kind}",
                settled_round=round_number,
                target_company_id="company_B",
                previous_price_cents=10_000,
                settled_price_cents=price,
                price_direction=direction,
                market_share_delta_ppm=share,
                public_sales_orders=sales,
                reputation_delta_ppm=reputation,
                public_shared_resilience_contribution_cents=contribution,
            )
        )
    return result


def _belief(kind: str) -> BeliefState:
    probabilities = {
        "growth": (800_000, 150_000, 50_000),
        "profit": (50_000, 150_000, 800_000),
        "defensive": (100_000, 800_000, 100_000),
        "cooperative": (200_000, 600_000, 200_000),
    }[kind]
    direction = {
        "growth": "price_cut",
        "profit": "price_raise",
        "defensive": "maintain",
        "cooperative": "maintain",
    }[kind]
    counts = {key: (10 if key == direction else 0) for key in (
        "price_cut", "maintain", "price_raise"
    )}
    return BeliefState(
        episode_id=f"benchmark-{kind}",
        observer_company_id="company_A",
        prediction_target_round=11,
        state_version=10,
        public_evidence_through_round=10,
        opponent_beliefs={
            "company_B": OpponentPriceBelief(
                opponent_company_id="company_B",
                prediction_target_round=11,
                evidence_count=10,
                latest_evidence_round=10,
                latest_observed_direction=direction,
                observed_counts=counts,
                next_price_direction=PriceDirectionDistribution(
                    price_cut_ppm=probabilities[0],
                    maintain_ppm=probabilities[1],
                    price_raise_ppm=probabilities[2],
                ),
            )
        },
    )


def _baseline_distribution(kind: str) -> dict[str, int]:
    observed = _belief(kind).opponent_beliefs[
        "company_B"
    ].latest_observed_direction
    mapped = {
        "price_cut": "growth",
        "price_raise": "profit",
        "maintain": "defensive",
    }[observed]
    return {strategy: (700_000 if strategy == mapped else 100_000) for strategy in STRATEGIES}


def _scores(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    accuracy = sum(row[f"{key}_prediction"] == row["truth"] for row in rows)
    brier = 0.0
    log_loss = 0.0
    for row in rows:
        distribution = row[f"{key}_distribution_ppm"]
        for strategy in STRATEGIES:
            probability = distribution[strategy] / 1_000_000
            target = 1.0 if strategy == row["truth"] else 0.0
            brier += (probability - target) ** 2
        log_loss -= math.log(
            max(1e-12, distribution[row["truth"]] / 1_000_000)
        )
    return {
        "accuracy_ppm": accuracy * 1_000_000 // len(rows),
        "mean_multiclass_brier": brier / len(rows),
        "mean_log_loss": log_loss / len(rows),
    }


def run(output: Path, closed_loop_summary: Path | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    advice_by_kind: dict[str, Any] = {}
    utility_distances: dict[str, int] = {}
    deterministic_hashes: dict[str, str] = {}
    for kind in STRATEGIES:
        strategy = build_strategy_model("company_B", _evidence(kind))
        model = OpponentModelState(
            episode_id=f"benchmark-{kind}",
            observer_company_id="company_A",
            prediction_target_round=11,
            state_version=10,
            public_evidence_through_round=10,
            opponent_models={"company_B": strategy},
        )
        model_hash = compute_opponent_model_hash(model)
        deterministic_hashes[kind] = model_hash
        utility, _ = OpponentUtilityInferer().infer(model)
        inferred = utility.opponent_utilities["company_B"]
        target = strategy_utility_template(kind)
        utility_distances[kind] = sum(
            abs(getattr(inferred, key).mean_ppm - value)
            for key, value in target.items()
        )
        advice = BayesianStrategyAdvisor().advise(
            belief_state=_belief(kind),
            opponent_model=model,
            utility_inference=utility,
            own_company={
                "commercial": {
                    "price_cents": 10_000,
                    "market_share_ppm": 250_000,
                },
                "operations": {"actual_unit_cost_cents": 7_000},
            },
            action_constraints={
                "bounds": {"price_cents": {"min": 8_000, "max": 12_000}}
            },
        )
        advice_by_kind[kind] = advice
        distribution = strategy.strategy_distribution.model_dump(mode="json")
        model_distribution = {
            key: distribution[f"{key}_ppm"] for key in STRATEGIES
        }
        baseline = _baseline_distribution(kind)
        rows.append(
            {
                "truth": kind,
                "opponent_model_prediction": strategy.strategy_distribution.top_strategy,
                "opponent_model_distribution_ppm": model_distribution,
                "price_frequency_prediction": max(baseline, key=baseline.get),
                "price_frequency_distribution_ppm": baseline,
                "confidence_ppm": strategy.confidence_ppm,
            }
        )

    opponent_scores = _scores(rows, "opponent_model")
    baseline_scores = _scores(rows, "price_frequency")
    aggressive = advice_by_kind["growth"]
    conservative = advice_by_kind["defensive"]
    aggressive_best = max(
        item.expected_utility_proxy for item in aggressive.candidate_actions
    )
    aggressive_maintain = next(
        item.expected_utility_proxy
        for item in aggressive.candidate_actions
        if item.action_label == "maintain"
    )
    baseline_regret = aggressive_best - aggressive_maintain

    strategist = RepeatedGameStrategist()
    trusted, _ = strategist.build(
        episode_id="benchmark-repeated",
        observer_company_id="company_A",
        round_number=5,
        cooperation_view={"cooperation_memory": {"company_B": {
            "credibility_ppm": 800_000,
            "accepted_by_opponent": 3,
            "fulfilled_by_opponent": 3,
            "partial_betrayals_by_opponent": 0,
            "betrayed_by_opponent": 0,
        }}},
    )
    betrayed, _ = strategist.build(
        episode_id="benchmark-repeated",
        observer_company_id="company_A",
        round_number=5,
        cooperation_view={"cooperation_memory": {"company_B": {
            "credibility_ppm": 200_000,
            "accepted_by_opponent": 3,
            "fulfilled_by_opponent": 1,
            "partial_betrayals_by_opponent": 0,
            "betrayed_by_opponent": 2,
        }}},
    )
    trusted_strategy = trusted.opponent_strategies["company_B"]
    betrayed_strategy = betrayed.opponent_strategies["company_B"]

    closed_loop = None
    if closed_loop_summary is not None:
        closed_loop = json.loads(closed_loop_summary.read_text(encoding="utf-8"))
    checks = {
        "same_public_evidence_same_hash": all(
            deterministic_hashes[kind]
            == compute_opponent_model_hash(
                OpponentModelState(
                    episode_id=f"benchmark-{kind}",
                    observer_company_id="company_A",
                    prediction_target_round=11,
                    state_version=10,
                    public_evidence_through_round=10,
                    opponent_models={
                        "company_B": build_strategy_model(
                            "company_B", _evidence(kind)
                        )
                    },
                )
            )
            for kind in STRATEGIES
        ),
        "opponent_accuracy_better_than_price_frequency": (
            opponent_scores["accuracy_ppm"] > baseline_scores["accuracy_ppm"]
        ),
        "opponent_brier_better_than_price_frequency": (
            opponent_scores["mean_multiclass_brier"]
            < baseline_scores["mean_multiclass_brier"]
        ),
        "opponent_log_loss_better_than_price_frequency": (
            opponent_scores["mean_log_loss"]
            < baseline_scores["mean_log_loss"]
        ),
        "advisor_improves_regret_in_growth_scenario": baseline_regret > 0,
        "advisor_is_non_binding_and_not_nash": all(
            item.recommendation_is_non_binding
            and item.approximate_best_response
            and not item.claims_nash_equilibrium
            for item in advice_by_kind.values()
        ),
        "aggressive_vs_conservative_action_differs": (
            aggressive.recommended_action != conservative.recommended_action
        ),
        "betrayal_changes_repeated_game_response": (
            trusted_strategy.recommended_stance == "cooperate"
            and betrayed_strategy.recommended_stance == "permanent_refusal"
            and trusted_strategy.contribution_multiplier_ppm
            > betrayed_strategy.contribution_multiplier_ppm
        ),
        "strategic_hidden_state_leak_zero": all(
            not getattr(
                advice_by_kind[kind], "uses_hidden_opponent_state"
            )
            for kind in STRATEGIES
        ),
        "closed_loop_all_replays_passed": (
            closed_loop is None
            or (
                closed_loop.get("passed") is True
                and closed_loop.get("game_theory_replay", {}).get(
                    "hidden_state_leak_count"
                ) == 0
            )
        ),
    }
    summary = {
        "acceptance_schema_version": "game-theory-enhancement-v1.0.0",
        "benchmark_kind": "deterministic_synthetic_engineering_acceptance",
        "synthetic_type_rows": rows,
        "prediction_scores": {
            "opponent_model": opponent_scores,
            "price_frequency_baseline": baseline_scores,
        },
        "utility_weight_l1_distance_ppm": utility_distances,
        "advisor_ablation": {
            "scenario": "growth_opponent",
            "no_advisor_action": "maintain",
            "advisor_action": aggressive.recommended_action,
            "no_advisor_regret_proxy": baseline_regret,
            "advisor_regret_proxy": 0,
            "metric_scope": "advisor_internal_expected_utility_proxy",
        },
        "strategic_adaptation": {
            "growth_opponent_action": aggressive.recommended_action,
            "defensive_opponent_action": conservative.recommended_action,
        },
        "repeated_game": {
            "trusted_stance": trusted_strategy.recommended_stance,
            "betrayed_stance": betrayed_strategy.recommended_stance,
            "trusted_contribution_multiplier_ppm": trusted_strategy.contribution_multiplier_ppm,
            "betrayed_contribution_multiplier_ppm": betrayed_strategy.contribution_multiplier_ppm,
        },
        "closed_loop_acceptance": (
            {
                "path": str(closed_loop_summary.resolve()),
                "sha256": _file_hash(closed_loop_summary),
                "passed": closed_loop["passed"],
                "protocol_checks": closed_loop["protocol_checks"],
                "game_theory_replay": closed_loop["game_theory_replay"],
            }
            if closed_loop is not None and closed_loop_summary is not None
            else None
        ),
        "checks": checks,
        "passed": all(checks.values()),
        "research_boundary": (
            "Engineering and synthetic directional evidence only. It does not "
            "establish Nash equilibrium, optimality, real-LLM understanding, "
            "or universal profit improvement."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--closed-loop-summary", type=Path)
    args = parser.parse_args()
    summary = run(args.output, args.closed_loop_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
