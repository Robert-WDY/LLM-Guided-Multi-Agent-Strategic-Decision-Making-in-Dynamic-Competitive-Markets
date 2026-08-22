"""Cross-score every Persona Pilot action under controlled settlements."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from game_theory_agent.agents import PersonaProfile, load_persona_registry
from game_theory_agent.experiments.persona_pilot import (
    _action_vector,
    _distance,
    build_scenarios,
)
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import CompanyAction, MarketEnv, MarketState, load_market_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _medoid(rows: list[dict[str, Any]], config: Any) -> dict[str, Any]:
    if len(rows) == 1:
        return rows[0]
    vectors = [_action_vector(row["final_action"], config) for row in rows]
    scores = [
        mean(
            _distance(vector, other)
            for other_index, other in enumerate(vectors)
            if other_index != index
        )
        for index, vector in enumerate(vectors)
    ]
    return rows[min(range(len(rows)), key=scores.__getitem__)]


def _condition_id(row: dict[str, Any]) -> str:
    return str(row.get("condition_id", row["persona_id"]))


def _profile(row: dict[str, Any], registry: Any) -> PersonaProfile:
    payload = row.get("persona_profile")
    if payload is not None:
        return PersonaProfile.model_validate(payload)
    return registry.get(str(row["persona_id"]))


def _scenario_states(
    experiment_dir: Path, config: Any
) -> dict[str, MarketState]:
    manifest_path = experiment_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            str(item["scenario_id"]): MarketState.from_dict(item["state"])
            for item in manifest["scenarios"]
        }
    return {item.scenario_id: item.state for item in build_scenarios(config)}


def _settle(
    row: dict[str, Any], state: MarketState, config: Any
) -> tuple[MarketState, dict[str, Any]]:
    actions = {
        company_id: build_rule_action(config, state, company_id)
        for company_id in state.company_ids
        if company_id != "company_A"
    }
    actions["company_A"] = CompanyAction.from_dict(row["final_action"])
    env = MarketEnv(config)
    env.load_state(state)
    result = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions
    )
    company = result.state_after.company("company_A")
    return result.state_after, {
        "condition_id": _condition_id(row),
        "persona_id": row["persona_id"],
        "ablation_mode": row.get("ablation_mode", "full"),
        "repetition": row["repetition"],
        "round_profit_cents": company.financial.round_profit_cents,
        "cash_balance_cents": company.financial.cash_balance_cents,
        "market_share_ppm": company.commercial.market_share_ppm,
        "sales_orders": company.commercial.sales_orders,
        "state_hash": result.state_after.state_hash,
    }


def analyze(experiment_dir: Path) -> dict[str, Any]:
    config_path = Path(
        os.environ.get(
            "MARKET_CONFIG_PATH", PROJECT_ROOT / "configs" / "market_v4.yaml"
        )
    )
    config = load_market_config(config_path)
    registry = load_persona_registry(config_path)
    scenarios = _scenario_states(experiment_dir, config)
    rows = [
        json.loads(line)
        for line in (experiment_dir / "decisions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    successful = [
        row for row in rows if row.get("success") and row.get("final_action")
    ]
    scenario_results: dict[str, Any] = {}
    aggregate_regrets: list[float] = []
    aggregate_wins = 0
    aggregate_decisions = 0
    aggregate_alignments: list[float] = []

    for scenario_id, state in scenarios.items():
        scenario_rows = [
            row for row in successful if row["scenario_id"] == scenario_id
        ]
        if not scenario_rows:
            continue
        if {row["state_hash"] for row in scenario_rows} != {state.state_hash}:
            raise ValueError(f"{scenario_id} rows do not match reconstructed state")
        conditions = tuple(sorted({_condition_id(row) for row in scenario_rows}))
        profiles = {
            condition: _profile(
                next(
                    row
                    for row in scenario_rows
                    if _condition_id(row) == condition
                ),
                registry,
            )
            for condition in conditions
        }
        settled = [(row, *_settle(row, state, config)) for row in scenario_rows]
        candidate_outcomes = [outcome for _row, _state, outcome in settled]

        utility_matrix: dict[str, dict[str, int]] = {}
        condition_metrics: dict[str, Any] = {}
        for evaluator_condition in conditions:
            evaluator = registry.evaluator(profiles[evaluator_condition])
            scored = [
                (
                    _condition_id(row),
                    evaluator.evaluate(
                        state, outcome_state, "company_A"
                    ).round_utility_ppm,
                )
                for row, outcome_state, _outcome in settled
            ]
            scores_by_condition = {
                action_condition: [
                    score
                    for candidate_condition, score in scored
                    if candidate_condition == action_condition
                ]
                for action_condition in conditions
            }
            utility_matrix[evaluator_condition] = {
                condition: round(mean(values))
                for condition, values in scores_by_condition.items()
            }
            best = max(score for _condition, score in scored)
            own = scores_by_condition[evaluator_condition]
            alternatives = [
                score
                for condition, score in scored
                if condition != evaluator_condition
            ]
            regrets = [best - score for score in own]
            wins = sum(score == best for score in own)
            baseline = scores_by_condition.get("none")
            alignment = mean(own) - mean(alternatives) if alternatives else 0.0
            condition_metrics[evaluator_condition] = {
                "sample_count": len(own),
                "mean_own_utility_ppm": round(mean(own), 2),
                "best_candidate_utility_ppm": best,
                "mean_regret_ppm": round(mean(regrets), 2),
                "max_regret_ppm": max(regrets),
                "strict_optimal_count": wins,
                "strict_optimal_rate": round(wins / len(own), 4),
                "alignment_vs_other_mean_ppm": round(alignment, 2),
                "utility_delta_vs_none_ppm": (
                    round(mean(own) - mean(baseline), 2)
                    if baseline is not None and evaluator_condition != "none"
                    else None
                ),
            }
            is_research_persona = (
                evaluator_condition != "none" and ":" not in evaluator_condition
            )
            if is_research_persona:
                aggregate_regrets.extend(regrets)
                aggregate_wins += wins
                aggregate_decisions += len(own)
                aggregate_alignments.append(alignment)

        representatives = {
            condition: _medoid(
                [
                    row
                    for row in scenario_rows
                    if _condition_id(row) == condition
                ],
                config,
            )
            for condition in conditions
        }
        representative_outcomes = {}
        for condition, row in representatives.items():
            _outcome_state, outcome = _settle(row, state, config)
            representative_outcomes[condition] = {
                key: value
                for key, value in outcome.items()
                if key not in {"condition_id", "persona_id", "ablation_mode"}
            }
        scenario_results[scenario_id] = {
            "candidate_outcomes": candidate_outcomes,
            "representative_outcomes": representative_outcomes,
            "utility_matrix_mean_ppm": utility_matrix,
            "condition_metrics": condition_metrics,
            "strict_optimal_count": sum(
                metric["strict_optimal_count"]
                for condition, metric in condition_metrics.items()
                if condition != "none" and ":" not in condition
            ),
            "strict_decision_count": sum(
                metric["sample_count"]
                for condition, metric in condition_metrics.items()
                if condition != "none" and ":" not in condition
            ),
        }

    result = {
        "analysis_schema_version": "persona-pilot-counterfactual-v2.0.0",
        "method": "all_samples_same_state_same_seed_rule_opponents_cross_scoring",
        "experiment_id": experiment_dir.name,
        "analysis_config_version": config.config_version,
        "scenario_results": scenario_results,
        "aggregate_research_personas": {
            "mean_regret_ppm": (
                round(mean(aggregate_regrets), 2) if aggregate_regrets else None
            ),
            "max_regret_ppm": max(aggregate_regrets) if aggregate_regrets else None,
            "strict_optimal_count": aggregate_wins,
            "decision_count": aggregate_decisions,
            "strict_optimal_rate": (
                round(aggregate_wins / aggregate_decisions, 4)
                if aggregate_decisions
                else None
            ),
            "mean_alignment_vs_other_mean_ppm": (
                round(mean(aggregate_alignments), 2)
                if aggregate_alignments
                else None
            ),
        },
    }
    output = experiment_dir / "counterfactual_v2.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = analyze(args.experiment_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
