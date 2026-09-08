"""Development/holdout empirical-game tournament for the final market."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from game_theory_agent.market import load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.self_play import (
    FrozenPromotionCandidate,
    operational_strategy_ids,
    role_rotated_matchups,
    strategy_versions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v6_final.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "final-market-self-play-v1"
DEVELOPMENT_SEEDS = (98001, 98002, 98003)
HOLDOUT_SEEDS = (99201, 99204)
ROUNDS = 20
BASE_POPULATION_STRATEGY_IDS = tuple(
    item.strategy_id
    for item in strategy_versions()
    if item.strategy_id != "contextual_defender_v1"
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _matrix(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["resident_strategy_id"]),
                str(row["mutant_strategy_id"]),
            )
        ].append(row)
    matrix = []
    for (resident, mutant), selected in sorted(grouped.items()):
        values = [int(row["focal_enterprise_value_cents"]) for row in selected]
        advantages = [
            int(row["focal_relative_advantage_cents"]) for row in selected
        ]
        matrix.append(
            {
                "resident_strategy_id": resident,
                "mutant_strategy_id": mutant,
                "sample_count": len(selected),
                "mean_focal_enterprise_value_cents": round(mean(values)),
                "worst_focal_enterprise_value_cents": min(values),
                "mean_relative_advantage_cents": round(mean(advantages)),
                "first_place_rate_ppm": (
                    sum(int(row["focal_rank"]) == 1 for row in selected)
                    * 1_000_000
                    // len(selected)
                ),
                "survival_rate_ppm": (
                    sum(bool(row["focal_survived"]) for row in selected)
                    * 1_000_000
                    // len(selected)
                ),
            }
        )
    return matrix


def _selection_metrics(
    matrix: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for strategy_id in operational_strategy_ids():
        cells = [
            row
            for row in matrix
            if row["mutant_strategy_id"] == strategy_id
        ]
        if not cells:
            continue
        result.append(
            {
                "strategy_id": strategy_id,
                "minimum_resident_mean_enterprise_value_cents": min(
                    int(row["mean_focal_enterprise_value_cents"])
                    for row in cells
                ),
                "mean_enterprise_value_cents": round(
                    mean(
                        int(row["mean_focal_enterprise_value_cents"])
                        for row in cells
                    )
                ),
                "minimum_survival_rate_ppm": min(
                    int(row["survival_rate_ppm"]) for row in cells
                ),
                "mean_first_place_rate_ppm": round(
                    mean(int(row["first_place_rate_ppm"]) for row in cells)
                ),
            }
        )
    return result


def _select_candidate(metrics: Sequence[Mapping[str, Any]]) -> str:
    selected = max(
        metrics,
        key=lambda row: (
            int(row["minimum_survival_rate_ppm"]),
            int(row["minimum_resident_mean_enterprise_value_cents"]),
            int(row["mean_enterprise_value_cents"]),
            int(row["mean_first_place_rate_ppm"]),
            str(row["strategy_id"]),
        ),
    )
    return str(selected["strategy_id"])


def _best_responses(
    matrix: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    residents = sorted({str(row["resident_strategy_id"]) for row in matrix})
    return {
        resident: str(
            max(
                (
                    row
                    for row in matrix
                    if row["resident_strategy_id"] == resident
                ),
                key=lambda row: (
                    int(row["mean_focal_enterprise_value_cents"]),
                    str(row["mutant_strategy_id"]),
                ),
            )["mutant_strategy_id"]
        )
        for resident in residents
    }


def _paired_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    baseline: str,
    value_getter: Any,
) -> list[int]:
    indexed = {
        (
            int(row["seed"]),
            str(row["resident_strategy_id"]),
            str(row["mutant_strategy_id"]),
            str(row["focal_company_id"]),
        ): row
        for row in rows
    }
    result = []
    for key, row in indexed.items():
        if key[2] != treatment:
            continue
        control_key = (key[0], key[1], baseline, key[3])
        if control_key in indexed:
            result.append(
                int(value_getter(row)) - int(value_getter(indexed[control_key]))
            )
    return result


def _phenomena(
    rows: Sequence[Mapping[str, Any]],
    matrix: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    unique_episodes = {
        (
            str(row["episode"]["episode_id"]),
            str(row["episode"]["final_state_hash"]),
        ): row["episode"]
        for row in rows
    }
    episodes = list(unique_episodes.values())
    best = _best_responses(matrix)

    indexed = {
        (
            int(row["seed"]),
            str(row["resident_strategy_id"]),
            str(row["mutant_strategy_id"]),
            str(row["focal_company_id"]),
        ): row
        for row in rows
    }
    free_rider_short: list[int] = []
    free_rider_welfare: list[int] = []
    for key, row in indexed.items():
        if (
            key[1] != "resilience_cooperator"
            or key[2] != "free_rider"
        ):
            continue
        control = indexed.get(
            (key[0], key[1], "resilience_cooperator", key[3])
        )
        if control is None:
            continue
        company_id = key[3]
        free_rider_short.append(
            int(row["episode"]["first_round_profit_cents"][company_id])
            - int(control["episode"]["first_round_profit_cents"][company_id])
        )
        free_rider_welfare.append(
            int(row["episode"]["cumulative_social_welfare_proxy_cents"])
            - int(control["episode"]["cumulative_social_welfare_proxy_cents"])
        )
    statuses = [
        status
        for episode in episodes
        for status in episode["coordination_statuses"]
    ]
    return {
        "unique_episode_count": len(episodes),
        "price_war_episode_count": sum(
            int(episode["price_war_rounds"]) > 0 for episode in episodes
        ),
        "bankruptcy_episode_count": sum(
            bool(episode["exited_company_ids"]) for episode in episodes
        ),
        "monopoly_episode_count": sum(
            "monopoly" in episode["concentration_regimes"]
            for episode in episodes
        ),
        "threshold_success_episode_count": sum(
            episode["threshold_project_status"] == "succeeded"
            for episode in episodes
        ),
        "threshold_failure_episode_count": sum(
            episode["threshold_project_status"] == "failed"
            for episode in episodes
        ),
        "mutual_aid_episode_count": sum(
            int(episode["mutual_aid_orders"]) > 0 for episode in episodes
        ),
        "coordination_honor_count": statuses.count("honored"),
        "coordination_betrayal_count": sum(
            status.startswith("undercut_by_") for status in statuses
        ),
        "regulatory_detection_count": sum(
            int(episode["detected_coordination_count"])
            for episode in episodes
        ),
        "best_response_by_resident": best,
        "unique_best_response_count": len(set(best.values())),
        "mean_free_rider_first_round_profit_advantage_cents": (
            round(mean(free_rider_short)) if free_rider_short else None
        ),
        "mean_free_rider_social_welfare_externality_cents": (
            round(mean(free_rider_welfare)) if free_rider_welfare else None
        ),
    }


def _run_rows(seeds: Sequence[int], split: str) -> list[dict[str, Any]]:
    config = load_market_config(CONFIG_PATH)
    ids = BASE_POPULATION_STRATEGY_IDS
    rows = role_rotated_matchups(
        config,
        seeds=seeds,
        strategy_ids=ids,
        rounds=ROUNDS,
        experiment_id=f"final-market-self-play-{split}-v1",
    )
    for row in rows:
        row["split"] = split
    return rows


def run_development(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    config = load_market_config(CONFIG_PATH)
    rows = _run_rows(DEVELOPMENT_SEEDS, "development")
    matrix = _matrix(rows)
    selection = _selection_metrics(matrix)
    candidate_id = _select_candidate(selection)
    phenomena = _phenomena(rows, matrix)
    evidence = {
        "evidence_schema_version": "final-market-self-play-development-v1.0.0",
        "config_id": config.config_id,
        "config_sha256": config.config_sha256,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "excluded_holdout_seeds": list(HOLDOUT_SEEDS),
        "rounds": ROUNDS,
        "strategy_versions": [
            item.model_dump(mode="json") for item in strategy_versions()
        ],
        "matrix": matrix,
        "selection_metrics": selection,
        "selected_candidate_strategy_id": candidate_id,
        "phenomena": phenomena,
        "invariants": {
            "replay_all": all(row["episode"]["replay_passed"] for row in rows),
            "demand_closure_all": all(
                row["episode"]["demand_closure_passed"] for row in rows
            ),
            "cash_non_negative_all": all(
                row["episode"]["non_negative_cash"] for row in rows
            ),
        },
        "provider_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    evidence_hash = sha256_hash(evidence)
    selected_version = next(
        item for item in strategy_versions() if item.strategy_id == candidate_id
    )
    promotion_payload = {
        "promotion_schema_version": "self-play-promotion-candidate-v1.0.0",
        "config_sha256": config.config_sha256,
        "development_result_hash": evidence_hash,
        "candidate_strategy_id": candidate_id,
        "candidate_strategy_hash": selected_version.strategy_hash,
        "selection_rule": (
            "lexicographic: minimum survival, worst-resident mean enterprise "
            "value, overall mean enterprise value, first-place rate"
        ),
        "development_seeds": DEVELOPMENT_SEEDS,
        "excluded_holdout_seeds": HOLDOUT_SEEDS,
        "uses_holdout_for_selection": False,
        "promotion_hash": "pending",
    }
    promotion_payload["promotion_hash"] = sha256_hash(
        {
            key: value
            for key, value in promotion_payload.items()
            if key != "promotion_hash"
        }
    )
    promotion = FrozenPromotionCandidate.model_validate(promotion_payload)
    summary = {
        **evidence,
        "development_evidence_hash": evidence_hash,
        "promotion_candidate": promotion.model_dump(mode="json"),
    }
    summary["result_hash"] = sha256_hash(summary)
    _write_jsonl(output / "development-rows.jsonl", rows)
    _write_json(output / "development-summary.json", summary)
    _write_json(
        output / "promotion-candidate.json", promotion.model_dump(mode="json")
    )
    return summary


def run_holdout(
    promotion_path: Path,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = load_market_config(CONFIG_PATH)
    promotion = FrozenPromotionCandidate.model_validate(
        json.loads(promotion_path.read_text(encoding="utf-8"))
    )
    versions = {item.strategy_id: item for item in strategy_versions()}
    if promotion.config_sha256 != config.config_sha256:
        raise ValueError("promotion candidate config hash mismatch")
    if promotion.candidate_strategy_id not in operational_strategy_ids():
        raise ValueError("research-only strategy cannot enter holdout promotion")
    if (
        versions[promotion.candidate_strategy_id].strategy_hash
        != promotion.candidate_strategy_hash
    ):
        raise ValueError("promotion candidate strategy hash mismatch")
    if tuple(HOLDOUT_SEEDS) != promotion.excluded_holdout_seeds:
        raise ValueError("holdout seeds differ from frozen exclusion list")

    rows = _run_rows(HOLDOUT_SEEDS, "holdout")
    matrix = _matrix(rows)
    phenomena = _phenomena(rows, matrix)
    candidate_id = promotion.candidate_strategy_id
    value_deltas = _paired_delta(
        rows,
        treatment=candidate_id,
        baseline="balanced_competitor",
        value_getter=lambda row: row["focal_enterprise_value_cents"],
    )
    survival_deltas = _paired_delta(
        rows,
        treatment=candidate_id,
        baseline="balanced_competitor",
        value_getter=lambda row: int(bool(row["focal_survived"])),
    )
    candidate_cells = [
        row for row in matrix if row["mutant_strategy_id"] == candidate_id
    ]
    baseline_cells = [
        row
        for row in matrix
        if row["mutant_strategy_id"] == "balanced_competitor"
    ]
    gates = {
        "development_holdout_seed_disjoint": not set(DEVELOPMENT_SEEDS).intersection(
            HOLDOUT_SEEDS
        ),
        "candidate_is_operational": candidate_id in operational_strategy_ids(),
        "replay_all": all(row["episode"]["replay_passed"] for row in rows),
        "demand_closure_all": all(
            row["episode"]["demand_closure_passed"] for row in rows
        ),
        "cash_non_negative_all": all(
            row["episode"]["non_negative_cash"] for row in rows
        ),
        "strategic_best_response_depends_on_opponent": (
            phenomena["unique_best_response_count"] >= 2
        ),
        "competition_bankruptcy_and_price_war_observed": (
            phenomena["price_war_episode_count"] > 0
            and phenomena["bankruptcy_episode_count"] > 0
        ),
        "cooperation_and_free_riding_observed": (
            phenomena["threshold_success_episode_count"] > 0
            and phenomena["threshold_failure_episode_count"] > 0
            and phenomena["mean_free_rider_first_round_profit_advantage_cents"]
            is not None
            and phenomena["mean_free_rider_first_round_profit_advantage_cents"]
            > 0
        ),
        "mutual_aid_observed": phenomena["mutual_aid_episode_count"] > 0,
        "coordination_betrayal_and_enforcement_observed": (
            phenomena["coordination_honor_count"] > 0
            and phenomena["coordination_betrayal_count"] > 0
            and phenomena["regulatory_detection_count"] > 0
        ),
        "candidate_mean_value_noninferior_to_baseline": (
            round(mean(value_deltas)) >= 0 if value_deltas else False
        ),
        "candidate_survival_noninferior_to_baseline": (
            sum(survival_deltas) >= 0 if survival_deltas else False
        ),
        "candidate_worst_resident_value_noninferior_to_baseline": (
            min(
                int(row["mean_focal_enterprise_value_cents"])
                for row in candidate_cells
            )
            >= min(
                int(row["mean_focal_enterprise_value_cents"])
                for row in baseline_cells
            )
        ),
        "candidate_tail_loss_bounded": (
            min(value_deltas) >= -5_000_000 if value_deltas else False
        ),
        "candidate_negative_pair_rate_at_most_ten_percent": (
            sum(value < 0 for value in value_deltas) * 10 <= len(value_deltas)
            if value_deltas
            else False
        ),
    }
    summary = {
        "result_schema_version": "final-market-self-play-holdout-v1.0.0",
        "config_id": config.config_id,
        "config_sha256": config.config_sha256,
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "rounds": ROUNDS,
        "promotion_candidate": promotion.model_dump(mode="json"),
        "matrix": matrix,
        "phenomena": phenomena,
        "promotion_comparison": {
            "pair_count": len(value_deltas),
            "mean_enterprise_value_delta_cents": (
                round(mean(value_deltas)) if value_deltas else None
            ),
            "worst_enterprise_value_delta_cents": (
                min(value_deltas) if value_deltas else None
            ),
            "positive_zero_negative": {
                "positive": sum(value > 0 for value in value_deltas),
                "zero": sum(value == 0 for value in value_deltas),
                "negative": sum(value < 0 for value in value_deltas),
            },
        },
        "gates": gates,
        "promotion_accepted": all(gates.values()),
        "provider_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    summary["result_hash"] = sha256_hash(summary)
    _write_jsonl(output / "holdout-rows.jsonl", rows)
    _write_json(output / "holdout-summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "holdout"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--promotion", type=Path)
    args = parser.parse_args()
    if args.phase == "development":
        result = run_development(args.output)
    else:
        promotion = args.promotion or args.output / "promotion-candidate.json"
        result = run_holdout(promotion.resolve(), args.output)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "selected_candidate_strategy_id": (
                    result.get("selected_candidate_strategy_id")
                    or result["promotion_candidate"]["candidate_strategy_id"]
                ),
                "promotion_accepted": result.get("promotion_accepted"),
                "phenomena": result["phenomena"],
                "failed_gates": [
                    key
                    for key, value in result.get("gates", {}).items()
                    if not value
                ],
                "result_hash": result["result_hash"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
