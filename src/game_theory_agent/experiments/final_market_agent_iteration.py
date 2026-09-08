"""Data-driven policy iteration with a disjoint final-market holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from game_theory_agent.market import load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.self_play import (
    COMPANY_IDS,
    FrozenPromotionCandidate,
    run_episode,
    strategy_versions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v6_final.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "final-market-agent-iteration-v1"
DEVELOPMENT_SEEDS = (98001, 98002, 98003)
HOLDOUT_SEEDS = (99101, 99104)
RESIDENT_STRATEGIES = (
    "balanced_competitor",
    "aggressive_growth",
    "premium_defender",
    "resilience_cooperator",
    "free_rider",
    "mutual_aid_reciprocal",
    "cartel_honorer",
    "cartel_undercutter",
    "predatory_price_stressor",
)
TREATMENTS = (
    "balanced_competitor",
    "premium_defender",
    "contextual_defender_v1",
)
ROUNDS = 20


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


def _rows(seeds: Sequence[int], split: str) -> list[dict[str, Any]]:
    config = load_market_config(CONFIG_PATH)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for resident in RESIDENT_STRATEGIES:
            for focal in COMPANY_IDS:
                for treatment in TREATMENTS:
                    assignments = {
                        company_id: (
                            treatment if company_id == focal else resident
                        )
                        for company_id in COMPANY_IDS
                    }
                    episode = run_episode(
                        config,
                        seed=seed,
                        assignments=assignments,
                        rounds=ROUNDS,
                        experiment_id=f"final-market-agent-iteration-{split}",
                        episode_id=(
                            f"agent-iteration-{split}-{seed}-"
                            f"resident-{resident}-focal-{focal}"
                        ),
                    )
                    rows.append(
                        {
                            "split": split,
                            "seed": seed,
                            "resident_strategy_id": resident,
                            "focal_company_id": focal,
                            "treatment_strategy_id": treatment,
                            "contextual_parent": (
                                "balanced_competitor"
                                if treatment == "contextual_defender_v1"
                                and episode["market_model"] == "value_oriented"
                                and focal in {"company_B", "company_D"}
                                else "premium_defender"
                                if treatment == "contextual_defender_v1"
                                else None
                            ),
                            "focal_enterprise_value_cents": int(
                                episode["enterprise_value_cents"][focal]
                            ),
                            "focal_rank": int(episode["rank"][focal]),
                            "focal_survived": (
                                episode["final_status"][focal] != "exited"
                            ),
                            "episode": episode,
                        }
                    )
    return rows


def _paired_comparison(
    rows: Sequence[Mapping[str, Any]], treatment: str, baseline: str
) -> dict[str, Any]:
    indexed = {
        (
            int(row["seed"]),
            str(row["resident_strategy_id"]),
            str(row["focal_company_id"]),
            str(row["treatment_strategy_id"]),
        ): row
        for row in rows
    }
    deltas: list[int] = []
    survival_deltas: list[int] = []
    rank_deltas: list[int] = []
    for key, row in indexed.items():
        if key[3] != treatment:
            continue
        control = indexed[(key[0], key[1], key[2], baseline)]
        deltas.append(
            int(row["focal_enterprise_value_cents"])
            - int(control["focal_enterprise_value_cents"])
        )
        survival_deltas.append(
            int(bool(row["focal_survived"]))
            - int(bool(control["focal_survived"]))
        )
        rank_deltas.append(
            int(control["focal_rank"]) - int(row["focal_rank"])
        )
    return {
        "pair_count": len(deltas),
        "mean_enterprise_value_delta_cents": round(mean(deltas)),
        "worst_enterprise_value_delta_cents": min(deltas),
        "negative_pair_count": sum(value < 0 for value in deltas),
        "negative_pair_rate_ppm": (
            sum(value < 0 for value in deltas) * 1_000_000 // len(deltas)
        ),
        "mean_rank_improvement_milli": round(mean(rank_deltas) * 1_000),
        "net_survival_improvement_count": sum(survival_deltas),
        "positive_zero_negative": {
            "positive": sum(value > 0 for value in deltas),
            "zero": sum(value == 0 for value in deltas),
            "negative": sum(value < 0 for value in deltas),
        },
    }


def _common_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    contextual = [
        row
        for row in rows
        if row["treatment_strategy_id"] == "contextual_defender_v1"
    ]
    parent_counts = {
        parent: sum(row["contextual_parent"] == parent for row in contextual)
        for parent in ("balanced_competitor", "premium_defender")
    }
    return {
        "contextual_vs_balanced": _paired_comparison(
            rows, "contextual_defender_v1", "balanced_competitor"
        ),
        "premium_vs_balanced": _paired_comparison(
            rows, "premium_defender", "balanced_competitor"
        ),
        "contextual_vs_premium": _paired_comparison(
            rows, "contextual_defender_v1", "premium_defender"
        ),
        "contextual_parent_counts": parent_counts,
        "replay_all": all(row["episode"]["replay_passed"] for row in rows),
        "demand_closure_all": all(
            row["episode"]["demand_closure_passed"] for row in rows
        ),
        "cash_non_negative_all": all(
            row["episode"]["non_negative_cash"] for row in rows
        ),
    }


def run_development(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    config = load_market_config(CONFIG_PATH)
    rows = _rows(DEVELOPMENT_SEEDS, "development")
    metrics = _common_summary(rows)
    vs_balanced = metrics["contextual_vs_balanced"]
    premium_vs_balanced = metrics["premium_vs_balanced"]
    gates = {
        "development_holdout_seed_disjoint": not set(DEVELOPMENT_SEEDS).intersection(
            HOLDOUT_SEEDS
        ),
        "replay_all": metrics["replay_all"],
        "demand_closure_all": metrics["demand_closure_all"],
        "cash_non_negative_all": metrics["cash_non_negative_all"],
        "contextual_mean_value_above_balanced": (
            vs_balanced["mean_enterprise_value_delta_cents"] > 0
        ),
        "contextual_worst_value_not_below_balanced": (
            vs_balanced["worst_enterprise_value_delta_cents"] >= 0
        ),
        "contextual_removes_premium_tail_failures": (
            vs_balanced["worst_enterprise_value_delta_cents"]
            > premium_vs_balanced["worst_enterprise_value_delta_cents"]
            and vs_balanced["negative_pair_count"]
            < premium_vs_balanced["negative_pair_count"]
        ),
        "both_contextual_parents_exercised": all(
            count > 0 for count in metrics["contextual_parent_counts"].values()
        ),
    }
    evidence = {
        "evidence_schema_version": "final-market-agent-iteration-development-v1.0.0",
        "config_id": config.config_id,
        "config_sha256": config.config_sha256,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "excluded_holdout_seeds": list(HOLDOUT_SEEDS),
        "resident_strategy_ids": list(RESIDENT_STRATEGIES),
        "treatment_strategy_ids": list(TREATMENTS),
        "rounds": ROUNDS,
        "metrics": metrics,
        "gates": gates,
        "development_passed": all(gates.values()),
        "provider_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    evidence_hash = sha256_hash(evidence)
    version = next(
        item
        for item in strategy_versions()
        if item.strategy_id == "contextual_defender_v1"
    )
    promotion_payload = {
        "promotion_schema_version": "self-play-promotion-candidate-v1.0.0",
        "config_sha256": config.config_sha256,
        "development_result_hash": evidence_hash,
        "candidate_strategy_id": version.strategy_id,
        "candidate_strategy_hash": version.strategy_hash,
        "selection_rule": (
            "development-only tail repair: use balanced in value-oriented "
            "markets for high-capacity roles, premium defense otherwise"
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
    version = next(
        item
        for item in strategy_versions()
        if item.strategy_id == "contextual_defender_v1"
    )
    if promotion.candidate_strategy_hash != version.strategy_hash:
        raise ValueError("iteration strategy hash mismatch")
    if tuple(HOLDOUT_SEEDS) != promotion.excluded_holdout_seeds:
        raise ValueError("holdout seeds differ from frozen exclusion list")
    rows = _rows(HOLDOUT_SEEDS, "holdout")
    metrics = _common_summary(rows)
    vs_balanced = metrics["contextual_vs_balanced"]
    gates = {
        "development_holdout_seed_disjoint": not set(DEVELOPMENT_SEEDS).intersection(
            HOLDOUT_SEEDS
        ),
        "candidate_hash_matches": promotion.candidate_strategy_hash
        == version.strategy_hash,
        "replay_all": metrics["replay_all"],
        "demand_closure_all": metrics["demand_closure_all"],
        "cash_non_negative_all": metrics["cash_non_negative_all"],
        "contextual_mean_value_noninferior_to_balanced": (
            vs_balanced["mean_enterprise_value_delta_cents"] >= 0
        ),
        "contextual_tail_loss_at_most_five_million_cents": (
            vs_balanced["worst_enterprise_value_delta_cents"] >= -5_000_000
        ),
        "contextual_negative_pair_rate_at_most_ten_percent": (
            vs_balanced["negative_pair_rate_ppm"] <= 100_000
        ),
        "contextual_survival_noninferior_to_balanced": (
            vs_balanced["net_survival_improvement_count"] >= 0
        ),
        "both_contextual_parents_exercised": all(
            count > 0 for count in metrics["contextual_parent_counts"].values()
        ),
    }
    summary = {
        "result_schema_version": "final-market-agent-iteration-holdout-v1.0.0",
        "config_id": config.config_id,
        "config_sha256": config.config_sha256,
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "promotion_candidate": promotion.model_dump(mode="json"),
        "metrics": metrics,
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
                "passed": result.get("development_passed")
                if args.phase == "development"
                else result["promotion_accepted"],
                "metrics": result["metrics"],
                "failed_gates": [
                    key for key, value in result["gates"].items() if not value
                ],
                "result_hash": result["result_hash"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
