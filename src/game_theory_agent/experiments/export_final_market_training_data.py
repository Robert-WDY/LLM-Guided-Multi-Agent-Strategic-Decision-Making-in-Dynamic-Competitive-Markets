"""Export development-only counterfactual preference data for future training."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from game_theory_agent.market.protocols import sha256_hash


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "runs" / "final-market-agent-iteration-v1"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "final-market-training-data-v1"
CANDIDATES = ("balanced_competitor", "premium_defender")
FORBIDDEN_SPLITS = {"holdout", "test", "final_holdout"}


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


def _capacity_segment(company_id: str) -> str:
    return "high" if company_id in {"company_B", "company_D"} else "low"


def _build_examples(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if any(str(row.get("split")) in FORBIDDEN_SPLITS for row in rows):
        raise ValueError("holdout/test rows cannot enter the training export")
    if any(str(row.get("split")) != "development" for row in rows):
        raise ValueError("every training source row must be development")
    indexed = {
        (
            int(row["seed"]),
            str(row["resident_strategy_id"]),
            str(row["focal_company_id"]),
            str(row["treatment_strategy_id"]),
        ): row
        for row in rows
    }
    contexts = sorted({key[:3] for key in indexed})
    examples = []
    for seed, resident, focal in contexts:
        selected = {
            strategy_id: indexed[(seed, resident, focal, strategy_id)]
            for strategy_id in CANDIDATES
        }
        outcomes = {
            strategy_id: int(row["focal_enterprise_value_cents"])
            for strategy_id, row in selected.items()
        }
        preferred = max(CANDIDATES, key=lambda item: (outcomes[item], item))
        market_model = str(
            selected["balanced_competitor"]["episode"]["market_model"]
        )
        input_context = {
            "market_model_id": market_model,
            "own_capacity_segment": _capacity_segment(focal),
            "round_horizon": 20,
            "information_scope": "public_market_plus_own_private",
        }
        source_binding = sha256_hash(
            {
                "seed": seed,
                "resident_strategy_id": resident,
                "focal_company_id": focal,
                "state_hashes": {
                    strategy_id: row["episode"]["final_state_hash"]
                    for strategy_id, row in selected.items()
                },
            }
        )
        payload = {
            "example_schema_version": "strategy-preference-example-v1.0.0",
            "example_id": source_binding,
            "source_split": "development",
            "input_context": input_context,
            "candidate_strategy_ids": list(CANDIDATES),
            "preferred_strategy_id": preferred,
            "enterprise_value_by_strategy_cents": outcomes,
            "absolute_preference_margin_cents": abs(
                outcomes[CANDIDATES[0]] - outcomes[CANDIDATES[1]]
            ),
            "uses_hidden_opponent_state": False,
            "contains_raw_llm_output": False,
            "example_hash": "pending",
        }
        payload["example_hash"] = sha256_hash(
            {
                key: value
                for key, value in payload.items()
                if key != "example_hash"
            }
        )
        examples.append(payload)
    return examples


def export(source: Path, output: Path) -> dict[str, Any]:
    summary_path = source / "development-summary.json"
    rows_path = source / "development-rows.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("development_passed"):
        raise ValueError("source development gate did not pass")
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    examples = _build_examples(rows)
    labels_by_context: dict[tuple[str, str], set[str]] = defaultdict(set)
    for example in examples:
        context = example["input_context"]
        labels_by_context[
            (context["market_model_id"], context["own_capacity_segment"])
        ].add(example["preferred_strategy_id"])
    collision_groups = sum(
        len(labels) > 1 for labels in labels_by_context.values()
    )
    contextual_matches = 0
    for example in examples:
        context = example["input_context"]
        contextual_choice = (
            "balanced_competitor"
            if context["market_model_id"] == "value_oriented"
            and context["own_capacity_segment"] == "high"
            else "premium_defender"
        )
        contextual_matches += (
            contextual_choice == example["preferred_strategy_id"]
        )
    manifest = {
        "dataset_schema_version": "final-market-training-dataset-v1.0.0",
        "source_development_result_hash": summary["result_hash"],
        "source_config_sha256": summary["config_sha256"],
        "source_development_seeds": summary["development_seeds"],
        "excluded_holdout_seeds": summary["excluded_holdout_seeds"],
        "example_count": len(examples),
        "candidate_strategy_ids": list(CANDIDATES),
        "input_fields": [
            "market_model_id",
            "own_capacity_segment",
            "round_horizon",
            "information_scope",
        ],
        "opponent_strategy_id_is_training_input": False,
        "contains_holdout_rows": False,
        "contains_raw_llm_output": False,
        "contains_real_market_data": False,
        "context_group_count": len(labels_by_context),
        "conflicting_label_context_group_count": collision_groups,
        "contextual_policy_label_match_rate_ppm": (
            contextual_matches * 1_000_000 // len(examples)
        ),
        "allowed_use": "offline strategy-selector prototyping",
        "fine_tuning_release_allowed": False,
        "fine_tuning_blockers": [
            "examples are synthetic rule-policy counterfactuals",
            "the compact observable context has conflicting labels",
            "the dataset has no real-market calibration",
            "the dataset has no sufficient real-LLM action diversity",
        ],
        "provider_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "dataset_hash": sha256_hash(
            [example["example_hash"] for example in examples]
        ),
        "manifest_hash": "pending",
    }
    manifest["manifest_hash"] = sha256_hash(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_hash"
        }
    )
    _write_jsonl(output / "strategy-preferences-development.jsonl", examples)
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = export(args.source.resolve(), args.output.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
