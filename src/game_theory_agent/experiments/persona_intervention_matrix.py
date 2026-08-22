"""Plan or execute blocked Persona intervention matrices with position rotation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from game_theory_agent.experiments.four_agent_acceptance import (
    EXPERIMENT_CONDITIONS,
    run as run_episode,
)


SEED_SPLITS = {
    "development": tuple(range(101, 111)),
    "validation": tuple(range(201, 221)),
    "final_holdout": tuple(range(1001, 1031)),
}


def build_matrix(
    *, conditions: tuple[str, ...], seed_split: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in conditions:
        personas = tuple(EXPERIMENT_CONDITIONS[condition]["personas"])
        rotations = (0,) if len(set(personas)) == 1 else (0, 1, 2, 3)
        for seed in SEED_SPLITS[seed_split]:
            for rotation in rotations:
                rows.append(
                    {
                        "condition": condition,
                        "seed_split": seed_split,
                        "seed": seed,
                        "rotation_index": rotation,
                        "personas_before_rotation": list(personas),
                        "primary_experiment_unit": "seed",
                        "blocking_factors": ["seed", "company_position"],
                    }
                )
    return rows


async def run(args: argparse.Namespace) -> int:
    conditions = tuple(
        item.strip() for item in args.conditions.split(",") if item.strip()
    )
    unknown = set(conditions) - set(EXPERIMENT_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    if args.seed_split == "final_holdout" and not args.confirm_final_holdout:
        raise ValueError(
            "final_holdout execution/inspection requires --confirm-final-holdout; "
            "do not use holdout seeds for calibration or persona weight search"
        )
    matrix = build_matrix(conditions=conditions, seed_split=args.seed_split)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "matrix_schema_version": "persona-intervention-matrix-v1.0.0",
        "seed_split": args.seed_split,
        "conditions": list(conditions),
        "episode_count": len(matrix),
        "primary_experiment_unit": "paired_seed",
        "decisions_are_nested_independent_samples": False,
        "automatic_weight_search_enabled": False,
        "holdout_policy": (
            "final holdout is never used for tuning; explicit confirmation required"
        ),
        "rows": matrix,
    }
    (args.output / "matrix-plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    results: list[dict[str, object]] = []
    for index, row in enumerate(matrix, start=1):
        condition = str(row["condition"])
        seed = int(row["seed"])
        rotation = int(row["rotation_index"])
        episode_output = (
            args.output / condition / f"seed-{seed}" / f"rotation-{rotation}"
        )
        print(
            f"[{index}/{len(matrix)}] condition={condition} seed={seed} "
            f"rotation={rotation}",
            flush=True,
        )
        episode_args = SimpleNamespace(
            episode_id=None,
            seed=seed,
            rounds=args.rounds,
            market_model=args.market_model,
            provider=args.provider,
            model=args.model,
            persona="balanced",
            personas=None,
            condition=condition,
            llm_count=4,
            rotation_index=rotation,
            decision_support_version="economic_v2",
            persona_semantics_version="economic_v2",
            diagnostic_mode="off",
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
            output=episode_output,
        )
        exit_code = await run_episode(episode_args)
        summary = json.loads(
            (episode_output / "summary.json").read_text(encoding="utf-8")
        )
        results.append(
            {
                "condition": condition,
                "seed": seed,
                "rotation_index": rotation,
                "exit_code": exit_code,
                "passed": bool(summary["passed"]),
                "summary_path": str((episode_output / "summary.json").resolve()),
            }
        )
        if exit_code != 0 and args.stop_on_failure:
            break
    aggregate = {
        **{key: value for key, value in plan.items() if key != "rows"},
        "executed_episode_count": len(results),
        "passed_episode_count": sum(bool(item["passed"]) for item in results),
        "results": results,
    }
    (args.output / "matrix-results.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0 if all(bool(item["passed"]) for item in results) else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        default="A_balanced_legacy,B_extreme_legacy,C_extreme_support,"
        "D_extreme_semantics,E_moderate_semantics,F_moderate_diagnostics",
    )
    parser.add_argument(
        "--seed-split", choices=tuple(SEED_SPLITS), default="development"
    )
    parser.add_argument("--confirm-final-holdout", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--provider", choices=("mock", "doubao", "deepseek"), default="mock")
    parser.add_argument("--model")
    parser.add_argument("--rounds", type=int, choices=(5, 10, 15, 20), default=20)
    parser.add_argument("--market-model", default="balanced")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    return asyncio.run(run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
