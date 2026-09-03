"""Twelve-call terminal-payoff cooperation prompt causality check."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from game_theory_agent.agents.personas import load_persona_registry
from game_theory_agent.cooperation.prompt_variants import (
    build_cooperation_prompt,
    economic_only_persona_view,
)
from game_theory_agent.experiments.canonical_validity_real import (
    PRICE_SNAPSHOT,
    PROJECT_ROOT,
    PROVIDERS,
    PROMPT_VARIANTS,
    REPETITIONS,
    _base_generation_row,
    _cost,
    _valid_integer,
    _write_json,
)
from game_theory_agent.experiments.real_json_client import RealJSONClient
from game_theory_agent.strategic_reliability import RealModelBudget, RealModelCostGuard


DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "cooperation-terminal-real-v1"


async def run(output: Path, *, persona_view_mode: str = "balanced_economic") -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "rows.jsonl"
    rows: list[dict[str, Any]] = []
    guard = RealModelCostGuard(
        RealModelBudget(
            max_calls=12,
            max_prompt_tokens=24_000,
            max_completion_tokens=4_800,
            max_estimated_cost_microunits=250_000,
        ),
        explicitly_authorized=True,
    )
    if persona_view_mode == "balanced_economic":
        persona_view = economic_only_persona_view(load_persona_registry().get("balanced"))
    elif persona_view_mode == "neutral_profit_only":
        persona_view = {
            "view_schema_version": "persona-economic-view-v1.0.0",
            "persona_id": "neutral_profit_only",
            "objective": "仅最大化本题明确列出的终局 private_value。",
            "utility_weights_ppm": {"profit": 1_000_000},
            "traits_ppm": {},
            "masked_fields": [
                "label",
                "long_term_objective",
                "reputation",
                "resilience",
                "cooperation_reputation",
                "commitment_honesty",
                "opportunism",
            ],
        }
    else:
        raise ValueError(f"unsupported persona_view_mode: {persona_view_mode}")
    manifest = {
        "schema_version": "cooperation-terminal-real-manifest-v1.0.0",
        "planned_calls": 12,
        "providers": list(PROVIDERS),
        "repetitions": list(REPETITIONS),
        "prompt_variants": list(PROMPT_VARIANTS),
        "incentive": "high_deviation_incentive",
        "payoff_scope": "terminal_private_value_only",
        "persona_view_mode": persona_view_mode,
        "analytic_private_optimum": 0,
        "price_snapshot": PRICE_SNAPSHOT,
        "research_boundary": "Three repetitions per provider/variant are directional evidence only.",
    }
    _write_json(output / "manifest.json", manifest)
    for provider in PROVIDERS:
        client = RealJSONClient(provider, temperature=0.0)
        for repetition in REPETITIONS:
            variants = PROMPT_VARIANTS if repetition % 2 else tuple(reversed(PROMPT_VARIANTS))
            for variant in variants:
                prompt, condition_hash = build_cooperation_prompt(
                    variant=variant,
                    incentive="high_deviation_incentive",
                    persona_view=persona_view,
                    repetition=repetition,
                    terminal_payoff_only=True,
                )
                guard.reserve(
                    prompt_tokens=2_000,
                    completion_tokens=400,
                    estimated_cost_microunits=_cost(provider, 2_000, 400),
                )
                generation = await client.generate(prompt)
                guard.record_actual(
                    prompt_tokens=generation.input_tokens,
                    completion_tokens=generation.output_tokens,
                    estimated_cost_microunits=_cost(provider, generation.input_tokens, generation.output_tokens),
                )
                call_id = f"terminal|{provider}|r{repetition}|{variant}"
                row = _base_generation_row(call_id, generation)
                contribution = _valid_integer((generation.parsed or {}).get("actual_contribution"), 0, 10)
                row.update(
                    {
                        "experiment": "cooperation_terminal",
                        "repetition": repetition,
                        "variant": variant,
                        "incentive": "high_deviation_incentive",
                        "condition_hash": condition_hash,
                        "actual_contribution": contribution,
                        "valid": contribution is not None,
                        "private_optimal_contribution": 0,
                    }
                )
                rows.append(row)
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                print(f"[{len(rows)}/12] {call_id} valid={row['valid']} tokens={generation.input_tokens + generation.output_tokens}", flush=True)
    cells: dict[str, Any] = {}
    for provider in PROVIDERS:
        for variant in PROMPT_VARIANTS:
            selected = [row for row in rows if row["provider"] == provider and row["variant"] == variant and row["valid"]]
            cells[f"{provider}|{variant}"] = {
                "n": len(selected),
                "values": [row["actual_contribution"] for row in selected],
                "mean_contribution": round(mean(row["actual_contribution"] for row in selected), 4) if selected else None,
                "private_optimal_action_rate": round(sum(row["actual_contribution"] == 0 for row in selected) / len(selected), 4) if selected else None,
            }
    summary = {
        "schema_version": "cooperation-terminal-real-summary-v1.0.0",
        "evidence_level": "directional_real_model_evidence_not_statistical_confirmation",
        "recorded_calls": len(rows),
        "valid_calls": sum(row["valid"] for row in rows),
        "cells": cells,
        "usage": asdict(guard.actual),
        "estimated_cost_cny_conservative": round(guard.actual.estimated_cost_microunits / 1_000_000, 6),
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--persona-view",
        choices=("balanced_economic", "neutral_profit_only"),
        default="balanced_economic",
    )
    parser.add_argument("--authorize-real-model", action="store_true")
    args = parser.parse_args()
    if not args.authorize_real_model:
        raise RuntimeError("--authorize-real-model is required")
    print(
        json.dumps(
            asyncio.run(run(args.output.resolve(), persona_view_mode=args.persona_view)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
