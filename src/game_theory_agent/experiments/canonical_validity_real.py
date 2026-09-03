"""Preregistered 60-call real-LLM core evaluation for Cournot and cooperation."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from game_theory_agent.agents.personas import load_persona_registry
from game_theory_agent.benchmark_games import CournotEnvironment
from game_theory_agent.cooperation.prompt_variants import (
    build_cooperation_prompt,
    cooperation_payoff,
    economic_only_persona_view,
    private_optimal_contribution,
)
from game_theory_agent.experiments.real_json_client import RealJSONClient
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.strategic_reliability import (
    RealModelBudget,
    RealModelCostGuard,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "canonical-validity-real-v1"
PROVIDERS = ("doubao", "deepseek")
REPETITIONS = (1, 2, 3)
OPPONENT_SCENARIOS = (6, 8, 12)
COURNOT_CONDITIONS = ("persona_only", "exact_cournot_advisor")
PROMPT_VARIANTS = ("explicit_options_v1", "neutral_numeric_v1")
INCENTIVES = ("high_deviation_incentive", "low_deviation_incentive")

# Conservative current public price snapshots, represented as CNY microunits/token.
# Doubao rounds 0.6 upward to 1 for guard safety. DeepSeek uses peak cache-miss prices.
PRICE_SNAPSHOT = {
    "doubao": {
        "input_microunits_per_token": 1,
        "output_microunits_per_token": 4,
        "basis": "Doubao Seed 2.0 Lite <=32k, conservative rounding",
        "source": "https://www.volcengine.com/docs/84458/1585097",
    },
    "deepseek": {
        "input_microunits_per_token": 3,
        "output_microunits_per_token": 9,
        "basis": "DeepSeek V4 Flash peak cache-miss",
        "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _cournot_prompt(opponent_quantity: int, condition: str) -> tuple[str, dict[str, Any]]:
    env = CournotEnvironment()
    belief = {opponent_quantity: 1_000_000}
    advice = env.advise(company_id="A", belief_ppm=belief)
    payload: dict[str, Any] = {
        "game": {
            "players": 2,
            "simultaneous_actions": True,
            "own_quantity_bounds": [0, 30],
            "price_formula": "max(30 - own_quantity - opponent_quantity, 0)",
            "own_profit_formula": "(price - 6) * own_quantity",
        },
        "history": {
            "opponent_previous_quantities": [opponent_quantity] * 3,
            "forecast_for_current_round": {str(opponent_quantity): 1.0},
            "current_opponent_action_visible": False,
        },
        "persona": {
            "objective": "最大化本轮可验证利润；相同利润时选择更小产量。"
        },
    }
    if condition == "exact_cournot_advisor":
        payload["advisor"] = {
            "recommended_quantity": advice.recommended_quantity,
            "expected_profit": advice.expected_profit_microunits / 1_000_000,
            "method": "对0到30全部整数候选逐一枚举",
            "advice_is_non_binding": True,
        }
    prompt = "\n".join(
        (
            "你是 Cournot 数量竞争中的公司 A。选择本轮产量。",
            "不得假装看见对手本轮同时行动；只能使用历史与给定预测。",
            "只输出 JSON：{\"quantity\":0到30的整数,\"reason\":\"不超过80字的可审计理由\"}。",
            "不要输出隐藏推理过程。",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )
    )
    return prompt, advice.model_dump(mode="json")


def _base_generation_row(call_id: str, generation: Any) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "provider": generation.provider,
        "model": generation.model,
        "prompt_hash": sha256_hash(generation.prompt),
        "raw_response": generation.raw_response,
        "parsed_output": generation.parsed,
        "parse_error": generation.parse_error,
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "latency_ms": generation.latency_ms,
        "request_started_at": generation.request_started_at,
        "response_received_at": generation.response_received_at,
        "request_id": generation.request_id,
        "response_model": generation.response_model,
    }


def _valid_integer(value: Any, low: int, high: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        return None
    return value


def _cost(provider: str, input_tokens: int, output_tokens: int) -> int:
    price = PRICE_SNAPSHOT[provider]
    return (
        input_tokens * int(price["input_microunits_per_token"])
        + output_tokens * int(price["output_microunits_per_token"])
    )


async def _call(
    *,
    client: RealJSONClient,
    guard: RealModelCostGuard,
    prompt: str,
) -> Any:
    provider = client.provider
    guard.reserve(
        prompt_tokens=2_000,
        completion_tokens=400,
        estimated_cost_microunits=_cost(provider, 2_000, 400),
    )
    generation = await client.generate(prompt)
    guard.record_actual(
        prompt_tokens=generation.input_tokens,
        completion_tokens=generation.output_tokens,
        estimated_cost_microunits=_cost(
            provider, generation.input_tokens, generation.output_tokens
        ),
    )
    return generation


def _summarize(rows: list[dict[str, Any]], guard: RealModelCostGuard) -> dict[str, Any]:
    cournot = [row for row in rows if row["experiment"] == "cournot"]
    cooperation = [row for row in rows if row["experiment"] == "cooperation"]
    cournot_cells: dict[str, Any] = {}
    for condition in COURNOT_CONDITIONS:
        selected = [row for row in cournot if row["condition"] == condition and row["valid"]]
        cournot_cells[condition] = {
            "n": len(selected),
            "mean_exact_regret": round(mean(row["exact_regret"] for row in selected), 4) if selected else None,
            "zero_regret_rate": round(sum(row["exact_regret"] == 0 for row in selected) / len(selected), 4) if selected else None,
            "mean_profit": round(mean(row["profit"] for row in selected), 4) if selected else None,
            "advisor_adoption_rate": round(sum(row.get("advisor_adopted", False) for row in selected) / len(selected), 4) if condition == "exact_cournot_advisor" and selected else None,
        }
    paired_deltas: list[int] = []
    for provider in PROVIDERS:
        for repetition in REPETITIONS:
            for opponent in OPPONENT_SCENARIOS:
                base = next((row for row in cournot if row["provider"] == provider and row["repetition"] == repetition and row["opponent_quantity"] == opponent and row["condition"] == "persona_only" and row["valid"]), None)
                treatment = next((row for row in cournot if row["provider"] == provider and row["repetition"] == repetition and row["opponent_quantity"] == opponent and row["condition"] == "exact_cournot_advisor" and row["valid"]), None)
                if base and treatment:
                    paired_deltas.append(base["exact_regret"] - treatment["exact_regret"])
    cooperation_cells: dict[str, Any] = {}
    for variant in PROMPT_VARIANTS:
        for incentive in INCENTIVES:
            selected = [row for row in cooperation if row["variant"] == variant and row["incentive"] == incentive and row["valid"]]
            key = f"{variant}|{incentive}"
            cooperation_cells[key] = {
                "n": len(selected),
                "mean_contribution": round(mean(row["actual_contribution"] for row in selected), 4) if selected else None,
                "full_fulfillment_rate": round(sum(row["actual_contribution"] == 10 for row in selected) / len(selected), 4) if selected else None,
                "zero_contribution_rate": round(sum(row["actual_contribution"] == 0 for row in selected) / len(selected), 4) if selected else None,
                "private_optimal_action_rate": round(sum(row["actual_contribution"] == row["private_optimal_contribution"] for row in selected) / len(selected), 4) if selected else None,
            }
    return {
        "schema_version": "canonical-validity-real-summary-v1.0.0",
        "evidence_level": "directional_real_model_evidence_not_statistical_confirmation",
        "recorded_calls": len(rows),
        "valid_calls": sum(row["valid"] for row in rows),
        "invalid_calls": sum(not row["valid"] for row in rows),
        "cournot": {
            "cells": cournot_cells,
            "complete_pairs": len(paired_deltas),
            "mean_paired_regret_reduction": round(mean(paired_deltas), 4) if paired_deltas else None,
            "positive_zero_negative_pairs": {
                "positive": sum(delta > 0 for delta in paired_deltas),
                "zero": sum(delta == 0 for delta in paired_deltas),
                "negative": sum(delta < 0 for delta in paired_deltas),
            },
        },
        "cooperation": {"cells": cooperation_cells},
        "usage": asdict(guard.actual),
        "estimated_cost_cny_conservative": round(guard.actual.estimated_cost_microunits / 1_000_000, 6),
    }


async def run(output: Path) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "rows.jsonl"
    existing: list[dict[str, Any]] = []
    if rows_path.exists():
        existing = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = {row["call_id"] for row in existing}
    rows = list(existing)
    manifest = {
        "schema_version": "canonical-validity-real-manifest-v1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "planned_calls": 60,
        "providers": list(PROVIDERS),
        "repetitions": list(REPETITIONS),
        "cournot": {"opponent_scenarios": list(OPPONENT_SCENARIOS), "conditions": list(COURNOT_CONDITIONS), "planned_calls": 36},
        "cooperation": {"prompt_variants": list(PROMPT_VARIANTS), "incentives": list(INCENTIVES), "persona_view": "economic_only", "planned_calls": 24},
        "price_snapshot": PRICE_SNAPSHOT,
        "research_boundary": "A 3-repetition core experiment is directional evidence, not statistical confirmation.",
    }
    _write_json(output / "manifest.json", manifest)
    guard = RealModelCostGuard(
        RealModelBudget(
            max_calls=60,
            max_prompt_tokens=120_000,
            max_completion_tokens=24_000,
            max_estimated_cost_microunits=1_000_000,
        ),
        explicitly_authorized=True,
    )
    # Re-reserve and record persisted calls so resume cannot bypass the budget.
    for row in existing:
        guard.reserve(prompt_tokens=2_000, completion_tokens=400, estimated_cost_microunits=_cost(row["provider"], 2_000, 400))
        guard.record_actual(prompt_tokens=int(row["input_tokens"]), completion_tokens=int(row["output_tokens"]), estimated_cost_microunits=_cost(row["provider"], int(row["input_tokens"]), int(row["output_tokens"])))
    registry = load_persona_registry()
    persona_view = economic_only_persona_view(registry.get("balanced"))
    total = manifest["planned_calls"]
    for provider in PROVIDERS:
        client = RealJSONClient(provider, temperature=0.0)
        for repetition in REPETITIONS:
            for scenario_index, opponent_quantity in enumerate(OPPONENT_SCENARIOS):
                conditions = COURNOT_CONDITIONS if (repetition + scenario_index) % 2 else tuple(reversed(COURNOT_CONDITIONS))
                for condition in conditions:
                    call_id = f"cournot|{provider}|r{repetition}|opp{opponent_quantity}|{condition}"
                    if call_id in completed:
                        continue
                    prompt, advice = _cournot_prompt(opponent_quantity, condition)
                    generation = await _call(client=client, guard=guard, prompt=prompt)
                    row = _base_generation_row(call_id, generation)
                    quantity = _valid_integer((generation.parsed or {}).get("quantity"), 0, 30)
                    row.update({"experiment": "cournot", "repetition": repetition, "condition": condition, "opponent_quantity": opponent_quantity, "advice": advice, "quantity": quantity, "valid": quantity is not None})
                    if quantity is not None:
                        env = CournotEnvironment()
                        row.update({"profit": env.profit(quantity, opponent_quantity), "exact_regret": env.regret(quantity, opponent_quantity), "advisor_adopted": quantity == advice["recommended_quantity"] if condition == "exact_cournot_advisor" else None})
                    rows.append(row)
                    _append_jsonl(rows_path, row)
                    print(f"[{len(rows)}/{total}] {call_id} valid={row['valid']} tokens={generation.input_tokens + generation.output_tokens}", flush=True)
        for repetition in REPETITIONS:
            for variant_index, variant in enumerate(PROMPT_VARIANTS):
                incentives = INCENTIVES if (repetition + variant_index) % 2 else tuple(reversed(INCENTIVES))
                for incentive in incentives:
                    call_id = f"cooperation|{provider}|r{repetition}|{variant}|{incentive}"
                    if call_id in completed:
                        continue
                    prompt, condition_hash = build_cooperation_prompt(variant=variant, incentive=incentive, persona_view=persona_view, repetition=repetition)
                    generation = await _call(client=client, guard=guard, prompt=prompt)
                    row = _base_generation_row(call_id, generation)
                    contribution = _valid_integer((generation.parsed or {}).get("actual_contribution"), 0, 10)
                    optimum = private_optimal_contribution(incentive)
                    row.update({"experiment": "cooperation", "repetition": repetition, "variant": variant, "incentive": incentive, "condition_hash": condition_hash, "actual_contribution": contribution, "valid": contribution is not None, "promised_contribution": 10, "private_optimal_contribution": optimum, "fulfillment_ratio_ppm": contribution * 100_000 if contribution is not None else None, "payoff": cooperation_payoff(contribution, incentive) if contribution is not None else None})
                    rows.append(row)
                    _append_jsonl(rows_path, row)
                    print(f"[{len(rows)}/{total}] {call_id} valid={row['valid']} tokens={generation.input_tokens + generation.output_tokens}", flush=True)
    summary = _summarize(rows, guard)
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--authorize-real-model", action="store_true")
    args = parser.parse_args()
    if not args.authorize_real_model:
        raise RuntimeError("--authorize-real-model is required")
    print(json.dumps(asyncio.run(run(args.output.resolve())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
