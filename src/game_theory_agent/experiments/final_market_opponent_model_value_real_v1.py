"""Three-call real-LLM check of correct versus identity-shuffled opponent models."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from dotenv import load_dotenv

from game_theory_agent.agents import AgentRuntime, EpisodeMemory
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.experiments.final_market_advisor_v9_validation import (
    CONFIG_PATH,
    FOCAL,
    HORIZON,
    SCENARIOS,
    _actual_outcome,
    _build_window,
)
from game_theory_agent.experiments.final_market_v9_real_llm_smoke import (
    _complete_observation,
    _is_research_only_coordination,
)
from game_theory_agent.market import load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.model_clients import BudgetedModelClient, DeepSeekModelClient
from game_theory_agent.opponent import compute_opponent_model_hash
from game_theory_agent.strategic_reliability import (
    PublicMarketRolloutAdvisor,
    RealModelBudget,
    RealModelCostGuard,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROWS = PROJECT_ROOT / "runs" / "final-market-real-core-v1" / "rows.json"
DEFAULT_SPEC = (
    PROJECT_ROOT
    / "experiment-specs"
    / "final-market-opponent-model-value-real-v1"
    / "PREREGISTRATION.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "final-market-opponent-model-value-real-v1"
SEED = 95102
WINDOW = "mature_shortage"
PERSONAS = ("balanced_v1", "aggressive_v1_extreme", "risk_guarded_v1")
RESERVED_PROMPT_TOKENS = 28_000
RESERVED_COMPLETION_TOKENS = 4_000
INPUT_PRICE_MICROUNITS = 3
OUTPUT_PRICE_MICROUNITS = 9


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_hash(payload: Mapping[str, Any], field: str) -> str:
    normalized = dict(payload)
    normalized.pop(field, None)
    return sha256_hash(normalized)


def _rotated_opponent_model(model: Any) -> Any:
    company_ids = sorted(model.opponent_models)
    source = [model.opponent_models[item] for item in company_ids]
    rotated = {
        company_id: source[(index + 1) % len(source)].model_copy(
            update={"opponent_company_id": company_id}
        )
        for index, company_id in enumerate(company_ids)
    }
    return model.model_copy(update={"opponent_models": rotated})


def _source_lookup() -> dict[tuple[int, str, str, str], dict[str, Any]]:
    rows = json.loads(SOURCE_ROWS.read_text(encoding="utf-8"))
    return {
        (
            int(row["seed"]),
            str(row["window"]),
            str(row["persona_id"]),
            str(row["condition"]),
        ): row
        for row in rows
    }


def _material(persona_id: str) -> dict[str, Any]:
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    state, belief, opponent, observation = _build_window(config, SEED, WINDOW)
    shuffled = _rotated_opponent_model(opponent)
    shuffled_observation = json.loads(json.dumps(observation, ensure_ascii=False))
    shuffled_observation["opponent_model_state"] = shuffled.model_dump(mode="json")
    shuffled_observation["opponent_model_hash"] = compute_opponent_model_hash(shuffled)
    advice = PublicMarketRolloutAdvisor(config).advise(
        observation=shuffled_observation,
        company_id=FOCAL,
        persona_profile=personas.get(persona_id),
        belief_state=belief,
        opponent_model=shuffled,
        horizon_rounds=HORIZON,
        scenario_count=SCENARIOS,
        advisor_mode="strategic_market_v9",
    ).model_dump(mode="json")
    if advice["execution_disposition"] != "recommend":
        raise AssertionError(f"shuffled treatment no longer recommends: {persona_id}")
    if advice["recommended_candidate_id"] != "price_increase":
        raise AssertionError(f"unexpected shuffled recommendation: {persona_id}")
    completed = _complete_observation(
        config, state, shuffled_observation, advice=advice
    )
    return {
        "config": config,
        "personas": personas,
        "state": state,
        "belief": belief,
        "opponent": shuffled,
        "advice": advice,
        "observation": completed,
    }


def prepare(spec_path: Path) -> dict[str, Any]:
    if spec_path.exists():
        raise RuntimeError("preregistration already exists; refusing to overwrite it")
    source = _source_lookup()
    cells: list[dict[str, Any]] = []
    for persona_id in PERSONAS:
        material = _material(persona_id)
        baseline = source[(SEED, WINDOW, persona_id, "baseline_without_advice")]
        correct = source[(SEED, WINDOW, persona_id, "treatment_with_v9_advice")]
        if not baseline.get("success") or not correct.get("success"):
            raise AssertionError(f"source evidence is incomplete: {persona_id}")
        if (correct.get("advisor_adoption") or {}).get("advisor_candidate_id") != "mutual_aid_request":
            raise AssertionError(f"correct source recommendation changed: {persona_id}")
        cells.append(
            {
                "persona_id": persona_id,
                "state_hash": material["state"].state_hash,
                "shuffled_opponent_model_hash": compute_opponent_model_hash(
                    material["opponent"]
                ),
                "shuffled_advice_hash": material["advice"]["advice_hash"],
                "shuffled_observation_hash": material["observation"]["observation_hash"],
                "correct_recommendation": "mutual_aid_request",
                "shuffled_recommendation": "price_increase",
                "baseline_enterprise_value_cents": int(
                    baseline["outcome"]["enterprise_value_cents"]
                ),
                "correct_model_enterprise_value_cents": int(
                    correct["outcome"]["enterprise_value_cents"]
                ),
            }
        )
    count = len(cells)
    spec: dict[str, Any] = {
        "experiment_schema_version": "opponent-model-value-prereg-v1.0.0",
        "evidence_level": "POST_HOC_DIRECTIONAL_REAL_LLM_MECHANISM_EVIDENCE",
        "provider": "deepseek",
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "temperature_ppm": 0,
        "top_p_ppm": 100_000,
        "seed": SEED,
        "window": WINDOW,
        "source_rows_sha256": _file_sha256(SOURCE_ROWS),
        "selection_rule": (
            "zero-token scan selected the first frozen state where rotating only "
            "opponent identities changed released advice for all three personas"
        ),
        "treatment": (
            "cyclically rotate public-evidence-derived opponent profiles across B/C/D; "
            "keep market, seed, persona and action belief unchanged"
        ),
        "maximum_logical_provider_calls": count,
        "reserved_prompt_tokens_per_call": RESERVED_PROMPT_TOKENS,
        "reserved_completion_tokens_per_call": RESERVED_COMPLETION_TOKENS,
        "maximum_estimated_cost_microunits": count
        * (
            RESERVED_PROMPT_TOKENS * INPUT_PRICE_MICROUNITS
            + RESERVED_COMPLETION_TOKENS * OUTPUT_PRICE_MICROUNITS
        ),
        "price_snapshot": {
            "input_price_microunits_per_token": INPUT_PRICE_MICROUNITS,
            "output_price_microunits_per_token": OUTPUT_PRICE_MICROUNITS,
            "currency": "CNY",
            "status": "conservative_project_estimate_not_provider_invoice",
        },
        "cells": cells,
    }
    spec["preregistration_hash"] = _normalized_hash(spec, "preregistration_hash")
    _write_json(spec_path, spec)
    return spec


async def run(spec_path: Path, output: Path) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec["preregistration_hash"] != _normalized_hash(spec, "preregistration_hash"):
        raise RuntimeError("preregistration hash mismatch")
    if spec["source_rows_sha256"] != _file_sha256(SOURCE_ROWS):
        raise RuntimeError("source rows changed after preregistration")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "plan.json", spec)
    rows_path = output / "rows.json"
    rows: list[dict[str, Any]] = (
        json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else []
    )
    completed = {str(row["persona_id"]) for row in rows}
    remaining = len(spec["cells"]) - len(rows)
    price = spec["price_snapshot"]
    guard = RealModelCostGuard(
        RealModelBudget(
            max_calls=max(1, remaining),
            max_prompt_tokens=max(1, remaining) * RESERVED_PROMPT_TOKENS,
            max_completion_tokens=max(1, remaining) * RESERVED_COMPLETION_TOKENS,
            max_estimated_cost_microunits=max(1, remaining)
            * (
                RESERVED_PROMPT_TOKENS * int(price["input_price_microunits_per_token"])
                + RESERVED_COMPLETION_TOKENS
                * int(price["output_price_microunits_per_token"])
            ),
        ),
        explicitly_authorized=True,
    )
    client = BudgetedModelClient(
        DeepSeekModelClient(
            model=str(spec["model"]),
            temperature=int(spec["temperature_ppm"]) / 1_000_000,
            top_p=int(spec["top_p_ppm"]) / 1_000_000,
            max_schema_attempts=1,
            max_transport_retries=0,
        ),
        cost_guard=guard,
        reserved_prompt_tokens_per_call=RESERVED_PROMPT_TOKENS,
        reserved_completion_tokens_per_call=RESERVED_COMPLETION_TOKENS,
        input_price_microunits_per_token=int(price["input_price_microunits_per_token"]),
        output_price_microunits_per_token=int(price["output_price_microunits_per_token"]),
    )
    cell_specs = {str(cell["persona_id"]): cell for cell in spec["cells"]}
    for persona_id, cell in cell_specs.items():
        if persona_id in completed:
            continue
        material = _material(persona_id)
        if material["observation"]["observation_hash"] != cell["shuffled_observation_hash"]:
            raise AssertionError(f"observation changed: {persona_id}")
        runtime = AgentRuntime(
            agent_id=f"opponent-model-value-{persona_id}",
            company_id=FOCAL,
            model_client=client,
            memory=EpisodeMemory(),
            persona_profile=material["personas"].get(persona_id),
            persona_registry=material["personas"],
        )
        result = await runtime.decide(material["observation"], timeout_seconds=90.0)
        row: dict[str, Any] = {
            "seed": SEED,
            "window": WINDOW,
            "persona_id": persona_id,
            "success": result.success,
            "input_tokens": int(result.input_tokens or 0),
            "output_tokens": int(result.output_tokens or 0),
            "model_name": result.model_name,
            "raw_response": result.raw_response,
            "error_code": result.error_code,
            "error_message": result.error_message,
        }
        if result.success and result.decision is not None:
            requested = result.decision.requested_action.model_dump(mode="json")
            resolution = resolve_action_request(
                material["config"],
                material["state"],
                FOCAL,
                requested,
                source="opponent-model-value-real-v1",
            )
            action = resolution.action.to_dict()
            outcome = _actual_outcome(
                material["config"], material["state"], "shuffled_model", action
            )
            ev = int(outcome["enterprise_value_cents"])
            row.update(
                {
                    "requested_action": requested,
                    "final_action": action,
                    "plan": result.decision.plan.model_dump(mode="json"),
                    "outcome": outcome,
                    "delta_vs_baseline_cents": ev
                    - int(cell["baseline_enterprise_value_cents"]),
                    "delta_vs_correct_model_cents": ev
                    - int(cell["correct_model_enterprise_value_cents"]),
                    "research_only_price_coordination_action": (
                        _is_research_only_coordination(action)
                    ),
                }
            )
        rows.append(row)
        _write_json(rows_path, rows)
        print(
            f"[{len(rows)}/{len(cell_specs)}] {persona_id} success={result.success} "
            f"tokens={row['input_tokens'] + row['output_tokens']}",
            flush=True,
        )
    successful = [row for row in rows if row.get("success") and row.get("outcome")]
    deltas_correct = [int(row["delta_vs_correct_model_cents"]) for row in successful]
    deltas_baseline = [int(row["delta_vs_baseline_cents"]) for row in successful]
    summary: dict[str, Any] = {
        "result_schema_version": "opponent-model-value-result-v1.0.0",
        "evidence_level": spec["evidence_level"],
        "preregistration_hash": spec["preregistration_hash"],
        "attempted_calls": len(rows),
        "successful_calls": len(successful),
        "mean_shuffled_minus_correct_model_ev_cents": (
            round(mean(deltas_correct)) if deltas_correct else None
        ),
        "mean_shuffled_minus_baseline_ev_cents": (
            round(mean(deltas_baseline)) if deltas_baseline else None
        ),
        "correct_model_better_count": sum(value < 0 for value in deltas_correct),
        "shuffled_model_better_count": sum(value > 0 for value in deltas_correct),
        "usage": {
            "calls": len(rows),
            "prompt_tokens": sum(int(row["input_tokens"]) for row in rows),
            "completion_tokens": sum(int(row["output_tokens"]) for row in rows),
            "estimated_cost_microunits": sum(
                int(row["input_tokens"])
                * int(price["input_price_microunits_per_token"])
                + int(row["output_tokens"])
                * int(price["output_price_microunits_per_token"])
                for row in rows
            ),
        },
        "gates": {
            "all_calls_recorded": len(rows) == len(spec["cells"]),
            "all_calls_successful": len(successful) == len(spec["cells"]),
            "correct_model_outperforms_shuffled_on_average": bool(deltas_correct)
            and mean(deltas_correct) < 0,
            "no_research_only_coordination_executed": not any(
                bool(row.get("research_only_price_coordination_action")) for row in rows
            ),
        },
        "rows": [
            {
                key: row.get(key)
                for key in (
                    "persona_id",
                    "success",
                    "delta_vs_baseline_cents",
                    "delta_vs_correct_model_cents",
                )
            }
            for row in rows
        ],
    }
    summary["result_hash"] = _normalized_hash(summary, "result_hash")
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--authorize-real-model", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.spec.resolve()), ensure_ascii=False, indent=2))
        return
    if not args.authorize_real_model:
        raise RuntimeError("--authorize-real-model is required")
    print(
        json.dumps(
            asyncio.run(run(args.spec.resolve(), args.output.resolve())),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
