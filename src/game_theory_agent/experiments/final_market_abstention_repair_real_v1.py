"""Targeted real-LLM verification of the Advisor abstention context repair.

This experiment is deliberately narrow and post-hoc.  It re-runs only the five
negative cells from final-market-real-core-v1 after hiding candidate anchors
when the Advisor disposition is ``defer_to_agent``.  It is repair evidence,
not an independent estimate of the Advisor's average effect.
"""

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
    _best_operational_outcome,
    _complete_observation,
    _is_research_only_coordination,
)
from game_theory_agent.market import load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.model_clients import BudgetedModelClient, DeepSeekModelClient
from game_theory_agent.information import seal_observation
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
    / "final-market-abstention-repair-real-v1"
    / "PREREGISTRATION.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "final-market-abstention-repair-real-v1"
CELLS = (
    (95102, "price_war", "risk_guarded_v1"),
    (95104, "pivotal_project", "balanced_v1"),
    (95104, "mature_shortage", "balanced_v1"),
    (95104, "mature_shortage", "aggressive_v1_extreme"),
    (95104, "mature_shortage", "risk_guarded_v1"),
)
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


def _historical_compact_abstention_view(advice: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce the intermediate v2 view frozen by this experiment's spec."""

    return {
        "agent_context_view_version": "advisor-agent-view-v2.0.0",
        "advice_schema_version": advice.get("advice_schema_version"),
        "advisor_mode": advice.get("advisor_mode"),
        "advice_hash": advice.get("advice_hash"),
        "execution_disposition": "defer_to_agent",
        "should_abstain": True,
        "recommended_candidate_id": None,
        "recommended_action": None,
        "candidate_actions": [],
        "research_only_candidate_ids": [],
        "full_advice_withheld": True,
        "withholding_reason": (
            "可靠性门禁未释放建议；候选身份、分数和动作不进入Agent上下文，"
            "以防止弃权后的锚定干预。"
        ),
    }


def _cell_material(
    seed: int,
    window: str,
    persona_id: str,
) -> dict[str, Any]:
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    state, belief, opponent, observation = _build_window(config, seed, window)
    advice = PublicMarketRolloutAdvisor(config).advise(
        observation=observation,
        company_id=FOCAL,
        persona_profile=personas.get(persona_id),
        belief_state=belief,
        opponent_model=opponent,
        horizon_rounds=HORIZON,
        scenario_count=SCENARIOS,
        advisor_mode="strategic_market_v9",
    ).model_dump(mode="json")
    if advice["execution_disposition"] != "defer_to_agent":
        raise AssertionError(f"repair cell no longer abstains: {seed}/{window}/{persona_id}")
    agent_view = _historical_compact_abstention_view(advice)
    if not agent_view.get("full_advice_withheld"):
        raise AssertionError("abstention view was not withheld")
    if agent_view.get("candidate_actions"):
        raise AssertionError("candidate anchors leaked into abstention view")
    completed_observation = _complete_observation(
        config, state, observation, advice=None
    )
    completed_observation["game_theory_advice"] = agent_view
    completed_observation = seal_observation(completed_observation)
    best_id, best_outcome = _best_operational_outcome(config, state, advice)
    return {
        "config": config,
        "personas": personas,
        "state": state,
        "advice": advice,
        "observation": completed_observation,
        "best_candidate_id": best_id,
        "best_outcome": best_outcome,
    }


def prepare(spec_path: Path) -> dict[str, Any]:
    if spec_path.exists():
        raise RuntimeError("preregistration already exists; refusing to overwrite it")
    source = _source_lookup()
    cells: list[dict[str, Any]] = []
    for seed, window, persona_id in CELLS:
        material = _cell_material(seed, window, persona_id)
        baseline = source[(seed, window, persona_id, "baseline_without_advice")]
        old_treatment = source[(seed, window, persona_id, "treatment_with_v9_advice")]
        if not baseline.get("success") or not old_treatment.get("success"):
            raise AssertionError(f"source pair is incomplete: {seed}/{window}/{persona_id}")
        cells.append(
            {
                "seed": seed,
                "window": window,
                "persona_id": persona_id,
                "state_hash": material["state"].state_hash,
                "full_advice_hash": material["advice"]["advice_hash"],
                "sanitized_observation_hash": material["observation"]["observation_hash"],
                "original_baseline_enterprise_value_cents": int(
                    baseline["outcome"]["enterprise_value_cents"]
                ),
                "original_treatment_enterprise_value_cents": int(
                    old_treatment["outcome"]["enterprise_value_cents"]
                ),
            }
        )
    call_count = len(cells)
    spec: dict[str, Any] = {
        "experiment_schema_version": "final-market-abstention-repair-prereg-v1.0.0",
        "evidence_level": "POST_HOC_DIRECTIONAL_REAL_LLM_REPAIR_EVIDENCE",
        "provider": "deepseek",
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "temperature_ppm": 0,
        "top_p_ppm": 100_000,
        "source_experiment": "final-market-real-core-v1",
        "source_rows_sha256": _file_sha256(SOURCE_ROWS),
        "selection_rule": "all five negative complete pairs in the frozen source summary",
        "intervention": (
            "withhold all candidate identities, actions and scores from the Agent when "
            "Advisor execution_disposition is defer_to_agent"
        ),
        "maximum_logical_provider_calls": call_count,
        "reserved_prompt_tokens_per_call": RESERVED_PROMPT_TOKENS,
        "reserved_completion_tokens_per_call": RESERVED_COMPLETION_TOKENS,
        "maximum_prompt_tokens": call_count * RESERVED_PROMPT_TOKENS,
        "maximum_completion_tokens": call_count * RESERVED_COMPLETION_TOKENS,
        "maximum_estimated_cost_microunits": call_count
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
        "primary_metrics": [
            "delta_vs_original_baseline_cents",
            "recovery_vs_unsanitized_treatment_cents",
            "negative_pair_count_after_repair",
        ],
        "success_criterion": (
            "candidate anchors absent in every prompt and negative-pair count lower than five"
        ),
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
    completed = {
        (int(row["seed"]), str(row["window"]), str(row["persona_id"]))
        for row in rows
    }
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
    cell_specs = {
        (int(cell["seed"]), str(cell["window"]), str(cell["persona_id"])): cell
        for cell in spec["cells"]
    }
    for key, cell_spec in cell_specs.items():
        if key in completed:
            continue
        material = _cell_material(*key)
        if material["state"].state_hash != cell_spec["state_hash"]:
            raise AssertionError(f"state hash changed: {key}")
        if material["advice"]["advice_hash"] != cell_spec["full_advice_hash"]:
            raise AssertionError(f"advice hash changed: {key}")
        if material["observation"]["observation_hash"] != cell_spec["sanitized_observation_hash"]:
            raise AssertionError(f"observation hash changed: {key}")
        runtime = AgentRuntime(
            agent_id=f"abstention-repair-{key[0]}-{key[1]}-{key[2]}",
            company_id=FOCAL,
            model_client=client,
            memory=EpisodeMemory(),
            persona_profile=material["personas"].get(key[2]),
            persona_registry=material["personas"],
        )
        result = await runtime.decide(material["observation"], timeout_seconds=90.0)
        row: dict[str, Any] = {
            "seed": key[0],
            "window": key[1],
            "persona_id": key[2],
            "state_hash": material["state"].state_hash,
            "observation_hash": material["observation"]["observation_hash"],
            "full_advice_hash": material["advice"]["advice_hash"],
            "candidate_anchors_visible": bool(
                material["observation"]["game_theory_advice"]["candidate_actions"]
            ),
            "full_advice_withheld": bool(
                material["observation"]["game_theory_advice"]["full_advice_withheld"]
            ),
            "success": result.success,
            "model_name": result.model_name,
            "prompt_version": result.prompt_version,
            "input_tokens": int(result.input_tokens or 0),
            "output_tokens": int(result.output_tokens or 0),
            "latency_ms": result.latency_ms,
            "error_code": result.error_code,
            "error_message": result.error_message,
            "raw_response": result.raw_response,
            "provider_audit": (
                result.provider_audit.model_dump(mode="json")
                if result.provider_audit
                else None
            ),
        }
        if result.success and result.decision is not None:
            requested = result.decision.requested_action.model_dump(mode="json")
            resolution = resolve_action_request(
                material["config"],
                material["state"],
                FOCAL,
                requested,
                source="abstention-repair-real-v1",
            )
            final_action = resolution.action.to_dict()
            outcome = _actual_outcome(
                material["config"], material["state"], "repaired_treatment", final_action
            )
            best_ev = int(material["best_outcome"]["enterprise_value_cents"])
            row.update(
                {
                    "requested_action": requested,
                    "final_action": final_action,
                    "plan": result.decision.plan.model_dump(mode="json"),
                    "outcome": outcome,
                    "candidate_regret_cents": max(
                        0, best_ev - int(outcome["enterprise_value_cents"])
                    ),
                    "research_only_price_coordination_action": (
                        _is_research_only_coordination(final_action)
                    ),
                    "delta_vs_original_baseline_cents": int(
                        outcome["enterprise_value_cents"]
                    )
                    - int(cell_spec["original_baseline_enterprise_value_cents"]),
                    "recovery_vs_unsanitized_treatment_cents": int(
                        outcome["enterprise_value_cents"]
                    )
                    - int(cell_spec["original_treatment_enterprise_value_cents"]),
                }
            )
        rows.append(row)
        _write_json(rows_path, rows)
        print(
            f"[{len(rows)}/{len(cell_specs)}] {key} success={result.success} "
            f"tokens={row['input_tokens'] + row['output_tokens']}",
            flush=True,
        )

    successful = [row for row in rows if row.get("success") and row.get("outcome")]
    deltas = [int(row["delta_vs_original_baseline_cents"]) for row in successful]
    recoveries = [
        int(row["recovery_vs_unsanitized_treatment_cents"]) for row in successful
    ]
    usage = {
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
    }
    summary: dict[str, Any] = {
        "result_schema_version": "final-market-abstention-repair-result-v1.0.0",
        "evidence_level": spec["evidence_level"],
        "preregistration_hash": spec["preregistration_hash"],
        "attempted_calls": len(rows),
        "successful_calls": len(successful),
        "negative_pair_count_after_repair": sum(value < 0 for value in deltas),
        "nonnegative_pair_count_after_repair": sum(value >= 0 for value in deltas),
        "mean_delta_vs_original_baseline_cents": round(mean(deltas)) if deltas else None,
        "mean_recovery_vs_unsanitized_treatment_cents": (
            round(mean(recoveries)) if recoveries else None
        ),
        "worst_delta_vs_original_baseline_cents": min(deltas) if deltas else None,
        "usage": usage,
        "gates": {
            "all_calls_recorded": len(rows) == len(spec["cells"]),
            "all_calls_successful": len(successful) == len(spec["cells"]),
            "candidate_anchors_absent": all(
                not bool(row["candidate_anchors_visible"]) for row in rows
            ),
            "full_advice_withheld": all(
                bool(row["full_advice_withheld"]) for row in rows
            ),
            "negative_pair_count_reduced": bool(deltas) and sum(v < 0 for v in deltas) < 5,
            "no_research_only_coordination_executed": not any(
                bool(row.get("research_only_price_coordination_action")) for row in rows
            ),
        },
        "rows": [
            {
                key: row.get(key)
                for key in (
                    "seed",
                    "window",
                    "persona_id",
                    "success",
                    "delta_vs_original_baseline_cents",
                    "recovery_vs_unsanitized_treatment_cents",
                    "candidate_anchors_visible",
                    "full_advice_withheld",
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
