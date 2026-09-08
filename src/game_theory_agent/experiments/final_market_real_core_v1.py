"""Pre-registered, cost-bounded real-LLM validation of final-market Advisor v9.

The experiment is intentionally paired and fixed-state.  It tests whether the
same model, persona and market state behave differently with v9 advice, then
settles both actions against the same deterministic three-round continuation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from dotenv import load_dotenv

from game_theory_agent.advisor import build_advisor_adoption_trace
from game_theory_agent.agents import AgentRuntime, EpisodeMemory
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.experiments.final_market_advisor_v9_validation import (
    CONFIG_PATH,
    FOCAL,
    HORIZON,
    REAL_CORE_WINDOWS,
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
from game_theory_agent.strategic_reliability import (
    PublicMarketRolloutAdvisor,
    RealModelBudget,
    RealModelCostGuard,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = (
    PROJECT_ROOT
    / "experiment-specs"
    / "final-market-real-core-v1"
    / "PREREGISTRATION.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "final-market-real-core-v1"
SEEDS = (95101, 95102, 95104)
PERSONAS = ("balanced_v1", "aggressive_v1_extreme", "risk_guarded_v1")
CONDITIONS = ("baseline_without_advice", "treatment_with_v9_advice")
PROVIDER = "deepseek"
RESERVED_PROMPT_TOKENS = 32_000
RESERVED_COMPLETION_TOKENS = 4_000
INPUT_PRICE_MICROUNITS = 3
OUTPUT_PRICE_MICROUNITS = 9


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _result_hash(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("result_hash", None)
    return sha256_hash(normalized)


def _preregistration_hash(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("preregistration_hash", None)
    return sha256_hash(normalized)


def prepare(spec_path: Path) -> dict[str, Any]:
    if spec_path.exists():
        raise RuntimeError("preregistration already exists; refusing to overwrite it")
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    advisor = PublicMarketRolloutAdvisor(config)
    cells: list[dict[str, Any]] = []
    for seed in SEEDS:
        for window in REAL_CORE_WINDOWS:
            state, belief, opponent, observation = _build_window(config, seed, window)
            if FOCAL not in state.strategic_market.active_company_ids:
                raise AssertionError(f"ineligible focal company for {seed}/{window}")
            for persona_id in PERSONAS:
                advice = advisor.advise(
                    observation=observation,
                    company_id=FOCAL,
                    persona_profile=personas.get(persona_id),
                    belief_state=belief,
                    opponent_model=opponent,
                    horizon_rounds=HORIZON,
                    scenario_count=SCENARIOS,
                    advisor_mode="strategic_market_v9",
                ).model_dump(mode="json")
                order = (
                    list(CONDITIONS)
                    if (seed + len(window) + len(persona_id)) % 2 == 0
                    else list(reversed(CONDITIONS))
                )
                cells.append(
                    {
                        "seed": seed,
                        "window": window,
                        "persona_id": persona_id,
                        "state_hash": state.state_hash,
                        "advice_hash": advice["advice_hash"],
                        "expected_disposition": advice["execution_disposition"],
                        "expected_recommended_candidate_id": advice[
                            "recommended_candidate_id"
                        ],
                        "call_order": order,
                    }
                )
    call_count = len(cells) * len(CONDITIONS)
    spec: dict[str, Any] = {
        "experiment_schema_version": (
            "final-market-real-core-preregistration-v1.0.0"
        ),
        "evidence_level": "DIRECTIONAL_REAL_LLM_PAIRED_EVIDENCE",
        "authorization": "user_explicitly_authorized_real_llm_experiments",
        "provider": PROVIDER,
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "temperature_ppm": 0,
        "top_p_ppm": 100_000,
        "seeds": list(SEEDS),
        "seed_eligibility_rule": (
            "first three seeds from 95101 upward for which the focal company "
            "is active in all four frozen windows; 95103 was mechanically "
            "ineligible because it exited before mature_shortage"
        ),
        "windows": list(REAL_CORE_WINDOWS),
        "personas": list(PERSONAS),
        "conditions": list(CONDITIONS),
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
        "frozen_controls": [
            "market_state",
            "market_seed",
            "persona",
            "belief",
            "opponent_model",
            "provider",
            "model",
            "three_round_rule_continuation",
        ],
        "primary_metrics": [
            "paired_enterprise_value_delta_cents",
            "paired_candidate_regret_reduction_cents",
            "action_change_rate",
            "advice_adoption_rate",
            "worst_paired_enterprise_value_delta_cents",
        ],
        "safety_metrics": [
            "research_only_price_coordination_action_count",
            "failed_call_count",
            "token_usage",
            "estimated_cost",
        ],
        "cells": cells,
    }
    spec["preregistration_hash"] = _preregistration_hash(spec)
    _write_json(spec_path, spec)
    return spec


def _load_cells(spec: Mapping[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    advisor = PublicMarketRolloutAdvisor(config)
    rebuilt: dict[tuple[int, str, str], dict[str, Any]] = {}
    for item in spec["cells"]:
        key = (int(item["seed"]), str(item["window"]), str(item["persona_id"]))
        state, belief, opponent, observation = _build_window(config, key[0], key[1])
        advice = advisor.advise(
            observation=observation,
            company_id=FOCAL,
            persona_profile=personas.get(key[2]),
            belief_state=belief,
            opponent_model=opponent,
            horizon_rounds=HORIZON,
            scenario_count=SCENARIOS,
            advisor_mode="strategic_market_v9",
        ).model_dump(mode="json")
        if state.state_hash != item["state_hash"]:
            raise AssertionError(f"state changed after preregistration: {key}")
        if advice["advice_hash"] != item["advice_hash"]:
            raise AssertionError(f"advice changed after preregistration: {key}")
        if advice["execution_disposition"] != item["expected_disposition"]:
            raise AssertionError(f"advice disposition changed: {key}")
        best_id, best_outcome = _best_operational_outcome(config, state, advice)
        rebuilt[key] = {
            "state": state,
            "observation": observation,
            "advice": advice,
            "best_candidate_id": best_id,
            "best_outcome": best_outcome,
        }
    return rebuilt


async def run(spec_path: Path, output: Path) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("preregistration_hash") != _preregistration_hash(spec):
        raise RuntimeError("preregistration hash mismatch")
    cells = _load_cells(spec)
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "plan.json", spec)
    rows_path = output / "rows.json"
    rows: list[dict[str, Any]] = (
        json.loads(rows_path.read_text(encoding="utf-8"))
        if rows_path.exists()
        else []
    )
    completed = {
        (row["seed"], row["window"], row["persona_id"], row["condition"])
        for row in rows
    }
    remaining = int(spec["maximum_logical_provider_calls"]) - len(rows)
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
        input_price_microunits_per_token=int(
            price["input_price_microunits_per_token"]
        ),
        output_price_microunits_per_token=int(
            price["output_price_microunits_per_token"]
        ),
    )
    total = int(spec["maximum_logical_provider_calls"])
    for item in spec["cells"]:
        key = (int(item["seed"]), str(item["window"]), str(item["persona_id"]))
        cell = cells[key]
        for condition in item["call_order"]:
            call_key = (*key, condition)
            if call_key in completed:
                continue
            advice = (
                cell["advice"] if condition == "treatment_with_v9_advice" else None
            )
            observation = _complete_observation(
                config, cell["state"], cell["observation"], advice=advice
            )
            runtime = AgentRuntime(
                agent_id=f"real-core-{key[0]}-{key[1]}-{key[2]}-{condition}",
                company_id=FOCAL,
                model_client=client,
                memory=EpisodeMemory(),
                persona_profile=personas.get(key[2]),
                persona_registry=personas,
            )
            result = await runtime.decide(observation, timeout_seconds=90.0)
            row: dict[str, Any] = {
                "seed": key[0],
                "window": key[1],
                "persona_id": key[2],
                "condition": condition,
                "state_hash": cell["state"].state_hash,
                "observation_hash": observation["observation_hash"],
                "advice_hash": cell["advice"]["advice_hash"] if advice else None,
                "success": result.success,
                "model_name": result.model_name,
                "prompt_version": result.prompt_version,
                "input_tokens": int(result.input_tokens or 0),
                "output_tokens": int(result.output_tokens or 0),
                "latency_ms": result.latency_ms,
                "retry_count": result.retry_count,
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
                    config,
                    cell["state"],
                    FOCAL,
                    requested,
                    source=f"real-core:{condition}",
                )
                final_action = resolution.action.to_dict()
                outcome = _actual_outcome(
                    config, cell["state"], condition, final_action
                )
                best_ev = int(cell["best_outcome"]["enterprise_value_cents"])
                adoption = build_advisor_adoption_trace(
                    advice=advice,
                    llm_requested_action=requested,
                    final_action=final_action,
                    planner_output=result.decision.plan.model_dump(mode="json"),
                )
                row.update(
                    {
                        "requested_action": requested,
                        "final_action": final_action,
                        "plan": result.decision.plan.model_dump(mode="json"),
                        "adjustments": [
                            adjustment.to_dict()
                            for adjustment in resolution.adjustments
                        ],
                        "outcome": outcome,
                        "candidate_regret_cents": max(
                            0, best_ev - int(outcome["enterprise_value_cents"])
                        ),
                        "research_only_price_coordination_action": (
                            _is_research_only_coordination(final_action)
                        ),
                        "advisor_adoption": (
                            adoption.model_dump(mode="json")
                            if adoption is not None
                            else None
                        ),
                    }
                )
            rows.append(row)
            completed.add(call_key)
            _write_json(rows_path, rows)
            print(
                f"[{len(rows)}/{total}] {call_key} success={result.success} "
                f"tokens={row['input_tokens'] + row['output_tokens']}",
                flush=True,
            )

    pairs: list[dict[str, Any]] = []
    for item in spec["cells"]:
        key = (int(item["seed"]), str(item["window"]), str(item["persona_id"]))
        selected = [
            row
            for row in rows
            if (row["seed"], row["window"], row["persona_id"]) == key
        ]
        baseline = next(
            (row for row in selected if row["condition"] == CONDITIONS[0]), None
        )
        treatment = next(
            (row for row in selected if row["condition"] == CONDITIONS[1]), None
        )
        if not baseline or not treatment or not baseline["success"] or not treatment["success"]:
            continue
        baseline_ev = int(baseline["outcome"]["enterprise_value_cents"])
        treatment_ev = int(treatment["outcome"]["enterprise_value_cents"])
        baseline_regret = int(baseline["candidate_regret_cents"])
        treatment_regret = int(treatment["candidate_regret_cents"])
        adoption = treatment.get("advisor_adoption") or {}
        pairs.append(
            {
                "seed": key[0],
                "window": key[1],
                "persona_id": key[2],
                "state_hash": baseline["state_hash"],
                "enterprise_value_delta_cents": treatment_ev - baseline_ev,
                "candidate_regret_reduction_cents": (
                    baseline_regret - treatment_regret
                ),
                "action_changed": (
                    baseline["final_action"] != treatment["final_action"]
                ),
                "advice_accepted": adoption.get("accepted"),
                "advice_candidate_id": item[
                    "expected_recommended_candidate_id"
                ],
                "disposition": item["expected_disposition"],
            }
        )
    actual = asdict(guard.actual)
    all_usage = {
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
        "latest_process_guard_usage": actual,
    }
    ev_deltas = [int(pair["enterprise_value_delta_cents"]) for pair in pairs]
    regret_reductions = [
        int(pair["candidate_regret_reduction_cents"]) for pair in pairs
    ]
    recommended_pairs = [
        pair for pair in pairs if pair["disposition"] == "recommend"
    ]
    summary: dict[str, Any] = {
        "result_schema_version": "final-market-real-core-result-v1.0.0",
        "evidence_level": spec["evidence_level"],
        "preregistration_hash": spec["preregistration_hash"],
        "attempted_logical_calls": len(rows),
        "successful_calls": sum(bool(row["success"]) for row in rows),
        "complete_pair_count": len(pairs),
        "pair_count_by_window": {
            window: sum(pair["window"] == window for pair in pairs)
            for window in REAL_CORE_WINDOWS
        },
        "mean_enterprise_value_delta_cents": (
            round(mean(ev_deltas)) if ev_deltas else None
        ),
        "median_enterprise_value_delta_cents": (
            sorted(ev_deltas)[len(ev_deltas) // 2] if ev_deltas else None
        ),
        "worst_enterprise_value_delta_cents": min(ev_deltas) if ev_deltas else None,
        "positive_pair_count": sum(value > 0 for value in ev_deltas),
        "equal_pair_count": sum(value == 0 for value in ev_deltas),
        "negative_pair_count": sum(value < 0 for value in ev_deltas),
        "mean_candidate_regret_reduction_cents": (
            round(mean(regret_reductions)) if regret_reductions else None
        ),
        "action_change_rate_ppm": (
            round(sum(pair["action_changed"] for pair in pairs) * 1_000_000 / len(pairs))
            if pairs
            else None
        ),
        "recommended_adoption_rate_ppm": (
            round(
                sum(pair["advice_accepted"] is True for pair in recommended_pairs)
                * 1_000_000
                / len(recommended_pairs)
            )
            if recommended_pairs
            else None
        ),
        "research_only_price_coordination_action_count": sum(
            bool(row.get("research_only_price_coordination_action")) for row in rows
        ),
        "usage": all_usage,
        "gates": {
            "all_preregistered_calls_recorded": len(rows)
            == int(spec["maximum_logical_provider_calls"]),
            "all_calls_successful": all(bool(row["success"]) for row in rows),
            "all_pairs_complete": len(pairs) == len(spec["cells"]),
            "no_research_only_coordination_executed": not any(
                bool(row.get("research_only_price_coordination_action"))
                for row in rows
            ),
            "mean_enterprise_value_non_decreasing": bool(ev_deltas)
            and mean(ev_deltas) >= 0,
            "mean_candidate_regret_non_increasing": bool(regret_reductions)
            and mean(regret_reductions) >= 0,
        },
        "pairs": pairs,
    }
    summary["result_hash"] = _result_hash(summary)
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
    summary = asyncio.run(run(args.spec.resolve(), args.output.resolve()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
