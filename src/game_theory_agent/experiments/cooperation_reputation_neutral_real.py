"""Native Agent fixed-state reputation experiment with neutral cooperation wording."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from game_theory_agent.agents import (
    AgentRuntime,
    DecisionContextBuilder,
    EpisodeMemory,
    load_persona_registry,
)
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.experiments.canonical_validity_real import PRICE_SNAPSHOT
from game_theory_agent.experiments.persona_pilot import PROJECT_ROOT, _model_client, _observation
from game_theory_agent.experiments.real_cooperation_counterfactual import (
    build_condition_observation,
)
from game_theory_agent.information import compute_observation_hash
from game_theory_agent.market import MarketConfig, MarketEnv, load_market_config
from game_theory_agent.model_clients import BudgetedModelClient
from game_theory_agent.strategic_reliability import RealModelBudget, RealModelCostGuard


CONDITIONS = ("no_history", "recent_betrayal", "repaired_after_betrayal")
PROVIDERS = ("doubao", "deepseek")
REPETITIONS = (1, 2, 3)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "cooperation-reputation-neutral-real-v1"


def cooperation_capability_overlay(base: MarketConfig) -> MarketConfig:
    """Enable the action capability without mutating the replay-stable v4 file.

    Utility weights remain unchanged at zero, so this grants permission to use
    cooperation but does not create an intrinsic reward for cooperating.
    """

    payload = base.to_dict()
    payload["persona_utilities"]["capabilities"]["cooperation"] = True
    return MarketConfig.from_mapping(payload)


def _credibility_record(condition: str) -> dict[str, Any]:
    if condition == "no_history":
        values = (0, 0, 0, 0, 0, 500_000)
    elif condition == "recent_betrayal":
        values = (1, 0, 0, 1, 1_000_000, 100_000)
    elif condition == "repaired_after_betrayal":
        values = (2, 1, 0, 1, 2_000_000, 700_000)
    else:
        raise ValueError(f"unsupported condition: {condition}")
    verified, fulfilled, partial, betrayed, promised, credibility = values
    actual = 0 if condition == "recent_betrayal" else 1_400_000 if condition == "repaired_after_betrayal" else 0
    return {
        "credibility_schema_version": "credibility-v1.0.0",
        "company_id": "company_A",
        "verified_commitment_count": verified,
        "fulfilled_count": fulfilled,
        "partial_betrayal_count": partial,
        "betrayed_count": betrayed,
        "total_promised_contribution_cents": promised,
        "total_actual_capped_contribution_cents": actual,
        "credibility_ppm": credibility,
    }


def _memory_record(condition: str) -> dict[str, Any]:
    credibility = _credibility_record(condition)
    return {
        "company_id": "company_A",
        "proposals_received": 1,
        "proposals_sent": 1,
        "accepted_by_self": 0,
        "accepted_by_opponent": 1,
        "commitments_by_opponent": credibility["verified_commitment_count"],
        "fulfilled_by_opponent": credibility["fulfilled_count"],
        "partial_betrayals_by_opponent": credibility["partial_betrayal_count"],
        "betrayed_by_opponent": credibility["betrayed_count"],
        "promised_by_opponent_cents": credibility["total_promised_contribution_cents"],
        "fulfilled_by_opponent_cents": credibility["total_actual_capped_contribution_cents"],
        "credibility_ppm": credibility["credibility_ppm"],
        "history_is_neutralized": False,
    }


def build_reputation_observation(base: dict[str, Any], condition: str) -> dict[str, Any]:
    observation = build_condition_observation(base, condition="cooperation_proposal")
    record = _credibility_record(condition)
    observation["cooperation"]["public_credibility"]["company_A"] = record
    observation["cooperation"]["cooperation_memory"] = {
        "company_A": _memory_record(condition)
    }
    observation["cooperation"]["reputation_treatment"] = {
        "condition": condition,
        "history_is_exogenous": True,
        "same_current_proposal_across_conditions": True,
    }
    observation["observation_hash"] = "pending"
    observation["observation_hash"] = compute_observation_hash(observation)
    return observation


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _disposition(result: Any) -> str | None:
    if not result.success or result.decision is None:
        return None
    responses = result.decision.message_responses
    return responses[0].disposition if responses else "ignored"


async def run(output: Path) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("output directory must be new and empty")
    output.mkdir(parents=True, exist_ok=True)
    config_path = Path(os.getenv("MARKET_CONFIG_PATH", PROJECT_ROOT / "configs" / "market_v4.yaml"))
    base_config = load_market_config(config_path)
    config = cooperation_capability_overlay(base_config)
    registry = load_persona_registry(config_path)
    profile = registry.get("balanced_v1")
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B", "company_C", "company_D"),
        episode_id="cooperation-reputation-neutral-920201",
        episode_seed=920_201,
        market_model="balanced",
        max_rounds=10,
        cooperation_mode="shared_resilience_v1",
    )
    base = _observation(config, state, "company_B")
    base["shared_resilience"] = state.shared_resilience.to_dict()
    observations = {
        condition: build_reputation_observation(base, condition)
        for condition in CONDITIONS
    }
    manifest = {
        "schema_version": "cooperation-reputation-neutral-real-manifest-v1.0.0",
        "planned_calls": 18,
        "providers": list(PROVIDERS),
        "repetitions": list(REPETITIONS),
        "conditions": list(CONDITIONS),
        "persona": "balanced_v1",
        "cooperation_prompt_variant": "neutral_numeric_v1",
        "config_overlay": {
            "base_config_id": base_config.config_id,
            "base_config_version": base_config.config_version,
            "base_config_sha256": base_config.config_sha256,
            "effective_config_sha256": config.config_sha256,
            "persona_cooperation_capability": True,
            "persona_cooperation_utility_weights_changed": False,
        },
        "state_hash": state.state_hash,
        "observation_hashes": {key: value["observation_hash"] for key, value in observations.items()},
        "price_snapshot": PRICE_SNAPSHOT,
        "evidence_level": "fixed_state_directional_real_model_evidence",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    guard = RealModelCostGuard(
        RealModelBudget(
            max_calls=18,
            max_prompt_tokens=360_000,
            max_completion_tokens=72_000,
            max_estimated_cost_microunits=2_000_000,
        ),
        explicitly_authorized=True,
    )
    clients: dict[str, Any] = {}
    for provider in PROVIDERS:
        price = PRICE_SNAPSHOT[provider]
        clients[provider] = BudgetedModelClient(
            _model_client(provider, None, temperature=0.0, top_p=1.0),
            cost_guard=guard,
            reserved_prompt_tokens_per_call=20_000,
            reserved_completion_tokens_per_call=4_000,
            input_price_microunits_per_token=int(price["input_microunits_per_token"]),
            output_price_microunits_per_token=int(price["output_microunits_per_token"]),
        )
    rows: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        for repetition in REPETITIONS:
            shift = (repetition - 1) % len(CONDITIONS)
            condition_order = CONDITIONS[shift:] + CONDITIONS[:shift]
            for condition in condition_order:
                runtime = AgentRuntime(
                    agent_id=f"cooperation-reputation-{provider}-{repetition}-{condition}",
                    company_id="company_B",
                    model_client=clients[provider],
                    memory=EpisodeMemory(),
                    context_builder=DecisionContextBuilder(
                        persona_profile=profile,
                        persona_registry=registry,
                        cooperation_history_mode="full",
                        cooperation_prompt_variant="neutral_numeric_v1",
                    ),
                )
                result = await runtime.decide(observations[condition], timeout_seconds=90.0)
                # Persist the model boundary before action resolution.
                row: dict[str, Any] = {
                    "provider": provider,
                    "model": result.model_name,
                    "repetition": repetition,
                    "condition": condition,
                    "state_hash": state.state_hash,
                    "observation_hash": observations[condition]["observation_hash"],
                    "success": result.success,
                    "error_code": result.error_code,
                    "error_message": result.error_message,
                    "prompt_version": result.prompt_version,
                    "raw_response": result.raw_response,
                    "input_tokens": int(result.input_tokens or 0),
                    "output_tokens": int(result.output_tokens or 0),
                    "latency_ms": result.latency_ms,
                    "disposition": _disposition(result),
                    "credibility_ppm": _credibility_record(condition)["credibility_ppm"],
                }
                _append_jsonl(output / "provider_results.jsonl", row)
                if result.success and result.decision is not None:
                    requested = result.decision.requested_action.model_dump(mode="json")
                    resolution = resolve_action_request(
                        config,
                        state,
                        "company_B",
                        requested,
                        source=f"cooperation-reputation-neutral:{condition}",
                    )
                    searchable = json.dumps(
                        {
                            "plan": result.decision.plan.model_dump(mode="json"),
                            "responses": [item.model_dump(mode="json") for item in result.decision.message_responses],
                        },
                        ensure_ascii=False,
                    ).lower()
                    row.update(
                        {
                            "requested_action": requested,
                            "final_action": resolution.action.to_dict(),
                            "contribution_cents": int(resolution.action.shared_resilience_contribution_cents or 0),
                            "credibility_explicitly_cited": any(term in searchable for term in ("信誉", "可信", "credibility", "履约", "背叛")),
                            "message_responses": [item.model_dump(mode="json") for item in result.decision.message_responses],
                            "plan": result.decision.plan.model_dump(mode="json"),
                        }
                    )
                rows.append(row)
                # Replace the boundary row with a derived record in a separate log.
                _append_jsonl(output / "derived_rows.jsonl", row)
                print(f"[{len(rows)}/18] {provider} r{repetition} {condition} disposition={row['disposition']} tokens={row['input_tokens'] + row['output_tokens']}", flush=True)
    cells: dict[str, Any] = {}
    for provider in PROVIDERS:
        for condition in CONDITIONS:
            selected = [row for row in rows if row["provider"] == provider and row["condition"] == condition and row["success"]]
            cells[f"{provider}|{condition}"] = {
                "n": len(selected),
                "dispositions": [row["disposition"] for row in selected],
                "acceptance_rate": sum(row["disposition"] == "accepted" for row in selected) / len(selected) if selected else None,
                "mean_contribution_cents": mean(row["contribution_cents"] for row in selected) if selected else None,
                "credibility_citation_rate": sum(row["credibility_explicitly_cited"] for row in selected) / len(selected) if selected else None,
            }
    summary = {
        "schema_version": "cooperation-reputation-neutral-real-summary-v1.0.0",
        "evidence_level": manifest["evidence_level"],
        "calls": len(rows),
        "successful_calls": sum(row["success"] for row in rows),
        "all_conditions_share_state_hash": len({row["state_hash"] for row in rows}) == 1,
        "cells": cells,
        "usage": asdict(guard.actual),
        "estimated_cost_cny_conservative": round(guard.actual.estimated_cost_microunits / 1_000_000, 6),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
