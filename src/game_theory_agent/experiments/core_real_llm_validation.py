"""Cost-bounded real-LLM validation of the two highest-priority open claims.

The suite deliberately does not repeat every historical real-model matrix.  It
tests (1) the reliable Pareto v5 repair on the five frozen Stage 6.5 failures
and (2) the causal effect of proposal credibility across two available
personas.  Every paid decision is bound to an immutable Agent Version.
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
from game_theory_agent.agent_registry import (
    AgentBehaviorSpec,
    AgentRegistry,
    AgentVersionManifest,
    StrategicStackSpec,
)
from game_theory_agent.agent_registry.bootstrap import CurrentAgentVersionBuilder
from game_theory_agent.agent_registry.runtime_factory import (
    RegisteredAgentRuntimeFactory,
)
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.experiments.persona_pilot import _observation
from game_theory_agent.experiments.real_cooperation_counterfactual import (
    build_condition_observation,
)
from game_theory_agent.experiments.stage66_failure_forensics import (
    DEFAULT_STAGE65_SUMMARY,
    FOCAL_COMPANY,
    PAID_ROUND,
    _canonical_rows,
    _events,
    _paid_trace,
    simulate_frozen_tape,
)
from game_theory_agent.experiments.stage66_reliable_repair import (
    NEGATIVE_KEYS,
    _build_advice,
)
from game_theory_agent.market import MarketEnv, MarketState, load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.strategic_reliability import (
    PublicMarketRolloutAdvisor,
    RealModelBudget,
    RealModelCostGuard,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v4.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "core-real-llm-validation-20260825"
MODEL = "deepseek-v4-flash"
STRATEGIC_CONDITIONS = ("strategy_utility_control", "pareto_reliable_v5")
COOPERATION_CONDITIONS = (
    "no_message",
    "high_credibility_proposal",
    "low_credibility_proposal",
)
COOPERATION_PERSONAS = ("balanced_v1", "profit_myopic")
COOPERATION_REPETITIONS = 2
RESERVED_PROMPT_TOKENS = 32_000
RESERVED_COMPLETION_TOKENS = 4_000
# DeepSeek official 2026-08-25 price snapshot, conservatively doubled for peak.
INPUT_PRICE_MICROUNITS_PER_TOKEN = 2
OUTPUT_PRICE_MICROUNITS_PER_TOKEN = 4


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_plan() -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for pair_index, (seed, persona_id) in enumerate(NEGATIVE_KEYS):
        order = (
            STRATEGIC_CONDITIONS
            if pair_index % 2 == 0
            else tuple(reversed(STRATEGIC_CONDITIONS))
        )
        for condition in order:
            plan.append(
                {
                    "experiment": "strategic_v5_failure_retest",
                    "seed": seed,
                    "persona_id": persona_id,
                    "condition": condition,
                    "paid_round": PAID_ROUND,
                }
            )
    for persona_index, persona_id in enumerate(COOPERATION_PERSONAS):
        for repetition in range(1, COOPERATION_REPETITIONS + 1):
            shift = (persona_index + repetition - 1) % len(
                COOPERATION_CONDITIONS
            )
            order = (
                COOPERATION_CONDITIONS[shift:]
                + COOPERATION_CONDITIONS[:shift]
            )
            for condition in order:
                plan.append(
                    {
                        "experiment": "cooperation_credibility_persona",
                        "seed": 20260825,
                        "persona_id": persona_id,
                        "condition": condition,
                        "repetition": repetition,
                        "paid_round": 1,
                    }
                )
    return plan


def _stack(
    *,
    belief: bool = False,
    advisor: bool = False,
    cooperation: bool = False,
) -> StrategicStackSpec:
    return StrategicStackSpec(
        belief_mode="public_action_v1" if belief else "off",
        belief_updater_version=(
            "dirichlet-public-price-v1.0.0" if belief else "none"
        ),
        opponent_model_mode="public_strategy_v1" if belief else "off",
        opponent_model_version=(
            "public-strategy-rule-bayes-v1.0.0" if belief else "none"
        ),
        utility_inference_mode="strategy_utility_v1" if belief else "off",
        utility_inference_version=(
            "strategy-mixture-utility-v1.0.0" if belief else "none"
        ),
        advisor_mode="pareto_reliable_v5" if advisor else "off",
        advisor_version=(
            "public-pareto-reliable-market-rollout-v1.0.0"
            if advisor
            else "none"
        ),
        repeated_game_mode="off",
        repeated_game_version="none",
        cooperation_mode="shared_resilience_v1" if cooperation else "off",
        cooperation_version=(
            "shared-resilience-cooperation-v1.0.0"
            if cooperation
            else "none"
        ),
    )


def _register_variant(
    *,
    registry: AgentRegistry,
    builder: CurrentAgentVersionBuilder,
    persona_id: str,
    family_id: str,
    condition: str,
    paid_round: int,
) -> AgentVersionManifest:
    advisor = condition == "pareto_reliable_v5"
    cooperation = condition == "cooperation_credibility_persona"
    policy_parameters: dict[str, int | str | bool | None] = {
        "paid_rounds_csv": str(paid_round),
        "reserved_prompt_tokens_per_call": RESERVED_PROMPT_TOKENS,
        "reserved_completion_tokens_per_call": RESERVED_COMPLETION_TOKENS,
        "input_price_microunits_per_token": (
            INPUT_PRICE_MICROUNITS_PER_TOKEN
        ),
        "output_price_microunits_per_token": (
            OUTPUT_PRICE_MICROUNITS_PER_TOKEN
        ),
        "max_transport_retries": 0,
        "communication_mode": "public_private" if cooperation else "off",
        "experiment_stack": condition,
    }
    base = builder.register(
        provider="deepseek",
        model=MODEL,
        model_revision=f"provider-alias:{MODEL}:2026-08-25",
        persona_id=persona_id,
        family_id=family_id,
        human_version=f"{condition}-{persona_id}-1.0.0",
        advisor_mode="pareto_reliable_v5" if advisor else "off",
        max_schema_attempts=1,
        policy_parameters=policy_parameters,
    )
    desired_stack = _stack(
        belief=condition in STRATEGIC_CONDITIONS,
        advisor=advisor,
        cooperation=cooperation,
    )
    if base.behavior_spec.strategic_stack == desired_stack:
        return base
    behavior = AgentBehaviorSpec.model_validate(
        {
            **base.behavior_spec.model_dump(mode="json"),
            "strategic_stack": desired_stack.model_dump(mode="json"),
        }
    )
    return registry.register_version(
        AgentVersionManifest.create(
            behavior_spec=behavior,
            human_version=f"{condition}-{persona_id}-1.0.0",
            reproducibility_tier="C_recorded_output",
        )
    )


def _audit(result: Any) -> dict[str, Any] | None:
    return (
        result.provider_audit.model_dump(mode="json")
        if result.provider_audit is not None
        else None
    )


def _decision_row(result: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "success": result.success,
        "error_code": result.error_code,
        "error_message": result.error_message,
        "model_name": result.model_name,
        "prompt_version": result.prompt_version,
        "latency_ms": result.latency_ms,
        "input_tokens": int(result.input_tokens or 0),
        "output_tokens": int(result.output_tokens or 0),
        "retry_count": result.retry_count,
        "provider_audit": _audit(result),
        "raw_model_output": result.raw_response,
    }
    if result.success and result.decision is not None:
        row["decision"] = result.decision.model_dump(mode="json")
    return row


def _frozen_strategic_cells(
    stage65_summary: Path,
    advisor: PublicMarketRolloutAdvisor,
    personas: PersonaRegistry,
) -> dict[tuple[int, str], dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for row in _canonical_rows(stage65_summary):
        grouped.setdefault(
            (int(row["seed"]), str(row["persona_id"])), {}
        )[str(row["condition"])] = row
    cells: dict[tuple[int, str], dict[str, Any]] = {}
    for seed, persona_id in NEGATIVE_KEYS:
        original = grouped[(seed, persona_id)]["pareto_v4"]
        events = _events(Path(original["directory"]))
        paid_event, trace = _paid_trace(events)
        advice = _build_advice(
            advisor=advisor,
            registry=personas,
            row=original,
            trace=trace,
        )
        cells[(seed, persona_id)] = {
            "source_directory": str(Path(original["directory"]).resolve()),
            "initial_state": MarketState.from_dict(paid_event.state_before),
            "observation": trace.observation,
            "v5_advice": advice.model_dump(mode="json"),
            "recorded_events": tuple(
                event for event in events if event.settled_round >= PAID_ROUND
            ),
        }
    return cells


async def _run_strategic_call(
    *,
    item: Mapping[str, Any],
    cell: Mapping[str, Any],
    manifest: AgentVersionManifest,
    factory: RegisteredAgentRuntimeFactory,
    config: Any,
) -> dict[str, Any]:
    condition = str(item["condition"])
    observation = json.loads(json.dumps(cell["observation"]))
    if condition == "pareto_reliable_v5":
        observation["game_theory_advice"] = cell["v5_advice"]
    else:
        observation.pop("game_theory_advice", None)
    input_hash = sha256_hash(
        {
            "protocol": "core-real-strategic-input-v1.0.0",
            "condition": condition,
            "observation": observation,
        }
    )
    runtime = factory.build(
        agent_version_id=manifest.agent_version_id,
        episode_id=str(observation["episode_id"]),
        company_id=FOCAL_COMPANY,
        agent_id=(
            f"core-strategic-{item['seed']}-{item['persona_id']}-{condition}"
        ),
    )
    result = await runtime.decide(observation, timeout_seconds=45.0)
    row = {
        **dict(item),
        "agent_version_id": manifest.agent_version_id,
        "manifest_hash": manifest.manifest_hash,
        "input_snapshot_hash": input_hash,
        "frozen_state_hash": cell["initial_state"].state_hash,
        "source_directory": cell["source_directory"],
        **_decision_row(result),
    }
    if not result.success or result.decision is None:
        return row
    requested = result.decision.requested_action.model_dump(mode="json")
    resolved = resolve_action_request(
        config,
        cell["initial_state"],
        FOCAL_COMPANY,
        requested,
        source=f"core-real-strategic:{condition}",
    )
    final_action = resolved.action.to_dict()
    path = simulate_frozen_tape(
        initial_state=cell["initial_state"],
        initial_action=final_action,
        recorded_events=cell["recorded_events"],
    )
    adoption = build_advisor_adoption_trace(
        advice=(
            cell["v5_advice"]
            if condition == "pareto_reliable_v5"
            else None
        ),
        llm_requested_action=requested,
        final_action=final_action,
        planner_output=result.decision.plan.model_dump(mode="json"),
    )
    row.update(
        {
            "requested_action": requested,
            "final_action": final_action,
            "resolution_adjustments": [
                value.to_dict() for value in resolved.adjustments
            ],
            "terminal_enterprise_value_cents": int(
                path["final_enterprise_value_cents"]
            ),
            "advisor_adoption": (
                adoption.model_dump(mode="json")
                if adoption is not None
                else None
            ),
        }
    )
    return row


def _cooperation_base(config: Any) -> tuple[MarketState, dict[str, Any]]:
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B", "company_C", "company_D"),
        episode_id="core-real-cooperation-credibility",
        episode_seed=20260825,
        market_model="balanced",
        max_rounds=5,
        cooperation_mode="shared_resilience_v1",
    )
    observation = _observation(config, state, "company_B")
    observation["shared_resilience"] = state.shared_resilience.to_dict()
    return state, observation


async def _run_cooperation_call(
    *,
    item: Mapping[str, Any],
    state: MarketState,
    base_observation: dict[str, Any],
    manifest: AgentVersionManifest,
    factory: RegisteredAgentRuntimeFactory,
    config: Any,
) -> dict[str, Any]:
    condition = str(item["condition"])
    observation = build_condition_observation(
        base_observation, condition=condition
    )
    input_hash = sha256_hash(
        {
            "protocol": "core-real-cooperation-input-v1.0.0",
            "persona_id": item["persona_id"],
            "condition": condition,
            "observation": observation,
        }
    )
    runtime = factory.build(
        agent_version_id=manifest.agent_version_id,
        episode_id=str(observation["episode_id"]),
        company_id="company_B",
        agent_id=(
            f"core-cooperation-{item['persona_id']}-{item['repetition']}-"
            f"{condition}"
        ),
    )
    result = await runtime.decide(observation, timeout_seconds=45.0)
    row = {
        **dict(item),
        "agent_version_id": manifest.agent_version_id,
        "manifest_hash": manifest.manifest_hash,
        "input_snapshot_hash": input_hash,
        "frozen_state_hash": state.state_hash,
        **_decision_row(result),
    }
    if not result.success or result.decision is None:
        return row
    requested = result.decision.requested_action.model_dump(mode="json")
    resolved = resolve_action_request(
        config,
        state,
        "company_B",
        requested,
        source=f"core-real-cooperation:{condition}",
    )
    responses = [
        value.model_dump(mode="json")
        for value in result.decision.message_responses
    ]
    row.update(
        {
            "requested_action": requested,
            "final_action": resolved.action.to_dict(),
            "message_responses": responses,
            "proposal_accepted": any(
                value["disposition"] == "accepted" for value in responses
            ),
            "proposal_rejected": any(
                value["disposition"] == "rejected" for value in responses
            ),
            "contribution_cents": int(
                resolved.action.shared_resilience_contribution_cents or 0
            ),
        }
    )
    return row


def _strategic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for seed, persona_id in NEGATIVE_KEYS:
        selected = [
            row
            for row in rows
            if row["experiment"] == "strategic_v5_failure_retest"
            and int(row["seed"]) == seed
            and row["persona_id"] == persona_id
        ]
        by_condition = {row["condition"]: row for row in selected}
        complete = all(
            condition in by_condition and by_condition[condition]["success"]
            for condition in STRATEGIC_CONDITIONS
        )
        pair: dict[str, Any] = {
            "seed": seed,
            "persona_id": persona_id,
            "complete": complete,
        }
        if complete:
            control = by_condition["strategy_utility_control"]
            treatment = by_condition["pareto_reliable_v5"]
            delta = int(treatment["terminal_enterprise_value_cents"]) - int(
                control["terminal_enterprise_value_cents"]
            )
            pair.update(
                {
                    "action_changed": treatment["final_action"]
                    != control["final_action"],
                    "treatment_minus_control_ev_cents": delta,
                    "adoption_status": treatment["advisor_adoption"][
                        "adoption_status"
                    ],
                    "target_alignment_ppm": treatment["advisor_adoption"][
                        "target_alignment_ppm"
                    ],
                }
            )
        pairs.append(pair)
    complete_pairs = [row for row in pairs if row["complete"]]
    deltas = [
        int(row["treatment_minus_control_ev_cents"])
        for row in complete_pairs
    ]
    return {
        "pair_count": len(pairs),
        "complete_pair_count": len(complete_pairs),
        "action_changed_pair_count": sum(
            bool(row.get("action_changed")) for row in complete_pairs
        ),
        "accepted_advice_pair_count": sum(
            row.get("adoption_status") in {"exact_action", "accepted_target"}
            for row in complete_pairs
        ),
        "mean_ev_delta_cents": round(mean(deltas)) if deltas else None,
        "worst_ev_delta_cents": min(deltas) if deltas else None,
        "positive_zero_negative": {
            "positive": sum(value > 0 for value in deltas),
            "zero": sum(value == 0 for value in deltas),
            "negative": sum(value < 0 for value in deltas),
        },
        "directional_gate": {
            "all_pairs_complete": len(complete_pairs) == len(NEGATIVE_KEYS),
            "mean_value_not_lower": bool(deltas) and mean(deltas) >= 0,
            "worst_value_not_lower": bool(deltas) and min(deltas) >= 0,
            "advice_changes_or_is_explicitly_adopted": any(
                row.get("action_changed")
                or row.get("adoption_status")
                in {"exact_action", "accepted_target"}
                for row in complete_pairs
            ),
        },
        "pairs": pairs,
    }


def _cooperation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = [
        row
        for row in rows
        if row["experiment"] == "cooperation_credibility_persona"
    ]
    cells: list[dict[str, Any]] = []
    for persona_id in COOPERATION_PERSONAS:
        for condition in COOPERATION_CONDITIONS:
            selected = [
                row
                for row in relevant
                if row["persona_id"] == persona_id
                and row["condition"] == condition
            ]
            successes = [row for row in selected if row["success"]]
            cells.append(
                {
                    "persona_id": persona_id,
                    "condition": condition,
                    "attempt_count": len(selected),
                    "success_count": len(successes),
                    "acceptance_rate_ppm": (
                        sum(bool(row.get("proposal_accepted")) for row in successes)
                        * 1_000_000
                        // len(successes)
                        if successes
                        else None
                    ),
                    "mean_contribution_cents": (
                        round(mean(int(row["contribution_cents"]) for row in successes))
                        if successes
                        else None
                    ),
                }
            )
    by_key = {(row["persona_id"], row["condition"]): row for row in cells}
    contrasts: list[dict[str, Any]] = []
    for persona_id in COOPERATION_PERSONAS:
        high = by_key[(persona_id, "high_credibility_proposal")]
        low = by_key[(persona_id, "low_credibility_proposal")]
        none = by_key[(persona_id, "no_message")]
        contrasts.append(
            {
                "persona_id": persona_id,
                "high_minus_low_acceptance_ppm": (
                    high["acceptance_rate_ppm"] - low["acceptance_rate_ppm"]
                    if high["acceptance_rate_ppm"] is not None
                    and low["acceptance_rate_ppm"] is not None
                    else None
                ),
                "high_minus_low_contribution_cents": (
                    high["mean_contribution_cents"]
                    - low["mean_contribution_cents"]
                    if high["mean_contribution_cents"] is not None
                    and low["mean_contribution_cents"] is not None
                    else None
                ),
                "high_minus_no_message_contribution_cents": (
                    high["mean_contribution_cents"]
                    - none["mean_contribution_cents"]
                    if high["mean_contribution_cents"] is not None
                    and none["mean_contribution_cents"] is not None
                    else None
                ),
            }
        )
    complete = all(
        row["success_count"] == COOPERATION_REPETITIONS for row in cells
    )
    return {
        "cell_count": len(cells),
        "all_cells_complete": complete,
        "cells": cells,
        "contrasts": contrasts,
        "directional_gate": {
            "all_cells_complete": complete,
            "high_credibility_acceptance_not_lower_in_each_persona": complete
            and all(
                value["high_minus_low_acceptance_ppm"] is not None
                and value["high_minus_low_acceptance_ppm"] >= 0
                for value in contrasts
            ),
            "high_credibility_contribution_not_lower_in_each_persona": complete
            and all(
                value["high_minus_low_contribution_cents"] is not None
                and value["high_minus_low_contribution_cents"] >= 0
                for value in contrasts
            ),
            "persona_changes_at_least_one_response_distribution": complete
            and any(
                by_key[(COOPERATION_PERSONAS[0], condition)][
                    "acceptance_rate_ppm"
                ]
                != by_key[(COOPERATION_PERSONAS[1], condition)][
                    "acceptance_rate_ppm"
                ]
                or by_key[(COOPERATION_PERSONAS[0], condition)][
                    "mean_contribution_cents"
                ]
                != by_key[(COOPERATION_PERSONAS[1], condition)][
                    "mean_contribution_cents"
                ]
                for condition in COOPERATION_CONDITIONS
            ),
        },
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    if not args.authorize_real_model:
        raise RuntimeError("--authorize-real-model is required")
    plan = build_plan()
    if len(plan) != 22:
        raise AssertionError("core real-LLM plan must contain exactly 22 calls")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("output exists; use a fresh directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "plan.json", plan)
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    advisor = PublicMarketRolloutAdvisor(config)
    budget = RealModelBudget(
        max_calls=len(plan),
        max_prompt_tokens=len(plan) * RESERVED_PROMPT_TOKENS,
        max_completion_tokens=len(plan) * RESERVED_COMPLETION_TOKENS,
        max_estimated_cost_microunits=len(plan)
        * (
            RESERVED_PROMPT_TOKENS * INPUT_PRICE_MICROUNITS_PER_TOKEN
            + RESERVED_COMPLETION_TOKENS * OUTPUT_PRICE_MICROUNITS_PER_TOKEN
        ),
    )
    guard = RealModelCostGuard(budget, explicitly_authorized=True)
    registry = AgentRegistry(output / "registry")
    builder = CurrentAgentVersionBuilder(registry, config, personas)
    factory = RegisteredAgentRuntimeFactory(
        registry,
        config,
        personas,
        real_model_cost_guard=guard,
    )
    manifests: dict[tuple[str, str], AgentVersionManifest] = {}
    for item in plan:
        key = (str(item["persona_id"]), str(item["condition"]))
        version_condition = (
            "cooperation_credibility_persona"
            if item["experiment"] == "cooperation_credibility_persona"
            else str(item["condition"])
        )
        version_key = (str(item["persona_id"]), version_condition)
        if version_key not in manifests:
            manifests[version_key] = _register_variant(
                registry=registry,
                builder=builder,
                persona_id=str(item["persona_id"]),
                family_id=(
                    "core-real-cooperation-agent"
                    if item["experiment"] == "cooperation_credibility_persona"
                    else "core-real-strategic-agent"
                ),
                condition=version_condition,
                paid_round=int(item["paid_round"]),
            )
        manifests[key] = manifests[version_key]
    registered_versions_path = output / "registered-versions.json"
    if not registered_versions_path.exists():
        _write_json(
            registered_versions_path,
            [
                manifest.model_dump(mode="json")
                for manifest in sorted(
                    {
                        value.agent_version_id: value
                        for value in manifests.values()
                    }.values(),
                    key=lambda value: value.agent_version_id,
                )
            ],
        )
    strategic_cells = _frozen_strategic_cells(
        Path(args.stage65_summary), advisor, personas
    )
    cooperation_state, cooperation_observation = _cooperation_base(config)
    rows: list[dict[str, Any]] = []
    rows_path = output / "rows.json"
    if args.resume and rows_path.exists():
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
        for row in rows:
            guard.reserve(
                prompt_tokens=RESERVED_PROMPT_TOKENS,
                completion_tokens=RESERVED_COMPLETION_TOKENS,
                estimated_cost_microunits=(
                    RESERVED_PROMPT_TOKENS * INPUT_PRICE_MICROUNITS_PER_TOKEN
                    + RESERVED_COMPLETION_TOKENS
                    * OUTPUT_PRICE_MICROUNITS_PER_TOKEN
                ),
            )
            if row.get("success"):
                guard.record_actual(
                    prompt_tokens=int(row["input_tokens"]),
                    completion_tokens=int(row["output_tokens"]),
                    estimated_cost_microunits=(
                        int(row["input_tokens"])
                        * INPUT_PRICE_MICROUNITS_PER_TOKEN
                        + int(row["output_tokens"])
                        * OUTPUT_PRICE_MICROUNITS_PER_TOKEN
                    ),
                )
    completed_keys = {
        (
            row["experiment"],
            row["seed"],
            row["persona_id"],
            row["condition"],
            row.get("repetition"),
        )
        for row in rows
    }
    for index, item in enumerate(plan, start=1):
        call_key = (
            item["experiment"],
            item["seed"],
            item["persona_id"],
            item["condition"],
            item.get("repetition"),
        )
        if call_key in completed_keys:
            continue
        manifest = manifests[(str(item["persona_id"]), str(item["condition"]))]
        if item["experiment"] == "strategic_v5_failure_retest":
            row = await _run_strategic_call(
                item=item,
                cell=strategic_cells[(int(item["seed"]), str(item["persona_id"]))],
                manifest=manifest,
                factory=factory,
                config=config,
            )
        else:
            row = await _run_cooperation_call(
                item=item,
                state=cooperation_state,
                base_observation=cooperation_observation,
                manifest=manifest,
                factory=factory,
                config=config,
            )
        rows.append(row)
        _write_json(rows_path, rows)
        _write_json(
            output / "partial-summary.json",
            {
                "completed": len(rows),
                "planned": len(plan),
                "latest": call_key,
                "reserved_usage": asdict(guard.reserved),
                "successful_usage": asdict(guard.actual),
            },
        )
        if not args.quiet:
            print(
                f"[{index}/{len(plan)}] {item['experiment']} "
                f"persona={item['persona_id']} condition={item['condition']} "
                f"success={row['success']} tokens="
                f"{row['input_tokens'] + row['output_tokens']}",
                flush=True,
            )
    strategic = _strategic_summary(rows)
    cooperation = _cooperation_summary(rows)
    all_success = len(rows) == len(plan) and all(row["success"] for row in rows)
    provider_audit_complete = all(
        isinstance(row.get("provider_audit"), dict)
        and bool(row["provider_audit"].get("request_id"))
        and bool(row["provider_audit"].get("response_model"))
        for row in rows
        if row["success"]
    )
    unique_versions = {
        row["agent_version_id"] for row in rows if row["success"]
    }
    if all_success and provider_audit_complete:
        for version_id in sorted(unique_versions):
            registry.promote(version_id, "candidate")
    actual = guard.actual
    reserved = guard.reserved
    summary: dict[str, Any] = {
        "experiment_schema_version": "core-real-llm-validation-v1.0.0",
        "provider": "deepseek",
        "model": MODEL,
        "evidence_level": "paired-small-sample-real-llm-directional",
        "plan": {
            "total_calls": len(plan),
            "strategic_calls": 10,
            "cooperation_calls": 12,
        },
        "engineering": {
            "all_planned_calls_recorded": len(rows) == len(plan),
            "all_decisions_successful": all_success,
            "provider_audit_complete": provider_audit_complete,
            "immutable_version_attribution_complete": all(
                bool(row.get("agent_version_id"))
                and bool(row.get("manifest_hash"))
                for row in rows
            ),
            "unique_called_agent_versions": len(unique_versions),
        },
        "usage": {
            "reserved": asdict(reserved),
            "successful": asdict(actual),
            "estimated_actual_cost_cny": (
                f"{actual.estimated_cost_microunits / 1_000_000:.6f}"
            ),
            "maximum_reserved_cost_cny": (
                f"{reserved.estimated_cost_microunits / 1_000_000:.6f}"
            ),
            "pricing_snapshot": (
                "DeepSeek official 2026-08-25; input 1/output 2 CNY per "
                "million tokens, conservatively doubled for peak"
            ),
            "schema_retries_per_call": 0,
            "transport_retries_per_call": 0,
        },
        "strategic_v5": strategic,
        "cooperation_credibility_persona": cooperation,
        "research_gate": {
            "strategic_directional_gate_passed": all(
                strategic["directional_gate"].values()
            ),
            "cooperation_directional_gate_passed": all(
                cooperation["directional_gate"].values()
            ),
        },
        "conclusion_limits": [
            "five strategic pairs are known historical failures, not unknown-seed generalization",
            "strategic continuation actions and opponent actions are frozen after the paid decision",
            "cooperation uses two repetitions per cell and supports directional, not statistical, inference",
            "the current persona catalog has no dedicated cooperator or free-rider utility profile",
            "provider outputs are recorded Tier C evidence and need not regenerate identically",
        ],
        "rows_file": str(rows_path.resolve()),
    }
    summary["report_hash"] = sha256_hash(summary)
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage65-summary", type=Path, default=DEFAULT_STAGE65_SUMMARY
    )
    parser.add_argument("--authorize-real-model", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    if args.print_plan:
        print(json.dumps(build_plan(), ensure_ascii=False, indent=2))
        return 0
    summary = asyncio.run(run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(summary["engineering"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
