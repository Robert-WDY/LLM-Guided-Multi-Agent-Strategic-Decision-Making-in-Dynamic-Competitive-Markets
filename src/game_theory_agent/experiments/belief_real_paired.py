"""Paired real-model Belief OFF/ON pilot with fixed-state counterfactuals."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import uuid
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from dotenv import dotenv_values, load_dotenv
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ENV = dotenv_values(PROJECT_ROOT / ".env")
if _PROJECT_ENV.get("MARKET_CONFIG_PATH"):
    os.environ.setdefault(
        "MARKET_CONFIG_PATH", str(_PROJECT_ENV["MARKET_CONFIG_PATH"])
    )

from game_theory_agent.agents import (
    AgentRuntime,
    DecisionContextBuilder,
    load_persona_registry,
)
from game_theory_agent.api import CONFIG, CONFIG_PATH, SESSIONS, agent_app, app
from game_theory_agent.belief import (
    compute_belief_calibration,
    verify_belief_replay,
)
from game_theory_agent.experiments.four_agent_acceptance import (
    _ControllerAdapter,
    _GatewayAdapter,
)
from game_theory_agent.experiments.market_metrics import compute_research_metrics
from game_theory_agent.experiments.persona_pilot import _model_client
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.information import seal_observation, verify_information_replay
from game_theory_agent.interaction.replay import verify_interaction_replay
from game_theory_agent.market import MarketEnv, MarketState
from game_theory_agent.market.replay import (
    EpisodeManifest,
    MarketTransition,
    verify_replay,
)
from game_theory_agent.orchestration import (
    JsonlRoundEventLogger,
    RoundCoordinator,
    RoundEvent,
)


COMPANIES = ("company_A", "company_B", "company_C", "company_D")
LLM_COMPANY = "company_A"
CONDITIONS = ("belief_off", "belief_on")
ACTION_FIELDS = (
    "price_cents",
    "advertising_budget_cents",
    "service_budget_cents",
    "capacity_investment_cents",
    "resilience_budget_cents",
    "shared_resilience_contribution_cents",
)
EXPLICIT_BELIEF_PATTERNS = (
    r"belief|信念|公开价格历史|价格方向",
    r"(?:对手|竞争对手|公司[BCD]|company_[bcd]|(?<![a-z_])[bcd](?![a-z_]))[，、\s\S]{0,24}(?:概率|大概率|预测)",
    r"(?:概率|大概率|预测)[，、\s\S]{0,24}(?:对手|降价|持平|涨价)",
)
DIRECTION_TERMS = ("降价", "持平", "涨价", "对手价格")


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("provide one or more unique comma-separated Seeds")
    return seeds


def build_plan(seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        order = CONDITIONS if seed_index % 2 == 0 else tuple(reversed(CONDITIONS))
        for condition in order:
            plan.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "belief_mode": (
                        "public_action_v1" if condition == "belief_on" else "off"
                    ),
                    "episode_id": f"belief-real-paired-{seed}",
                    "primary_experiment_unit": "paired_seed",
                }
            )
    return plan


def _trace_text(trace: Any) -> str:
    return json.dumps(
        trace.planner_output or {}, ensure_ascii=False, sort_keys=True
    ).lower()


def _belief_reference(trace: Any) -> dict[str, Any]:
    text = _trace_text(trace)
    matched_patterns = [
        pattern
        for pattern in EXPLICIT_BELIEF_PATTERNS
        if re.search(pattern, text)
    ]
    direction_terms = [term for term in DIRECTION_TERMS if term in text]
    return {
        "referenced": bool(matched_patterns),
        "matched_patterns": matched_patterns,
        "direction_language_terms": direction_terms,
    }


def _action_metrics(actions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "decision_count": len(actions),
        "mean": {
            field: mean(float(action.get(field) or 0) for action in actions)
            if actions
            else None
            for field in ACTION_FIELDS
        },
        "population_std": {
            field: (
                pstdev(float(action.get(field) or 0) for action in actions)
                if len(actions) > 1
                else 0.0
            )
            for field in ACTION_FIELDS
        },
        "total": {
            field: sum(int(action.get(field) or 0) for action in actions)
            for field in ACTION_FIELDS
        },
    }


def _token_usage_from_events(events: list[Any]) -> dict[str, Any]:
    traces = [
        trace
        for event in events
        for trace in event.traces
        if trace.company_id == LLM_COMPANY and trace.agent_type == "model"
    ]
    input_tokens = sum(int(trace.input_tokens or 0) for trace in traces)
    output_tokens = sum(int(trace.output_tokens or 0) for trace in traces)
    return {
        "token_usage_schema_version": "model-token-usage-v1.0.0",
        "llm_call_count": len(traces),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "calls_without_token_usage": sum(
            trace.input_tokens is None or trace.output_tokens is None
            for trace in traces
        ),
        "models_used": sorted(
            {trace.model_name for trace in traces if trace.model_name}
        ),
    }


def _sum_token_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "token_usage_schema_version": "model-token-usage-v1.0.0",
        "llm_call_count": sum(int(item["llm_call_count"]) for item in rows),
        "input_tokens": sum(int(item["input_tokens"]) for item in rows),
        "output_tokens": sum(int(item["output_tokens"]) for item in rows),
        "total_tokens": sum(int(item["total_tokens"]) for item in rows),
        "calls_without_token_usage": sum(
            int(item["calls_without_token_usage"]) for item in rows
        ),
        "models_used": sorted(
            {model for item in rows for model in item.get("models_used", [])}
        ),
    }


def _company_round_profit(event: Any, company_id: str) -> int:
    companies = event.state_after["companies"]
    company = (
        companies[company_id]
        if isinstance(companies, dict)
        else next(item for item in companies if item["company_id"] == company_id)
    )
    return int(company["financial"]["round_profit_cents"])


def _transitions_from_events(events: list[RoundEvent]) -> list[MarketTransition]:
    return [
        MarketTransition.from_dict(
            {
                "step_id": event.step_result["step_id"],
                "settled_round": event.settled_round,
                "state_before_hash": event.state_before_hash,
                "state_before": event.state_before,
                "final_actions": event.joint_action,
                "joint_action_hash": event.joint_action_hash,
                "random_draw_summary": event.random_draw_summary,
                "invariant_results": event.step_result.get(
                    "invariant_results", []
                ),
                "state_after": event.state_after,
            }
        )
        for event in events
    ]


def _summarize_episode_records(
    *,
    events: list[RoundEvent],
    manifest: EpisodeManifest,
    transitions: list[MarketTransition],
    seed: int,
    condition: str,
    provider: str,
    model: str | None,
    persona: str,
    rounds: int,
) -> dict[str, Any]:
    belief_mode = "public_action_v1" if condition == "belief_on" else "off"
    economic = verify_replay(MarketEnv(CONFIG), manifest, transitions)
    interaction = verify_interaction_replay(events)
    information = verify_information_replay(events, manifest)
    beliefs = verify_belief_replay(events, manifest)
    model_traces = [
        next(trace for trace in event.traces if trace.company_id == LLM_COMPANY)
        for event in events
    ]
    requested_actions = [dict(trace.requested_action or {}) for trace in model_traces]
    final_actions = [dict(trace.final_action) for trace in model_traces]
    references = [_belief_reference(trace) for trace in model_traces]
    research_metrics = compute_research_metrics(events, CONFIG)
    replay = {
        "economic": len(economic) == rounds + 1,
        "interaction": len(interaction) == rounds,
        "information": len(information) == rounds,
        "belief": len(beliefs) == (rounds if belief_mode != "off" else 0),
    }
    final_state_hash = (
        events[-1].state_after_hash if events else manifest.initial_state.state_hash
    )
    return {
        "episode_schema_version": "belief-real-episode-v1.1.0",
        "seed": seed,
        "condition": condition,
        "belief_mode": belief_mode,
        "episode_id": f"belief-real-paired-{seed}",
        "provider": provider,
        "model": model,
        "persona": persona,
        "rounds": rounds,
        "llm_rule_composition": "1 LLM + 3 Rule",
        "completed_rounds": len(events),
        "replay": replay,
        "llm_fallback_count": sum(
            trace.decision_status != "submitted" for trace in model_traces
        ),
        "requested_action_metrics": _action_metrics(requested_actions),
        "final_action_metrics": _action_metrics(final_actions),
        "belief_reference_count": sum(item["referenced"] for item in references),
        "belief_reference_rate_ppm": round(
            1_000_000 * sum(item["referenced"] for item in references) / rounds
        ),
        "belief_references": references,
        "belief_calibration": compute_belief_calibration(events),
        "market_total_profit_cents": research_metrics["market_total_profit_cents"],
        "llm_company_total_profit_cents": sum(
            _company_round_profit(event, LLM_COMPANY) for event in events
        ),
        "final_state_hash": final_state_hash,
        "token_usage": _token_usage_from_events(events),
        "passed": (
            len(events) == rounds
            and all(replay.values())
            and all(trace.decision_status == "submitted" for trace in model_traces)
        ),
        "artifacts": {"round_events": "round-events.jsonl"},
    }


def _recover_episode_summary(
    *,
    episode_dir: Path,
    seed: int,
    condition: str,
    provider: str,
    model: str | None,
    persona: str,
    rounds: int,
) -> dict[str, Any]:
    event_path = episode_dir / "round-events.jsonl"
    events = list(JsonlRoundEventLogger(event_path).read_all())
    if len(events) != rounds:
        raise ValueError(
            f"cannot recover incomplete event log ({len(events)}/{rounds}): {event_path}"
        )
    initial_state = MarketState.from_dict(events[0].state_before)
    belief_mode = "public_action_v1" if condition == "belief_on" else "off"
    manifest = EpisodeManifest.create(
        MarketEnv(CONFIG),
        initial_state,
        experiment_id="belief-real-paired-recovered",
        information_mode="public",
        communication_mode="off",
        cooperation_mode="off",
        belief_mode=belief_mode,
    )
    summary = _summarize_episode_records(
        events=events,
        manifest=manifest,
        transitions=_transitions_from_events(events),
        seed=seed,
        condition=condition,
        provider=provider,
        model=model,
        persona=persona,
        rounds=rounds,
    )
    summary["recovered_from_complete_round_event_log"] = True
    (episode_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


async def _run_episode(
    *,
    output: Path,
    seed: int,
    condition: str,
    provider: str,
    model: str | None,
    persona: str,
    rounds: int,
    temperature: float | None,
    top_p: float | None,
) -> dict[str, Any]:
    belief_mode = "public_action_v1" if condition == "belief_on" else "off"
    episode_id = f"belief-real-paired-{seed}"
    episode_dir = output / f"seed-{seed}" / condition
    event_path = episode_dir / "round-events.jsonl"
    if event_path.exists() and event_path.stat().st_size:
        raise ValueError(f"refusing to append to non-empty log: {event_path}")
    episode_dir.mkdir(parents=True, exist_ok=True)

    controller_token = f"belief-real-{uuid.uuid4().hex}"
    os.environ["MARKET_CONTROLLER_TOKEN"] = controller_token
    SESSIONS.clear()
    registry = load_persona_registry(CONFIG_PATH)
    profile = registry.get(persona)
    agent_id = f"{provider}-belief-{LLM_COMPANY}"
    created_response = TestClient(app).post(
        "/api/episodes",
        headers={"X-Controller-Token": controller_token},
        json={
            "episode_id": episode_id,
            "episode_seed": seed,
            "company_ids": list(COMPANIES),
            "max_rounds": rounds,
            "market_model": "balanced",
            "information_mode": "public",
            "communication_mode": "off",
            "cooperation_mode": "off",
            "belief_mode": belief_mode,
            "agent_configs": {
                LLM_COMPANY: {
                    "agent_id": agent_id,
                    "provider": provider,
                    "model": model,
                    "persona_id": persona,
                    "persona_profile_hash": profile.profile_hash,
                }
            },
        },
    )
    created_response.raise_for_status()
    created = created_response.json()
    runtime = AgentRuntime(
        agent_id,
        LLM_COMPANY,
        _model_client(provider, model, temperature, top_p),
        context_builder=DecisionContextBuilder(
            persona_profile=profile, persona_registry=registry
        ),
        persona_profile=profile,
        persona_registry=registry,
    )
    coordinator = RoundCoordinator(
        _ControllerAdapter(controller_token),
        _GatewayAdapter(created.get("agent_tokens", {})),
        {LLM_COMPANY: runtime},
        event_logger=JsonlRoundEventLogger(event_path),
        decision_timeout_seconds=90.0,
    )
    coordinated = await coordinator.run_episode(episode_id)
    events = [item.event for item in coordinated]
    session = SESSIONS[episode_id]

    summary = _summarize_episode_records(
        events=events,
        manifest=session.manifest,
        transitions=list(session.transitions),
        seed=seed,
        condition=condition,
        provider=provider,
        model=model,
        persona=persona,
        rounds=rounds,
    )
    (episode_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _prepare_fixed_observations(
    *, output: Path, seed: int, persona: str, provider: str, model: str | None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    controller_token = f"belief-fixed-{uuid.uuid4().hex}"
    os.environ["MARKET_CONTROLLER_TOKEN"] = controller_token
    SESSIONS.clear()
    episode_id = f"belief-real-fixed-{seed}"
    created_response = TestClient(app).post(
        "/api/episodes",
        headers={"X-Controller-Token": controller_token},
        json={
            "episode_id": episode_id,
            "episode_seed": seed,
            "company_ids": list(COMPANIES),
            "max_rounds": 5,
            "market_model": "balanced",
            "information_mode": "public",
            "belief_mode": "public_action_v1",
            "agent_configs": {
                LLM_COMPANY: {
                    "agent_id": f"{provider}-belief-fixed-{LLM_COMPANY}",
                    "provider": provider,
                    "model": model,
                    "persona_id": persona,
                }
            },
        },
    )
    created_response.raise_for_status()
    created = created_response.json()
    session = SESSIONS[episode_id]
    control = TestClient(app)
    for _ in range(3):
        state = session.env.get_state()
        actions = {
            company_id: build_rule_action(CONFIG, state, company_id).to_dict()
            for company_id in state.company_ids
        }
        actions["company_B"]["price_cents"] = max(
            int(
                session.env.get_action_constraints(
                    "company_B", state.state_version
                )["bounds"]["price_cents"]["min"]
            ),
            state.company("company_B").commercial.price_cents - 200,
        )
        response = control.post(
            f"/api/episodes/{episode_id}/steps",
            json={
                "step_id": f"{episode_id}:{state.round}:{state.state_version}",
                "joint_action": actions,
            },
        )
        response.raise_for_status()
    token = created["agent_tokens"][LLM_COMPANY]
    on_observation = TestClient(agent_app).get(
        f"/v1/episodes/{episode_id}/companies/{LLM_COMPANY}/observation",
        headers={"X-Agent-Token": token},
    ).json()
    off_observation = deepcopy(on_observation)
    off_observation["belief_schema_version"] = "none"
    off_observation["belief_hash"] = None
    off_observation["belief_state"] = None
    off_observation["episode_config"]["belief_mode"] = "off"
    off_observation["visibility_policy"]["belief_schema_version"] = "none"
    off_observation = seal_observation(off_observation)
    fixed_metadata = {
        "episode_id": episode_id,
        "seed": seed,
        "round": on_observation["round"],
        "state_version": on_observation["state_version"],
        "state_hash": on_observation["state_hash"],
        "on_observation_hash": on_observation["observation_hash"],
        "off_observation_hash": off_observation["observation_hash"],
        "company_B_price_cut_ppm": on_observation["belief_state"][
            "opponent_beliefs"
        ]["company_B"]["next_price_direction"]["price_cut_ppm"],
        "all_non_belief_fields_fixed": True,
    }
    fixed_dir = output / "fixed-state"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    (fixed_dir / "context-metadata.json").write_text(
        json.dumps(fixed_metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return off_observation, on_observation, fixed_metadata


def _fixed_summary_from_calls(
    *,
    calls: list[dict[str, Any]],
    metadata: dict[str, Any],
    provider: str,
    model: str | None,
    persona: str,
    repeats: int,
) -> dict[str, Any]:
    # Recompute text classification from the recorded structured planner output;
    # old keyword classifications are never trusted when the metric evolves.
    for item in calls:
        trace_like = type(
            "FixedTrace", (), {"planner_output": item.get("planner_output")}
        )()
        item["belief_reference"] = _belief_reference(trace_like)
    by_condition = {
        condition: [item for item in calls if item["condition"] == condition]
        for condition in CONDITIONS
    }
    action_metrics = {
        condition: _action_metrics(
            [
                item["requested_action"]
                for item in rows
                if item["requested_action"] is not None
            ]
        )
        for condition, rows in by_condition.items()
    }
    paired = []
    for repeat in range(1, repeats + 1):
        off = next(
            item
            for item in by_condition["belief_off"]
            if item["repeat"] == repeat
        )
        on = next(
            item
            for item in by_condition["belief_on"]
            if item["repeat"] == repeat
        )
        paired.append(
            {
                "repeat": repeat,
                "both_success": off["success"] and on["success"],
                "requested_action_changed": (
                    off["requested_action"] != on["requested_action"]
                ),
                "field_deltas_on_minus_off": {
                    field: int((on["requested_action"] or {}).get(field) or 0)
                    - int((off["requested_action"] or {}).get(field) or 0)
                    for field in ACTION_FIELDS
                },
            }
        )
    token_rows = [
        {
            "llm_call_count": 1,
            "input_tokens": int(item["input_tokens"] or 0),
            "output_tokens": int(item["output_tokens"] or 0),
            "total_tokens": int(item["input_tokens"] or 0)
            + int(item["output_tokens"] or 0),
            "calls_without_token_usage": int(
                item["input_tokens"] is None or item["output_tokens"] is None
            ),
            "models_used": [item["model_name"]] if item["model_name"] else [],
        }
        for item in calls
    ]
    explicit_rates = {
        condition: round(
            1_000_000
            * sum(item["belief_reference"]["referenced"] for item in rows)
            / repeats
        )
        for condition, rows in by_condition.items()
    }
    paired_mean_deltas = {
        field: mean(item["field_deltas_on_minus_off"][field] for item in paired)
        for field in ACTION_FIELDS
    }
    price_deltas = [
        item["field_deltas_on_minus_off"]["price_cents"] for item in paired
    ]
    return {
        "counterfactual_schema_version": "belief-real-fixed-v1.1.0",
        "provider": provider,
        "model": model,
        "persona": persona,
        "repeats_per_condition": repeats,
        "metadata": metadata,
        "action_metrics": action_metrics,
        "explicit_belief_reference_rate_ppm": explicit_rates,
        # Backward-compatible alias with the now stricter semantics.
        "belief_reference_rate_ppm": explicit_rates,
        "paired_results": paired,
        "paired_mean_field_deltas_on_minus_off": paired_mean_deltas,
        "price_delta_same_direction": (
            all(value < 0 for value in price_deltas)
            or all(value > 0 for value in price_deltas)
        ),
        "changed_pair_count": sum(
            item["requested_action_changed"] for item in paired
        ),
        "token_usage": _sum_token_usage(token_rows),
        "all_calls_succeeded": all(item["success"] for item in calls),
    }


async def _run_fixed_counterfactual(
    *,
    output: Path,
    seed: int,
    repeats: int,
    provider: str,
    model: str | None,
    persona: str,
    temperature: float | None,
    top_p: float | None,
) -> dict[str, Any]:
    off_observation, on_observation, metadata = _prepare_fixed_observations(
        output=output,
        seed=seed,
        persona=persona,
        provider=provider,
        model=model,
    )
    registry = load_persona_registry(CONFIG_PATH)
    profile = registry.get(persona)
    calls: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        order = CONDITIONS if repeat % 2 else tuple(reversed(CONDITIONS))
        for condition in order:
            observation = (
                on_observation if condition == "belief_on" else off_observation
            )
            runtime = AgentRuntime(
                f"{provider}-belief-fixed-{LLM_COMPANY}",
                LLM_COMPANY,
                _model_client(provider, model, temperature, top_p),
                context_builder=DecisionContextBuilder(
                    persona_profile=profile, persona_registry=registry
                ),
                persona_profile=profile,
                persona_registry=registry,
            )
            result = await runtime.decide(observation, timeout_seconds=90.0)
            action = (
                result.decision.requested_action.model_dump(mode="json")
                if result.decision is not None
                else None
            )
            planner = (
                result.decision.model_dump(mode="json")
                if result.decision is not None
                else None
            )
            trace_like = type(
                "FixedTrace", (), {"planner_output": planner}
            )()
            reference = _belief_reference(trace_like)
            calls.append(
                {
                    "repeat": repeat,
                    "condition": condition,
                    "success": result.success,
                    "model_name": result.model_name,
                    "prompt_version": result.prompt_version,
                    "requested_action": action,
                    "planner_output": planner,
                    "belief_reference": reference,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "latency_ms": result.latency_ms,
                    "error_code": result.error_code,
                    "error_message": result.error_message,
                }
            )
            print(
                f"[fixed {repeat}/{repeats}] {condition} "
                f"success={result.success} tokens="
                f"{int(result.input_tokens or 0) + int(result.output_tokens or 0)}",
                flush=True,
            )
    summary = _fixed_summary_from_calls(
        calls=calls,
        metadata=metadata,
        provider=provider,
        model=model,
        persona=persona,
        repeats=repeats,
    )
    fixed_dir = output / "fixed-state"
    (fixed_dir / "calls.json").write_text(
        json.dumps(calls, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (fixed_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _paired_episode_comparison(
    seeds: tuple[int, ...], results: list[dict[str, Any]]
) -> dict[str, Any]:
    by_key = {(item["seed"], item["condition"]): item for item in results}
    seed_rows = []
    for seed in seeds:
        off = by_key[(seed, "belief_off")]
        on = by_key[(seed, "belief_on")]
        seed_rows.append(
            {
                "seed": seed,
                "market_profit_delta_on_minus_off_cents": (
                    int(on["market_total_profit_cents"])
                    - int(off["market_total_profit_cents"])
                ),
                "llm_company_profit_delta_on_minus_off_cents": (
                    int(on["llm_company_total_profit_cents"])
                    - int(off["llm_company_total_profit_cents"])
                ),
                "requested_action_mean_deltas": {
                    field: (
                        float(on["requested_action_metrics"]["mean"][field])
                        - float(off["requested_action_metrics"]["mean"][field])
                    )
                    for field in ACTION_FIELDS
                },
                "belief_reference_rate_delta_ppm": (
                    int(on["belief_reference_rate_ppm"])
                    - int(off["belief_reference_rate_ppm"])
                ),
                "final_state_hash_equal": (
                    on["final_state_hash"] == off["final_state_hash"]
                ),
            }
        )
    return {
        "seed_rows": seed_rows,
        "mean_market_profit_delta_on_minus_off_cents": mean(
            row["market_profit_delta_on_minus_off_cents"] for row in seed_rows
        ),
        "mean_llm_company_profit_delta_on_minus_off_cents": mean(
            row["llm_company_profit_delta_on_minus_off_cents"] for row in seed_rows
        ),
        "mean_requested_action_deltas": {
            field: mean(
                row["requested_action_mean_deltas"][field] for row in seed_rows
            )
            for field in ACTION_FIELDS
        },
        "mean_belief_reference_rate_delta_ppm": mean(
            row["belief_reference_rate_delta_ppm"] for row in seed_rows
        ),
        "different_final_state_seed_count": sum(
            not row["final_state_hash_equal"] for row in seed_rows
        ),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    seeds = parse_seeds(args.seeds)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = build_plan(seeds)
    (output / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.plan_only:
        return {"plan_only": True, "run_count": len(plan), "plan": plan}

    fixed_path = output / "fixed-state" / "summary.json"
    if fixed_path.exists() and args.recompute_summaries:
        calls_path = output / "fixed-state" / "calls.json"
        metadata_path = output / "fixed-state" / "context-metadata.json"
        fixed_calls = json.loads(calls_path.read_text(encoding="utf-8"))
        fixed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fixed = _fixed_summary_from_calls(
            calls=fixed_calls,
            metadata=fixed_metadata,
            provider=args.provider,
            model=args.model,
            persona=args.persona,
            repeats=args.fixed_repeats,
        )
        calls_path.write_text(
            json.dumps(fixed_calls, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        fixed_path.write_text(
            json.dumps(fixed, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    elif fixed_path.exists():
        fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    else:
        fixed = await _run_fixed_counterfactual(
            output=output,
            seed=args.fixed_seed,
            repeats=args.fixed_repeats,
            provider=args.provider,
            model=args.model,
            persona=args.persona,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    results: list[dict[str, Any]] = []
    for index, row in enumerate(plan, start=1):
        episode_dir = output / f"seed-{row['seed']}" / row["condition"]
        summary_path = episode_dir / "summary.json"
        print(
            f"[{index}/{len(plan)}] {row['condition']} seed={row['seed']} starting",
            flush=True,
        )
        if summary_path.exists() and args.recompute_summaries:
            result = _recover_episode_summary(
                episode_dir=episode_dir,
                seed=row["seed"],
                condition=row["condition"],
                provider=args.provider,
                model=args.model,
                persona=args.persona,
                rounds=args.rounds,
            )
            print(f"[{index}/{len(plan)}] summary recomputed", flush=True)
        elif summary_path.exists():
            result = json.loads(summary_path.read_text(encoding="utf-8"))
            expected = (
                row["seed"], row["condition"], args.rounds, args.provider
            )
            actual = (
                result.get("seed"),
                result.get("condition"),
                result.get("rounds"),
                result.get("provider"),
            )
            if actual != expected or "token_usage" not in result:
                raise ValueError(f"incompatible resume summary: {summary_path}")
            print(f"[{index}/{len(plan)}] resumed from disk", flush=True)
        elif (
            (episode_dir / "round-events.jsonl").exists()
            and (episode_dir / "round-events.jsonl").stat().st_size
        ):
            result = _recover_episode_summary(
                episode_dir=episode_dir,
                seed=row["seed"],
                condition=row["condition"],
                provider=args.provider,
                model=args.model,
                persona=args.persona,
                rounds=args.rounds,
            )
            print(
                f"[{index}/{len(plan)}] recovered from complete event log",
                flush=True,
            )
        else:
            result = await _run_episode(
                output=output,
                seed=row["seed"],
                condition=row["condition"],
                provider=args.provider,
                model=args.model,
                persona=args.persona,
                rounds=args.rounds,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        result["_summary_path"] = str(summary_path)
        results.append(result)
        print(
            f"[{index}/{len(plan)}] completed; "
            f"tokens={result['token_usage']['total_tokens']} passed={result['passed']}",
            flush=True,
        )

    clean_results = [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in results
    ]
    episode_tokens = _sum_token_usage(
        [item["token_usage"] for item in clean_results]
    )
    tokens_by_condition = {
        condition: _sum_token_usage(
            [
                item["token_usage"]
                for item in clean_results
                if item["condition"] == condition
            ]
        )
        for condition in CONDITIONS
    }
    on_results = [
        item for item in clean_results if item["condition"] == "belief_on"
    ]
    aggregate = {
        "experiment_schema_version": "belief-real-paired-v1.1.0",
        "primary_experiment_unit": "paired_seed",
        "provider": args.provider,
        "model": args.model,
        "persona": args.persona,
        "seeds": list(seeds),
        "conditions": list(CONDITIONS),
        "rounds_per_episode": args.rounds,
        "llm_rule_composition": "1 LLM + 3 Rule",
        "run_count": len(clean_results),
        "passed_run_count": sum(item["passed"] for item in clean_results),
        "all_engineering_checks_passed": (
            all(item["passed"] for item in clean_results)
            and fixed["all_calls_succeeded"]
        ),
        "fixed_state_counterfactual": fixed,
        "paired_episode_comparison": _paired_episode_comparison(seeds, results),
        "belief_on_calibration_means": {
            "top1_accuracy_ppm": mean(
                float(item["belief_calibration"]["top1_accuracy_ppm"])
                for item in on_results
            ),
            "mean_brier_score": mean(
                float(item["belief_calibration"]["mean_brier_score"])
                for item in on_results
            ),
            "mean_log_loss": mean(
                float(item["belief_calibration"]["mean_log_loss"])
                for item in on_results
            ),
        },
        "token_usage": _sum_token_usage(
            [episode_tokens, fixed["token_usage"]]
        ),
        "token_usage_by_stage": {
            "fixed_state": fixed["token_usage"],
            "multi_seed_episodes": episode_tokens,
        },
        "token_usage_by_condition": tokens_by_condition,
        "belief_on_token_overhead_vs_off": {
            "input_tokens": (
                tokens_by_condition["belief_on"]["input_tokens"]
                - tokens_by_condition["belief_off"]["input_tokens"]
            ),
            "output_tokens": (
                tokens_by_condition["belief_on"]["output_tokens"]
                - tokens_by_condition["belief_off"]["output_tokens"]
            ),
            "total_tokens": (
                tokens_by_condition["belief_on"]["total_tokens"]
                - tokens_by_condition["belief_off"]["total_tokens"]
            ),
        },
        "results": clean_results,
        "interpretation_boundary": (
            "Fixed-state repetitions are the direct message-free Belief treatment; "
            "multi-round deltas include path-mediated effects and model sampling noise."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("deepseek", "doubao"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--persona", default="balanced_v1")
    parser.add_argument("--seeds", default="701,702,703")
    parser.add_argument("--rounds", type=int, choices=(5, 10, 15, 20), default=5)
    parser.add_argument("--fixed-seed", type=int, default=700)
    parser.add_argument("--fixed-repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--output", default="runs/belief-real-paired-20260821"
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--recompute-summaries", action="store_true")
    args = parser.parse_args()
    summary = asyncio.run(run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("plan_only") or summary["all_engineering_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
