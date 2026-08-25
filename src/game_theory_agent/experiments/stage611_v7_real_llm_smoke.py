"""Six-call exploratory real-LLM smoke for Pareto v7 advice adoption."""

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

from game_theory_agent.advisor import build_advisor_adoption_trace
from game_theory_agent.agents import AgentRuntime, EpisodeMemory, MarketRegimeEvaluator
from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.belief import BeliefLedger
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.economics import decision_support_metrics
from game_theory_agent.experiments.persona_pilot import _model_client
from game_theory_agent.experiments.stage610_v7_zero_token_holdout import (
    CONFIG_PATH,
    FOCAL_COMPANY,
    _build_rule_episode,
    _simulate_closed_loop,
)
from game_theory_agent.information import compute_observation_hash
from game_theory_agent.market import MarketEnv, MarketState, load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.opponent import OpponentModelLedger
from game_theory_agent.strategic_reliability import (
    PublicMarketRolloutAdvisor,
    RealModelBudget,
    RealModelCostGuard,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = PROJECT_ROOT / "experiment-specs" / "stage6.11-v7-real-llm-smoke" / "PREREGISTRATION.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "stage6.11-v7-real-llm-smoke"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _observation(config: Any, state: MarketState, belief: Any, belief_hash: str, opponent: Any, opponent_hash: str) -> dict[str, Any]:
    env = MarketEnv(config)
    env.load_state(state)
    observation = ObservationBuilder().build(
        state,
        FOCAL_COMPANY,
        "public",
        belief_state=belief.model_dump(mode="json"),
        belief_hash=belief_hash,
        belief_schema_version=belief.belief_schema_version,
    )
    observation.update(
        {
            "observation_schema_version": "agent-observation-v1.8.0",
            "episode_id": state.episode_id,
            "round": state.round,
            "rounds_remaining": state.rounds_remaining,
            "state_version": state.state_version,
            "state_hash": state.state_hash,
            "information_mode": "public",
            "communication_mode": "off",
            "cooperation_mode": "off",
            "market_regime": MarketRegimeEvaluator(config).evaluate(state),
            "decision_support": decision_support_metrics(config, state, FOCAL_COMPANY),
            "action_constraints": env.get_action_constraints(FOCAL_COMPANY, state.state_version),
            "opponent_model_state": opponent.model_dump(mode="json"),
            "opponent_model_hash": opponent_hash,
            "utility_inference_state": None,
            "utility_inference_hash": None,
            "observation_hash": "pending",
        }
    )
    observation["observation_hash"] = compute_observation_hash(observation)
    return observation


def _cells(spec: dict[str, Any], config: Any, personas: PersonaRegistry) -> dict[tuple[int, int, str], dict[str, Any]]:
    wanted = {(int(x["seed"]), int(x["round"]), str(x["persona_id"])) for x in spec["paired_cells"]}
    advisor = PublicMarketRolloutAdvisor(config)
    result: dict[tuple[int, int, str], dict[str, Any]] = {}
    for seed in sorted({x[0] for x in wanted}):
        _initial, transitions, _ = _build_rule_episode(config, seed)
        belief_ledger = BeliefLedger(episode_id=transitions[0].state_before.episode_id, company_ids=transitions[0].state_before.company_ids)
        opponent_ledger = OpponentModelLedger(episode_id=transitions[0].state_before.episode_id, company_ids=transitions[0].state_before.company_ids)
        for index, transition in enumerate(transitions):
            state = transition.state_before
            belief, belief_hash = belief_ledger.company_view(observer_company_id=FOCAL_COMPANY, round_number=state.round, state_version=state.state_version)
            opponent, opponent_hash = opponent_ledger.company_view(observer_company_id=FOCAL_COMPANY, round_number=state.round, state_version=state.state_version)
            for key in sorted(wanted):
                if key[0] != seed or key[1] != state.round:
                    continue
                observation = _observation(config, state, belief, belief_hash, opponent, opponent_hash)
                advice = advisor.advise(
                    observation=observation,
                    company_id=FOCAL_COMPANY,
                    persona_profile=personas.get(key[2]),
                    belief_state=belief,
                    opponent_model=opponent,
                    horizon_rounds=min(3, len(transitions) - index),
                    scenario_count=5,
                    advisor_mode="pareto_reliable_v7",
                )
                if advice.execution_disposition != "recommend":
                    raise AssertionError(f"selected cell no longer recommends: {key}")
                result[key] = {"state": state, "observation": observation, "advice": advice.model_dump(mode="json")}
            belief_ledger.update_after_settlement(state, dict(transition.joint_action))
            opponent_ledger.update_after_settlement(state, transition.state_after, dict(transition.joint_action))
    if set(result) != wanted:
        raise AssertionError("failed to rebuild every preregistered cell")
    return result


async def run(spec_path: Path, output: Path) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not os.environ.get("ARK_API_KEY"):
        raise RuntimeError("ARK_API_KEY is required")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "plan.json", spec)
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    cells = _cells(spec, config, personas)
    price = spec["price_snapshot"]
    incident_path = output / "incident.json"
    incident = json.loads(incident_path.read_text(encoding="utf-8")) if incident_path.exists() else None
    lost_calls = int(incident["lost_provider_call_count"]) if incident else 0
    guard = RealModelCostGuard(
        RealModelBudget(
            max_calls=int(spec["maximum_provider_calls"]) - lost_calls,
            max_prompt_tokens=int(spec["maximum_prompt_tokens"]),
            max_completion_tokens=int(spec["maximum_completion_tokens"]),
            max_estimated_cost_microunits=int(spec["maximum_estimated_cost_microunits"]),
        ),
        explicitly_authorized=True,
    )
    rows: list[dict[str, Any]] = []
    for cell_spec in spec["paired_cells"]:
        key = (int(cell_spec["seed"]), int(cell_spec["round"]), str(cell_spec["persona_id"]))
        cell = cells[key]
        for condition in cell_spec["call_order"]:
            if incident and [key[0], key[1], key[2], condition] == incident["lost_call_key"]:
                continue
            observation = json.loads(json.dumps(cell["observation"]))
            if condition == "treatment_with_v7_advice":
                observation["game_theory_advice"] = cell["advice"]
            guard.reserve(
                prompt_tokens=int(spec["reserved_prompt_tokens_per_call"]),
                completion_tokens=int(spec["reserved_completion_tokens_per_call"]),
                estimated_cost_microunits=int(spec["reserved_prompt_tokens_per_call"]) * int(price["input_price_microunits_per_token"]) + int(spec["reserved_completion_tokens_per_call"]) * int(price["output_price_microunits_per_token"]),
            )
            runtime = AgentRuntime(
                agent_id=f"stage611-{key[0]}-{key[1]}-{key[2]}-{condition}",
                company_id=FOCAL_COMPANY,
                model_client=_model_client("doubao", str(spec["model"]), float(spec["temperature"]), float(spec["top_p"])),
                memory=EpisodeMemory(),
                persona_profile=personas.get(key[2]),
                persona_registry=personas,
            )
            decision = await runtime.decide(observation, timeout_seconds=60.0)
            row: dict[str, Any] = {
                "seed": key[0], "round": key[1], "persona_id": key[2], "condition": condition,
                "state_hash": cell["state"].state_hash, "observation_hash": observation["observation_hash"],
                "input_snapshot_hash": sha256_hash(observation), "success": decision.success,
                "model_name": decision.model_name, "prompt_version": decision.prompt_version,
                "input_tokens": int(decision.input_tokens or 0), "output_tokens": int(decision.output_tokens or 0),
                "latency_ms": decision.latency_ms, "retry_count": decision.retry_count,
                "error_code": decision.error_code, "error_message": decision.error_message,
                "raw_response": decision.raw_response,
                "provider_audit": decision.provider_audit.model_dump(mode="json") if decision.provider_audit else None,
            }
            if decision.success:
                actual_cost = row["input_tokens"] * int(price["input_price_microunits_per_token"]) + row["output_tokens"] * int(price["output_price_microunits_per_token"])
                guard.record_actual(prompt_tokens=row["input_tokens"], completion_tokens=row["output_tokens"], estimated_cost_microunits=actual_cost)
                requested = decision.decision.requested_action.model_dump(mode="json")
                resolved = resolve_action_request(config, cell["state"], FOCAL_COMPANY, requested, source=f"stage611:{condition}")
                final_action = resolved.action.to_dict()
                outcome = _simulate_closed_loop(config=config, initial_state=cell["state"], first_focal_action=final_action, horizon_rounds=min(3, cell["state"].rounds_remaining))
                adoption = build_advisor_adoption_trace(advice=cell["advice"] if condition.startswith("treatment") else None, llm_requested_action=requested, final_action=final_action, planner_output=decision.decision.plan.model_dump(mode="json"))
                row.update({"requested_action": requested, "final_action": final_action, "plan": decision.decision.plan.model_dump(mode="json"), "enterprise_value_cents": outcome["enterprise_value_cents"], "cumulative_profit_cents": outcome["cumulative_profit_cents"], "advisor_adoption": adoption.model_dump(mode="json") if adoption else None})
            rows.append(row)
            _write_json(output / "rows.json", rows)
            print(f"[{len(rows)}/6] {key} {condition} success={decision.success} tokens={row['input_tokens'] + row['output_tokens']}", flush=True)
    paired: list[dict[str, Any]] = []
    for cell_spec in spec["paired_cells"]:
        key = (int(cell_spec["seed"]), int(cell_spec["round"]), str(cell_spec["persona_id"]))
        selected = [x for x in rows if (x["seed"], x["round"], x["persona_id"]) == key]
        base = next((x for x in selected if x["condition"].startswith("baseline")), None)
        treat = next((x for x in selected if x["condition"].startswith("treatment")), None)
        if base is None or treat is None:
            continue
        if base["success"] and treat["success"]:
            paired.append({"seed": key[0], "round": key[1], "persona_id": key[2], "action_changed": base["final_action"] != treat["final_action"], "enterprise_value_delta_cents": treat["enterprise_value_cents"] - base["enterprise_value_cents"], "profit_delta_cents": treat["cumulative_profit_cents"] - base["cumulative_profit_cents"], "advisor_adoption": treat["advisor_adoption"]})
    deltas = [x["enterprise_value_delta_cents"] for x in paired]
    summary = {
        "experiment_schema_version": "stage6.11-v7-real-llm-smoke-v1.0.0",
        "evidence_level": spec["evidence_level"], "provider": "doubao", "model": spec["model"],
        "planned_calls": 6, "lost_provider_calls": lost_calls, "recorded_calls": len(rows), "successful_calls": sum(x["success"] for x in rows),
        "paired_complete": len(paired), "pairs": paired,
        "action_change_pairs": sum(x["action_changed"] for x in paired),
        "mean_enterprise_value_delta_cents": round(mean(deltas)) if deltas else None,
        "worst_enterprise_value_delta_cents": min(deltas) if deltas else None,
        "positive_zero_negative_pairs": {"positive": sum(x > 0 for x in deltas), "zero": sum(x == 0 for x in deltas), "negative": sum(x < 0 for x in deltas)},
        "usage": asdict(guard.actual), "reserved_usage": asdict(guard.reserved),
        "research_boundary": spec["research_boundary"],
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--authorize-real-model", action="store_true")
    args = parser.parse_args()
    if not args.authorize_real_model:
        raise RuntimeError("--authorize-real-model is required")
    print(json.dumps(asyncio.run(run(args.spec.resolve(), args.output.resolve())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
