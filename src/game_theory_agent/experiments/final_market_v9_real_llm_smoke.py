"""Cost-bounded paired real-LLM smoke for final-market Advisor v9."""

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

from game_theory_agent.advisor import (
    build_advisor_adoption_trace,
    build_agent_advice_view,
)
from game_theory_agent.agents import AgentRuntime, EpisodeMemory, MarketRegimeEvaluator
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.economics import decision_support_metrics
from game_theory_agent.experiments.final_market_advisor_v9_validation import (
    CONFIG_PATH,
    FOCAL,
    HORIZON,
    SCENARIOS,
    _actual_outcome,
    _build_window,
)
from game_theory_agent.experiments.persona_pilot import _model_client
from game_theory_agent.gameplay import build_company_analysis
from game_theory_agent.information import seal_observation
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.strategic_reliability import (
    PublicMarketRolloutAdvisor,
    RealModelBudget,
    RealModelCostGuard,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = (
    PROJECT_ROOT
    / "experiment-specs"
    / "final-market-v9-real-llm-smoke"
    / "PREREGISTRATION.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "final-market-v9-real-llm-smoke"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _complete_observation(
    config: Any,
    state: Any,
    observation: Mapping[str, Any],
    *,
    advice: Mapping[str, Any] | None,
) -> dict[str, Any]:
    env = MarketEnv(config)
    env.load_state(state)
    payload = json.loads(json.dumps(observation, ensure_ascii=False))
    payload.update(
        {
            "observation_schema_version": "agent-observation-v1.8.0",
            "episode_id": state.episode_id,
            "round": state.round,
            "decision_round": None if state.terminal else state.round,
            "last_settled_round": min(state.state_version, state.max_rounds),
            "rounds_remaining": state.rounds_remaining,
            "state_version": state.state_version,
            "state_hash": state.state_hash,
            "terminal": state.terminal,
            "episode_config": {
                "max_rounds": state.max_rounds,
                "market_model_id": state.market.market_model_id,
                "information_mode": "public",
                "market_information_mode": "public",
                "company_scoped_information_treatment": False,
                "communication_mode": "off",
                "cooperation_mode": "combined_v1",
                "belief_mode": "simple_v1",
                "advisor_mode": "off",
                "opponent_model_mode": "strategy_utility_v1",
                "utility_inference_mode": "strategy_utility_v1",
                "repeated_game_mode": "off",
            },
            "information_mode": "public",
            "communication_mode": "off",
            "cooperation_mode": "combined_v1",
            "market_regime": MarketRegimeEvaluator(config).evaluate(
                state, information_mode="public"
            ),
            "decision_support": decision_support_metrics(config, state, FOCAL),
            "action_constraints": env.get_action_constraints(
                FOCAL, state.state_version
            ),
            "company_analysis": build_company_analysis(state, FOCAL, config),
            "utility_inference_state": None,
            "utility_inference_hash": None,
            "repeated_game_strategy": None,
            "repeated_game_strategy_hash": None,
        }
    )
    if advice is not None:
        agent_advice = build_agent_advice_view(advice)
        if agent_advice is not None:
            payload["game_theory_advice"] = agent_advice
            payload["episode_config"]["advisor_mode"] = "strategic_market_v9"
    return seal_observation(payload)


def _is_research_only_coordination(action: Mapping[str, Any]) -> bool:
    return bool(
        action.get("price_coordination_partner_company_id")
        or action.get("price_coordination_target_cents") is not None
    )


def _best_operational_outcome(
    config: Any,
    state: Any,
    advice: Mapping[str, Any],
) -> tuple[str, dict[str, int]]:
    research_only = set(advice.get("research_only_candidate_ids") or ())
    outcomes: dict[str, dict[str, int]] = {}
    for row in advice["candidate_actions"]:
        candidate = row["candidate"]
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in research_only:
            continue
        outcomes[candidate_id] = _actual_outcome(
            config,
            state,
            candidate_id,
            candidate["action"],
        )
    best_id = max(
        outcomes,
        key=lambda item: (outcomes[item]["enterprise_value_cents"], item),
    )
    return best_id, outcomes[best_id]


async def run(spec_path: Path, output: Path) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    provider = str(spec["provider"])
    credential = "ARK_API_KEY" if provider == "doubao" else "DEEPSEEK_API_KEY"
    if not os.environ.get(credential):
        raise RuntimeError(f"{credential} is required")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "plan.json", spec)
    incident_path = output / "incident.json"
    incident = (
        json.loads(incident_path.read_text(encoding="utf-8"))
        if incident_path.exists()
        else None
    )
    lost_calls = int(incident["lost_provider_call_count"]) if incident else 0

    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    advisor = PublicMarketRolloutAdvisor(config)
    price = spec["price_snapshot"]
    guard = RealModelCostGuard(
        RealModelBudget(
            max_calls=int(spec["maximum_provider_calls"]) - lost_calls,
            max_prompt_tokens=int(spec["maximum_prompt_tokens"]),
            max_completion_tokens=int(spec["maximum_completion_tokens"]),
            max_estimated_cost_microunits=int(
                spec["maximum_estimated_cost_microunits"]
            ),
        ),
        explicitly_authorized=True,
    )
    cells: dict[tuple[int, str, str], dict[str, Any]] = {}
    for cell_spec in spec["paired_cells"]:
        key = (
            int(cell_spec["seed"]),
            str(cell_spec["window"]),
            str(cell_spec["persona_id"]),
        )
        state, belief, opponent, observation = _build_window(
            config, key[0], key[1]
        )
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
        expected = str(cell_spec["expected_disposition"])
        if advice["execution_disposition"] != expected:
            raise AssertionError(
                f"preregistered disposition changed for {key}: "
                f"{advice['execution_disposition']} != {expected}"
            )
        best_id, best_outcome = _best_operational_outcome(
            config, state, advice
        )
        cells[key] = {
            "state": state,
            "observation": observation,
            "advice": advice,
            "best_candidate_id": best_id,
            "best_outcome": best_outcome,
        }

    rows_path = output / "rows.json"
    rows: list[dict[str, Any]] = (
        json.loads(rows_path.read_text(encoding="utf-8"))
        if rows_path.exists()
        else []
    )
    completed_keys = {
        (
            int(row["seed"]),
            str(row["window"]),
            str(row["persona_id"]),
            str(row["condition"]),
        )
        for row in rows
    }
    lost_key = tuple(incident["lost_call_key"]) if incident else None
    total = sum(len(cell["call_order"]) for cell in spec["paired_cells"])
    for cell_spec in spec["paired_cells"]:
        key = (
            int(cell_spec["seed"]),
            str(cell_spec["window"]),
            str(cell_spec["persona_id"]),
        )
        cell = cells[key]
        for condition in cell_spec["call_order"]:
            call_key = (key[0], key[1], key[2], condition)
            if call_key in completed_keys or call_key == lost_key:
                continue
            advice = (
                cell["advice"]
                if condition == "treatment_with_v9_advice"
                else None
            )
            observation = _complete_observation(
                config,
                cell["state"],
                cell["observation"],
                advice=advice,
            )
            reserved_input = int(spec["reserved_prompt_tokens_per_call"])
            reserved_output = int(spec["reserved_completion_tokens_per_call"])
            guard.reserve(
                prompt_tokens=reserved_input,
                completion_tokens=reserved_output,
                estimated_cost_microunits=(
                    reserved_input
                    * int(price["input_price_microunits_per_token"])
                    + reserved_output
                    * int(price["output_price_microunits_per_token"])
                ),
            )
            runtime = AgentRuntime(
                agent_id=(
                    f"v9-real-{key[0]}-{key[1]}-{key[2]}-{condition}"
                ),
                company_id=FOCAL,
                model_client=_model_client(
                    provider,
                    str(spec["model"]),
                    float(spec["temperature"]),
                    float(spec["top_p"]),
                ),
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
                "input_snapshot_hash": sha256_hash(observation),
                "advice_hash": (
                    cell["advice"]["advice_hash"] if advice else None
                ),
                "advice_disposition": (
                    cell["advice"]["execution_disposition"] if advice else None
                ),
                "advice_candidate_id": (
                    cell["advice"]["recommended_candidate_id"] if advice else None
                ),
                "best_operational_candidate_id": cell["best_candidate_id"],
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
                actual_cost = (
                    row["input_tokens"]
                    * int(price["input_price_microunits_per_token"])
                    + row["output_tokens"]
                    * int(price["output_price_microunits_per_token"])
                )
                guard.record_actual(
                    prompt_tokens=row["input_tokens"],
                    completion_tokens=row["output_tokens"],
                    estimated_cost_microunits=actual_cost,
                )
                requested = result.decision.requested_action.model_dump(
                    mode="json"
                )
                resolved = resolve_action_request(
                    config,
                    cell["state"],
                    FOCAL,
                    requested,
                    source=f"final-v9-real:{condition}",
                )
                final_action = resolved.action.to_dict()
                outcome = _actual_outcome(
                    config,
                    cell["state"],
                    condition,
                    final_action,
                )
                candidate_gap = (
                    int(cell["best_outcome"]["enterprise_value_cents"])
                    - int(outcome["enterprise_value_cents"])
                )
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
                            item.to_dict() for item in resolved.adjustments
                        ],
                        "outcome": outcome,
                        "candidate_opportunity_gap_cents": candidate_gap,
                        "candidate_regret_cents": max(0, candidate_gap),
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
            _write_json(rows_path, rows)
            print(
                f"[{len(rows)}/{total}] {key} {condition} "
                f"success={result.success} "
                f"tokens={row['input_tokens'] + row['output_tokens']}",
                flush=True,
            )

    pairs: list[dict[str, Any]] = []
    for cell_spec in spec["paired_cells"]:
        key = (
            int(cell_spec["seed"]),
            str(cell_spec["window"]),
            str(cell_spec["persona_id"]),
        )
        selected = [
            row
            for row in rows
            if (row["seed"], row["window"], row["persona_id"]) == key
        ]
        baseline = next(
            (
                row
                for row in selected
                if row["condition"] == "baseline_without_advice"
            ),
            None,
        )
        treatment = next(
            (
                row
                for row in selected
                if row["condition"] == "treatment_with_v9_advice"
            ),
            None,
        )
        if not baseline or not treatment:
            continue
        if not baseline["success"] or not treatment["success"]:
            continue
        pairs.append(
            {
                "seed": key[0],
                "window": key[1],
                "persona_id": key[2],
                "advice_disposition": treatment["advice_disposition"],
                "advice_candidate_id": treatment["advice_candidate_id"],
                "action_changed": (
                    baseline["final_action"] != treatment["final_action"]
                ),
                "enterprise_value_delta_cents": (
                    treatment["outcome"]["enterprise_value_cents"]
                    - baseline["outcome"]["enterprise_value_cents"]
                ),
                "candidate_regret_reduction_cents": (
                    baseline["candidate_regret_cents"]
                    - treatment["candidate_regret_cents"]
                ),
                "advisor_adoption": treatment["advisor_adoption"],
            }
        )

    ev_deltas = [int(pair["enterprise_value_delta_cents"]) for pair in pairs]
    regret_deltas = [
        int(pair["candidate_regret_reduction_cents"]) for pair in pairs
    ]
    safety_count = sum(
        bool(row.get("research_only_price_coordination_action"))
        for row in rows
    )
    successful = sum(bool(row["success"]) for row in rows)
    actionable = [
        pair for pair in pairs if pair["advice_disposition"] == "recommend"
    ]
    summary = {
        "result_schema_version": "final-market-v9-real-llm-smoke-v1.0.0",
        "evidence_level": spec["evidence_level"],
        "provider": provider,
        "model": spec["model"],
        "planned_calls": total,
        "lost_provider_calls": lost_calls,
        "attempted_provider_calls": len(rows) + lost_calls,
        "recorded_calls": len(rows),
        "successful_calls": successful,
        "failed_call_count": len(rows) - successful,
        "paired_complete": len(pairs),
        "pairs": pairs,
        "action_change_pairs": sum(bool(pair["action_changed"]) for pair in pairs),
        "actionable_pair_count": len(actionable),
        "actionable_adoption_count": sum(
            bool((pair.get("advisor_adoption") or {}).get("accepted"))
            for pair in actionable
        ),
        "mean_enterprise_value_delta_cents": (
            round(mean(ev_deltas)) if ev_deltas else None
        ),
        "worst_enterprise_value_delta_cents": (
            min(ev_deltas) if ev_deltas else None
        ),
        "mean_candidate_regret_reduction_cents": (
            round(mean(regret_deltas)) if regret_deltas else None
        ),
        "research_only_price_coordination_action_count": safety_count,
        "provider_audit_complete": all(
            isinstance(row.get("provider_audit"), dict)
            and bool(row["provider_audit"].get("request_id"))
            and bool(row["provider_audit"].get("response_model"))
            for row in rows
            if row["success"]
        ),
        "usage": asdict(guard.actual),
        "usage_is_lower_bound": bool(lost_calls),
        "reserved_usage": asdict(guard.reserved),
        "interpretation_boundary": spec["success_interpretation"],
    }
    summary["result_hash"] = sha256_hash(summary)
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
    summary = asyncio.run(run(args.spec.resolve(), args.output.resolve()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
