"""Paired 2-LLM + 2-Rule cooperation experiment over common Seeds."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path
from statistics import mean
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
from game_theory_agent.api import CONFIG, CONFIG_PATH, SESSIONS, app
from game_theory_agent.cooperation.replay import verify_cooperation_replay
from game_theory_agent.experiments.cooperation_metrics import (
    compute_cooperation_metrics,
)
from game_theory_agent.experiments.four_agent_acceptance import (
    _ControllerAdapter,
    _GatewayAdapter,
)
from game_theory_agent.experiments.persona_pilot import _model_client
from game_theory_agent.interaction.replay import verify_interaction_replay
from game_theory_agent.information import verify_information_replay
from game_theory_agent.market import MarketEnv
from game_theory_agent.market.replay import verify_replay
from game_theory_agent.orchestration import JsonlRoundEventLogger, RoundCoordinator


COMPANIES = ("company_A", "company_B", "company_C", "company_D")
LLM_COMPANIES = COMPANIES[:2]
CONDITIONS = {
    "action_only": {
        "communication_mode": "off",
        "cooperation_history_mode": "none",
    },
    "communication_no_history": {
        "communication_mode": "public_private",
        "cooperation_history_mode": "none",
    },
    "communication_with_history": {
        "communication_mode": "public_private",
        "cooperation_history_mode": "full",
    },
}
RESEARCH_METRICS = (
    "proposal_rate_ppm",
    "peer_acceptance_rate_ppm",
    "commitment_rate_ppm",
    "amount_weighted_fulfillment_ppm",
    "betrayal_rate_ppm",
    "total_contribution_cents",
    "free_rider_company_round_rate_ppm",
    "mean_industry_resilience_after_ppm",
    "mean_public_protection_applied_ppm",
    "market_total_round_profit_cents",
    "event_exposed_total_profit_cents",
    "company_incident_observation_count",
    "cumulative_lost_after_stockout_orders",
)


def _token_usage(events: list[Any]) -> dict[str, Any]:
    usage = {
        "token_usage_schema_version": "model-token-usage-v1.0.0",
        "decision_input_tokens": 0,
        "decision_output_tokens": 0,
        "communication_input_tokens": 0,
        "communication_output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "llm_call_count": 0,
        "calls_without_token_usage": 0,
        "by_company": {},
        "models_used": [],
    }
    models: set[str] = set()
    for event in events:
        records = [("decision", trace) for trace in event.traces]
        phase = event.communication_phase
        # In off mode the coordinator emits model-attributed disabled/silence
        # audit rows, but no provider request is made.  They are not LLM calls.
        if phase is not None and phase.mode != "off":
            records.extend(
                ("communication", trace)
                for trace in phase.generation_traces
            )
        for kind, trace in records:
            if trace.agent_type != "model":
                continue
            usage["llm_call_count"] += 1
            if trace.model_name:
                models.add(trace.model_name)
            company = usage["by_company"].setdefault(
                trace.company_id,
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
            input_tokens = trace.input_tokens
            output_tokens = trace.output_tokens
            if input_tokens is None or output_tokens is None:
                usage["calls_without_token_usage"] += 1
            input_value = int(input_tokens or 0)
            output_value = int(output_tokens or 0)
            usage[f"{kind}_input_tokens"] += input_value
            usage[f"{kind}_output_tokens"] += output_value
            company["input_tokens"] += input_value
            company["output_tokens"] += output_value
            company["total_tokens"] += input_value + output_value
            usage["estimated_cost_usd"] += float(
                trace.estimated_cost_usd or 0.0
            )
    usage["input_tokens"] = (
        usage["decision_input_tokens"]
        + usage["communication_input_tokens"]
    )
    usage["output_tokens"] = (
        usage["decision_output_tokens"]
        + usage["communication_output_tokens"]
    )
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    usage["estimated_cost_usd"] = round(usage["estimated_cost_usd"], 8)
    usage["models_used"] = sorted(models)
    return usage


def _sum_token_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = (
        "decision_input_tokens",
        "decision_output_tokens",
        "communication_input_tokens",
        "communication_output_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "llm_call_count",
        "calls_without_token_usage",
    )
    result = {key: sum(int(row[key]) for row in rows) for key in numeric}
    result["estimated_cost_usd"] = round(
        sum(float(row["estimated_cost_usd"]) for row in rows), 8
    )
    result["models_used"] = sorted(
        {model for row in rows for model in row["models_used"]}
    )
    return result


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("provide one or more unique comma-separated Seeds")
    return seeds


def build_plan(seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "condition": condition,
            "communication_mode": settings["communication_mode"],
            "cooperation_history_mode": settings["cooperation_history_mode"],
            "primary_experiment_unit": "paired_seed",
        }
        for seed in seeds
        for condition, settings in CONDITIONS.items()
    ]


async def _run_one(
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
    settings = CONDITIONS[condition]
    episode_dir = output / f"seed-{seed}" / condition
    event_path = episode_dir / "round-events.jsonl"
    if event_path.exists() and event_path.stat().st_size:
        raise ValueError(f"refusing to append to non-empty log: {event_path}")
    episode_dir.mkdir(parents=True, exist_ok=True)
    token = f"cooperation-real-{uuid.uuid4().hex}"
    os.environ["MARKET_CONTROLLER_TOKEN"] = token
    SESSIONS.clear()
    episode_id = f"cooperation-{seed}-{condition}"
    registry = load_persona_registry(CONFIG_PATH)
    profile = registry.get(persona)
    agent_configs = {
        company_id: {
            "agent_id": f"{provider}-{company_id}",
            "provider": provider,
            "model": model,
            "persona_id": persona,
            "persona_profile_hash": profile.profile_hash,
            "cooperation_history_mode": settings[
                "cooperation_history_mode"
            ],
        }
        for company_id in LLM_COMPANIES
    }
    created_response = TestClient(app).post(
        "/api/episodes",
        headers={"X-Controller-Token": token},
        json={
            "episode_id": episode_id,
            "episode_seed": seed,
            "company_ids": list(COMPANIES),
            "max_rounds": rounds,
            "market_model": "balanced",
            "information_mode": "perfect",
            "communication_mode": settings["communication_mode"],
            "cooperation_mode": "shared_resilience_v1",
            "agent_configs": agent_configs,
        },
    )
    created_response.raise_for_status()
    created = created_response.json()
    runtimes = {
        company_id: AgentRuntime(
            agent_configs[company_id]["agent_id"],
            company_id,
            _model_client(provider, model, temperature, top_p),
            context_builder=DecisionContextBuilder(
                persona_profile=profile,
                persona_registry=registry,
                cooperation_history_mode=settings[
                    "cooperation_history_mode"
                ],
            ),
        )
        for company_id in LLM_COMPANIES
    }
    coordinator = RoundCoordinator(
        _ControllerAdapter(token),
        _GatewayAdapter(created["agent_tokens"]),
        runtimes,
        event_logger=JsonlRoundEventLogger(event_path),
    )
    coordinated = await coordinator.run_episode(episode_id)
    events = [item.event for item in coordinated]
    session = SESSIONS[episode_id]
    interaction = verify_interaction_replay(events)
    information = verify_information_replay(events, session.manifest)
    cooperation = verify_cooperation_replay(events, MarketEnv(CONFIG))
    economic = verify_replay(MarketEnv(CONFIG), session.manifest, session.transitions)
    metrics = compute_cooperation_metrics(events)
    token_usage = _token_usage(events)
    summary = {
        "episode_schema_version": "cooperation-real-episode-v1.0.0",
        "seed": seed,
        "condition": condition,
        "provider": provider,
        "model": model,
        "persona": persona,
        "rounds": rounds,
        "llm_count": 2,
        "rule_count": 2,
        "communication_mode": settings["communication_mode"],
        "cooperation_history_mode": settings["cooperation_history_mode"],
        "completed_rounds": len(events),
        "replay": {
            "economic": len(economic) == rounds + 1,
            "interaction": len(interaction) == rounds,
            "information": len(information)
            == rounds
            * len(LLM_COMPANIES)
            * (2 if settings["communication_mode"] != "off" else 1),
            "cooperation": len(cooperation) == rounds,
        },
        "metrics": metrics,
        "token_usage": token_usage,
        "passed": (
            len(events) == rounds
            and len(economic) == rounds + 1
            and len(interaction) == rounds
            and len(information)
            == rounds
            * len(LLM_COMPANIES)
            * (2 if settings["communication_mode"] != "off" else 1)
            and len(cooperation) == rounds
        ),
        "artifacts": {"round_events": "round-events.jsonl"},
    }
    (episode_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


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
    results = []
    for index, row in enumerate(plan, start=1):
        summary_path = (
            output / f"seed-{row['seed']}" / row["condition"] / "summary.json"
        )
        print(
            f"[{index}/{len(plan)}] {row['condition']} seed={row['seed']} starting",
            flush=True,
        )
        if summary_path.exists():
            result = json.loads(summary_path.read_text(encoding="utf-8"))
            expected = (row["seed"], row["condition"], args.rounds, args.provider)
            actual = (
                result.get("seed"), result.get("condition"),
                result.get("rounds"), result.get("provider"),
            )
            if actual != expected or "token_usage" not in result:
                raise ValueError(f"incompatible resume summary: {summary_path}")
            print(f"[{index}/{len(plan)}] resumed from disk", flush=True)
        else:
            result = await _run_one(
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
        results.append(result)
        print(
            f"[{index}/{len(plan)}] completed; "
            f"tokens={result['token_usage']['total_tokens']}",
            flush=True,
        )
    by_key = {
        (item["seed"], item["condition"]): item for item in results
    }
    condition_means = {
        condition: {
            metric: mean(
                float(item["metrics"][metric])
                for item in results
                if item["condition"] == condition
            )
            for metric in RESEARCH_METRICS
        }
        for condition in CONDITIONS
    }
    paired_comparisons = {}
    for label, treatment, control in (
        (
            "communication_effect_without_history",
            "communication_no_history",
            "action_only",
        ),
        (
            "commitment_history_effect",
            "communication_with_history",
            "communication_no_history",
        ),
    ):
        seed_rows = [
            {
                "seed": seed,
                "deltas": {
                    metric: (
                        float(by_key[(seed, treatment)]["metrics"][metric])
                        - float(by_key[(seed, control)]["metrics"][metric])
                    )
                    for metric in RESEARCH_METRICS
                },
            }
            for seed in seeds
        ]
        paired_comparisons[label] = {
            "treatment": treatment,
            "control": control,
            "seed_deltas": seed_rows,
            "mean_delta": {
                metric: mean(row["deltas"][metric] for row in seed_rows)
                for metric in RESEARCH_METRICS
            },
        }
    aggregate = {
        "experiment_schema_version": "cooperation-real-multiseed-v1.1.0",
        "primary_experiment_unit": "paired_seed",
        "seeds": list(seeds),
        "conditions": list(CONDITIONS),
        "rounds_per_episode": args.rounds,
        "llm_rule_composition": "2 LLM + 2 Rule",
        "run_count": len(results),
        "passed_run_count": sum(item["passed"] for item in results),
        "all_engineering_checks_passed": all(item["passed"] for item in results),
        "condition_means": condition_means,
        "paired_comparisons": paired_comparisons,
        "token_usage": _sum_token_usage(
            [item["token_usage"] for item in results]
        ),
        "token_usage_by_condition": {
            condition: _sum_token_usage(
                [
                    item["token_usage"]
                    for item in results
                    if item["condition"] == condition
                ]
            )
            for condition in CONDITIONS
        },
        "results": results,
        "interpretation_boundary": (
            "Behavior directions are research results, not engineering pass criteria."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("deepseek", "doubao"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--persona", default="balanced_v1")
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in range(101, 111))
    )
    parser.add_argument("--rounds", type=int, choices=(5, 10, 15, 20), default=10)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--output", default="runs/cooperation-real-multiseed")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    summary = asyncio.run(run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("plan_only") or summary["all_engineering_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
