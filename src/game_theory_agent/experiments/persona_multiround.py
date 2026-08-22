"""Multi-round Persona benchmark against deterministic rule opponents."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from game_theory_agent.agents import (
    AgentRuntime,
    DecisionContextBuilder,
    EpisodeMemory,
    ResultAnalyzer,
    load_persona_registry,
)
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.experiments.persona_pilot import _model_client, _observation
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import MarketEnv, load_market_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PERSONAS = (
    "none",
    "balanced",
    "aggressive",
    "conservative",
    "selfish_long_term",
    "profit_myopic",
)


async def _run_episode(
    *,
    provider: str,
    client: object,
    config: Any,
    registry: Any,
    persona_id: str,
    seed: int,
    rounds: int,
    market_model: str,
    timeout: float,
    context_mode: str = "full",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episode_id = f"persona-multiround-{persona_id}-{seed}-{uuid.uuid4().hex[:6]}"
    env = MarketEnv(config)
    initial = env.reset(
        episode_id=episode_id,
        episode_seed=seed,
        market_model=market_model,
        max_rounds=rounds,
    )
    profile = registry.get(persona_id)
    runtime = AgentRuntime(
        agent_id=f"persona-multiround-{provider}-{persona_id}",
        company_id="company_A",
        model_client=client,
        memory=EpisodeMemory(),
        context_builder=DecisionContextBuilder(
            persona_profile=profile,
            persona_registry=registry,
            context_mode=context_mode,
        ),
    )
    analyzer = ResultAnalyzer()
    rows: list[dict[str, Any]] = []
    minimum_cash = initial.company("company_A").financial.cash_balance_cents
    total_spend = 0
    fallbacks = 0

    while not env.get_state().terminal:
        before = env.get_state()
        observation = _observation(config, before)
        result = await runtime.decide(observation, timeout_seconds=timeout)
        adjustments: list[dict[str, Any]] = []
        if result.success and result.decision is not None:
            requested = result.decision.requested_action.model_dump(mode="json")
            resolution = resolve_action_request(
                config,
                before,
                "company_A",
                requested,
                source=f"persona-multiround:{persona_id}",
                action_id=(
                    f"persona-multiround:{episode_id}:{before.round}:"
                    f"{before.state_version}"
                ),
            )
            own_action = resolution.action
            adjustments = [item.to_dict() for item in resolution.adjustments]
        else:
            own_action = build_rule_action(config, before, "company_A")
            fallbacks += 1
            runtime.memory.record_fallback()
        actions = {
            company_id: build_rule_action(config, before, company_id)
            for company_id in before.company_ids
            if company_id != "company_A"
        }
        actions["company_A"] = own_action
        step = env.step(
            f"{before.episode_id}:{before.round}:{before.state_version}", actions
        )
        after = step.state_after
        utility = runtime.assess_persona_utility(before, after)
        decision = result.decision if result.success else None
        analysis = analyzer.analyze(
            before,
            after,
            "company_A",
            decision.plan.expected_outcome if decision else None,
            adjustments,
            decision.plan.success_criteria if decision else None,
        )
        if decision is not None:
            runtime.memory.record(
                decision, own_action.to_dict(), analysis, utility
            )
        else:
            runtime.memory.record_fallback_outcome(
                None,
                own_action.to_dict(),
                analysis,
                result.error_code,
                utility,
            )
        company = after.company("company_A")
        minimum_cash = min(minimum_cash, company.financial.cash_balance_cents)
        total_spend += own_action.fixed_spend_cents
        rows.append(
            {
                "row_schema_version": "persona-multiround-row-v1.0.0",
                "episode_id": episode_id,
                "provider": provider,
                "persona_id": persona_id,
                "context_mode": context_mode,
                "market_seed": seed,
                "settled_round": before.round,
                "state_before_hash": before.state_hash,
                "state_after_hash": after.state_hash,
                "success": result.success,
                "error_code": result.error_code,
                "model_name": result.model_name,
                "prompt_version": result.prompt_version,
                "decision_context_mode": result.context.context_mode,
                "current_plan": result.context.current_plan,
                "recent_round_count": len(result.context.recent_rounds),
                "critical_event_count": len(result.context.critical_events),
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "action": own_action.to_dict(),
                "planner_strategy_summary": (
                    decision.plan.strategy_summary if decision else None
                ),
                "resolution_adjustments": adjustments,
                "round_profit_cents": company.financial.round_profit_cents,
                "cash_balance_cents": company.financial.cash_balance_cents,
                "market_share_ppm": company.commercial.market_share_ppm,
                "reputation_ppm": company.brand.reputation_ppm,
                "resilience_ppm": company.risk.resilience_ppm,
                "discounted_round_utility_ppm": (
                    utility.discounted_round_utility_ppm
                ),
                "cumulative_discounted_utility_ppm": (
                    utility.cumulative_discounted_utility_ppm
                ),
            }
        )

    final = env.get_state()
    company = final.company("company_A")
    terminal_values = dict(final.terminal_enterprise_values_cents)
    action_fields = (
        "price_cents",
        "advertising_budget_cents",
        "service_budget_cents",
        "capacity_investment_cents",
        "resilience_budget_cents",
        "incident_response",
    )
    action_signatures = [
        json.dumps(
            {field: row["action"][field] for field in action_fields},
            sort_keys=True,
        )
        for row in rows
    ]
    repeated_actions = sum(
        current == previous
        for previous, current in zip(action_signatures, action_signatures[1:])
    )
    replan_count = sum(
        bool((row.get("current_plan") or {}).get("replanned")) for row in rows
    )
    episode_summary = {
        "episode_id": episode_id,
        "persona_id": persona_id,
        "context_mode": context_mode,
        "condition_id": f"{persona_id}:{context_mode}",
        "market_seed": seed,
        "rounds": len(rows),
        "fallback_count": fallbacks,
        "consecutive_repeated_action_count": repeated_actions,
        "consecutive_repeated_action_rate": round(
            repeated_actions / max(1, len(rows) - 1), 4
        ),
        "replan_count": replan_count,
        "distinct_plan_count": len(
            {
                plan_id
                for row in rows
                if (plan_id := (row.get("current_plan") or {}).get("plan_id"))
            }
        ),
        "cumulative_profit_cents": company.financial.cumulative_profit_cents,
        "final_cash_cents": company.financial.cash_balance_cents,
        "minimum_cash_cents": minimum_cash,
        "final_market_share_ppm": company.commercial.market_share_ppm,
        "final_reputation_ppm": company.brand.reputation_ppm,
        "final_resilience_ppm": company.risk.resilience_ppm,
        "terminal_enterprise_value_cents": terminal_values.get("company_A"),
        "total_fixed_spend_cents": total_spend,
        "cumulative_discounted_utility_ppm": rows[-1][
            "cumulative_discounted_utility_ppm"
        ],
        "final_state_hash": final.state_hash,
    }
    return rows, episode_summary


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "consecutive_repeated_action_rate",
        "replan_count",
        "distinct_plan_count",
        "cumulative_profit_cents",
        "final_cash_cents",
        "minimum_cash_cents",
        "final_market_share_ppm",
        "final_reputation_ppm",
        "final_resilience_ppm",
        "terminal_enterprise_value_cents",
        "total_fixed_spend_cents",
        "cumulative_discounted_utility_ppm",
    )
    result: dict[str, Any] = {}
    for condition_id in sorted({item["condition_id"] for item in episodes}):
        group = [item for item in episodes if item["condition_id"] == condition_id]
        result[condition_id] = {
            "episode_count": len(group),
            "fallback_count": sum(item["fallback_count"] for item in group),
            **{
                f"mean_{field}": round(
                    mean(float(item[field]) for item in group), 2
                )
                for field in fields
            },
        }
    return result


async def run(args: argparse.Namespace) -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config_path = Path(
        os.environ.get(
            "MARKET_CONFIG_PATH", PROJECT_ROOT / "configs" / "market_v4.yaml"
        )
    )
    config = load_market_config(config_path)
    registry = load_persona_registry(config_path)
    personas = tuple(item.strip() for item in args.personas.split(",") if item.strip())
    for persona_id in personas:
        registry.get(persona_id)
    seeds = tuple(int(item.strip()) for item in args.market_seeds.split(",") if item.strip())
    context_modes = tuple(
        item.strip() for item in args.context_modes.split(",") if item.strip()
    )
    if not context_modes or set(context_modes) - {"full", "state_only"}:
        raise ValueError("context modes must be full or state_only")
    output = args.output or (
        PROJECT_ROOT
        / "runs"
        / f"persona-multiround-{args.provider}-{uuid.uuid4().hex[:8]}"
    )
    output.mkdir(parents=True, exist_ok=False)
    client = _model_client(
        args.provider, args.model, args.temperature, args.top_p
    )
    all_rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    total = len(personas) * len(seeds) * len(context_modes)
    index = 0
    for seed in seeds:
        for context_mode in context_modes:
            for persona_id in personas:
                index += 1
                print(
                    f"[{index}/{total}] seed={seed} persona={persona_id} "
                    f"context={context_mode}",
                    flush=True,
                )
                rows, episode = await _run_episode(
                    provider=args.provider,
                    client=client,
                    config=config,
                    registry=registry,
                    persona_id=persona_id,
                    seed=seed,
                    rounds=args.rounds,
                    market_model=args.market_model,
                    timeout=args.timeout,
                    context_mode=context_mode,
                )
                all_rows.extend(rows)
                episodes.append(episode)
                with (output / "rounds.jsonl").open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                        handle.write("\n")
    summary = {
        "summary_schema_version": "persona-multiround-summary-v1.0.0",
        "experiment_id": output.name,
        "provider": args.provider,
        "model": args.model,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "provider_seed": None,
        },
        "personas": list(personas),
        "context_modes": list(context_modes),
        "market_seeds": list(seeds),
        "rounds_per_episode": args.rounds,
        "market_model": args.market_model,
        "rule_opponent_policy": "state_dependent_deterministic_v1",
        "episode_count": len(episodes),
        "decision_count": len(all_rows),
        "successful_decisions": sum(row["success"] for row in all_rows),
        "fallback_count": sum(not row["success"] for row in all_rows),
        "episodes": episodes,
        "persona_aggregates": _aggregate(episodes),
        "config_version": config.config_version,
        "config_sha256": config.config_sha256,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"OUTPUT_DIR={output.resolve()}", flush=True)
    return 0 if summary["fallback_count"] == 0 else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("mock", "doubao", "deepseek"), default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--personas", default=",".join(DEFAULT_PERSONAS))
    parser.add_argument("--context-modes", default="full")
    parser.add_argument("--market-seeds", default="42")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--market-model", default="balanced")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    return asyncio.run(run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
