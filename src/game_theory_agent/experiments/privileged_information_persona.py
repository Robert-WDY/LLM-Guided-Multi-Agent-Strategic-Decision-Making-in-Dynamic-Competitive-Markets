"""Paired persona experiment for one privileged observer in a public market.

The market remains an incomplete-information (``public``) episode.  Company A
is the only model-controlled company and is compared under two treatments:

* ``public_control``: Company A sees the same public opponent summaries.
* ``privileged_perfect``: Company A sees every opponent CompanyState field.

The market, Seed, persona and deterministic rule opponents are paired.  Winning
is a measured outcome, never an engineering acceptance condition.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

from dotenv import dotenv_values, load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ENV = dotenv_values(PROJECT_ROOT / ".env")
if _PROJECT_ENV.get("MARKET_CONFIG_PATH"):
    os.environ.setdefault(
        "MARKET_CONFIG_PATH", str(_PROJECT_ENV["MARKET_CONFIG_PATH"])
    )

from game_theory_agent.api import CONFIG  # noqa: E402
from game_theory_agent.experiments.four_agent_acceptance import (  # noqa: E402
    run as run_episode,
)
from game_theory_agent.gameplay import build_terminal_rankings  # noqa: E402
from game_theory_agent.market import MarketState  # noqa: E402
from game_theory_agent.orchestration import JsonlRoundEventLogger  # noqa: E402


TARGET_COMPANY = "company_A"
CONDITIONS = ("public_control", "privileged_perfect")
DEFAULT_PERSONAS = (
    "balanced_v1",
    "aggressive_v1_extreme",
    "conservative_v1_extreme",
    "selfish_long_term_v1",
    "profit_myopic",
)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items or len(items) != len(set(items)):
        raise ValueError("Seeds 必须是一个或多个不重复整数")
    return items


def parse_csv_strings(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items or len(items) != len(set(items)):
        raise ValueError("人格必须是一个或多个不重复名称")
    return items


def build_plan(
    seeds: tuple[int, ...], personas: tuple[str, ...]
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for persona_index, persona in enumerate(personas):
        for seed_index, seed in enumerate(seeds):
            order = (
                CONDITIONS
                if (persona_index + seed_index) % 2 == 0
                else tuple(reversed(CONDITIONS))
            )
            for call_order, condition in enumerate(order, start=1):
                plan.append(
                    {
                        "seed": seed,
                        "persona": persona,
                        "condition": condition,
                        "condition_call_order": call_order,
                    }
                )
    return plan


def _episode_args(
    args: argparse.Namespace, item: dict[str, Any], output: Path
) -> SimpleNamespace:
    privileged = (
        TARGET_COMPANY
        if item["condition"] == "privileged_perfect"
        else None
    )
    return SimpleNamespace(
        # Paired conditions intentionally reuse the same Episode ID.  Some RNG
        # components bind the episode namespace, so including the treatment in
        # the ID would create a different stochastic market path.
        episode_id=f"privileged-info-{item['persona']}-{item['seed']}",
        seed=item["seed"],
        rounds=args.rounds,
        market_model=args.market_model,
        information_mode="public",
        privileged_observer_company_id=privileged,
        provider=args.provider,
        belief_mode="off",
        opponent_model_mode="off",
        utility_inference_mode="off",
        advisor_mode="off",
        repeated_game_mode="off",
        cooperation_mode="off",
        honor_game_theory_advice=False,
        model=args.model,
        persona=item["persona"],
        condition=None,
        personas=None,
        llm_count=1,
        rotation_index=0,
        decision_support_version="economic_v2",
        persona_semantics_version="economic_v2",
        diagnostic_mode="off",
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        communication_mode="off",
        communication_timeout=30.0,
        mock_communication_scenario="silence",
        quiet=True,
        output=output,
    )


def _token_usage(events: list[Any]) -> dict[str, Any]:
    traces = [
        trace
        for event in events
        for trace in event.traces
        if trace.company_id == TARGET_COMPANY and trace.agent_type != "rule"
    ]
    input_tokens = sum(int(trace.input_tokens or 0) for trace in traces)
    output_tokens = sum(int(trace.output_tokens or 0) for trace in traces)
    return {
        "model_call_count": len(traces),
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


def summarize_episode(
    directory: Path, item: dict[str, Any]
) -> dict[str, Any]:
    base_summary = json.loads(
        (directory / "summary.json").read_text(encoding="utf-8")
    )
    events = list(
        JsonlRoundEventLogger(directory / "round-events.jsonl").read_all()
    )
    final_state = MarketState.from_dict(events[-1].state_after)
    rankings = build_terminal_rankings(final_state, CONFIG)
    composite = rankings["composite"]
    assets = rankings["total_assets"]
    target_composite = next(
        row for row in composite if row["company_id"] == TARGET_COMPANY
    )
    target_assets = next(
        row for row in assets if row["company_id"] == TARGET_COMPANY
    )
    target = final_state.company(TARGET_COMPANY)
    target_traces = [
        next(
            trace
            for trace in event.traces
            if trace.company_id == TARGET_COMPANY
        )
        for event in events
    ]
    expected_mode = (
        "perfect"
        if item["condition"] == "privileged_perfect"
        else "public"
    )
    observation_mode_matches = all(
        trace.information_snapshot is not None
        and trace.information_snapshot.information_mode == expected_mode
        for trace in target_traces
    )
    opponent_private_state_visible = all(
        trace.observation is not None
        and all(
            "financial" in opponent
            and "operations" in opponent
            and "risk" in opponent
            and "persona" in opponent
            for opponent in trace.observation["competitors"]
        )
        for trace in target_traces
    )
    public_control_private_leak_count = sum(
        any(
            private_key in opponent
            for private_key in ("financial", "operations", "risk", "persona")
        )
        for trace in target_traces
        for opponent in (trace.observation or {}).get("competitors", [])
    )
    row = {
        "episode_schema_version": "privileged-information-episode-v1.0.0",
        **item,
        "target_company_id": TARGET_COMPANY,
        "market_information_mode": "public",
        "target_information_mode": expected_mode,
        "rounds": len(events),
        "passed": base_summary["passed"],
        "protocol_checks": base_summary["protocol_checks"],
        "observation_mode_matches": observation_mode_matches,
        "private_visibility_check": (
            opponent_private_state_visible
            if expected_mode == "perfect"
            else public_control_private_leak_count == 0
        ),
        "public_control_private_leak_count": (
            public_control_private_leak_count
        ),
        "target_composite_rank": target_composite["rank"],
        "target_asset_rank": target_assets["rank"],
        "target_won_composite": target_composite["rank"] == 1,
        "target_won_assets": target_assets["rank"] == 1,
        "target_enterprise_value_cents": target_composite["value_cents"],
        "target_total_assets_cents": target_assets["value_cents"],
        "target_cumulative_profit_cents": (
            target.financial.cumulative_profit_cents
        ),
        "target_market_share_ppm": target.commercial.market_share_ppm,
        "composite_ranking": composite,
        "asset_ranking": assets,
        "token_usage": _token_usage(events),
        "artifacts": {
            "round_events": str(
                (directory / "round-events.jsonl").resolve()
            ),
            "base_summary": str((directory / "summary.json").resolve()),
        },
    }
    (directory / "privileged-information-summary.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return row


def aggregate(
    rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
    personas: tuple[str, ...],
) -> dict[str, Any]:
    by_pair = {
        (persona, seed): {
            row["condition"]: row
            for row in rows
            if row["persona"] == persona and row["seed"] == seed
        }
        for persona in personas
        for seed in seeds
    }
    paired = []
    for (persona, seed), conditions in by_pair.items():
        control = conditions["public_control"]
        treatment = conditions["privileged_perfect"]
        paired.append(
            {
                "persona": persona,
                "seed": seed,
                "control_rank": control["target_composite_rank"],
                "treatment_rank": treatment["target_composite_rank"],
                "rank_improvement": (
                    control["target_composite_rank"]
                    - treatment["target_composite_rank"]
                ),
                "enterprise_value_delta_cents": (
                    treatment["target_enterprise_value_cents"]
                    - control["target_enterprise_value_cents"]
                ),
                "profit_delta_cents": (
                    treatment["target_cumulative_profit_cents"]
                    - control["target_cumulative_profit_cents"]
                ),
                "market_share_delta_ppm": (
                    treatment["target_market_share_ppm"]
                    - control["target_market_share_ppm"]
                ),
                "treatment_won": treatment["target_won_composite"],
            }
        )
    treatment_rows = [
        row for row in rows if row["condition"] == "privileged_perfect"
    ]
    control_rows = [
        row for row in rows if row["condition"] == "public_control"
    ]
    persona_results = {
        persona: {
            "episode_count": len(seeds),
            "first_place_count": sum(
                row["target_won_composite"]
                for row in treatment_rows
                if row["persona"] == persona
            ),
            "all_seeds_first": all(
                row["target_won_composite"]
                for row in treatment_rows
                if row["persona"] == persona
            ),
        }
        for persona in personas
    }
    tokens = {
        key: sum(int(row["token_usage"][key]) for row in rows)
        for key in (
            "model_call_count",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "calls_without_token_usage",
        )
    }
    all_privileged_first = all(
        row["target_won_composite"] for row in treatment_rows
    )
    return {
        "experiment_schema_version": (
            "privileged-information-persona-v1.0.0"
        ),
        "primary_experiment_unit": "paired_persona_seed",
        "hypothesis": (
            "在不完全信息市场中，单一完全信息 Agent 是否能跨人格稳定获得第一名"
        ),
        "target_company_id": TARGET_COMPANY,
        "market_information_mode": "public",
        "seeds": list(seeds),
        "personas": list(personas),
        "episode_count": len(rows),
        "engineering_passed": all(
            row["passed"]
            and row["observation_mode_matches"]
            and row["private_visibility_check"]
            for row in rows
        ),
        "public_control_first_place_rate": (
            sum(row["target_won_composite"] for row in control_rows)
            / len(control_rows)
        ),
        "privileged_first_place_rate": (
            sum(row["target_won_composite"] for row in treatment_rows)
            / len(treatment_rows)
        ),
        "all_privileged_persona_seed_cases_first": all_privileged_first,
        "research_hypothesis_supported": all_privileged_first,
        "persona_results": persona_results,
        "paired_results": paired,
        "mean_paired_effect": {
            "rank_improvement": mean(
                row["rank_improvement"] for row in paired
            ),
            "enterprise_value_delta_cents": mean(
                row["enterprise_value_delta_cents"] for row in paired
            ),
            "profit_delta_cents": mean(
                row["profit_delta_cents"] for row in paired
            ),
            "market_share_delta_ppm": mean(
                row["market_share_delta_ppm"] for row in paired
            ),
        },
        "token_usage": tokens,
        "episodes": rows,
        "interpretation_boundary": (
            "工程通过只证明信息被正确隔离和送达；第一名是研究结果。"
            "完全信息不包含对手尚未提交的同轮行动，也不保证模型会正确利用信息。"
        ),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    seeds = parse_csv_ints(args.seeds)
    personas = parse_csv_strings(args.personas)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = build_plan(seeds, personas)
    (output / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    for item in plan:
        directory = (
            output
            / item["persona"]
            / f"seed-{item['seed']}"
            / item["condition"]
        )
        episode_summary = directory / "privileged-information-summary.json"
        if episode_summary.exists():
            if not args.resume:
                raise FileExistsError(
                    f"已有结果，继续运行请添加 --resume：{episode_summary}"
                )
            rows.append(json.loads(episode_summary.read_text(encoding="utf-8")))
            continue
        exit_code = await run_episode(_episode_args(args, item, directory))
        if exit_code != 0:
            raise RuntimeError(
                "Episode 工程验收失败："
                f"persona={item['persona']} seed={item['seed']} "
                f"condition={item['condition']}"
            )
        rows.append(summarize_episode(directory, item))
        partial = {
            "completed_episode_count": len(rows),
            "planned_episode_count": len(plan),
            "completed": [
                {
                    "persona": row["persona"],
                    "seed": row["seed"],
                    "condition": row["condition"],
                }
                for row in rows
            ],
        }
        (output / "partial-summary.json").write_text(
            json.dumps(partial, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = aggregate(rows, seeds, personas)
    summary.update(
        {
            "provider": args.provider,
            "model": args.model,
            "rounds": args.rounds,
            "market_model": args.market_model,
            "passed": summary["engineering_passed"],
        }
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=("mock", "doubao", "deepseek"), default="mock"
    )
    parser.add_argument("--model")
    parser.add_argument("--seeds", default="6101,6102,6103")
    parser.add_argument("--personas", default=",".join(DEFAULT_PERSONAS))
    parser.add_argument(
        "--rounds", type=int, choices=(5, 10, 15, 20), default=10
    )
    parser.add_argument("--market-model", default="balanced")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = asyncio.run(run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
