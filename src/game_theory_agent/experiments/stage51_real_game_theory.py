"""Paired real-LLM Stage 5.1 Game Theory ablation and evaluation.

The four treatments share Seed, model, persona, market and deterministic rule
opponents.  They differ only in strategic information made available to the
LLM company.  All research claims use Seed-level paired summaries.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace
from typing import Any

from dotenv import dotenv_values, load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
# The market config is loaded at api import time.  Resolve only its selector
# first; credentials remain loaded later when the experiment actually runs.
_PROJECT_ENV = dotenv_values(PROJECT_ROOT / ".env")
if _PROJECT_ENV.get("MARKET_CONFIG_PATH"):
    os.environ.setdefault(
        "MARKET_CONFIG_PATH", str(_PROJECT_ENV["MARKET_CONFIG_PATH"])
    )

from game_theory_agent.api import CONFIG
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.experiments.four_agent_acceptance import run as run_episode
from game_theory_agent.gameplay import _value_breakdown
from game_theory_agent.market import CompanyAction, MarketEnv, MarketState
from game_theory_agent.orchestration import JsonlRoundEventLogger


LLM_COMPANY = "company_A"
CONDITIONS: dict[str, dict[str, str]] = {
    "A_persona_only": {
        "belief_mode": "off",
        "opponent_model_mode": "off",
        "utility_inference_mode": "off",
        "advisor_mode": "off",
    },
    "B_action_belief": {
        "belief_mode": "public_action_v1",
        "opponent_model_mode": "off",
        "utility_inference_mode": "off",
        "advisor_mode": "off",
    },
    "C_opponent_model": {
        "belief_mode": "public_action_v1",
        "opponent_model_mode": "public_strategy_v1",
        "utility_inference_mode": "off",
        "advisor_mode": "off",
    },
    "D_utility_advisor": {
        "belief_mode": "public_action_v1",
        "opponent_model_mode": "public_strategy_v1",
        "utility_inference_mode": "strategy_utility_v1",
        "advisor_mode": "bayesian_strategy_v2",
    },
}
ACTION_FIELDS = (
    "price_cents",
    "advertising_budget_cents",
    "service_budget_cents",
    "capacity_investment_cents",
    "resilience_budget_cents",
)


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("provide one or more unique comma-separated Seeds")
    return seeds


def build_plan(seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    names = tuple(CONDITIONS)
    plan: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        shift = index % len(names)
        order = names[shift:] + names[:shift]
        for call_order, condition in enumerate(order, start=1):
            plan.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "condition_call_order": call_order,
                    **CONDITIONS[condition],
                }
            )
    return plan


def _company(state: MarketState, company_id: str = LLM_COMPANY):
    return state.company(company_id)


def _outcome(state: MarketState) -> dict[str, int]:
    company = _company(state)
    value = _value_breakdown(company, CONFIG)
    return {
        "round_profit_cents": company.financial.round_profit_cents,
        "cash_balance_cents": company.financial.cash_balance_cents,
        "market_share_ppm": company.commercial.market_share_ppm,
        "resilience_ppm": company.risk.resilience_ppm,
        "round_incident_cost_cents": (
            company.financial.round_incident_cost_cents
        ),
        "enterprise_value_cents": value["enterprise_value_cents"],
    }


def _simulate(
    state_before: MarketState,
    joint_action: dict[str, dict[str, Any]],
    replacements: dict[str, dict[str, Any]],
    step_id: str,
) -> tuple[MarketState, dict[str, dict[str, Any]]]:
    actions: dict[str, CompanyAction] = {}
    resolved: dict[str, dict[str, Any]] = {}
    for company_id in state_before.company_ids:
        raw = dict(joint_action[company_id])
        raw.update(replacements.get(company_id, {}))
        resolution = resolve_action_request(
            CONFIG,
            state_before,
            company_id,
            raw,
            source="stage51-counterfactual",
            action_id=(
                f"stage51:{state_before.episode_id}:{state_before.round}:"
                f"{company_id}:{len(replacements)}"
            ),
        )
        actions[company_id] = resolution.action
        resolved[company_id] = resolution.action.to_dict()
    env = MarketEnv(CONFIG)
    env.load_state(state_before)
    result = env.step(step_id, actions)
    return result.state_after, resolved


def evaluate_round_counterfactual(event: Any) -> dict[str, Any]:
    """Exact one-step market counterfactuals over bounded price actions."""

    before = MarketState.from_dict(event.state_before)
    after = MarketState.from_dict(event.state_after)
    actual = _outcome(after)
    actual_action = dict(event.joint_action[LLM_COMPANY])
    bounds = CONFIG.mapping("action", "bounds", "price_cents")
    current_price = before.company(LLM_COMPANY).commercial.price_cents
    advice = next(
        trace for trace in event.traces if trace.company_id == LLM_COMPANY
    ).advisor_output
    advice_prices = (
        [int(item["price_cents"]) for item in advice["candidate_actions"]]
        if isinstance(advice, dict) and advice.get("candidate_actions")
        else []
    )
    own_prices = sorted(
        {
            int(bounds["min"]),
            max(int(bounds["min"]), current_price - 500),
            current_price,
            min(int(bounds["max"]), current_price + 500),
            int(bounds["max"]),
            *advice_prices,
        }
    )
    own_rows = []
    for price in own_prices:
        state, resolved = _simulate(
            before,
            event.joint_action,
            {LLM_COMPANY: {"price_cents": price}},
            event.step_result["step_id"],
        )
        own_rows.append(
            {
                "requested_price_cents": price,
                "resolved_price_cents": resolved[LLM_COMPANY]["price_cents"],
                **_outcome(state),
            }
        )
    best = max(
        own_rows,
        key=lambda row: (
            row["round_profit_cents"],
            row["enterprise_value_cents"],
            row["market_share_ppm"],
        ),
    )
    advisor_row = None
    if isinstance(advice, dict):
        recommended = int(advice["recommended_price_cents"])
        advisor_row = next(
            row for row in own_rows if row["requested_price_cents"] == recommended
        )

    adversarial_rows = []
    for opponent in before.company_ids:
        if opponent == LLM_COMPANY:
            continue
        opponent_price = before.company(opponent).commercial.price_cents
        for price in sorted(
            {
                int(bounds["min"]),
                max(int(bounds["min"]), opponent_price - 500),
                opponent_price,
                min(int(bounds["max"]), opponent_price + 500),
                int(bounds["max"]),
            }
        ):
            state, resolved = _simulate(
                before,
                event.joint_action,
                {opponent: {"price_cents": price}},
                event.step_result["step_id"],
            )
            adversarial_rows.append(
                {
                    "opponent_company_id": opponent,
                    "requested_price_cents": price,
                    "resolved_price_cents": resolved[opponent]["price_cents"],
                    **_outcome(state),
                }
            )
    worst = min(
        adversarial_rows,
        key=lambda row: (
            row["round_profit_cents"],
            row["enterprise_value_cents"],
        ),
    )
    return {
        "counterfactual_schema_version": "stage51-market-cf-v1.0.0",
        "method": "same_state_same_seed_fixed_other_actions_price_grid",
        "state_before_hash": before.state_hash,
        "actual_state_after_hash": after.state_hash,
        "actual_action": actual_action,
        "actual_outcome": actual,
        "best_response_proxy": best,
        "regret_profit_cents": max(
            0, best["round_profit_cents"] - actual["round_profit_cents"]
        ),
        "counterfactual_gain_enterprise_value_cents": (
            best["enterprise_value_cents"] - actual["enterprise_value_cents"]
        ),
        "advisor_counterfactual": advisor_row,
        "advisor_profit_delta_vs_actual_cents": (
            advisor_row["round_profit_cents"] - actual["round_profit_cents"]
            if advisor_row is not None
            else None
        ),
        "bounded_price_exploitability_proxy_cents": max(
            0, actual["round_profit_cents"] - worst["round_profit_cents"]
        ),
        "worst_opponent_price_response": worst,
        "own_price_candidate_count": len(own_rows),
        "opponent_price_response_count": len(adversarial_rows),
    }


def _token_usage(events: list[Any]) -> dict[str, Any]:
    traces = [
        trace
        for event in events
        for trace in event.traces
        if trace.company_id == LLM_COMPANY and trace.agent_type == "model"
    ]
    input_tokens = sum(int(trace.input_tokens or 0) for trace in traces)
    output_tokens = sum(int(trace.output_tokens or 0) for trace in traces)
    return {
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


def _reference_counts(events: list[Any]) -> dict[str, int]:
    texts = [
        json.dumps(
            next(
                trace for trace in event.traces
                if trace.company_id == LLM_COMPANY
            ).planner_output,
            ensure_ascii=False,
        ).lower()
        for event in events
    ]
    terms = {
        "belief": ("belief", "信念", "概率", "价格方向"),
        "opponent_model": (
            "opponent model", "对手模型", "策略分布", "价格进攻性",
            "增长型", "利润型", "防御型", "合作型",
            "growth player", "profit player", "defensive player",
            "cooperative player",
        ),
        "utility": ("utility", "效用", "权重"),
        "advisor": (
            "advisor", "博弈建议", "预期效用代理", "best response",
            "最佳响应",
        ),
    }
    return {
        key: sum(any(term in text for term in patterns) for text in texts)
        for key, patterns in terms.items()
    }


def summarize_episode(
    directory: Path, condition: str, seed: int
) -> dict[str, Any]:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    events = list(
        JsonlRoundEventLogger(directory / "round-events.jsonl").read_all()
    )
    counterfactuals = [evaluate_round_counterfactual(event) for event in events]
    states = [MarketState.from_dict(event.state_after) for event in events]
    final_state = states[-1]
    final_company = _company(final_state)
    traces = [
        next(trace for trace in event.traces if trace.company_id == LLM_COMPANY)
        for event in events
    ]
    requested_actions = [dict(trace.requested_action or {}) for trace in traces]
    final_actions = [dict(trace.final_action) for trace in traces]
    prior_prices = [
        MarketState.from_dict(event.state_before)
        .company(LLM_COMPANY)
        .commercial.price_cents
        for event in events
    ]
    row = {
        "episode_schema_version": "stage51-real-episode-v1.0.0",
        "seed": seed,
        "condition": condition,
        "treatment": CONDITIONS[condition],
        "provider": summary["provider"],
        "model": summary["model"],
        "persona": summary["persona"],
        "rounds": len(events),
        "passed": summary["passed"],
        "protocol_checks": summary["protocol_checks"],
        "llm_fallback_count": sum(
            trace.decision_status != "submitted" for trace in traces
        ),
        "market_total_profit_cents": summary["research_metrics"][
            "market_total_profit_cents"
        ],
        "llm_total_profit_cents": sum(
            state.company(LLM_COMPANY).financial.round_profit_cents
            for state in states
        ),
        "final_enterprise_value_cents": _value_breakdown(
            final_company, CONFIG
        )["enterprise_value_cents"],
        "final_market_share_ppm": final_company.commercial.market_share_ppm,
        "final_resilience_ppm": final_company.risk.resilience_ppm,
        "total_risk_loss_cents": sum(
            state.company(LLM_COMPANY).financial.round_incident_cost_cents
            for state in states
        ),
        "price_cut_rounds": sum(
            int(action["price_cents"]) < prior
            for action, prior in zip(final_actions, prior_prices, strict=True)
        ),
        "mean_requested_action": {
            field: mean(float(action.get(field) or 0) for action in requested_actions)
            for field in ACTION_FIELDS
        },
        "total_regret_profit_cents": sum(
            item["regret_profit_cents"] for item in counterfactuals
        ),
        "mean_regret_profit_cents": mean(
            item["regret_profit_cents"] for item in counterfactuals
        ),
        "mean_bounded_price_exploitability_proxy_cents": mean(
            item["bounded_price_exploitability_proxy_cents"]
            for item in counterfactuals
        ),
        "advisor_counterfactual_profit_delta_cents": sum(
            int(item["advisor_profit_delta_vs_actual_cents"] or 0)
            for item in counterfactuals
        ),
        "strategic_reference_counts": _reference_counts(events),
        "token_usage": _token_usage(events),
        "counterfactuals": counterfactuals,
        "artifacts": {
            "round_events": str((directory / "round-events.jsonl").resolve()),
            "base_summary": str((directory / "summary.json").resolve()),
        },
    }
    row.update(
        {
            f"mean_requested_{field}": row["mean_requested_action"][field]
            for field in ACTION_FIELDS
        }
    )
    (directory / "stage51-summary.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return row


def _exact_sign_test(differences: list[float]) -> dict[str, Any]:
    nonzero = [value for value in differences if value != 0]
    positives = sum(value > 0 for value in nonzero)
    n = len(nonzero)
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(positives, n - positives) + 1))
        p_value = min(1.0, 2 * tail / (2**n))
    return {
        "nonzero_pair_count": n,
        "positive_pair_count": positives,
        "two_sided_exact_sign_p_value": p_value,
    }


def aggregate(rows: list[dict[str, Any]], seeds: tuple[int, ...]) -> dict[str, Any]:
    metrics = (
        "llm_total_profit_cents",
        "final_enterprise_value_cents",
        "final_market_share_ppm",
        "total_risk_loss_cents",
        "mean_regret_profit_cents",
        "mean_bounded_price_exploitability_proxy_cents",
        "mean_requested_price_cents",
        "mean_requested_advertising_budget_cents",
        "mean_requested_service_budget_cents",
        "mean_requested_capacity_investment_cents",
        "mean_requested_resilience_budget_cents",
    )
    by_seed = {
        seed: {row["condition"]: row for row in rows if row["seed"] == seed}
        for seed in seeds
    }
    contrasts = {
        "B_belief_vs_A_persona": ("B_action_belief", "A_persona_only"),
        "C_opponent_vs_A_persona": ("C_opponent_model", "A_persona_only"),
        "C_opponent_vs_B_action_belief": (
            "C_opponent_model", "B_action_belief"
        ),
        "D_advisor_vs_C_opponent": (
            "D_utility_advisor", "C_opponent_model"
        ),
        "D_advisor_vs_A_persona": (
            "D_utility_advisor", "A_persona_only"
        ),
    }
    paired: dict[str, Any] = {}
    for label, (treatment, control) in contrasts.items():
        differences = {
            metric: [
                float(by_seed[seed][treatment][metric])
                - float(by_seed[seed][control][metric])
                for seed in seeds
            ]
            for metric in metrics
        }
        paired[label] = {
            "treatment": treatment,
            "control": control,
            "metrics": {
                metric: {
                    "deltas": values,
                    "mean_delta": mean(values),
                    "median_delta": median(values),
                    "sign_test": _exact_sign_test(values),
                }
                for metric, values in differences.items()
            },
        }
    tokens = {
        key: sum(int(row["token_usage"][key]) for row in rows)
        for key in (
            "llm_call_count", "input_tokens", "output_tokens", "total_tokens",
            "calls_without_token_usage",
        )
    }
    return {
        "experiment_schema_version": "stage51-real-game-theory-v1.0.0",
        "primary_experiment_unit": "paired_seed",
        "seeds": list(seeds),
        "conditions": CONDITIONS,
        "episode_count": len(rows),
        "all_episodes_passed": all(row["passed"] for row in rows),
        "all_replays_passed": all(
            all(row["protocol_checks"].get(key, False) for key in (
                "replay_match_100pct",
                "interaction_replay_match_100pct",
                "information_replay_match_100pct",
                "belief_replay_match_100pct",
                "game_theory_replay_match_100pct",
            ))
            for row in rows
        ),
        "paired_contrasts": paired,
        "token_usage": tokens,
        "episodes": rows,
        "research_boundary": (
            "Paired real-model evidence. Regret is an exact one-step bounded "
            "price-grid counterfactual; exploitability is a bounded one-opponent "
            "price-response proxy, not formal game-theoretic exploitability."
        ),
    }


def _episode_args(args: argparse.Namespace, item: dict[str, Any], output: Path):
    return SimpleNamespace(
        episode_id=f"stage51-{item['seed']}-{item['condition']}",
        seed=item["seed"],
        rounds=args.rounds,
        market_model=args.market_model,
        information_mode="public",
        provider=args.provider,
        belief_mode=item["belief_mode"],
        opponent_model_mode=item["opponent_model_mode"],
        utility_inference_mode=item["utility_inference_mode"],
        advisor_mode=item["advisor_mode"],
        repeated_game_mode="off",
        cooperation_mode="off",
        honor_game_theory_advice=False,
        model=args.model,
        persona=args.persona,
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


async def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    seeds = parse_seeds(args.seeds)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = build_plan(seeds)
    (output / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for item in plan:
        directory = output / f"seed-{item['seed']}" / item["condition"]
        stage_summary = directory / "stage51-summary.json"
        if stage_summary.exists() and not args.recompute:
            if not args.resume:
                raise FileExistsError(
                    f"existing result requires --resume: {stage_summary}"
                )
            rows.append(json.loads(stage_summary.read_text(encoding="utf-8")))
            continue
        base_summary = directory / "summary.json"
        if not base_summary.exists():
            exit_code = await run_episode(
                _episode_args(args, item, directory)
            )
            if exit_code != 0:
                raise RuntimeError(
                    f"episode failed: seed={item['seed']} condition={item['condition']}"
                )
        rows.append(summarize_episode(directory, item["condition"], item["seed"]))
        partial = {
            "experiment_schema_version": "stage51-real-partial-v1.0.0",
            "completed_episode_count": len(rows),
            "planned_episode_count": len(plan),
            "completed": [
                {"seed": row["seed"], "condition": row["condition"]}
                for row in rows
            ],
            "token_usage": {
                key: sum(int(row["token_usage"][key]) for row in rows)
                for key in (
                    "llm_call_count", "input_tokens", "output_tokens",
                    "total_tokens", "calls_without_token_usage",
                )
            },
        }
        (output / "partial-summary.json").write_text(
            json.dumps(partial, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = aggregate(rows, seeds)
    summary.update(
        {
            "provider": args.provider,
            "model": args.model,
            "persona": args.persona,
            "rounds": args.rounds,
            "market_model": args.market_model,
            "passed": summary["all_episodes_passed"]
            and summary["all_replays_passed"]
            and summary["token_usage"]["calls_without_token_usage"] == 0,
        }
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("doubao", "deepseek", "mock"), default="doubao")
    parser.add_argument("--model")
    parser.add_argument("--persona", default="balanced_v1")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--rounds", type=int, choices=(5, 10, 15, 20), default=5)
    parser.add_argument("--market-model", default="balanced")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="recompute evaluation summaries from saved RoundEvents without model calls",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = asyncio.run(run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
