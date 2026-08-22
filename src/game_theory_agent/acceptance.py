"""Repeatable P0/P1/P2 acceptance evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.economics import decision_support_metrics
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import MarketConfig, MarketEnv, load_market_config


COMPANIES = ("company_A", "company_B", "company_C", "company_D")
MARKET_MODELS = (
    "balanced",
    "value_oriented",
    "quality_oriented",
    "service_oriented",
    "random",
)


def _rng(*parts: object) -> random.Random:
    payload = "|".join(str(item) for item in parts).encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(seed)


def _random_request(config: MarketConfig, state, company_id: str) -> dict[str, Any]:
    rng = _rng(state.episode_seed, state.round, company_id, "acceptance-random")
    bounds = config.mapping("action", "bounds")

    def draw(field: str) -> int:
        return rng.randint(int(bounds[field]["min"]), int(bounds[field]["max"]))

    return {
        "price_cents": draw("price_cents"),
        "advertising_budget_cents": draw("advertising_budget_cents"),
        "service_budget_cents": draw("service_budget_cents"),
        "capacity_investment_cents": (
            draw("capacity_investment_cents") if state.rounds_remaining > 1 else 0
        ),
        "resilience_budget_cents": (
            draw("resilience_budget_cents") if state.rounds_remaining > 1 else 0
        ),
        "incident_response": {"mode": "wait", "repair_budget_cents": 0},
        "strategy_summary": "acceptance reproducible random request",
    }


def replay_recorded_tape(
    config: MarketConfig,
    tape_path: Path,
    *,
    episode_seed: int,
    market_model: str = "balanced",
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in tape_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    env = MarketEnv(config)
    state = env.reset(
        COMPANIES,
        episode_id="acceptance-recorded-tape",
        episode_seed=episode_seed,
        market_model=market_model,
        max_rounds=len(records),
    )
    adjustments: Counter[str] = Counter()
    invariants_passed = True
    negative_margin_executions = 0
    liquidity_reserve_violations = 0
    for record in records:
        requests = {
            "company_A": record["agent_requested_action"],
            **record["random_requested_actions"],
        }
        actions = {}
        for company_id in state.company_ids:
            support = decision_support_metrics(config, state, company_id)
            decision = resolve_action_request(
                config,
                state,
                company_id,
                requests[company_id],
                source="acceptance-recorded-tape",
                action_id=(
                    f"acceptance-recorded-tape:{state.round}:{company_id}"
                ),
            )
            actions[company_id] = decision.action
            adjustments.update(item.reason_code for item in decision.adjustments)
            if decision.action.price_cents < int(
                support["minimum_safe_price_cents"]
            ):
                negative_margin_executions += 1
            if decision.action.fixed_spend_cents > int(
                support["maximum_discretionary_budget_cents"]
            ):
                liquidity_reserve_violations += 1
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", actions
        )
        invariants_passed &= result.invariant_results == ("all_passed",)
        state = result.state_after
    values = dict(state.terminal_enterprise_values_cents)
    ranking = sorted(state.company_ids, key=values.get, reverse=True)
    company = state.company("company_A")
    return {
        "method": "recorded requests with current controller policy",
        "episode_seed": episode_seed,
        "rounds": len(records),
        "terminal_rank": ranking.index("company_A") + 1,
        "terminal_enterprise_value_cents": values["company_A"],
        "terminal_cash_cents": company.financial.cash_balance_cents,
        "cumulative_profit_cents": company.financial.cumulative_profit_cents,
        "negative_margin_executions": negative_margin_executions,
        "liquidity_reserve_violations": liquidity_reserve_violations,
        "adjustments": dict(adjustments),
        "all_invariants_passed": invariants_passed,
    }


def evaluate_safety_matrix(
    config: MarketConfig,
    *,
    seed_start: int = 0,
    seed_count: int = 200,
    rounds: int = 20,
) -> dict[str, Any]:
    if not 1 <= seed_count <= 500:
        raise ValueError("seed_count must be in [1, 500]")
    total_episodes = 0
    completed = 0
    invariant_failures = 0
    negative_margin_executions = 0
    liquidity_reserve_violations = 0
    recovery_price_cuts = 0
    ranks: Counter[int] = Counter()
    terminal_values: list[int] = []
    per_market: dict[str, dict[str, Any]] = {}
    for market_model in MARKET_MODELS:
        market_ranks: Counter[int] = Counter()
        market_values: list[int] = []
        for seed in range(seed_start, seed_start + seed_count):
            total_episodes += 1
            episode_id = f"acceptance-{market_model}-{seed}"
            env = MarketEnv(config)
            state = env.reset(
                COMPANIES,
                episode_id=episode_id,
                episode_seed=seed,
                market_model=market_model,
                max_rounds=rounds,
            )
            while not state.terminal:
                actions = {}
                for company_id in state.company_ids:
                    support = decision_support_metrics(config, state, company_id)
                    raw = (
                        build_rule_action(config, state, company_id).to_dict()
                        if company_id == "company_A"
                        else _random_request(config, state, company_id)
                    )
                    decision = resolve_action_request(
                        config,
                        state,
                        company_id,
                        raw,
                        source="acceptance-matrix",
                        action_id=f"{episode_id}:{state.round}:{company_id}",
                    )
                    actions[company_id] = decision.action
                    if decision.action.price_cents < int(
                        support["minimum_safe_price_cents"]
                    ):
                        negative_margin_executions += 1
                    if decision.action.fixed_spend_cents > int(
                        support["maximum_discretionary_budget_cents"]
                    ):
                        liquidity_reserve_violations += 1
                    if (
                        support["strategic_phase"] != "growth"
                        and decision.action.price_cents
                        < state.company(company_id).commercial.price_cents
                    ):
                        recovery_price_cuts += 1
                result = env.step(
                    f"{episode_id}:{state.round}:{state.state_version}", actions
                )
                if result.invariant_results != ("all_passed",):
                    invariant_failures += 1
                state = result.state_after
            completed += 1
            values = dict(state.terminal_enterprise_values_cents)
            ranking = sorted(state.company_ids, key=values.get, reverse=True)
            rank = ranking.index("company_A") + 1
            ranks[rank] += 1
            market_ranks[rank] += 1
            terminal_values.append(values["company_A"])
            market_values.append(values["company_A"])
        per_market[market_model] = {
            "episodes": seed_count,
            "average_rank": sum(
                rank * count for rank, count in market_ranks.items()
            )
            / seed_count,
            "last_place_ratio": market_ranks[4] / seed_count,
            "average_terminal_value_cents": round(mean(market_values)),
        }
    average_rank = sum(rank * count for rank, count in ranks.items()) / max(
        1, total_episodes
    )
    checks = {
        "episode_completion_100pct": completed == total_episodes,
        "market_invariants_100pct": invariant_failures == 0,
        "negative_margin_executions_zero": negative_margin_executions == 0,
        "liquidity_reserve_violations_zero": liquidity_reserve_violations == 0,
        "recovery_price_cuts_zero": recovery_price_cuts == 0,
        "average_rank_at_most_2_5": average_rank <= 2.5,
        "last_place_ratio_at_most_35pct": ranks[4] / total_episodes <= 0.35,
    }
    return {
        "method": "company_A adaptive rule vs reproducible random requests",
        "seed_start": seed_start,
        "seed_count_per_market": seed_count,
        "rounds": rounds,
        "market_models": list(MARKET_MODELS),
        "total_episodes": total_episodes,
        "completed_episodes": completed,
        "invariant_failures": invariant_failures,
        "negative_margin_executions": negative_margin_executions,
        "liquidity_reserve_violations": liquidity_reserve_violations,
        "recovery_price_cuts": recovery_price_cuts,
        "rank_counts": {str(rank): ranks[rank] for rank in range(1, 5)},
        "average_rank": average_rank,
        "last_place_ratio": ranks[4] / total_episodes,
        "average_terminal_value_cents": round(mean(terminal_values)),
        "per_market": per_market,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run P0/P1/P2 acceptance checks")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recorded-tape", type=Path)
    parser.add_argument("--episode-seed", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = load_market_config(root / "configs" / "market_v4.yaml")
    report: dict[str, Any] = {
        "safety_matrix": evaluate_safety_matrix(
            config,
            seed_start=args.seed_start,
            seed_count=args.seed_count,
            rounds=args.rounds,
        )
    }
    if args.recorded_tape is not None:
        if args.episode_seed is None:
            parser.error("--episode-seed is required with --recorded-tape")
        report["recorded_tape"] = replay_recorded_tape(
            config,
            args.recorded_tape,
            episode_seed=args.episode_seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["safety_matrix"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
