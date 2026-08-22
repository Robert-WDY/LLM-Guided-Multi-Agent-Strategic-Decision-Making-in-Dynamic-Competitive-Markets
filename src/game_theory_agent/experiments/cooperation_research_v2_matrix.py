"""Deterministic high-risk public-good matrix for Cooperation Research v2."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

from game_theory_agent.market import CompanyAction, MarketEnv, MarketEvent
from game_theory_agent.market.protocols import state_hash

from game_theory_agent.api import CONFIG


COMPANIES = ("company_A", "company_B", "company_C", "company_D")
CASES = ("all_cooperate", "one_free_rider", "all_defect")


def _forced_disaster(state: Any) -> Any:
    event = MarketEvent(
        event_id=f"v2-forced-disaster:{state.round}",
        event_type="supply_disruption",
        severity="high",
        started_round=state.round,
        remaining_rounds=1,
        demand_multiplier_ppm=900_000,
        supply_cost_multiplier_ppm=1_700_000,
        capacity_multiplier_ppm=500_000,
        advertising_multiplier_ppm=700_000,
        service_penalty_ppm=250_000,
        reputation_penalty_ppm=150_000,
    )
    changed = replace(state, active_market_events=(event,), state_hash="")
    return replace(changed, state_hash=state_hash(changed.to_dict()))


def _contributions(case: str, contribution_cents: int) -> dict[str, int]:
    if case == "all_cooperate":
        return {company_id: contribution_cents for company_id in COMPANIES}
    if case == "one_free_rider":
        return {
            company_id: (0 if company_id == "company_A" else contribution_cents)
            for company_id in COMPANIES
        }
    if case == "all_defect":
        return {company_id: 0 for company_id in COMPANIES}
    raise ValueError(f"unknown case: {case}")


def _actions(state: Any, contributions: dict[str, int]) -> dict[str, CompanyAction]:
    # A contribution only creates protection for the following round.  The
    # market contract therefore (correctly) rejects contributions in the
    # terminal round.
    effective = (
        {company_id: 0 for company_id in COMPANIES}
        if state.rounds_remaining <= 1
        else contributions
    )
    return {
        company_id: CompanyAction(
            action_id=f"v2:{state.episode_id}:{state.round}:{company_id}",
            episode_id=state.episode_id,
            agent_id=company_id,
            round=state.round,
            state_version=state.state_version,
            # Keep the controlled firms solvent under the forced cost shock;
            # otherwise a below-cost price would turn this into a bankruptcy
            # experiment before the repeated public-good effect can emerge.
            price_cents=15_000,
            shared_resilience_contribution_cents=effective[company_id],
        )
        for company_id in COMPANIES
    }


def run_case(
    *,
    seed: int,
    case: str,
    rounds: int,
    contribution_cents: int,
) -> dict[str, Any]:
    env = MarketEnv(CONFIG)
    state = env.reset(
        COMPANIES,
        episode_id=f"cooperation-v2-{seed}-{case}",
        episode_seed=seed,
        market_model="balanced",
        max_rounds=rounds,
        cooperation_mode="shared_resilience_v1",
    )
    cumulative_benefit = {company_id: 0 for company_id in COMPANIES}
    cumulative_free_rider_benefit = {
        company_id: 0 for company_id in COMPANIES
    }
    resilience_path: list[int] = []
    lost_orders = 0
    first_round_profits: dict[str, int] | None = None
    contributions = _contributions(case, contribution_cents)
    for _ in range(rounds):
        state = _forced_disaster(state)
        env.load_state(state)
        actions = _actions(state, contributions)
        shadow = env.counterfactual_without_public_resilience(state, actions)
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", actions
        )
        after = result.state_after
        if first_round_profits is None:
            first_round_profits = {
                company_id: after.company(
                    company_id
                ).financial.round_profit_cents
                for company_id in COMPANIES
            }
        source = dict(
            state.shared_resilience.last_contribution_by_company_cents
        )
        protection = int(
            CONFIG.mapping("shared_resilience")[
                "public_protection_weight_ppm"
            ]
        ) * state.shared_resilience.industry_resilience_ppm // 1_000_000
        for company_id in COMPANIES:
            benefit = max(
                0,
                after.company(company_id).financial.round_profit_cents
                - shadow.state_after.company(
                    company_id
                ).financial.round_profit_cents,
            )
            cumulative_benefit[company_id] += benefit
            if protection > 0 and int(source.get(company_id, 0)) == 0:
                cumulative_free_rider_benefit[company_id] += benefit
        resilience_path.append(
            after.shared_resilience.industry_resilience_ppm
        )
        lost_orders += after.market.lost_after_stockout_orders
        state = after
    assert first_round_profits is not None
    return {
        "seed": seed,
        "case": case,
        "rounds": rounds,
        "contribution_cents_per_company_round": contribution_cents,
        "first_round_profit_by_company_cents": first_round_profits,
        "cumulative_profit_by_company_cents": {
            company_id: state.company(
                company_id
            ).financial.cumulative_profit_cents
            for company_id in COMPANIES
        },
        "total_contribution_by_company_cents": {
            company_id: contributions[company_id] * max(0, rounds - 1)
            for company_id in COMPANIES
        },
        "cumulative_public_benefit_by_company_cents": cumulative_benefit,
        "cumulative_free_rider_benefit_by_company_cents": (
            cumulative_free_rider_benefit
        ),
        "industry_resilience_path_ppm": resilience_path,
        "final_industry_resilience_ppm": resilience_path[-1],
        "cumulative_lost_after_stockout_orders": lost_orders,
        "final_enterprise_value_by_company_cents": dict(
            state.terminal_enterprise_values_cents
        ),
    }


def run_matrix(
    *,
    seeds: tuple[int, ...],
    rounds: int,
    contribution_cents: int,
) -> dict[str, Any]:
    results = [
        run_case(
            seed=seed,
            case=case,
            rounds=rounds,
            contribution_cents=contribution_cents,
        )
        for seed in seeds
        for case in CASES
    ]
    indexed = {(row["seed"], row["case"]): row for row in results}
    paired = []
    for seed in seeds:
        cooperate = indexed[(seed, "all_cooperate")]
        free_ride = indexed[(seed, "one_free_rider")]
        defect = indexed[(seed, "all_defect")]
        contributor_mean_profit = sum(
            free_ride["cumulative_profit_by_company_cents"][company_id]
            for company_id in COMPANIES[1:]
        ) // 3
        paired.append(
            {
                "seed": seed,
                "free_rider_vs_contributor_profit_gap_cents": (
                    free_ride["cumulative_profit_by_company_cents"]["company_A"]
                    - contributor_mean_profit
                ),
                "free_rider_vs_own_all_cooperate_profit_delta_cents": (
                    free_ride["cumulative_profit_by_company_cents"]["company_A"]
                    - cooperate["cumulative_profit_by_company_cents"]["company_A"]
                ),
                "free_rider_public_benefit_cents": (
                    free_ride[
                        "cumulative_free_rider_benefit_by_company_cents"
                    ]["company_A"]
                ),
                "free_rider_resilience_delta_vs_all_cooperate_ppm": (
                    free_ride["final_industry_resilience_ppm"]
                    - cooperate["final_industry_resilience_ppm"]
                ),
                "all_defect_resilience_delta_vs_all_cooperate_ppm": (
                    defect["final_industry_resilience_ppm"]
                    - cooperate["final_industry_resilience_ppm"]
                ),
                "all_defect_lost_order_delta_vs_all_cooperate": (
                    defect["cumulative_lost_after_stockout_orders"]
                    - cooperate["cumulative_lost_after_stockout_orders"]
                ),
                "all_defect_total_profit_delta_vs_all_cooperate_cents": (
                    sum(defect["cumulative_profit_by_company_cents"].values())
                    - sum(
                        cooperate["cumulative_profit_by_company_cents"].values()
                    )
                ),
            }
        )
    checks = {
        "all_cases_all_seeds_completed": len(results) == len(seeds) * len(CASES),
        "free_rider_saves_first_round_private_cost": all(
            indexed[(seed, "one_free_rider")][
                "first_round_profit_by_company_cents"
            ]["company_A"]
            - indexed[(seed, "all_cooperate")][
                "first_round_profit_by_company_cents"
            ]["company_A"]
            == contribution_cents
            for seed in seeds
        ),
        "free_rider_receives_public_benefit": all(
            row["free_rider_public_benefit_cents"] > 0 for row in paired
        ),
        "one_free_rider_reduces_public_stock": all(
            row["free_rider_resilience_delta_vs_all_cooperate_ppm"] < 0
            for row in paired
        ),
        "all_defect_reduces_public_stock": all(
            row["all_defect_resilience_delta_vs_all_cooperate_ppm"] < 0
            for row in paired
        ),
        "all_defect_increases_stockout_loss": all(
            row["all_defect_lost_order_delta_vs_all_cooperate"] > 0
            for row in paired
        ),
        "all_defect_reduces_total_long_term_profit": all(
            row["all_defect_total_profit_delta_vs_all_cooperate_cents"] < 0
            for row in paired
        ),
    }
    mean_deltas = {
        key: mean(float(row[key]) for row in paired)
        for key in paired[0]
        if key != "seed"
    }
    return {
        "matrix_schema_version": "cooperation-research-v2-matrix-v1.0.0",
        "environment": "forced_high_supply_disruption_every_round",
        "controlled_price_cents": 15_000,
        "terminal_round_contribution_cents": 0,
        "seeds": list(seeds),
        "rounds": rounds,
        "cases": list(CASES),
        "checks": checks,
        "passed": all(checks.values()),
        "paired_results": paired,
        "paired_mean_deltas": mean_deltas,
        "results": results,
        "interpretation_boundary": (
            "A free public benefit is an engineering fact; whether the free "
            "rider has higher cumulative profit is a measured research result."
        ),
    }


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("provide unique comma-separated seeds")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="301,302,303,304,305")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--contribution-cents", type=int, default=500_000)
    parser.add_argument(
        "--output",
        default="runs/cooperation-research-v2-free-rider-matrix",
    )
    args = parser.parse_args()
    summary = run_matrix(
        seeds=_parse_seeds(args.seeds),
        rounds=args.rounds,
        contribution_cents=args.contribution_cents,
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
