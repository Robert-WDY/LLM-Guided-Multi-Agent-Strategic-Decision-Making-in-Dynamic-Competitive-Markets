"""Twenty-round integration gate for every Strategic Market v6 mechanism."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import CompanyAction, MarketEnv, MarketState
from game_theory_agent.market import load_market_config
from game_theory_agent.market.protocols import sha256_hash, state_hash
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition
from game_theory_agent.market.replay import verify_replay


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v6_final.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "strategic-market-v6-stage75"
SEEDS = tuple(range(75001, 75021))
CONDITIONS = ("combined_strategies", "inactive_controls")
FORBIDDEN_OPPONENT_FIELDS = {
    "financial",
    "operations",
    "risk",
    "persona",
    "cash_balance_cents",
    "actual_unit_cost_cents",
}


def _rehash(state: MarketState) -> MarketState:
    cleared = replace(state, state_hash="")
    return replace(cleared, state_hash=state_hash(cleared.to_dict()))


def _initial_state(config, seed: int) -> tuple[MarketEnv, MarketState]:
    env = MarketEnv(config)
    state = env.reset(
        episode_id=f"stage75-{seed}",
        episode_seed=seed,
        max_rounds=20,
        cooperation_mode="combined_v1",
    )
    rebuilt = []
    for company in state.companies:
        if company.company_id == "company_A":
            company = replace(
                company,
                operations=replace(
                    company.operations,
                    base_capacity_orders=500,
                    effective_capacity_orders=500,
                ),
                brand=replace(
                    company.brand,
                    brand_awareness_ppm=850_000,
                    service_quality_ppm=850_000,
                    reputation_ppm=850_000,
                ),
            )
        elif company.company_id == "company_C":
            company = replace(
                company,
                operations=replace(
                    company.operations,
                    base_capacity_orders=10_000,
                    effective_capacity_orders=10_000,
                ),
            )
        elif company.company_id == "company_D":
            company = replace(
                company,
                financial=replace(
                    company.financial,
                    cash_balance_cents=200_000,
                    capacity_book_value_cents=1_000_000,
                ),
                operations=replace(
                    company.operations,
                    base_unit_cost_cents=9_000,
                    actual_unit_cost_cents=9_000,
                ),
            )
        rebuilt.append(company)
    state = _rehash(replace(state, companies=tuple(rebuilt)))
    env.load_state(state)
    return env, state


def _active_action(
    state: MarketState, company_id: str, condition: str
) -> CompanyAction:
    round_number = state.round
    prices = {
        "company_A": 11_500,
        "company_B": 11_500,
        "company_C": 10_000,
        "company_D": 7_500,
    }
    if condition == "combined_strategies" and round_number == 2:
        prices["company_A"] = 10_000
    action = CompanyAction(
        action_id=f"fixed:{state.episode_id}:{round_number}:{company_id}",
        episode_id=state.episode_id,
        agent_id=company_id,
        round=round_number,
        state_version=state.state_version,
        price_cents=prices[company_id],
        shared_resilience_contribution_cents=0,
        threshold_project_contribution_cents=0,
        mutual_aid_capacity_offer_orders=0,
        mutual_aid_capacity_request_orders=0,
        strategy_summary="combined strategic market fixed policy",
    )
    if condition != "combined_strategies":
        return action
    if round_number == 1:
        project_contributions = {
            "company_A": 3_000_000,
            "company_B": 3_000_000,
            "company_C": 2_000_000,
            "company_D": 0,
        }
        action = replace(
            action,
            threshold_project_contribution_cents=(
                project_contributions[company_id]
            ),
        )
    if round_number <= 5 and state.rounds_remaining > 1:
        action = replace(
            action,
            shared_resilience_contribution_cents=(
                250_000 if company_id in {"company_A", "company_B", "company_C"} else 0
            ),
        )
    if company_id == "company_A":
        action = replace(
            action,
            mutual_aid_partner_company_id="company_C",
            mutual_aid_capacity_request_orders=1_500,
        )
    elif company_id == "company_C":
        action = replace(
            action,
            mutual_aid_partner_company_id="company_A",
            mutual_aid_capacity_offer_orders=1_500,
        )
    if round_number <= 2 and company_id in {"company_A", "company_B"}:
        action = replace(
            action,
            price_coordination_partner_company_id=(
                "company_B" if company_id == "company_A" else "company_A"
            ),
            price_coordination_target_cents=11_500,
        )
    return action


def _actions(config, state: MarketState, condition: str):
    active_ids = set(state.strategic_market.active_company_ids)
    return {
        company_id: (
            _active_action(state, company_id, condition)
            if company_id in active_ids
            else build_rule_action(config, state, company_id)
        )
        for company_id in state.company_ids
    }


def _information_audit(state: MarketState) -> tuple[int, bool]:
    builder = ObservationBuilder()
    views = {
        company_id: builder.build(state, company_id, "public")
        for company_id in state.company_ids
    }
    public_states = [view["public_state"] for view in views.values()]
    public_consistent = all(item == public_states[0] for item in public_states)
    leaks = 0
    for company_id, view in views.items():
        for competitor in view["competitors"]:
            leaks += len(FORBIDDEN_OPPONENT_FIELDS.intersection(competitor))
            if competitor["company_id"] == company_id:
                leaks += 1
        if view["private_state"]["company_id"] != company_id:
            leaks += 1
    return leaks, public_consistent


def _run_condition(config, seed: int, condition: str) -> dict[str, object]:
    env, state = _initial_state(config, seed)
    initial = state
    manifest = EpisodeManifest.create(
        env, initial, experiment_id="stage75-combined-market"
    )
    transitions = []
    total_mutual_aid_orders = 0
    coordination_statuses = []
    detected_count = 0
    closure_passed = True
    information_leaks = 0
    public_state_consistent = True
    for _ in range(20):
        leaks, public_consistent = _information_audit(state)
        information_leaks += leaks
        public_state_consistent = public_state_consistent and public_consistent
        actions = _actions(config, state, condition)
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}",
            actions,
        )
        transitions.append(MarketTransition.create(state, actions, result))
        state = result.state_after
        strategic = state.strategic_market
        total_mutual_aid_orders += sum(
            item.fulfilled_orders
            for item in strategic.last_mutual_aid_transfers
        )
        coordination_statuses.extend(
            item.status.value
            for item in strategic.last_price_coordination_outcomes
        )
        detected_count += sum(
            item.detected
            for item in strategic.last_price_coordination_outcomes
        )
        closure_passed = closure_passed and (
            state.market.no_purchase_orders
            + state.market.lost_after_stockout_orders
            + sum(
                company.commercial.sales_orders
                for company in state.companies
            )
            == state.market.realized_demand_orders
        )
    replayed = verify_replay(MarketEnv(config), manifest, tuple(transitions))[-1]
    project = state.strategic_market.threshold_project
    credibility = dict(
        state.strategic_market.coordination_credibility_by_company_ppm
    )
    return {
        "seed": seed,
        "condition": condition,
        "project_status": project.status.value,
        "company_d_status": state.strategic_market.lifecycle(
            "company_D"
        ).status.value,
        "total_mutual_aid_orders": total_mutual_aid_orders,
        "coordination_statuses": coordination_statuses,
        "detected_coordination_count": detected_count,
        "company_a_coordination_credibility_ppm": credibility["company_A"],
        "company_b_coordination_credibility_ppm": credibility["company_B"],
        "final_industry_resilience_ppm": (
            state.shared_resilience.industry_resilience_ppm
        ),
        "terminal_market_value_cents": sum(
            value for _, value in state.terminal_enterprise_values_cents
        ),
        "cumulative_social_welfare_proxy_cents": (
            state.strategic_market.cumulative_social_welfare_proxy_cents
        ),
        "closure_passed": closure_passed,
        "information_leaks": information_leaks,
        "public_state_consistent": public_state_consistent,
        "non_negative_cash": all(
            company.financial.cash_balance_cents >= 0
            for company in state.companies
        ),
        "replay_passed": replayed.state_hash == state.state_hash,
    }


def _mean(values) -> int:
    values = tuple(int(value) for value in values)
    return sum(values) // len(values)


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    config = load_market_config(CONFIG_PATH)
    rows = [
        _run_condition(config, seed, condition)
        for seed in SEEDS
        for condition in CONDITIONS
    ]
    indexed = {
        (int(row["seed"]), str(row["condition"])): row for row in rows
    }
    paired = []
    for seed in SEEDS:
        combined = indexed[(seed, "combined_strategies")]
        control = indexed[(seed, "inactive_controls")]
        paired.append(
            {
                "seed": seed,
                "resilience_gain_ppm": (
                    int(combined["final_industry_resilience_ppm"])
                    - int(control["final_industry_resilience_ppm"])
                ),
                "market_value_delta_cents": (
                    int(combined["terminal_market_value_cents"])
                    - int(control["terminal_market_value_cents"])
                ),
                "social_welfare_delta_cents": (
                    int(combined["cumulative_social_welfare_proxy_cents"])
                    - int(control["cumulative_social_welfare_proxy_cents"])
                ),
            }
        )
    combined_rows = [
        indexed[(seed, "combined_strategies")] for seed in SEEDS
    ]
    gates = {
        "twenty_round_replay_40_of_40": all(
            row["replay_passed"] for row in rows
        ),
        "demand_and_cash_invariants_40_of_40": all(
            row["closure_passed"] and row["non_negative_cash"]
            for row in rows
        ),
        "public_information_leak_zero": all(
            row["information_leaks"] == 0
            and row["public_state_consistent"]
            for row in rows
        ),
        "bankruptcy_exit_40_of_40": all(
            row["company_d_status"] == "exited" for row in rows
        ),
        "threshold_project_success_20_of_20": all(
            row["project_status"] == "succeeded" for row in combined_rows
        ),
        "inactive_project_fails_20_of_20": all(
            indexed[(seed, "inactive_controls")]["project_status"]
            == "failed"
            for seed in SEEDS
        ),
        "mutual_aid_executes_20_of_20": all(
            row["total_mutual_aid_orders"] > 0 for row in combined_rows
        ),
        "honor_and_betrayal_observed_20_of_20": all(
            "honored" in row["coordination_statuses"]
            and "undercut_by_a" in row["coordination_statuses"]
            for row in combined_rows
        ),
        "betrayer_credibility_lower_20_of_20": all(
            row["company_a_coordination_credibility_ppm"]
            < row["company_b_coordination_credibility_ppm"]
            for row in combined_rows
        ),
        "shared_resilience_higher_20_of_20": all(
            row["resilience_gain_ppm"] > 0 for row in paired
        ),
        "regulatory_detection_occurs": sum(
            int(row["detected_coordination_count"])
            for row in combined_rows
        )
        > 0,
    }
    summary: dict[str, object] = {
        "result_schema_version": "strategic-market-v6-stage75-result-v1.0.0",
        "config_id": config.config_id,
        "config_sha256": config.config_sha256,
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "rounds": 20,
        "provider_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "gates": gates,
        "aggregate": {
            "mean_combined_mutual_aid_orders": _mean(
                row["total_mutual_aid_orders"] for row in combined_rows
            ),
            "total_detected_coordination_events": sum(
                int(row["detected_coordination_count"])
                for row in combined_rows
            ),
            "mean_resilience_gain_ppm": _mean(
                row["resilience_gain_ppm"] for row in paired
            ),
            "mean_market_value_delta_cents": _mean(
                row["market_value_delta_cents"] for row in paired
            ),
            "mean_social_welfare_delta_cents": _mean(
                row["social_welfare_delta_cents"] for row in paired
            ),
        },
        "paired_rows": paired,
        "episode_rows": rows,
    }
    summary["all_gates_passed"] = all(gates.values())
    summary["result_hash"] = sha256_hash(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.output)
    print(
        json.dumps(
            {
                "all_gates_passed": summary["all_gates_passed"],
                "aggregate": summary["aggregate"],
                "failed_gates": [
                    key for key, value in summary["gates"].items() if not value
                ],
                "result_hash": summary["result_hash"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
