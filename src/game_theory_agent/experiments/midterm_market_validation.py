"""Zero-token market, consumer, and native cooperation-incentive validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from game_theory_agent.gameplay import build_rule_action, build_terminal_rankings
from game_theory_agent.market import CompanyAction, MarketEnv, MarketEvent, load_market_config
from game_theory_agent.market.protocols import sha256_hash, state_hash


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v5_cooperation.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "midterm-market-validation-v1"
COMPANIES = ("company_A", "company_B", "company_C", "company_D")
MARKET_MODELS = (
    "balanced",
    "value_oriented",
    "quality_oriented",
    "service_oriented",
)
SEEDS = tuple(range(20))


def _action(
    state: Any,
    company_id: str,
    *,
    price_cents: int = 10_000,
    advertising_budget_cents: int = 0,
    service_budget_cents: int = 0,
    capacity_investment_cents: int = 0,
    resilience_budget_cents: int = 0,
    contribution_cents: int = 0,
) -> CompanyAction:
    return CompanyAction(
        action_id=(
            f"{state.episode_id}:{state.round}:{company_id}:"
            f"{price_cents}:{advertising_budget_cents}:{service_budget_cents}:"
            f"{capacity_investment_cents}:{resilience_budget_cents}:"
            f"{contribution_cents}"
        ),
        episode_id=state.episode_id,
        agent_id=company_id,
        round=state.round,
        state_version=state.state_version,
        price_cents=price_cents,
        advertising_budget_cents=advertising_budget_cents,
        service_budget_cents=service_budget_cents,
        capacity_investment_cents=capacity_investment_cents,
        resilience_budget_cents=resilience_budget_cents,
        shared_resilience_contribution_cents=contribution_cents,
    )


def _joint(state: Any, focal: Mapping[str, int] | None = None) -> dict[str, CompanyAction]:
    focal = dict(focal or {})
    return {
        company_id: _action(
            state,
            company_id,
            **(focal if company_id == "company_A" else {}),
        )
        for company_id in state.company_ids
    }


def _paired_state(config: Any, state: Any, focal: Mapping[str, int]) -> Any:
    env = MarketEnv(config)
    env.load_state(state)
    return env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}",
        _joint(state, focal),
    ).state_after


def _accounting_ok(state: Any, config: Any) -> bool:
    fixed_overhead = config.integer("operating_costs", "fixed_overhead_cents")
    fulfillment = config.integer(
        "operating_costs", "fulfillment_cost_per_order_cents"
    )
    for company in state.companies:
        expected_operating = fixed_overhead + company.commercial.sales_orders * fulfillment
        expected_profit = (
            company.financial.round_revenue_cents
            - company.financial.round_variable_cost_cents
            - company.financial.round_fixed_spend_cents
            - company.financial.round_incident_cost_cents
            - company.financial.round_operating_cost_cents
        )
        if (
            company.financial.round_operating_cost_cents != expected_operating
            or company.financial.round_profit_cents != expected_profit
            or company.financial.cash_balance_cents < 0
        ):
            return False
    return True


def validate_market_and_consumers(config: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for market_model in MARKET_MODELS:
            env = MarketEnv(config)
            state = env.reset(
                COMPANIES,
                episode_id=f"market-validation-{seed}",
                episode_seed=seed,
                market_model=market_model,
                max_rounds=10,
                cooperation_mode="shared_resilience_v1",
            )
            baseline = _paired_state(config, state, {})
            low_price = _paired_state(config, state, {"price_cents": 9_000})
            high_price = _paired_state(config, state, {"price_cents": 11_000})
            advertising = _paired_state(
                config, state, {"advertising_budget_cents": 1_000_000}
            )
            service = _paired_state(
                config, state, {"service_budget_cents": 1_000_000}
            )
            capacity = _paired_state(
                config, state, {"capacity_investment_cents": 1_000_000}
            )
            resilience = _paired_state(
                config, state, {"resilience_budget_cents": 1_000_000}
            )
            demand_closure = (
                baseline.market.no_purchase_orders
                + baseline.market.lost_after_stockout_orders
                + sum(item.commercial.sales_orders for item in baseline.companies)
                == baseline.market.realized_demand_orders
            )
            rows.append(
                {
                    "seed": seed,
                    "market_model": market_model,
                    "lower_price_increases_potential_demand": (
                        low_price.company("company_A").commercial.potential_demand_orders
                        > baseline.company("company_A").commercial.potential_demand_orders
                    ),
                    "higher_price_decreases_potential_demand": (
                        high_price.company("company_A").commercial.potential_demand_orders
                        < baseline.company("company_A").commercial.potential_demand_orders
                    ),
                    "advertising_increases_awareness": (
                        advertising.company("company_A").brand.brand_awareness_ppm
                        > baseline.company("company_A").brand.brand_awareness_ppm
                    ),
                    "service_increases_quality": (
                        service.company("company_A").brand.service_quality_ppm
                        > baseline.company("company_A").brand.service_quality_ppm
                    ),
                    "capacity_is_next_round_asset": (
                        capacity.company("company_A").operations.base_capacity_orders
                        == baseline.company("company_A").operations.base_capacity_orders
                        + 100
                        and capacity.company(
                            "company_A"
                        ).operations.effective_capacity_orders
                        == baseline.company(
                            "company_A"
                        ).operations.effective_capacity_orders
                    ),
                    "resilience_increases_private_stock": (
                        resilience.company("company_A").risk.resilience_ppm
                        > baseline.company("company_A").risk.resilience_ppm
                    ),
                    "demand_closes_exactly": demand_closure,
                    "shares_sum_to_one": (
                        sum(
                            item.commercial.market_share_ppm
                            for item in baseline.companies
                        )
                        == 1_000_000
                    ),
                    "profit_and_cash_reconcile": _accounting_ok(baseline, config),
                }
            )

    dynamic_checks = 0
    same_direction_share_violations = 0
    replay_hashes: list[str] = []
    for seed in SEEDS:
        env = MarketEnv(config)
        state = env.reset(
            COMPANIES,
            episode_id=f"market-dynamic-{seed}",
            episode_seed=seed,
            market_model="balanced",
            max_rounds=10,
        )
        while not state.terminal:
            before_shares = {
                item.company_id: item.commercial.market_share_ppm
                for item in state.companies
            }
            joint = {
                company_id: build_rule_action(config, state, company_id)
                for company_id in state.company_ids
            }
            state = env.step(
                f"{state.episode_id}:{state.round}:{state.state_version}", joint
            ).state_after
            after_shares = {
                item.company_id: item.commercial.market_share_ppm
                for item in state.companies
            }
            changes = [after_shares[key] - before_shares[key] for key in COMPANIES]
            if all(value > 0 for value in changes) or all(value < 0 for value in changes):
                same_direction_share_violations += 1
            if (
                sum(after_shares.values()) == 1_000_000
                and _accounting_ok(state, config)
                and state.market.no_purchase_orders
                + state.market.lost_after_stockout_orders
                + sum(item.commercial.sales_orders for item in state.companies)
                == state.market.realized_demand_orders
            ):
                dynamic_checks += 1
        env.assert_invariants(state)
        replay_hashes.append(state.state_hash)

    check_names = [
        key for key in rows[0] if key not in {"seed", "market_model"}
    ]
    counts = {
        key: sum(bool(row[key]) for row in rows)
        for key in check_names
    }
    return {
        "cell_count": len(rows),
        "directional_pass_counts": counts,
        "all_paired_checks_passed": all(value == len(rows) for value in counts.values()),
        "dynamic_round_count": len(SEEDS) * 10,
        "dynamic_invariant_pass_count": dynamic_checks,
        "same_direction_share_violation_count": same_direction_share_violations,
        "terminal_state_hash_digest": sha256_hash(replay_hashes),
    }


def _forced_disaster(state: Any) -> Any:
    event = MarketEvent(
        event_id="midterm-forced-high-weather",
        event_type="extreme_weather",
        severity="high",
        started_round=state.round,
        remaining_rounds=3,
        demand_multiplier_ppm=820_000,
        supply_cost_multiplier_ppm=1_350_000,
        capacity_multiplier_ppm=600_000,
        advertising_multiplier_ppm=650_000,
        service_penalty_ppm=220_000,
        reputation_penalty_ppm=70_000,
    )
    changed = replace(state, active_market_events=(event,), state_hash="")
    return replace(changed, state_hash=state_hash(changed.to_dict()))


def _cooperation_path(
    config: Any,
    *,
    seed: int,
    focal_contribution_cents: int,
    other_contribution_cents: int,
    forced_disaster: bool,
) -> dict[str, int]:
    env = MarketEnv(config)
    state = env.reset(
        COMPANIES,
        episode_id=f"cooperation-roi-{seed}",
        episode_seed=seed,
        market_model="balanced",
        max_rounds=5,
        cooperation_mode="shared_resilience_v1",
    )
    first = {
        company_id: _action(
            state,
            company_id,
            contribution_cents=(
                focal_contribution_cents
                if company_id == "company_A"
                else other_contribution_cents
            ),
        )
        for company_id in COMPANIES
    }
    state = env.step(f"{state.episode_id}:1:0", first).state_after
    if forced_disaster:
        state = _forced_disaster(state)
        env.load_state(state)
    for _ in range(3):
        state = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}",
            _joint(state),
        ).state_after
    value = next(
        int(item["value_cents"])
        for item in build_terminal_rankings(state, config)["composite"]
        if item["company_id"] == "company_A"
    )
    company = state.company("company_A")
    return {
        "enterprise_value_cents": value,
        "cash_cents": company.financial.cash_balance_cents,
        "cumulative_profit_cents": company.financial.cumulative_profit_cents,
        "industry_resilience_ppm": state.shared_resilience.industry_resilience_ppm,
    }


def validate_native_cooperation_incentive(config: Any) -> dict[str, Any]:
    conditions = (
        ("normal_no_other_contributors", False, 0),
        ("normal_three_other_contributors", False, 1_000_000),
        ("severe_disaster_no_other_contributors", True, 0),
        ("severe_disaster_three_other_contributors", True, 1_000_000),
    )
    cells: dict[str, Any] = {}
    for name, disaster, others in conditions:
        deltas: list[int] = []
        for seed in SEEDS:
            contribute = _cooperation_path(
                config,
                seed=seed,
                focal_contribution_cents=1_000_000,
                other_contribution_cents=others,
                forced_disaster=disaster,
            )
            defect = _cooperation_path(
                config,
                seed=seed,
                focal_contribution_cents=0,
                other_contribution_cents=others,
                forced_disaster=disaster,
            )
            deltas.append(
                contribute["enterprise_value_cents"]
                - defect["enterprise_value_cents"]
            )
        cells[name] = {
            "seed_count": len(deltas),
            "positive_zero_negative": {
                "positive": sum(value > 0 for value in deltas),
                "zero": sum(value == 0 for value in deltas),
                "negative": sum(value < 0 for value in deltas),
            },
            "mean_contribute_minus_defect_ev_cents": sum(deltas) // len(deltas),
            "minimum_delta_cents": min(deltas),
            "maximum_delta_cents": max(deltas),
        }
    low = cells["normal_no_other_contributors"]
    high = cells["severe_disaster_no_other_contributors"]
    direction_gate = (
        low["positive_zero_negative"]["negative"] == len(SEEDS)
        and high["positive_zero_negative"]["positive"] == len(SEEDS)
    )
    return {
        "contribution_cents": 1_000_000,
        "future_rounds_evaluated": 3,
        "cells": cells,
        "economic_direction_separation_gate_passed": direction_gate,
        "real_llm_incentive_experiment_gate_open": direction_gate,
        "interpretation": (
            "The current native market does not provide one privately positive and one privately negative cooperation condition. Paid causal testing must remain closed."
            if not direction_gate
            else "Native economic conditions separate contribution incentives."
        ),
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    config = load_market_config(CONFIG_PATH)
    summary = {
        "schema_version": "midterm-market-validation-v1.0.0",
        "config_id": config.config_id,
        "config_sha256": config.config_sha256,
        "market_and_consumer_validation": validate_market_and_consumers(config),
        "native_cooperation_incentive": validate_native_cooperation_incentive(config),
        "real_model_calls": 0,
        "token_usage": 0,
        "estimated_cost_cny": 0,
        "report_hash": "pending",
    }
    summary["report_hash"] = sha256_hash(
        {key: value for key, value in summary.items() if key != "report_hash"}
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
