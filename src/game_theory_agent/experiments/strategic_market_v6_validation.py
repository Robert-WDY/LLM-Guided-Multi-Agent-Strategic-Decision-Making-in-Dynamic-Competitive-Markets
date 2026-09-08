"""Zero-token causal validation for Stage 7.1 strategic market mechanics."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import (
    CompanyLifecycleState,
    CompanyOperatingStatus,
    ConcentrationRegime,
    MarketEnv,
    MarketState,
    load_market_config,
)
from game_theory_agent.market.protocols import sha256_hash, state_hash
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition, verify_replay


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "market_v6_strategic.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "strategic-market-v6-stage71"
SEEDS = tuple(range(71001, 71021))


def _mean_int(values) -> int:
    items = tuple(int(value) for value in values)
    return sum(items) // len(items)


def _rehash(state: MarketState) -> MarketState:
    cleared = replace(state, state_hash="")
    return replace(cleared, state_hash=state_hash(cleared.to_dict()))


def _actions(config, state):
    return {
        company_id: build_rule_action(config, state, company_id)
        for company_id in state.company_ids
    }


def _settle(config, state, actions):
    env = MarketEnv(config)
    env.load_state(state)
    result = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions
    )
    return result, env


def _bankruptcy_case(config, seed: int) -> dict[str, object]:
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B"),
        episode_id=f"stage71-bankruptcy-{seed}",
        episode_seed=seed,
        max_rounds=5,
    )
    weak = state.company("company_A")
    weak = replace(
        weak,
        financial=replace(
            weak.financial,
            cash_balance_cents=200_000,
            capacity_book_value_cents=1_000_000,
        ),
        operations=replace(
            weak.operations,
            base_capacity_orders=0,
            effective_capacity_orders=0,
            financial_capacity_orders=0,
        ),
    )
    state = _rehash(replace(state, companies=(weak, state.company("company_B"))))
    env.load_state(state)
    manifest = EpisodeManifest.create(env, state, experiment_id="stage71-bankruptcy")
    actions_1 = _actions(config, state)
    result_1 = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions_1
    )
    after_exit = result_1.state_after
    actions_2 = _actions(config, after_exit)
    result_2 = env.step(
        (
            f"{after_exit.episode_id}:{after_exit.round}:"
            f"{after_exit.state_version}"
        ),
        actions_2,
    )
    transitions = (
        MarketTransition.create(state, actions_1, result_1),
        MarketTransition.create(after_exit, actions_2, result_2),
    )
    replayed = verify_replay(MarketEnv(config), manifest, transitions)[-1]
    final = result_2.state_after
    lifecycle = final.strategic_market.lifecycle("company_A")
    return {
        "seed": seed,
        "exit_status": lifecycle.status.value,
        "exit_round": lifecycle.exit_round,
        "concentration_regime": final.strategic_market.concentration_regime.value,
        "active_company_count": final.strategic_market.active_company_count,
        "exited_company_sales_next_round": final.company(
            "company_A"
        ).commercial.sales_orders,
        "exited_company_spend_next_round": actions_2["company_A"].fixed_spend_cents,
        "liquidated_book_value_cents": final.company(
            "company_A"
        ).financial.capacity_book_value_cents,
        "replay_passed": replayed.state_hash == final.state_hash,
    }


def _price_pressure_case(config, seed: int) -> dict[str, object]:
    base_env = MarketEnv(config)
    state = base_env.reset(
        episode_id=f"stage71-price-{seed}",
        episode_seed=seed,
        max_rounds=5,
    )
    costly = state.company("company_A")
    costly = replace(
        costly,
        operations=replace(costly.operations, base_unit_cost_cents=8_000),
    )
    state = _rehash(
        replace(
            state,
            companies=(costly, *state.companies[1:]),
        )
    )
    common = _actions(config, state)
    variants = {}
    for label, price in (("sustainable", 10_000), ("below_cost", 7_500)):
        actions = dict(common)
        actions["company_A"] = replace(
            actions["company_A"],
            action_id=f"{actions['company_A'].action_id}:{label}",
            price_cents=price,
            advertising_budget_cents=0,
            service_budget_cents=0,
            capacity_investment_cents=0,
            resilience_budget_cents=0,
        )
        result, _ = _settle(config, state, actions)
        after = result.state_after
        focal = after.company("company_A")
        variants[label] = {
            "share_ppm": focal.commercial.market_share_ppm,
            "profit_cents": focal.financial.round_profit_cents,
            "cash_cents": focal.financial.cash_balance_cents,
            "consumer_surplus_proxy_cents": (
                after.strategic_market.consumer_surplus_proxy_cents
            ),
            "regulatory_pressure_ppm": (
                after.strategic_market.regulatory_pressure_ppm
            ),
            "predatory_flag": "company_A"
            in after.strategic_market.predatory_pricing_company_ids,
        }
    return {
        "seed": seed,
        "sustainable": variants["sustainable"],
        "below_cost": variants["below_cost"],
        "share_delta_ppm": (
            variants["below_cost"]["share_ppm"]
            - variants["sustainable"]["share_ppm"]
        ),
        "profit_delta_cents": (
            variants["below_cost"]["profit_cents"]
            - variants["sustainable"]["profit_cents"]
        ),
    }


def _monopoly_price_case(config, seed: int) -> dict[str, object]:
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B"),
        episode_id=f"stage71-monopoly-{seed}",
        episode_seed=seed,
        max_rounds=5,
    )
    lifecycle = (
        CompanyLifecycleState(company_id="company_A"),
        CompanyLifecycleState(
            company_id="company_B",
            status=CompanyOperatingStatus.EXITED,
            exit_round=0,
            exit_reason="frozen_scenario",
        ),
    )
    strategic = replace(
        state.strategic_market,
        company_lifecycle=lifecycle,
        active_company_count=1,
        hhi_ppm=1_000_000,
        concentration_regime=ConcentrationRegime.MONOPOLY,
        dominant_company_id="company_A",
        dominant_share_ppm=1_000_000,
    )
    inactive = state.company("company_B")
    inactive = replace(
        inactive,
        commercial=replace(inactive.commercial, market_share_ppm=0),
    )
    active = replace(
        state.company("company_A"),
        commercial=replace(
            state.company("company_A").commercial, market_share_ppm=1_000_000
        ),
        operations=replace(
            state.company("company_A").operations,
            base_capacity_orders=20_000,
            effective_capacity_orders=20_000,
        ),
    )
    state = _rehash(
        replace(state, companies=(active, inactive), strategic_market=strategic)
    )
    common = _actions(config, state)
    variants = {}
    anchor_offer = max(7_500, state.market.price_anchor_cents - 500)
    markup_offer = min(18_000, state.market.price_anchor_cents + 2_500)
    for label, price in (("anchor", anchor_offer), ("markup", markup_offer)):
        actions = dict(common)
        actions["company_A"] = replace(
            actions["company_A"],
            action_id=f"{actions['company_A'].action_id}:{label}",
            price_cents=price,
            advertising_budget_cents=0,
            service_budget_cents=0,
            capacity_investment_cents=0,
            resilience_budget_cents=0,
        )
        result, _ = _settle(config, state, actions)
        after = result.state_after
        variants[label] = {
            "sales_orders": after.company("company_A").commercial.sales_orders,
            "profit_cents": after.company(
                "company_A"
            ).financial.round_profit_cents,
            "no_purchase_orders": after.market.no_purchase_orders,
            "consumer_surplus_proxy_cents": (
                after.strategic_market.consumer_surplus_proxy_cents
            ),
            "market_power_markup_ppm": (
                after.strategic_market.market_power_markup_ppm
            ),
            "regulatory_pressure_ppm": (
                after.strategic_market.regulatory_pressure_ppm
            ),
        }
    return {
        "seed": seed,
        "anchor": variants["anchor"],
        "markup": variants["markup"],
        "sales_delta": (
            variants["markup"]["sales_orders"]
            - variants["anchor"]["sales_orders"]
        ),
        "profit_delta_cents": (
            variants["markup"]["profit_cents"]
            - variants["anchor"]["profit_cents"]
        ),
    }


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    config = load_market_config(DEFAULT_CONFIG)
    bankruptcy = [_bankruptcy_case(config, seed) for seed in SEEDS]
    pressure = [_price_pressure_case(config, seed) for seed in SEEDS]
    monopoly = [_monopoly_price_case(config, seed) for seed in SEEDS]
    gates = {
        "bankruptcy_exit_20_of_20": all(
            row["exit_status"] == "exited" for row in bankruptcy
        ),
        "exit_is_absorbing_and_inert_20_of_20": all(
            row["exited_company_sales_next_round"] == 0
            and row["exited_company_spend_next_round"] == 0
            for row in bankruptcy
        ),
        "bankruptcy_creates_monopoly_20_of_20": all(
            row["concentration_regime"] == "monopoly" for row in bankruptcy
        ),
        "strategic_replay_20_of_20": all(
            row["replay_passed"] for row in bankruptcy
        ),
        "below_cost_gains_share_20_of_20": all(
            row["share_delta_ppm"] > 0 for row in pressure
        ),
        "below_cost_reduces_profit_20_of_20": all(
            row["profit_delta_cents"] < 0 for row in pressure
        ),
        "below_cost_is_audited_20_of_20": all(
            row["below_cost"]["predatory_flag"]
            and row["below_cost"]["regulatory_pressure_ppm"]
            > row["sustainable"]["regulatory_pressure_ppm"]
            for row in pressure
        ),
        "monopoly_markup_reduces_sales_20_of_20": all(
            row["sales_delta"] < 0 for row in monopoly
        ),
        "monopoly_markup_reduces_consumer_surplus_20_of_20": all(
            row["markup"]["consumer_surplus_proxy_cents"]
            < row["anchor"]["consumer_surplus_proxy_cents"]
            for row in monopoly
        ),
        "monopoly_markup_raises_pressure_20_of_20": all(
            row["markup"]["regulatory_pressure_ppm"]
            > row["anchor"]["regulatory_pressure_ppm"]
            for row in monopoly
        ),
    }
    summary: dict[str, object] = {
        "result_schema_version": "strategic-market-v6-stage71-result-v1.0.0",
        "config_id": config.config_id,
        "config_sha256": config.config_sha256,
        "seeds": list(SEEDS),
        "provider_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "gates": gates,
        "aggregate": {
            "mean_below_cost_share_delta_ppm": _mean_int(
                row["share_delta_ppm"] for row in pressure
            ),
            "mean_below_cost_profit_delta_cents": _mean_int(
                row["profit_delta_cents"] for row in pressure
            ),
            "mean_monopoly_markup_sales_delta": _mean_int(
                row["sales_delta"] for row in monopoly
            ),
            "mean_monopoly_markup_profit_delta_cents": _mean_int(
                row["profit_delta_cents"] for row in monopoly
            ),
        },
        "bankruptcy_rows": bankruptcy,
        "price_pressure_rows": pressure,
        "monopoly_rows": monopoly,
    }
    summary["all_gates_passed"] = all(gates.values())
    summary["result_hash"] = sha256_hash(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "summary.json"
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(target)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps({
        "all_gates_passed": summary["all_gates_passed"],
        "aggregate": summary["aggregate"],
        "result_hash": summary["result_hash"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
