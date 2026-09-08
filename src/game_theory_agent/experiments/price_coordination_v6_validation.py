"""Paired cartel, betrayal and regulatory-enforcement validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from game_theory_agent.market import CompanyAction, MarketEnv, MarketState
from game_theory_agent.market import load_market_config
from game_theory_agent.market.protocols import sha256_hash, state_hash
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition
from game_theory_agent.market.replay import verify_replay


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v6_final.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "strategic-market-v6-stage74"
SEEDS = tuple(range(74001, 74021))
CONDITIONS = ("competition", "cartel_honor", "cartel_undercut")


def _rehash(state: MarketState) -> MarketState:
    cleared = replace(state, state_hash="")
    return replace(cleared, state_hash=state_hash(cleared.to_dict()))


def _frozen_state(config, seed: int) -> tuple[MarketEnv, MarketState]:
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B"),
        episode_id=f"stage74-{seed}",
        episode_seed=seed,
        max_rounds=5,
        cooperation_modes=("price_coordination_v1",),
    )
    companies = tuple(
        replace(
            company,
            operations=replace(
                company.operations,
                base_capacity_orders=20_000,
                effective_capacity_orders=20_000,
            ),
        )
        for company in state.companies
    )
    state = _rehash(replace(state, companies=companies))
    env.load_state(state)
    return env, state


def _actions(state: MarketState, condition: str) -> dict[str, CompanyAction]:
    prices = {
        "competition": {"company_A": 10_000, "company_B": 10_000},
        "cartel_honor": {"company_A": 12_000, "company_B": 12_000},
        "cartel_undercut": {"company_A": 10_000, "company_B": 12_000},
    }[condition]
    actions = {
        company_id: CompanyAction(
            action_id=f"fixed:{state.episode_id}:{state.round}:{company_id}",
            episode_id=state.episode_id,
            agent_id=company_id,
            round=state.round,
            state_version=state.state_version,
            price_cents=prices[company_id],
            strategy_summary="frozen price-coordination action tape",
        )
        for company_id in state.company_ids
    }
    if condition != "competition":
        actions["company_A"] = replace(
            actions["company_A"],
            price_coordination_partner_company_id="company_B",
            price_coordination_target_cents=12_000,
        )
        actions["company_B"] = replace(
            actions["company_B"],
            price_coordination_partner_company_id="company_A",
            price_coordination_target_cents=12_000,
        )
    return actions


def _run_condition(config, seed: int, condition: str) -> dict[str, object]:
    env, state = _frozen_state(config, seed)
    manifest = EpisodeManifest.create(
        env, state, experiment_id="stage74-price-coordination"
    )
    actions = _actions(state, condition)
    result = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions
    )
    after = result.state_after
    replayed = verify_replay(
        MarketEnv(config),
        manifest,
        (MarketTransition.create(state, actions, result),),
    )[-1]
    outcome = (
        after.strategic_market.last_price_coordination_outcomes[0]
        if after.strategic_market.last_price_coordination_outcomes
        else None
    )
    fines = sum(
        int(company.financial.round_regulatory_fine_cents or 0)
        for company in after.companies
    )
    joint_profit = sum(
        company.financial.round_profit_cents for company in after.companies
    )
    return {
        "seed": seed,
        "condition": condition,
        "average_paid_price_cents": after.market.average_paid_price_cents,
        "no_purchase_orders": after.market.no_purchase_orders,
        "consumer_surplus_proxy_cents": (
            after.strategic_market.consumer_surplus_proxy_cents
        ),
        "joint_profit_cents": joint_profit,
        "gross_joint_profit_before_fine_cents": joint_profit + fines,
        "regulatory_fine_cents": fines,
        "regulatory_pressure_ppm": (
            after.strategic_market.regulatory_pressure_ppm
        ),
        "company_a_share_ppm": (
            after.company("company_A").commercial.market_share_ppm
        ),
        "company_b_share_ppm": (
            after.company("company_B").commercial.market_share_ppm
        ),
        "company_a_profit_cents": (
            after.company("company_A").financial.round_profit_cents
        ),
        "company_b_profit_cents": (
            after.company("company_B").financial.round_profit_cents
        ),
        "company_a_credibility_ppm": dict(
            after.strategic_market.coordination_credibility_by_company_ppm
        )["company_A"],
        "company_b_credibility_ppm": dict(
            after.strategic_market.coordination_credibility_by_company_ppm
        )["company_B"],
        "coordination_status": outcome.status.value if outcome else None,
        "detected": outcome.detected if outcome else False,
        "detection_probability_ppm": (
            outcome.detection_probability_ppm if outcome else 0
        ),
        "replay_passed": replayed.state_hash == after.state_hash,
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
        competition = indexed[(seed, "competition")]
        honor = indexed[(seed, "cartel_honor")]
        undercut = indexed[(seed, "cartel_undercut")]
        paired.append(
            {
                "seed": seed,
                "cartel_price_increase_cents": (
                    int(honor["average_paid_price_cents"])
                    - int(competition["average_paid_price_cents"])
                ),
                "cartel_no_purchase_increase_orders": (
                    int(honor["no_purchase_orders"])
                    - int(competition["no_purchase_orders"])
                ),
                "cartel_consumer_surplus_loss_cents": (
                    int(competition["consumer_surplus_proxy_cents"])
                    - int(honor["consumer_surplus_proxy_cents"])
                ),
                "cartel_gross_profit_gain_cents": (
                    int(honor["gross_joint_profit_before_fine_cents"])
                    - int(competition["gross_joint_profit_before_fine_cents"])
                ),
                "cartel_net_profit_gain_cents": (
                    int(honor["joint_profit_cents"])
                    - int(competition["joint_profit_cents"])
                ),
                "undercutter_share_advantage_ppm": (
                    int(undercut["company_a_share_ppm"])
                    - int(undercut["company_b_share_ppm"])
                ),
                "undercutter_profit_advantage_cents": (
                    int(undercut["company_a_profit_cents"])
                    - int(undercut["company_b_profit_cents"])
                ),
                "betrayal_credibility_loss_ppm": (
                    int(honor["company_a_credibility_ppm"])
                    - int(undercut["company_a_credibility_ppm"])
                ),
            }
        )
    detected_count = sum(
        bool(indexed[(seed, "cartel_honor")]["detected"])
        for seed in SEEDS
    )
    gates = {
        "honored_status_20_of_20": all(
            indexed[(seed, "cartel_honor")]["coordination_status"]
            == "honored"
            for seed in SEEDS
        ),
        "undercut_status_20_of_20": all(
            indexed[(seed, "cartel_undercut")]["coordination_status"]
            == "undercut_by_a"
            for seed in SEEDS
        ),
        "cartel_raises_price_20_of_20": all(
            row["cartel_price_increase_cents"] > 0 for row in paired
        ),
        "cartel_does_not_raise_consumer_surplus_20_of_20": all(
            row["cartel_consumer_surplus_loss_cents"] >= 0 for row in paired
        ),
        "cartel_reduces_consumer_surplus_on_average": _mean(
            row["cartel_consumer_surplus_loss_cents"] for row in paired
        )
        > 0,
        "cartel_increases_no_purchase_20_of_20": all(
            row["cartel_no_purchase_increase_orders"] > 0 for row in paired
        ),
        "cartel_gross_profit_rises_20_of_20": all(
            row["cartel_gross_profit_gain_cents"] > 0 for row in paired
        ),
        "undercutter_gains_share_20_of_20": all(
            row["undercutter_share_advantage_ppm"] > 0 for row in paired
        ),
        "undercutter_loses_credibility_20_of_20": all(
            row["betrayal_credibility_loss_ppm"] > 0 for row in paired
        ),
        "regulatory_detection_is_stochastic": 0 < detected_count < len(SEEDS),
        "detected_cases_pay_fines": all(
            int(indexed[(seed, "cartel_honor")]["regulatory_fine_cents"])
            > 0
            for seed in SEEDS
            if indexed[(seed, "cartel_honor")]["detected"]
        ),
        "replay_60_of_60": all(row["replay_passed"] for row in rows),
    }
    summary: dict[str, object] = {
        "result_schema_version": "strategic-market-v6-stage74-result-v1.0.0",
        "config_id": config.config_id,
        "config_sha256": config.config_sha256,
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "provider_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "detected_honored_cartels": detected_count,
        "gates": gates,
        "aggregate": {
            key: _mean(row[key] for row in paired)
            for key in (
                "cartel_price_increase_cents",
                "cartel_no_purchase_increase_orders",
                "cartel_consumer_surplus_loss_cents",
                "cartel_gross_profit_gain_cents",
                "cartel_net_profit_gain_cents",
                "undercutter_share_advantage_ppm",
                "undercutter_profit_advantage_cents",
                "betrayal_credibility_loss_ppm",
            )
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
                "detected_honored_cartels": summary[
                    "detected_honored_cartels"
                ],
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
