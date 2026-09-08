"""Paired multi-seed validation for bilateral emergency fulfilment."""

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
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "strategic-market-v6-stage73"
SEEDS = tuple(range(73001, 73021))
CONDITIONS = (
    "matched_mutual_aid",
    "unilateral_request",
    "unilateral_offer",
    "no_mutual_aid",
)


def _rehash(state: MarketState) -> MarketState:
    cleared = replace(state, state_hash="")
    return replace(cleared, state_hash=state_hash(cleared.to_dict()))


def _frozen_state(config, seed: int) -> tuple[MarketEnv, MarketState]:
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B"),
        episode_id=f"stage73-{seed}",
        episode_seed=seed,
        max_rounds=5,
        cooperation_modes=("mutual_aid_v1",),
    )
    receiver = state.company("company_A")
    donor = state.company("company_B")
    receiver = replace(
        receiver,
        operations=replace(
            receiver.operations,
            base_capacity_orders=300,
            effective_capacity_orders=300,
        ),
        brand=replace(
            receiver.brand,
            brand_awareness_ppm=850_000,
            service_quality_ppm=850_000,
            reputation_ppm=850_000,
        ),
    )
    donor = replace(
        donor,
        operations=replace(
            donor.operations,
            base_capacity_orders=10_000,
            effective_capacity_orders=10_000,
        ),
        brand=replace(
            donor.brand,
            brand_awareness_ppm=250_000,
            service_quality_ppm=250_000,
            reputation_ppm=250_000,
        ),
    )
    state = _rehash(replace(state, companies=(receiver, donor)))
    env.load_state(state)
    return env, state


def _actions(state: MarketState, condition: str) -> dict[str, CompanyAction]:
    result = {
        company_id: CompanyAction(
            action_id=f"fixed:{state.episode_id}:{state.round}:{company_id}",
            episode_id=state.episode_id,
            agent_id=company_id,
            round=state.round,
            state_version=state.state_version,
            price_cents=10_000,
            mutual_aid_capacity_offer_orders=0,
            mutual_aid_capacity_request_orders=0,
            strategy_summary="frozen mutual-aid action tape",
        )
        for company_id in state.company_ids
    }
    if condition in {"matched_mutual_aid", "unilateral_request"}:
        result["company_A"] = replace(
            result["company_A"],
            mutual_aid_partner_company_id="company_B",
            mutual_aid_capacity_request_orders=2_000,
        )
    if condition in {"matched_mutual_aid", "unilateral_offer"}:
        result["company_B"] = replace(
            result["company_B"],
            mutual_aid_partner_company_id="company_A",
            mutual_aid_capacity_offer_orders=2_000,
        )
    return result


def _run_condition(config, seed: int, condition: str) -> dict[str, object]:
    env, state = _frozen_state(config, seed)
    initial = state
    manifest = EpisodeManifest.create(
        env, initial, experiment_id="stage73-mutual-aid"
    )
    actions = _actions(state, condition)
    result = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions
    )
    after = result.state_after
    transition = MarketTransition.create(state, actions, result)
    replayed = verify_replay(MarketEnv(config), manifest, (transition,))[-1]
    receiver = after.company("company_A")
    donor = after.company("company_B")
    return {
        "seed": seed,
        "condition": condition,
        "transfer_orders": sum(
            item.fulfilled_orders
            for item in after.strategic_market.last_mutual_aid_transfers
        ),
        "receiver_sales_orders": receiver.commercial.sales_orders,
        "receiver_profit_cents": receiver.financial.round_profit_cents,
        "receiver_reputation_ppm": receiver.brand.reputation_ppm,
        "donor_profit_cents": donor.financial.round_profit_cents,
        "joint_profit_cents": sum(
            company.financial.round_profit_cents
            for company in after.companies
        ),
        "lost_after_stockout_orders": after.market.lost_after_stockout_orders,
        "social_welfare_proxy_cents": (
            after.strategic_market.social_welfare_proxy_cents
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
        matched = indexed[(seed, "matched_mutual_aid")]
        baseline = indexed[(seed, "no_mutual_aid")]
        paired.append(
            {
                "seed": seed,
                "receiver_sales_gain_orders": (
                    int(matched["receiver_sales_orders"])
                    - int(baseline["receiver_sales_orders"])
                ),
                "receiver_profit_gain_cents": (
                    int(matched["receiver_profit_cents"])
                    - int(baseline["receiver_profit_cents"])
                ),
                "receiver_reputation_gain_ppm": (
                    int(matched["receiver_reputation_ppm"])
                    - int(baseline["receiver_reputation_ppm"])
                ),
                "donor_profit_gain_cents": (
                    int(matched["donor_profit_cents"])
                    - int(baseline["donor_profit_cents"])
                ),
                "joint_profit_gain_cents": (
                    int(matched["joint_profit_cents"])
                    - int(baseline["joint_profit_cents"])
                ),
                "stockout_reduction_orders": (
                    int(baseline["lost_after_stockout_orders"])
                    - int(matched["lost_after_stockout_orders"])
                ),
                "social_welfare_gain_cents": (
                    int(matched["social_welfare_proxy_cents"])
                    - int(baseline["social_welfare_proxy_cents"])
                ),
            }
        )
    gates = {
        "matched_transfer_20_of_20": all(
            indexed[(seed, "matched_mutual_aid")]["transfer_orders"] > 0
            for seed in SEEDS
        ),
        "unilateral_request_zero_transfer_20_of_20": all(
            indexed[(seed, "unilateral_request")]["transfer_orders"] == 0
            for seed in SEEDS
        ),
        "unilateral_offer_zero_transfer_20_of_20": all(
            indexed[(seed, "unilateral_offer")]["transfer_orders"] == 0
            for seed in SEEDS
        ),
        "receiver_sales_improve_20_of_20": all(
            row["receiver_sales_gain_orders"] > 0 for row in paired
        ),
        "receiver_profit_improves_20_of_20": all(
            row["receiver_profit_gain_cents"] > 0 for row in paired
        ),
        "receiver_reputation_improves_20_of_20": all(
            row["receiver_reputation_gain_ppm"] > 0 for row in paired
        ),
        "stockout_falls_20_of_20": all(
            row["stockout_reduction_orders"] > 0 for row in paired
        ),
        "joint_profit_improves_20_of_20": all(
            row["joint_profit_gain_cents"] > 0 for row in paired
        ),
        "social_welfare_improves_20_of_20": all(
            row["social_welfare_gain_cents"] > 0 for row in paired
        ),
        "replay_80_of_80": all(row["replay_passed"] for row in rows),
    }
    summary: dict[str, object] = {
        "result_schema_version": "strategic-market-v6-stage73-result-v1.0.0",
        "config_id": config.config_id,
        "config_sha256": config.config_sha256,
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "provider_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "gates": gates,
        "aggregate": {
            key: _mean(row[key] for row in paired)
            for key in (
                "receiver_sales_gain_orders",
                "receiver_profit_gain_cents",
                "receiver_reputation_gain_ppm",
                "donor_profit_gain_cents",
                "joint_profit_gain_cents",
                "stockout_reduction_orders",
                "social_welfare_gain_cents",
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
