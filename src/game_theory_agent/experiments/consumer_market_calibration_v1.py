"""Zero-token economic-shape calibration for legacy and v7 consumer choice."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.protocols import sha256_hash, state_hash


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "consumer-market-calibration-v1"
CONFIGS = {
    "legacy_v6": PROJECT_ROOT / "configs" / "market_v6_final.yaml",
    "consumer_v7": PROJECT_ROOT / "configs" / "market_v7_consumer_calibrated.yaml",
}
SEEDS = tuple(range(97001, 97021))
PRICES = (8000, 9000, 10000, 11000, 12000, 14000, 16000, 18000)
SPENDS = (0, 500_000, 1_000_000, 2_000_000, 5_000_000)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _result_hash(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("result_hash", None)
    return sha256_hash(normalized)


def _base(config: Any, seed: int, label: str) -> Any:
    state = MarketEnv(config).reset(
        episode_id=f"consumer-calibration-{seed}",
        episode_seed=seed,
        market_model="balanced",
        max_rounds=5,
        cooperation_mode="combined_v1",
    )
    companies = tuple(
        replace(
            company,
            operations=replace(
                company.operations,
                base_capacity_orders=50_000,
                effective_capacity_orders=50_000,
            ),
        )
        for company in state.companies
    )
    cleared = replace(state, companies=companies, state_hash="")
    return replace(cleared, state_hash=state_hash(cleared.to_dict()))


def _settle(
    config: Any,
    state: Any,
    *,
    focal_price: int = 10_000,
    all_price: int | None = None,
    focal_advertising: int = 0,
    focal_service: int = 0,
) -> Any:
    env = MarketEnv(config)
    env.load_state(state)
    actions = {}
    for company_id in state.company_ids:
        action = build_rule_action(config, state, company_id)
        actions[company_id] = replace(
            action,
            price_cents=(
                all_price
                if all_price is not None
                else focal_price
                if company_id == "company_A"
                else 10_000
            ),
            advertising_budget_cents=(
                focal_advertising if company_id == "company_A" else 0
            ),
            service_budget_cents=(
                focal_service if company_id == "company_A" else 0
            ),
            capacity_investment_cents=0,
            resilience_budget_cents=0,
            shared_resilience_contribution_cents=0,
            threshold_project_contribution_cents=0,
            mutual_aid_partner_company_id=None,
            mutual_aid_capacity_offer_orders=0,
            mutual_aid_capacity_request_orders=0,
            price_coordination_partner_company_id=None,
            price_coordination_target_cents=None,
        )
    return env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}", actions
    ).state_after


def _ppm_ratio(numerator: int, denominator: int) -> int:
    return 0 if denominator <= 0 else round(numerator * 1_000_000 / denominator)


def _config_result(label: str, path: Path) -> dict[str, Any]:
    config = load_market_config(path)
    own_price: dict[int, list[int]] = {price: [] for price in PRICES}
    all_price_outside: dict[int, list[int]] = {price: [] for price in PRICES}
    all_price_demand: dict[int, list[int]] = {price: [] for price in PRICES}
    service: dict[int, list[int]] = {spend: [] for spend in SPENDS}
    advertising: dict[int, list[int]] = {spend: [] for spend in SPENDS}
    closure_failures = 0
    for seed in SEEDS:
        state = _base(config, seed, label)
        for price in PRICES:
            own = _settle(config, state, focal_price=price)
            all_same = _settle(config, state, all_price=price)
            own_price[price].append(
                own.company("company_A").commercial.potential_demand_orders
            )
            all_price_outside[price].append(all_same.market.no_purchase_orders)
            all_price_demand[price].append(all_same.market.realized_demand_orders)
            closure = (
                all_same.market.no_purchase_orders
                + all_same.market.lost_after_stockout_orders
                + sum(item.commercial.sales_orders for item in all_same.companies)
            )
            closure_failures += closure != all_same.market.realized_demand_orders
        for spend in SPENDS:
            service_state = _settle(config, state, focal_service=spend)
            advertising_state = _settle(config, state, focal_advertising=spend)
            service[spend].append(
                service_state.company("company_A").commercial.potential_demand_orders
            )
            advertising[spend].append(
                advertising_state.company("company_A").commercial.potential_demand_orders
            )
    mean_own = {str(key): round(mean(value)) for key, value in own_price.items()}
    mean_outside = {
        str(key): round(mean(value)) for key, value in all_price_outside.items()
    }
    mean_outside_share = {
        str(key): _ppm_ratio(
            sum(all_price_outside[key]), sum(all_price_demand[key])
        )
        for key in PRICES
    }
    mean_service = {str(key): round(mean(value)) for key, value in service.items()}
    mean_advertising = {
        str(key): round(mean(value)) for key, value in advertising.items()
    }
    q9 = mean(own_price[9000])
    q11 = mean(own_price[11000])
    elasticity_abs_ppm = round(
        abs(((q11 - q9) / ((q11 + q9) / 2)) / 0.2) * 1_000_000
    )
    own_paths_monotone = sum(
        all(
            own_price[PRICES[index]][seed_index]
            >= own_price[PRICES[index + 1]][seed_index]
            for index in range(len(PRICES) - 1)
        )
        for seed_index in range(len(SEEDS))
    )
    outside_paths_monotone = sum(
        all(
            all_price_outside[PRICES[index]][seed_index]
            <= all_price_outside[PRICES[index + 1]][seed_index]
            for index in range(len(PRICES) - 1)
        )
        for seed_index in range(len(SEEDS))
    )
    return {
        "label": label,
        "config_id": config.config_id,
        "config_sha256": config.config_sha256,
        "mean_focal_demand_by_price": mean_own,
        "mean_all_price_outside_orders": mean_outside,
        "mean_all_price_outside_share_ppm": mean_outside_share,
        "mean_focal_demand_by_service_spend": mean_service,
        "mean_focal_demand_by_advertising_spend": mean_advertising,
        "own_price_arc_elasticity_abs_ppm_9000_11000": elasticity_abs_ppm,
        "own_price_monotone_seed_count": own_paths_monotone,
        "all_price_outside_monotone_seed_count": outside_paths_monotone,
        "high_price_16000_to_18000_outside_order_delta": (
            mean_outside["18000"] - mean_outside["16000"]
        ),
        "service_0_to_2m_demand_delta": mean_service["2000000"]
        - mean_service["0"],
        "advertising_0_to_2m_demand_delta": mean_advertising["2000000"]
        - mean_advertising["0"],
        "demand_closure_failure_count": closure_failures,
    }


def run(output: Path) -> dict[str, Any]:
    results = {
        label: _config_result(label, path) for label, path in CONFIGS.items()
    }
    v7 = results["consumer_v7"]
    target = load_market_config(CONFIGS["consumer_v7"]).mapping(
        "calibration_metadata", "target_moments"
    )
    gates = {
        "all_demand_closes": all(
            item["demand_closure_failure_count"] == 0 for item in results.values()
        ),
        "v7_own_price_monotone_all_seeds": v7["own_price_monotone_seed_count"]
        == len(SEEDS),
        "v7_outside_option_monotone_all_seeds": v7[
            "all_price_outside_monotone_seed_count"
        ]
        == len(SEEDS),
        "v7_high_price_plateau_removed": v7[
            "high_price_16000_to_18000_outside_order_delta"
        ]
        > 0,
        "v7_elasticity_inside_synthetic_target": int(
            target["balanced_market_own_price_arc_elasticity_absolute_min_ppm"]
        )
        <= int(v7["own_price_arc_elasticity_abs_ppm_9000_11000"])
        <= int(
            target["balanced_market_own_price_arc_elasticity_absolute_max_ppm"]
        ),
        "v7_18000_outside_share_meets_floor": int(
            v7["mean_all_price_outside_share_ppm"]["18000"]
        )
        >= int(target["all_firms_18000_no_purchase_share_min_ppm"]),
        "v7_service_direction_positive": v7["service_0_to_2m_demand_delta"] > 0,
        "v7_advertising_direction_positive": v7[
            "advertising_0_to_2m_demand_delta"
        ]
        > 0,
    }
    summary: dict[str, Any] = {
        "result_schema_version": "consumer-market-calibration-result-v1.0.0",
        "evidence_level": "DETERMINISTIC_SYNTHETIC_ECONOMIC_SHAPE_EVIDENCE",
        "seeds": list(SEEDS),
        "prices_cents": list(PRICES),
        "spends_cents": list(SPENDS),
        "calibration_claim": (
            "shape and internally declared synthetic targets only; not real demand calibration"
        ),
        "results": results,
        "gates": gates,
        "passed": all(gates.values()),
        "real_model_calls": 0,
        "tokens": 0,
    }
    summary["result_hash"] = _result_hash(summary)
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
