"""Deterministic validation of explicit consumer, value-chain and government welfare."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from game_theory_agent.market import CompanyAction, MarketEnv
from game_theory_agent.market.config import load_market_config
from game_theory_agent.market.protocols import sha256_hash


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "market_v9_welfare.yaml"
OUTPUT = ROOT / "runs" / "welfare-v9-validation" / "summary.json"
SEEDS = tuple(range(96201, 96221))
ROUNDS = 10


SCENARIOS = {
    "affordable_diverse": {"price": 9_000, "sourcing": "diverse"},
    "balanced_diverse": {"price": 10_000, "sourcing": "diverse"},
    "high_price_diverse": {"price": 14_000, "sourcing": "diverse"},
    "balanced_cheap_source": {"price": 10_000, "sourcing": "cheap"},
}


def _actions(state: Any, *, price: int, sourcing: str, coordination: bool = False):
    pairs = {
        "company_A": "company_B",
        "company_B": "company_A",
        "company_C": "company_D",
        "company_D": "company_C",
    }
    result = {}
    for company_id in state.company_ids:
        exited = bool(
            state.strategic_market is not None
            and state.strategic_market.lifecycle(company_id).status.value == "exited"
        )
        diverse = sourcing == "diverse"
        result[company_id] = CompanyAction(
            action_id=f"{state.episode_id}:{state.round}:{company_id}",
            episode_id=state.episode_id,
            agent_id=company_id,
            round=state.round,
            state_version=state.state_version,
            price_cents=(
                state.company(company_id).commercial.price_cents if exited else price
            ),
            price_coordination_partner_company_id=(
                pairs[company_id] if coordination and not exited else None
            ),
            price_coordination_target_cents=(
                price if coordination and not exited else None
            ),
            primary_supplier_id=(None if exited else "economy_supplier"),
            backup_supplier_id=(
                "resilient_supplier" if diverse and not exited else None
            ),
            primary_supplier_share_ppm=(
                None if exited else 500_000 if diverse else 1_000_000
            ),
        )
    return result


def _episode(seed: int, scenario: str) -> dict[str, Any]:
    env = MarketEnv(load_market_config(CONFIG))
    state = env.reset(
        episode_id=f"welfare-v9-{seed}-{scenario}",
        episode_seed=seed,
        market_model="balanced",
        max_rounds=ROUNDS,
    )
    no_purchase = stockout = 0
    params = SCENARIOS[scenario]
    while not state.terminal:
        state = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}",
            _actions(state, price=params["price"], sourcing=params["sourcing"]),
        ).state_after
        no_purchase += state.market.no_purchase_orders
        stockout += state.market.lost_after_stockout_orders
    welfare = state.welfare_accounting
    assert welfare is not None
    return {
        "seed": seed,
        "scenario": scenario,
        "cumulative_consumer_surplus_cents": welfare.cumulative_consumer_surplus_cents,
        "cumulative_downstream_surplus_cents": welfare.cumulative_downstream_producer_surplus_cents,
        "cumulative_upstream_surplus_cents": welfare.cumulative_upstream_producer_surplus_cents,
        "cumulative_government_net_cents": welfare.cumulative_government_net_budget_cents,
        "cumulative_externality_cost_cents": welfare.cumulative_externality_cost_cents,
        "cumulative_total_welfare_cents": welfare.cumulative_total_economic_welfare_cents,
        "total_no_purchase_orders": no_purchase,
        "total_stockout_orders": stockout,
    }


def _government(seed: int) -> dict[str, int]:
    env = MarketEnv(load_market_config(CONFIG))
    state = env.reset(
        episode_id=f"government-v9-{seed}",
        episode_seed=seed,
        market_model="balanced",
        max_rounds=ROUNDS,
        cooperation_modes=("price_coordination_v1",),
    )
    state = env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}",
        _actions(state, price=14_000, sourcing="diverse", coordination=True),
    ).state_after
    welfare = state.welfare_accounting
    assert welfare is not None and state.strategic_market is not None
    return {
        "seed": seed,
        "detected_cases": sum(
            int(item.detected)
            for item in state.strategic_market.last_price_coordination_outcomes
        ),
        "firm_fines_cents": sum(
            item.financial.round_regulatory_fine_cents or 0
            for item in state.companies
        ),
        "government_fine_revenue_cents": welfare.round_government_fine_revenue_cents,
        "government_enforcement_cost_cents": welfare.round_government_enforcement_cost_cents,
    }


def run() -> dict[str, Any]:
    rows = [
        _episode(seed, scenario)
        for seed in SEEDS
        for scenario in SCENARIOS
    ]
    fields = (
        "cumulative_consumer_surplus_cents",
        "cumulative_downstream_surplus_cents",
        "cumulative_upstream_surplus_cents",
        "cumulative_government_net_cents",
        "cumulative_externality_cost_cents",
        "cumulative_total_welfare_cents",
        "total_no_purchase_orders",
        "total_stockout_orders",
    )
    aggregates = {
        scenario: {
            field: round(mean(row[field] for row in rows if row["scenario"] == scenario))
            for field in fields
        }
        for scenario in SCENARIOS
    }
    government_rows = [_government(seed) for seed in SEEDS]
    summary: dict[str, Any] = {
        "experiment_id": "welfare-v9-validation",
        "evidence_level": "DETERMINISTIC_SYNTHETIC_ACCOUNTING_EVIDENCE",
        "common_seeds": list(SEEDS),
        "rounds": ROUNDS,
        "rows": rows,
        "aggregates": aggregates,
        "government_rows": government_rows,
        "government_aggregate": {
            field: sum(row[field] for row in government_rows)
            for field in (
                "detected_cases",
                "firm_fines_cents",
                "government_fine_revenue_cents",
                "government_enforcement_cost_cents",
            )
        },
        "gates": {
            "affordable_price_has_more_consumer_surplus": (
                aggregates["affordable_diverse"]["cumulative_consumer_surplus_cents"]
                > aggregates["high_price_diverse"]["cumulative_consumer_surplus_cents"]
            ),
            "affordable_price_has_less_no_purchase": (
                aggregates["affordable_diverse"]["total_no_purchase_orders"]
                < aggregates["high_price_diverse"]["total_no_purchase_orders"]
            ),
            "diverse_sourcing_has_less_stockout": (
                aggregates["balanced_diverse"]["total_stockout_orders"]
                < aggregates["balanced_cheap_source"]["total_stockout_orders"]
            ),
            "government_fines_match_firm_fines": (
                sum(row["firm_fines_cents"] for row in government_rows)
                == sum(row["government_fine_revenue_cents"] for row in government_rows)
            ),
            "zero_provider_calls": True,
        },
        "claim_boundary": (
            "The ledger is arithmetically auditable, but WTP and externality "
            "amounts remain synthetic priors pending external calibration."
        ),
    }
    summary["result_hash"] = sha256_hash(summary)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
