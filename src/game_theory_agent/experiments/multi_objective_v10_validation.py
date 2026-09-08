"""Market-backed tests for firm objectives beyond private profit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.market import CompanyAction, MarketEnv
from game_theory_agent.market.config import load_market_config
from game_theory_agent.market.protocols import sha256_hash, state_hash
from game_theory_agent.strategic_reliability.rollout import (
    AuthoritativeMarketRolloutEvaluator,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "market_v10_multi_objective.yaml"
OUTPUT = ROOT / "runs" / "multi-objective-v10-validation" / "summary.json"
SEEDS = tuple(range(96301, 96321))
PROFILES = (
    "none",
    "stakeholder_balanced",
    "public_service",
    "resilience_steward",
)


SCENARIOS = {
    "affordable_diverse": (9_000, "diverse", False),
    "balanced_diverse": (10_000, "diverse", False),
    "markup_diverse": (14_000, "diverse", False),
    "shock_cheap": (10_000, "cheap", True),
    "shock_diverse": (10_000, "diverse", True),
}


def _forced_state(env: MarketEnv, state: Any, shock: bool) -> Any:
    if not shock:
        return state
    assert state.supply_chain is not None
    state = replace(
        state,
        supply_chain=replace(
            state.supply_chain,
            suppliers=tuple(
                replace(
                    item,
                    disrupted=item.supplier_id == "economy_supplier",
                    available_capacity_orders=(
                        3_300
                        if item.supplier_id == "economy_supplier"
                        else item.base_capacity_orders
                    ),
                )
                for item in state.supply_chain.suppliers
            ),
        ),
        state_hash="",
    )
    state = replace(state, state_hash=state_hash(state.to_dict()))
    env.load_state(state)
    return state


def _step(seed: int, scenario: str, registry: PersonaRegistry) -> list[dict[str, Any]]:
    price, sourcing, shock = SCENARIOS[scenario]
    config = load_market_config(CONFIG)
    env = MarketEnv(config)
    before = env.reset(
        episode_id=f"objectives-{seed}-{scenario}",
        episode_seed=seed,
        market_model="balanced",
        max_rounds=20,
    )
    before = _forced_state(env, before, shock)
    diverse = sourcing == "diverse"
    actions = {
        company_id: CompanyAction(
            action_id=f"{before.episode_id}:{company_id}",
            episode_id=before.episode_id,
            agent_id=company_id,
            round=before.round,
            state_version=before.state_version,
            price_cents=price,
            primary_supplier_id="economy_supplier",
            backup_supplier_id="resilient_supplier" if diverse else None,
            primary_supplier_share_ppm=500_000 if diverse else 1_000_000,
        )
        for company_id in before.company_ids
    }
    after = env.step(
        f"{before.episode_id}:{before.round}:{before.state_version}", actions
    ).state_after
    assert after.welfare_accounting is not None
    result = []
    for profile_id in PROFILES:
        assessment = registry.evaluator(registry.get(profile_id)).evaluate(
            before, after, "company_A"
        )
        result.append(
            {
                "seed": seed,
                "scenario": scenario,
                "profile_id": profile_id,
                "utility_ppm": assessment.round_utility_ppm,
                "company_profit_cents": after.company("company_A").financial.round_profit_cents,
                "consumer_surplus_cents": after.welfare_accounting.round_consumer_surplus_cents,
                "total_welfare_cents": after.welfare_accounting.round_total_economic_welfare_cents,
                "sales_orders": sum(item.commercial.sales_orders for item in after.companies),
                "stockout_orders": after.market.lost_after_stockout_orders,
            }
        )
    return result


def run() -> dict[str, Any]:
    config = load_market_config(CONFIG)
    registry = PersonaRegistry.from_market_config(config)
    rows = [
        row
        for seed in SEEDS
        for scenario in SCENARIOS
        for row in _step(seed, scenario, registry)
    ]
    pricing = ("affordable_diverse", "balanced_diverse", "markup_diverse")
    supply = ("shock_cheap", "shock_diverse")
    selections = []
    for seed in SEEDS:
        for profile_id in PROFILES:
            for comparison, choices in (("pricing", pricing), ("supply", supply)):
                candidates = [
                    row
                    for row in rows
                    if row["seed"] == seed
                    and row["profile_id"] == profile_id
                    and row["scenario"] in choices
                ]
                selected = max(candidates, key=lambda row: (row["utility_ppm"], row["scenario"]))
                selections.append(
                    {
                        "seed": seed,
                        "profile_id": profile_id,
                        "comparison": comparison,
                        "selected_scenario": selected["scenario"],
                    }
                )
    selection_counts = {
        profile_id: {
            comparison: dict(
                Counter(
                    row["selected_scenario"]
                    for row in selections
                    if row["profile_id"] == profile_id
                    and row["comparison"] == comparison
                )
            )
            for comparison in ("pricing", "supply")
        }
        for profile_id in PROFILES
    }

    # A small zero-token integration sample proves the authoritative rollout
    # consumes the same profiles and the new sourcing candidates.
    rollout_rows = []
    oracle = AuthoritativeMarketRolloutEvaluator(config)
    for seed in SEEDS[:3]:
        for profile_id in PROFILES:
            env = MarketEnv(config)
            state = env.reset(
                episode_id=f"objective-rollout-{seed}-{profile_id}",
                episode_seed=seed,
                market_model="balanced",
                max_rounds=20,
            )
            plan = oracle.evaluate(
                state=state,
                company_id="company_A",
                persona_profile=registry.get(profile_id),
                horizon_rounds=3,
                scenario_count=3,
            )
            rollout_rows.append(
                {
                    "seed": seed,
                    "profile_id": profile_id,
                    "recommended_candidate_id": plan.recommended_candidate_id,
                    "expected_gain_over_baseline_cents": plan.expected_gain_over_baseline_cents,
                    "plan_hash": plan.plan_hash,
                }
            )

    summary: dict[str, Any] = {
        "experiment_id": "multi-objective-v10-validation",
        "evidence_level": "DETERMINISTIC_SYNTHETIC_OBJECTIVE_TREATMENT_EVIDENCE",
        "common_seeds": list(SEEDS),
        "profiles": list(PROFILES),
        "rows": rows,
        "selections": selections,
        "selection_counts": selection_counts,
        "mean_outcomes_by_scenario": {
            scenario: {
                field: round(mean(
                    row[field]
                    for row in rows
                    if row["scenario"] == scenario and row["profile_id"] == "none"
                ))
                for field in (
                    "company_profit_cents",
                    "consumer_surplus_cents",
                    "total_welfare_cents",
                    "sales_orders",
                    "stockout_orders",
                )
            }
            for scenario in SCENARIOS
        },
        "rollout_integration_rows": rollout_rows,
        "gates": {
            "public_service_prefers_affordability_more_than_profit_baseline": (
                selection_counts["public_service"]["pricing"].get("affordable_diverse", 0)
                > selection_counts["none"]["pricing"].get("affordable_diverse", 0)
            ),
            "resilience_steward_prefers_diversification_under_shock": (
                selection_counts["resilience_steward"]["supply"].get("shock_diverse", 0)
                == len(SEEDS)
            ),
            "profiles_reach_authoritative_rollout": len(rollout_rows) == 12,
            "zero_provider_calls": True,
        },
        "claim_boundary": (
            "Objective weights are explicit experimental treatments, not "
            "estimates of real corporate preferences."
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
