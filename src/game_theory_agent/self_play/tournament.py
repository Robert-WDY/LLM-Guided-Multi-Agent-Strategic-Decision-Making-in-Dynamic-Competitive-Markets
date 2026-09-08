"""Role-rotated empirical-game tournament on the final strategic market."""

from __future__ import annotations

from dataclasses import replace
from statistics import mean
from typing import Any, Mapping, Sequence

from game_theory_agent.gameplay import build_terminal_rankings
from game_theory_agent.market import MarketConfig, MarketEnv, MarketState
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.market.replay import (
    EpisodeManifest,
    MarketTransition,
    verify_replay,
)

from .policies import build_strategy_action


COMPANY_IDS = ("company_A", "company_B", "company_C", "company_D")
MARKET_MODELS = (
    "balanced",
    "value_oriented",
    "quality_oriented",
    "service_oriented",
)


def _rehash(state: MarketState) -> MarketState:
    cleared = replace(state, state_hash="")
    return replace(cleared, state_hash=state_hash(cleared.to_dict()))


def _initial_state(
    config: MarketConfig,
    *,
    episode_id: str,
    seed: int,
    rounds: int,
) -> tuple[MarketEnv, MarketState]:
    env = MarketEnv(config)
    market_model = MARKET_MODELS[seed % len(MARKET_MODELS)]
    state = env.reset(
        COMPANY_IDS,
        episode_id=episode_id,
        episode_seed=seed,
        market_model=market_model,
        max_rounds=rounds,
        cooperation_mode="combined_v1",
    )
    companies = []
    for index, company in enumerate(state.companies):
        if index % 2 == 0:
            company = replace(
                company,
                operations=replace(
                    company.operations,
                    base_capacity_orders=900,
                    effective_capacity_orders=900,
                ),
                brand=replace(
                    company.brand,
                    brand_awareness_ppm=800_000,
                    service_quality_ppm=800_000,
                    reputation_ppm=800_000,
                ),
            )
        else:
            company = replace(
                company,
                operations=replace(
                    company.operations,
                    base_capacity_orders=8_000,
                    effective_capacity_orders=8_000,
                ),
                brand=replace(
                    company.brand,
                    brand_awareness_ppm=450_000,
                    service_quality_ppm=500_000,
                    reputation_ppm=500_000,
                ),
            )
        companies.append(company)
    state = _rehash(replace(state, companies=tuple(companies)))
    env.load_state(state)
    return env, state


def run_episode(
    config: MarketConfig,
    *,
    seed: int,
    assignments: Mapping[str, str],
    rounds: int = 20,
    experiment_id: str = "final-market-self-play-v1",
    episode_id: str | None = None,
) -> dict[str, Any]:
    if set(assignments) != set(COMPANY_IDS):
        raise ValueError("self-play assignments must cover exactly four companies")
    assignment_key = "-".join(assignments[item] for item in COMPANY_IDS)
    resolved_episode_id = episode_id or f"self-play-{seed}-{assignment_key}"
    env, state = _initial_state(
        config, episode_id=resolved_episode_id, seed=seed, rounds=rounds
    )
    manifest = EpisodeManifest.create(env, state, experiment_id=experiment_id)
    transitions: list[MarketTransition] = []
    price_war_rounds = 0
    coordination_statuses: list[str] = []
    detected_coordination_count = 0
    mutual_aid_orders = 0
    exited_company_ids: set[str] = set()
    concentration_regimes: list[str] = []
    closure_passed = True
    first_round_profit_cents: dict[str, int] = {}
    first_round_public_contribution_cents: dict[str, int] = {}
    for _ in range(rounds):
        actions = {
            company_id: build_strategy_action(
                config,
                state,
                company_id,
                assignments[company_id],
            )
            for company_id in COMPANY_IDS
        }
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", actions
        )
        transitions.append(MarketTransition.create(state, actions, result))
        state = result.state_after
        if len(transitions) == 1:
            first_round_profit_cents = {
                company.company_id: company.financial.round_profit_cents
                for company in state.companies
            }
            first_round_public_contribution_cents = {
                company_id: int(
                    actions[company_id].shared_resilience_contribution_cents
                    or 0
                )
                + int(
                    actions[company_id].threshold_project_contribution_cents
                    or 0
                )
                for company_id in COMPANY_IDS
            }
        strategic = state.strategic_market
        price_war_rounds += strategic.price_war_intensity_ppm > 0
        concentration_regimes.append(strategic.concentration_regime.value)
        exited_company_ids.update(
            item.company_id
            for item in strategic.company_lifecycle
            if item.status.value == "exited"
        )
        mutual_aid_orders += sum(
            transfer.fulfilled_orders
            for transfer in strategic.last_mutual_aid_transfers
        )
        coordination_statuses.extend(
            outcome.status.value
            for outcome in strategic.last_price_coordination_outcomes
        )
        detected_coordination_count += sum(
            bool(outcome.detected)
            for outcome in strategic.last_price_coordination_outcomes
        )
        closure_passed = closure_passed and (
            state.market.no_purchase_orders
            + state.market.lost_after_stockout_orders
            + sum(company.commercial.sales_orders for company in state.companies)
            == state.market.realized_demand_orders
        )
        if result.done:
            break
    replayed = verify_replay(MarketEnv(config), manifest, tuple(transitions))[-1]
    ranking_rows = build_terminal_rankings(state, config)["composite"]
    enterprise_value = {
        str(row["company_id"]): int(row["value_cents"])
        for row in ranking_rows
    }
    rank = {
        str(row["company_id"]): index
        for index, row in enumerate(ranking_rows, start=1)
    }
    project = state.strategic_market.threshold_project
    return {
        "episode_id": resolved_episode_id,
        "trajectory_id": (
            f"{resolved_episode_id}:assignment:{assignment_key}"
        ),
        "seed": seed,
        "market_model": state.market.market_model_id,
        "rounds_completed": len(transitions),
        "assignments": dict(assignments),
        "enterprise_value_cents": enterprise_value,
        "rank": rank,
        "cumulative_profit_cents": {
            company.company_id: company.financial.cumulative_profit_cents
            for company in state.companies
        },
        "first_round_profit_cents": first_round_profit_cents,
        "first_round_public_contribution_cents": (
            first_round_public_contribution_cents
        ),
        "final_market_share_ppm": {
            company.company_id: company.commercial.market_share_ppm
            for company in state.companies
        },
        "final_status": {
            company_id: state.strategic_market.lifecycle(company_id).status.value
            for company_id in COMPANY_IDS
        },
        "exited_company_ids": sorted(exited_company_ids),
        "final_hhi_ppm": state.strategic_market.hhi_ppm,
        "concentration_regimes": concentration_regimes,
        "price_war_rounds": price_war_rounds,
        "threshold_project_status": project.status.value if project else None,
        "final_industry_resilience_ppm": (
            state.shared_resilience.industry_resilience_ppm
            if state.shared_resilience is not None
            else None
        ),
        "mutual_aid_orders": mutual_aid_orders,
        "coordination_statuses": coordination_statuses,
        "detected_coordination_count": detected_coordination_count,
        "cumulative_social_welfare_proxy_cents": (
            state.strategic_market.cumulative_social_welfare_proxy_cents
        ),
        "demand_closure_passed": closure_passed,
        "non_negative_cash": all(
            company.financial.cash_balance_cents >= 0
            for company in state.companies
        ),
        "replay_passed": replayed.state_hash == state.state_hash,
        "final_state_hash": state.state_hash,
    }


def role_rotated_matchups(
    config: MarketConfig,
    *,
    seeds: Sequence[int],
    strategy_ids: Sequence[str],
    rounds: int = 20,
    experiment_id: str = "final-market-self-play-v1",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for resident in strategy_ids:
            for mutant in strategy_ids:
                for focal_company_id in COMPANY_IDS:
                    assignments = {
                        company_id: (
                            mutant
                            if company_id == focal_company_id
                            else resident
                        )
                        for company_id in COMPANY_IDS
                    }
                    episode = run_episode(
                        config,
                        seed=seed,
                        assignments=assignments,
                        rounds=rounds,
                        experiment_id=experiment_id,
                        episode_id=(
                            f"self-play-{seed}-resident-{resident}-"
                            f"focal-{focal_company_id}"
                        ),
                    )
                    opponent_values = [
                        int(episode["enterprise_value_cents"][company_id])
                        for company_id in COMPANY_IDS
                        if company_id != focal_company_id
                    ]
                    focal_value = int(
                        episode["enterprise_value_cents"][focal_company_id]
                    )
                    rows.append(
                        {
                            "split": "unspecified",
                            "seed": seed,
                            "resident_strategy_id": resident,
                            "mutant_strategy_id": mutant,
                            "focal_company_id": focal_company_id,
                            "focal_enterprise_value_cents": focal_value,
                            "focal_relative_advantage_cents": (
                                focal_value - round(mean(opponent_values))
                            ),
                            "focal_rank": int(
                                episode["rank"][focal_company_id]
                            ),
                            "focal_survived": (
                                episode["final_status"][focal_company_id]
                                != "exited"
                            ),
                            "episode": episode,
                        }
                    )
    return rows
