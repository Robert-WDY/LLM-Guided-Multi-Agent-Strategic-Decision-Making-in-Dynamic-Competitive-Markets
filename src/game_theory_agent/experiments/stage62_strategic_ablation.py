"""Stage 6.2 zero-LLM downstream ablation across personas and opponent pools."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.agents.personas import PersonaRegistry, PersonaUtilityTracker
from game_theory_agent.belief import BeliefLedger
from game_theory_agent.gameplay import build_rule_action, build_terminal_rankings
from game_theory_agent.market import (
    CompanyAction,
    IncidentResponse,
    IncidentResponseMode,
    MarketEnv,
    MarketState,
    load_market_config,
)
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.opponent import (
    OpponentModelLedger,
    OpponentModelState,
    StrategyDistribution,
    compute_opponent_model_hash,
)
from game_theory_agent.strategic_reliability import (
    PublicMarketRolloutAdvisor,
    build_calibrated_opponent_state_v2,
    deterministic_dirichlet_weights,
    generate_public_overlay_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v4.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "stage6.2-ablation" / "summary.json"
COMPANIES = ("company_A", "company_B", "company_C", "company_D")
PERSONAS = (
    "aggressive_v1_extreme",
    "profit_myopic",
    "risk_guarded_v1",
)
CONDITIONS = (
    "rule_baseline",
    "persona_planner",
    "belief",
    "opponent_v1",
    "opponent_v2_candidate",
)
HOLDOUT_IDS = ("unknown_070", "unknown_084", "unknown_099")


def _distribution(
    growth: int, profit: int, defensive: int, cooperative: int
) -> StrategyDistribution:
    return StrategyDistribution(
        growth_ppm=growth,
        profit_ppm=profit,
        defensive_ppm=defensive,
        cooperative_ppm=cooperative,
    )


def opponent_pools() -> dict[str, dict[str, StrategyDistribution]]:
    return {
        "known": {
            "company_B": _distribution(850_000, 50_000, 50_000, 50_000),
            "company_C": _distribution(50_000, 850_000, 50_000, 50_000),
            "company_D": _distribution(50_000, 50_000, 850_000, 50_000),
        },
        "mixed": {
            "company_B": _distribution(600_000, 400_000, 0, 0),
            "company_C": _distribution(0, 500_000, 500_000, 0),
            "company_D": _distribution(400_000, 0, 300_000, 300_000),
        },
        "holdout": {
            company_id: deterministic_dirichlet_weights(20260824, case_id)
            for company_id, case_id in zip(
                COMPANIES[1:], HOLDOUT_IDS, strict=True
            )
        },
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError("seeds must be a non-empty unique list")
    return parsed


def parse_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError("values must be a non-empty unique list")
    return parsed


def _select_strategy(
    distribution: StrategyDistribution,
    *,
    seed: int,
    pool_id: str,
    company_id: str,
    round_number: int,
) -> str:
    digest = sha256_hash(
        {
            "protocol": "stage6.2-opponent-policy-v1",
            "seed": seed,
            "pool_id": pool_id,
            "company_id": company_id,
            "round": round_number,
        }
    )
    draw = int(digest.rsplit(":", 1)[-1][-16:], 16) % 1_000_000
    cumulative = 0
    for strategy in ("growth", "profit", "defensive", "cooperative"):
        cumulative += int(getattr(distribution, f"{strategy}_ppm"))
        if draw < cumulative:
            return strategy
    return "cooperative"


def _opponent_action(
    config,
    state: MarketState,
    company_id: str,
    *,
    seed: int,
    pool_id: str,
    distribution: StrategyDistribution,
) -> tuple[CompanyAction, str]:
    base = build_rule_action(config, state, company_id)
    env = MarketEnv(config)
    env.load_state(state)
    constraints = env.get_action_constraints(company_id, state.state_version)
    strategy = _select_strategy(
        distribution,
        seed=seed,
        pool_id=pool_id,
        company_id=company_id,
        round_number=state.round,
    )
    bounds = config.mapping("action", "bounds", "price_cents")
    price = base.price_cents
    advertising = base.advertising_budget_cents
    service = base.service_budget_cents
    capacity = base.capacity_investment_cents
    resilience = base.resilience_budget_cents
    shared = base.shared_resilience_contribution_cents or 0
    remaining = max(
        0,
        state.company(company_id).financial.cash_balance_cents - base.fixed_spend_cents,
    )

    if strategy == "growth":
        price -= 600
        increment = min(300_000, remaining)
        advertising += increment
        remaining -= increment
        if constraints["capacity_investment_enabled"]:
            capacity += min(250_000, remaining)
    elif strategy == "profit":
        price += 600
    elif strategy == "defensive":
        price += 200
        if constraints["resilience_investment_enabled"]:
            increment = min(350_000, remaining)
            resilience += increment
            remaining -= increment
        service += min(150_000, remaining)
    else:
        increment = min(300_000, remaining)
        service += increment
        remaining -= increment
        if constraints["shared_resilience_contribution_enabled"]:
            shared += min(400_000, remaining)

    action = replace(
        base,
        action_id=(
            f"stage6.2-opponent:{state.episode_id}:{state.round}:"
            f"{company_id}:{strategy}"
        ),
        price_cents=min(max(price, int(bounds["min"])), int(bounds["max"])),
        advertising_budget_cents=advertising,
        service_budget_cents=service,
        capacity_investment_cents=capacity,
        resilience_budget_cents=resilience,
        shared_resilience_contribution_cents=shared,
        strategy_summary=f"Stage 6.2 latent utility policy: {strategy}",
    )
    validated = env.validate_action(action, company_id)
    if not validated.valid or validated.action is None:
        raise ValueError(
            f"latent opponent action is illegal for {company_id}: {validated.errors}"
        )
    return validated.action, strategy


def _candidate_to_action(candidate, state: MarketState) -> CompanyAction:
    payload = candidate.action
    return CompanyAction(
        action_id=(
            f"stage6.2-candidate:{state.episode_id}:{state.round}:"
            f"company_A:{candidate.candidate_id}"
        ),
        episode_id=state.episode_id,
        agent_id="company_A",
        round=state.round,
        state_version=state.state_version,
        price_cents=payload.price_cents,
        advertising_budget_cents=payload.advertising_budget_cents,
        service_budget_cents=payload.service_budget_cents,
        capacity_investment_cents=payload.capacity_investment_cents,
        resilience_budget_cents=payload.resilience_budget_cents,
        shared_resilience_contribution_cents=(
            payload.shared_resilience_contribution_cents
        ),
        incident_response=IncidentResponse(
            IncidentResponseMode(payload.incident_response_mode),
            payload.repair_budget_cents,
        ),
        strategy_summary=f"Stage 6.2 candidate: {candidate.candidate_id}",
    )


def _advice_to_action(advice, state: MarketState) -> CompanyAction:
    payload = advice.recommended_action
    return CompanyAction(
        action_id=f"stage6.2-advice:{state.episode_id}:{state.round}:company_A",
        episode_id=state.episode_id,
        agent_id="company_A",
        round=state.round,
        state_version=state.state_version,
        price_cents=int(payload["price_cents"]),
        advertising_budget_cents=int(payload["advertising_budget_cents"]),
        service_budget_cents=int(payload["service_budget_cents"]),
        capacity_investment_cents=int(payload["capacity_investment_cents"]),
        resilience_budget_cents=int(payload["resilience_budget_cents"]),
        shared_resilience_contribution_cents=payload.get(
            "shared_resilience_contribution_cents"
        ),
        incident_response=IncidentResponse(
            IncidentResponseMode(payload["incident_response_mode"]),
            int(payload["repair_budget_cents"]),
        ),
        strategy_summary=f"Stage 6.2 advice: {advice.recommended_candidate_id}",
    )


def _enterprise_value(state: MarketState, config) -> int:
    return next(
        int(item["value_cents"])
        for item in build_terminal_rankings(state, config)["composite"]
        if item["company_id"] == "company_A"
    )


def _rank(state: MarketState, config) -> int:
    return next(
        int(item["rank"])
        for item in build_terminal_rankings(state, config)["composite"]
        if item["company_id"] == "company_A"
    )


def _true_one_step_regret(
    config,
    state: MarketState,
    *,
    persona_profile,
    opponent_actions: Mapping[str, CompanyAction],
    chosen_action: CompanyAction,
) -> int:
    registry = PersonaRegistry.from_market_config(config)
    evaluator = registry.evaluator(persona_profile)
    values: dict[str, int] = {}
    for candidate in generate_public_overlay_candidates(config, state, "company_A"):
        env = MarketEnv(config)
        env.load_state(state)
        actions = dict(opponent_actions)
        actions["company_A"] = _candidate_to_action(candidate, state)
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", actions
        )
        assessment = evaluator.evaluate(state, result.state_after, "company_A")
        loss = (
            assessment.realized_incident_loss_cents
            + assessment.realized_unserved_contribution_loss_cents
        )
        persona_bonus = (
            assessment.round_utility_ppm * registry.profit_scale_cents // 1_000_000
        )
        values[candidate.candidate_id] = (
            _enterprise_value(result.state_after, config) + persona_bonus - loss
        )
    # Advice is generated from a public forecast state.  Near a price or cash
    # boundary, its candidate id can legitimately be deduplicated out of the
    # candidate set regenerated from the hidden true state.  Regret must score
    # the action that was actually chosen, rather than looking it up by a label
    # in a different candidate set.
    chosen_env = MarketEnv(config)
    chosen_env.load_state(state)
    chosen_actions = dict(opponent_actions)
    chosen_actions["company_A"] = chosen_action
    chosen_result = chosen_env.step(
        f"{state.episode_id}:{state.round}:{state.state_version}",
        chosen_actions,
    )
    chosen_assessment = evaluator.evaluate(
        state, chosen_result.state_after, "company_A"
    )
    chosen_loss = (
        chosen_assessment.realized_incident_loss_cents
        + chosen_assessment.realized_unserved_contribution_loss_cents
    )
    chosen_bonus = (
        chosen_assessment.round_utility_ppm
        * registry.profit_scale_cents
        // 1_000_000
    )
    chosen_value = (
        _enterprise_value(chosen_result.state_after, config)
        + chosen_bonus
        - chosen_loss
    )
    return max(0, max((*values.values(), chosen_value)) - chosen_value)


def _v2_state(
    v1: OpponentModelState, ledger: OpponentModelLedger
) -> OpponentModelState:
    return build_calibrated_opponent_state_v2(v1, ledger.evidence())


def run_episode(
    config,
    *,
    seed: int,
    persona_id: str,
    pool_id: str,
    pool: Mapping[str, StrategyDistribution],
    condition: str,
    rounds: int,
    horizon_rounds: int,
    scenario_count: int,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    episode_id = f"stage6.2-{pool_id}-seed-{seed}"
    env = MarketEnv(config)
    state = env.reset(
        COMPANIES,
        episode_id=episode_id,
        episode_seed=seed,
        market_model="balanced",
        max_rounds=rounds,
        cooperation_mode="shared_resilience_v1",
    )
    registry = PersonaRegistry.from_market_config(config)
    profile = registry.get(persona_id)
    tracker = PersonaUtilityTracker(registry.evaluator(profile))
    belief_ledger = BeliefLedger(
        episode_id=episode_id, company_ids=state.company_ids
    )
    opponent_ledger = OpponentModelLedger(
        episode_id=episode_id, company_ids=state.company_ids
    )
    advisor = PublicMarketRolloutAdvisor(config)
    regrets: list[int] = []
    recommendations: list[str] = []
    opponent_strategies: Counter[str] = Counter()
    realized_loss = 0
    all_actions_legal = True

    while not state.terminal:
        belief, belief_hash = belief_ledger.company_view(
            observer_company_id="company_A",
            round_number=state.round,
            state_version=state.state_version,
        )
        v1, v1_hash = opponent_ledger.company_view(
            observer_company_id="company_A",
            round_number=state.round,
            state_version=state.state_version,
        )
        v2 = _v2_state(v1, opponent_ledger)
        opponent_actions: dict[str, CompanyAction] = {}
        for company_id in state.company_ids[1:]:
            action, strategy = _opponent_action(
                config,
                state,
                company_id,
                seed=seed,
                pool_id=pool_id,
                distribution=pool[company_id],
            )
            opponent_actions[company_id] = action
            opponent_strategies[strategy] += 1

        selected_belief = belief if condition != "persona_planner" else None
        selected_model: OpponentModelState | None = None
        if condition == "opponent_v1":
            selected_model = v1
        elif condition == "opponent_v2_candidate":
            selected_model = v2
        if condition == "rule_baseline":
            chosen_id = "maintain"
            focal_action = build_rule_action(config, state, "company_A")
        else:
            observation = ObservationBuilder().build(
                state,
                "company_A",
                "public",
                belief_state=(
                    selected_belief.model_dump(mode="json")
                    if selected_belief is not None
                    else None
                ),
                belief_hash=belief_hash if selected_belief is not None else None,
                belief_schema_version=(
                    selected_belief.belief_schema_version
                    if selected_belief is not None
                    else "none"
                ),
            )
            if selected_model is not None:
                observation["opponent_model_hash"] = (
                    v1_hash
                    if condition == "opponent_v1"
                    else compute_opponent_model_hash(v2)
                )
                observation["opponent_model_state"] = selected_model.model_dump(
                    mode="json"
                )
            advice = advisor.advise(
                observation=observation,
                company_id="company_A",
                persona_profile=profile,
                belief_state=selected_belief,
                opponent_model=selected_model,
                horizon_rounds=min(horizon_rounds, state.rounds_remaining),
                scenario_count=scenario_count,
            )
            chosen_id = advice.recommended_candidate_id
            focal_action = _advice_to_action(advice, state)
        recommendations.append(chosen_id)
        regrets.append(
            _true_one_step_regret(
                config,
                state,
                persona_profile=profile,
                opponent_actions=opponent_actions,
                chosen_action=focal_action,
            )
        )
        actions = {"company_A": focal_action, **opponent_actions}
        validation = env.validate_action(focal_action, "company_A")
        all_actions_legal = all_actions_legal and validation.valid
        if not validation.valid or validation.action is None:
            raise ValueError(f"illegal focal action: {validation.errors}")
        actions["company_A"] = validation.action
        before = state
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", actions
        )
        assessment = tracker.record(before, result.state_after, "company_A")
        realized_loss += (
            assessment.realized_incident_loss_cents
            + assessment.realized_unserved_contribution_loss_cents
        )
        belief_ledger.update_after_settlement(before, actions)
        opponent_ledger.update_after_settlement(before, result.state_after, actions)
        state = result.state_after

    company = state.company("company_A")
    return {
        "seed": seed,
        "persona_id": persona_id,
        "opponent_pool": pool_id,
        "condition": condition,
        "enterprise_value_cents": _enterprise_value(state, config),
        "cumulative_profit_cents": company.financial.cumulative_profit_cents,
        "final_market_share_ppm": company.commercial.market_share_ppm,
        "final_rank": _rank(state, config),
        "realized_loss_cents": realized_loss,
        "mean_true_one_step_regret_cents": round(mean(regrets)),
        "recommendations": recommendations,
        "opponent_realized_strategies": dict(sorted(opponent_strategies.items())),
        "all_actions_legal": all_actions_legal,
        "final_state_hash": state.state_hash,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(rows),
        "mean_enterprise_value_cents": round(
            mean(item["enterprise_value_cents"] for item in rows)
        ),
        "mean_cumulative_profit_cents": round(
            mean(item["cumulative_profit_cents"] for item in rows)
        ),
        "mean_market_share_ppm": round(
            mean(item["final_market_share_ppm"] for item in rows)
        ),
        "mean_rank": mean(item["final_rank"] for item in rows),
        "first_place_rate_ppm": (
            sum(item["final_rank"] == 1 for item in rows) * 1_000_000 // len(rows)
        ),
        "mean_realized_loss_cents": round(
            mean(item["realized_loss_cents"] for item in rows)
        ),
        "mean_true_one_step_regret_cents": round(
            mean(item["mean_true_one_step_regret_cents"] for item in rows)
        ),
        "worst_enterprise_value_cents": min(
            item["enterprise_value_cents"] for item in rows
        ),
    }


def _marginal(
    rows: list[dict[str, Any]], treatment: str, control: str
) -> dict[str, Any]:
    treated_rows = [item for item in rows if item["condition"] == treatment]
    control_by_key = {
        (item["seed"], item["persona_id"], item["opponent_pool"]): item
        for item in rows
        if item["condition"] == control
    }
    if not treated_rows or len(treated_rows) != len(control_by_key):
        raise ValueError("marginal requires complete paired treatment rows")

    paired: list[dict[str, int]] = []
    for item in treated_rows:
        key = (item["seed"], item["persona_id"], item["opponent_pool"])
        baseline = control_by_key.get(key)
        if baseline is None:
            raise ValueError(f"missing paired control row: {key}")
        paired.append(
            {
                "enterprise_value": (
                    item["enterprise_value_cents"]
                    - baseline["enterprise_value_cents"]
                ),
                "profit": (
                    item["cumulative_profit_cents"]
                    - baseline["cumulative_profit_cents"]
                ),
                "regret": (
                    item["mean_true_one_step_regret_cents"]
                    - baseline["mean_true_one_step_regret_cents"]
                ),
            }
        )
    ev_deltas = [item["enterprise_value"] for item in paired]
    profit_deltas = [item["profit"] for item in paired]
    regret_deltas = [item["regret"] for item in paired]
    return {
        "treatment": treatment,
        "control": control,
        "pair_count": len(paired),
        "mean_enterprise_value_delta_cents": round(mean(ev_deltas)),
        "median_enterprise_value_delta_cents": round(median(ev_deltas)),
        "mean_profit_delta_cents": round(mean(profit_deltas)),
        "mean_true_regret_delta_cents": round(mean(regret_deltas)),
        # Kept for compatibility with the initial Stage 6.2 result schema.
        "true_regret_delta_cents": round(mean(regret_deltas)),
        "worst_paired_enterprise_value_delta_cents": min(ev_deltas),
        "best_paired_enterprise_value_delta_cents": max(ev_deltas),
        "positive_enterprise_value_pairs": sum(value > 0 for value in ev_deltas),
        "zero_enterprise_value_pairs": sum(value == 0 for value in ev_deltas),
        "negative_enterprise_value_pairs": sum(value < 0 for value in ev_deltas),
        "non_negative_enterprise_value_pair_rate_ppm": (
            sum(value >= 0 for value in ev_deltas) * 1_000_000 // len(ev_deltas)
        ),
    }


def run(
    *,
    seeds: tuple[int, ...] = (1, 2, 3),
    personas: tuple[str, ...] = PERSONAS,
    pool_ids: tuple[str, ...] = ("known", "mixed", "holdout"),
    rounds: int = 5,
    horizon_rounds: int = 2,
    scenario_count: int = 2,
) -> dict[str, Any]:
    config = load_market_config(CONFIG_PATH)
    registry = PersonaRegistry.from_market_config(config)
    for persona_id in personas:
        registry.get(persona_id)
    pools = opponent_pools()
    unknown_pools = set(pool_ids) - set(pools)
    if unknown_pools:
        raise ValueError(f"unknown opponent pools: {sorted(unknown_pools)}")
    if rounds not in set(config.get("episode_options", "round_options")):
        raise ValueError("rounds must be an allowed episode length")

    rows = [
        run_episode(
            config,
            seed=seed,
            persona_id=persona_id,
            pool_id=pool_id,
            pool=pools[pool_id],
            condition=condition,
            rounds=rounds,
            horizon_rounds=horizon_rounds,
            scenario_count=scenario_count,
        )
        for seed in seeds
        for pool_id in pool_ids
        for persona_id in personas
        for condition in CONDITIONS
    ]
    by_condition = {
        condition: _aggregate(
            [item for item in rows if item["condition"] == condition]
        )
        for condition in CONDITIONS
    }
    by_cell = {
        f"{condition}|{persona_id}|{pool_id}": _aggregate(
            [
                item
                for item in rows
                if item["condition"] == condition
                and item["persona_id"] == persona_id
                and item["opponent_pool"] == pool_id
            ]
        )
        for condition in CONDITIONS
        for persona_id in personas
        for pool_id in pool_ids
    }

    comparisons = {
        "persona_planner_vs_rule": ("persona_planner", "rule_baseline"),
        "belief_vs_persona_planner": ("belief", "persona_planner"),
        "opponent_v1_vs_belief": ("opponent_v1", "belief"),
        "opponent_v2_vs_v1": ("opponent_v2_candidate", "opponent_v1"),
    }
    marginals = {
        key: _marginal(rows, treatment, control)
        for key, (treatment, control) in comparisons.items()
    }
    marginals_by_persona = {
        key: {
            persona_id: _marginal(
                [item for item in rows if item["persona_id"] == persona_id],
                treatment,
                control,
            )
            for persona_id in personas
        }
        for key, (treatment, control) in comparisons.items()
    }
    marginals_by_opponent_pool = {
        key: {
            pool_id: _marginal(
                [item for item in rows if item["opponent_pool"] == pool_id],
                treatment,
                control,
            )
            for pool_id in pool_ids
        }
        for key, (treatment, control) in comparisons.items()
    }
    action_counts_by_condition = {
        condition: dict(
            sorted(
                Counter(
                    recommendation
                    for item in rows
                    if item["condition"] == condition
                    for recommendation in item["recommendations"]
                ).items()
            )
        )
        for condition in CONDITIONS
    }

    def promotion_gate(key: str) -> dict[str, bool]:
        item = marginals[key]
        return {
            "mean_enterprise_value_non_negative": (
                item["mean_enterprise_value_delta_cents"] >= 0
            ),
            "true_regret_not_higher": item["true_regret_delta_cents"] <= 0,
            "worst_pair_non_negative": (
                item["worst_paired_enterprise_value_delta_cents"] >= 0
            ),
        }

    promotion_gates = {
        "persona_planner": promotion_gate("persona_planner_vs_rule"),
        "belief": promotion_gate("belief_vs_persona_planner"),
        "opponent_v1": promotion_gate("opponent_v1_vs_belief"),
        "opponent_v2_candidate": promotion_gate("opponent_v2_vs_v1"),
    }
    recommendations_by_matched_state: dict[str, set[tuple[str, ...]]] = {}
    for item in rows:
        if item["condition"] != "persona_planner":
            continue
        key = f"{item['seed']}|{item['opponent_pool']}"
        recommendations_by_matched_state.setdefault(key, set()).add(
            tuple(item["recommendations"])
        )
    engineering_checks = {
        "real_llm_calls_zero": True,
        "real_llm_tokens_zero": True,
        "real_llm_estimated_cost_zero": True,
        "all_actions_legal": all(item["all_actions_legal"] for item in rows),
        "matrix_complete": len(rows)
        == len(seeds) * len(pool_ids) * len(personas) * len(CONDITIONS),
        "holdout_ids_are_disjoint_from_development": all(
            int(case_id.rsplit("_", 1)[-1]) >= 70 for case_id in HOLDOUT_IDS
        ),
        "persona_changes_recommendations": any(
            len(values) > 1 for values in recommendations_by_matched_state.values()
        ),
    }
    return {
        "experiment_schema_version": "stage6.2-strategic-ablation-v1.1.0",
        "evidence_level": "deterministic_rule_mock_synthetic_benchmark",
        "comparison_design": {
            "actual_market_seed_matched": True,
            "advisor_common_random_scenarios_on_same_observation": True,
            "opponent_policy_matched_but_state_responsive": True,
            "one_step_regret_uses_same_true_state_and_opponent_actions": True,
        },
        "seeds": list(seeds),
        "personas": list(personas),
        "opponent_pools": list(pool_ids),
        "conditions": list(CONDITIONS),
        "rounds": rounds,
        "horizon_rounds": horizon_rounds,
        "scenario_count": scenario_count,
        "real_model_usage": {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost": 0,
            "currency": "CNY",
        },
        "opponent_pool_distributions": {
            pool_id: {
                company_id: distribution.model_dump(mode="json")
                for company_id, distribution in pools[pool_id].items()
            }
            for pool_id in pool_ids
        },
        "rows": rows,
        "by_condition": by_condition,
        "by_cell": by_cell,
        "marginals": marginals,
        "marginals_by_persona": marginals_by_persona,
        "marginals_by_opponent_pool": marginals_by_opponent_pool,
        "action_counts_by_condition": action_counts_by_condition,
        "promotion_gates": promotion_gates,
        "promotion_recommendations": {
            component: all(checks.values())
            for component, checks in promotion_gates.items()
        },
        "engineering_checks": engineering_checks,
        "engineering_passed": all(engineering_checks.values()),
        "conclusion_limits": [
            "five-round deterministic episodes are an engineering ablation",
            "true one-step regret is exact hindsight for one settled round, not equilibrium",
            "opponent pools are synthetic latent utility policies",
            "no result is evidence of real-LLM treatment adoption",
            "a candidate failing any gate is not promoted automatically",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--personas", default=",".join(PERSONAS))
    parser.add_argument("--opponent-pools", default="known,mixed,holdout")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--horizon-rounds", type=int, default=2)
    parser.add_argument("--scenario-count", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(
        seeds=parse_csv_ints(args.seeds),
        personas=parse_csv(args.personas),
        pool_ids=parse_csv(args.opponent_pools),
        rounds=args.rounds,
        horizon_rounds=args.horizon_rounds,
        scenario_count=args.scenario_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "engineering_passed": summary["engineering_passed"],
                "real_model_usage": summary["real_model_usage"],
                "marginals": summary["marginals"],
                "promotion_recommendations": summary[
                    "promotion_recommendations"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
