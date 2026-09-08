"""Zero-token development/holdout validation for final-market Advisor v9."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.belief import BeliefLedger
from game_theory_agent.gameplay import build_rule_action, build_terminal_rankings
from game_theory_agent.market import (
    CompanyAction,
    CompanyOperatingStatus,
    IncidentResponse,
    IncidentResponseMode,
    MarketConfig,
    MarketEnv,
    MarketState,
    load_market_config,
)
from game_theory_agent.market.protocols import sha256_hash, state_hash
from game_theory_agent.opponent import OpponentModelLedger
from game_theory_agent.strategic_reliability import PublicMarketRolloutAdvisor


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v6_final.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "final-market-advisor-v9"
COMPANIES = ("company_A", "company_B", "company_C", "company_D")
FOCAL = "company_A"
DEVELOPMENT_SEEDS = (
    91001,
    91002,
    91003,
    91004,
    92001,
    92002,
    92003,
    92004,
    92005,
    92006,
)
HOLDOUT_SEEDS = (94001, 94002, 94003, 94004, 94005, 94006)
PERSONAS = ("balanced_v1", "aggressive_v1_extreme", "risk_guarded_v1")
WINDOWS = ("pivotal_project", "mature_shortage")
REAL_CORE_WINDOWS = (
    "pivotal_project",
    "mature_shortage",
    "price_war",
    "distress",
)
HORIZON = 3
SCENARIOS = 5


def _enterprise_value(
    state: MarketState, company_id: str, config: MarketConfig
) -> int:
    return next(
        int(item["value_cents"])
        for item in build_terminal_rankings(state, config)["composite"]
        if item["company_id"] == company_id
    )


def _market_action(
    state: MarketState,
    company_id: str,
    candidate_id: str,
    payload: Mapping[str, Any],
) -> CompanyAction:
    return CompanyAction(
        action_id=(
            f"v9-actual:{state.episode_id}:{state.round}:"
            f"{company_id}:{candidate_id}"
        ),
        episode_id=state.episode_id,
        agent_id=company_id,
        round=state.round,
        state_version=state.state_version,
        price_cents=int(payload["price_cents"]),
        advertising_budget_cents=int(
            payload.get("advertising_budget_cents", 0)
        ),
        service_budget_cents=int(payload.get("service_budget_cents", 0)),
        capacity_investment_cents=int(
            payload.get("capacity_investment_cents", 0)
        ),
        resilience_budget_cents=int(
            payload.get("resilience_budget_cents", 0)
        ),
        shared_resilience_contribution_cents=payload.get(
            "shared_resilience_contribution_cents"
        ),
        threshold_project_contribution_cents=payload.get(
            "threshold_project_contribution_cents"
        ),
        mutual_aid_partner_company_id=payload.get(
            "mutual_aid_partner_company_id"
        ),
        mutual_aid_capacity_offer_orders=payload.get(
            "mutual_aid_capacity_offer_orders"
        ),
        mutual_aid_capacity_request_orders=payload.get(
            "mutual_aid_capacity_request_orders"
        ),
        price_coordination_partner_company_id=payload.get(
            "price_coordination_partner_company_id"
        ),
        price_coordination_target_cents=payload.get(
            "price_coordination_target_cents"
        ),
        incident_response=IncidentResponse(
            IncidentResponseMode(payload.get("incident_response_mode", "wait")),
            int(payload.get("repair_budget_cents", 0)),
        ),
        strategy_summary=f"v9 actual counterfactual: {candidate_id}",
    )


def _history_actions(
    config: MarketConfig, state: MarketState, window: str
) -> dict[str, CompanyAction]:
    actions = {
        company_id: build_rule_action(config, state, company_id)
        for company_id in state.company_ids
    }
    if state.shared_resilience is not None and state.rounds_remaining > 1:
        actions["company_B"] = replace(
            actions["company_B"],
            shared_resilience_contribution_cents=250_000,
        )
    actions["company_C"] = replace(
        actions["company_C"],
        price_cents=max(7_000, actions["company_C"].price_cents - 500),
        shared_resilience_contribution_cents=0,
    )
    if window == "pivotal_project" and state.round <= 2:
        for company_id in ("company_B", "company_C"):
            actions[company_id] = replace(
                actions[company_id],
                threshold_project_contribution_cents=1_000_000,
            )
    if window == "price_war":
        for company_id in ("company_B", "company_C"):
            actions[company_id] = replace(
                actions[company_id],
                price_cents=max(7_500, actions[company_id].price_cents - 1_800),
                advertising_budget_cents=min(
                    1_500_000,
                    actions[company_id].advertising_budget_cents + 300_000,
                ),
            )
    return actions


def _build_window(
    config: MarketConfig, seed: int, window: str
) -> tuple[MarketState, Any, Any, dict[str, Any]]:
    episode_id = f"advisor-v9-{seed}-{window}"
    env = MarketEnv(config)
    state = env.reset(
        COMPANIES,
        episode_id=episode_id,
        episode_seed=seed,
        market_model="balanced",
        max_rounds=20,
        cooperation_mode="combined_v1",
    )
    if window not in REAL_CORE_WINDOWS:
        raise ValueError(f"unsupported final-market window: {window}")
    if window in {"pivotal_project", "mature_shortage"}:
        companies = []
        for company in state.companies:
            if company.company_id == FOCAL:
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
            elif company.company_id == "company_B":
                company = replace(
                    company,
                    operations=replace(
                        company.operations,
                        base_capacity_orders=30_000,
                        effective_capacity_orders=30_000,
                    ),
                )
            companies.append(company)
        unsealed = replace(state, companies=tuple(companies), state_hash="")
        state = replace(unsealed, state_hash=state_hash(unsealed.to_dict()))
        env.load_state(state)
    beliefs = BeliefLedger(episode_id=episode_id, company_ids=COMPANIES)
    opponents = OpponentModelLedger(
        episode_id=episode_id,
        company_ids=COMPANIES,
    )
    history_rounds = {
        "pivotal_project": 2,
        "mature_shortage": 7,
        "price_war": 4,
        "distress": 4,
    }[window]
    for _ in range(history_rounds):
        before = state
        actions = _history_actions(config, before, window)
        result = env.step(
            f"{episode_id}:{before.round}:{before.state_version}",
            actions,
        )
        beliefs.update_after_settlement(before, actions)
        opponents.update_after_settlement(before, result.state_after, actions)
        state = result.state_after
    if window == "distress":
        focal = state.company(FOCAL)
        focal = replace(
            focal,
            financial=replace(
                focal.financial,
                cash_balance_cents=4_000_000,
            ),
            history=replace(
                focal.history,
                recent_profit_cents=(-1_800_000, -1_200_000, -900_000),
            ),
        )
        strategic = state.strategic_market
        if strategic is None:
            raise ValueError("distress window requires strategic market state")
        lifecycle = tuple(
            replace(
                item,
                status=CompanyOperatingStatus.DISTRESSED,
                distress_streak=1,
                first_distress_round=state.round - 1,
            )
            if item.company_id == FOCAL
            else item
            for item in strategic.company_lifecycle
        )
        unsealed = replace(
            state,
            companies=tuple(
                focal if item.company_id == FOCAL else item
                for item in state.companies
            ),
            strategic_market=replace(strategic, company_lifecycle=lifecycle),
            state_hash="",
        )
        state = replace(unsealed, state_hash=state_hash(unsealed.to_dict()))
        env.load_state(state)
    belief, belief_hash = beliefs.company_view(
        observer_company_id=FOCAL,
        round_number=state.round,
        state_version=state.state_version,
    )
    opponent, opponent_hash = opponents.company_view(
        observer_company_id=FOCAL,
        round_number=state.round,
        state_version=state.state_version,
    )
    observation = ObservationBuilder().build(
        state,
        FOCAL,
        "public",
        belief_state=belief.model_dump(mode="json"),
        belief_hash=belief_hash,
        belief_schema_version=belief.belief_schema_version,
    )
    observation["opponent_model_state"] = opponent.model_dump(mode="json")
    observation["opponent_model_hash"] = opponent_hash
    return state, belief, opponent, observation


def _responsive_first_round(
    config: MarketConfig,
    state: MarketState,
    focal: CompanyAction,
) -> dict[str, CompanyAction]:
    actions = {
        company_id: build_rule_action(config, state, company_id)
        for company_id in state.company_ids
    }
    actions[FOCAL] = focal
    project = state.strategic_market.threshold_project
    focal_project = int(focal.threshold_project_contribution_cents or 0)
    if project is not None and focal_project > 0:
        gap = max(
            0,
            project.required_total_contribution_cents
            - project.accumulated_total_contribution_cents
            - focal_project,
        )
        for index, company_id in enumerate(("company_B", "company_C")):
            remaining_peers = 2 - index
            contribution = min(
                2_000_000,
                (gap + remaining_peers - 1) // remaining_peers,
            )
            actions[company_id] = replace(
                actions[company_id],
                threshold_project_contribution_cents=contribution,
            )
            gap = max(0, gap - contribution)
    mutual_partner = focal.mutual_aid_partner_company_id
    if mutual_partner in actions:
        if int(focal.mutual_aid_capacity_request_orders or 0) > 0:
            actions[mutual_partner] = replace(
                actions[mutual_partner],
                mutual_aid_partner_company_id=FOCAL,
                mutual_aid_capacity_offer_orders=(
                    focal.mutual_aid_capacity_request_orders
                ),
                mutual_aid_capacity_request_orders=0,
            )
        elif int(focal.mutual_aid_capacity_offer_orders or 0) > 0:
            actions[mutual_partner] = replace(
                actions[mutual_partner],
                mutual_aid_partner_company_id=FOCAL,
                mutual_aid_capacity_offer_orders=0,
                mutual_aid_capacity_request_orders=(
                    focal.mutual_aid_capacity_offer_orders
                ),
            )
    coordination_partner = focal.price_coordination_partner_company_id
    target = focal.price_coordination_target_cents
    if coordination_partner in actions and target is not None:
        partner_price = (
            target
            if (state.episode_seed + state.round) % 3
            else max(7_000, target - 500)
        )
        actions[coordination_partner] = replace(
            actions[coordination_partner],
            price_cents=partner_price,
            price_coordination_partner_company_id=FOCAL,
            price_coordination_target_cents=target,
        )
    return actions


def _actual_outcome(
    config: MarketConfig,
    initial: MarketState,
    candidate_id: str,
    payload: Mapping[str, Any],
) -> dict[str, int]:
    env = MarketEnv(config)
    env.load_state(initial)
    initial_profit = initial.company(FOCAL).financial.cumulative_profit_cents
    initial_welfare = initial.strategic_market.cumulative_social_welfare_proxy_cents
    event_loss = 0
    fine = 0
    for offset in range(min(HORIZON, initial.rounds_remaining)):
        state = env.get_state()
        if offset == 0:
            focal = _market_action(state, FOCAL, candidate_id, payload)
            actions = _responsive_first_round(config, state, focal)
        else:
            actions = {
                company_id: build_rule_action(config, state, company_id)
                for company_id in state.company_ids
            }
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}",
            actions,
        )
        company = result.state_after.company(FOCAL)
        event_loss += company.financial.round_incident_cost_cents
        fine += int(company.financial.round_regulatory_fine_cents or 0)
        if result.done:
            break
    final = env.get_state()
    return {
        "enterprise_value_cents": _enterprise_value(final, FOCAL, config),
        "profit_delta_cents": (
            final.company(FOCAL).financial.cumulative_profit_cents
            - initial_profit
        ),
        "event_loss_cents": event_loss,
        "regulatory_fine_cents": fine,
        "social_welfare_delta_cents": (
            final.strategic_market.cumulative_social_welfare_proxy_cents
            - initial_welfare
        ),
        "state_hash_int": int(final.state_hash.rsplit(":", 1)[-1][-15:], 16),
    }


def _row(
    config: MarketConfig,
    advisor: PublicMarketRolloutAdvisor,
    personas: PersonaRegistry,
    seed: int,
    split: str,
    window: str,
    persona_id: str,
) -> dict[str, Any]:
    state, belief, opponent, observation = _build_window(config, seed, window)
    profile = personas.get(persona_id)
    advice_by_mode = {
        mode: advisor.advise(
            observation=observation,
            company_id=FOCAL,
            persona_profile=profile,
            belief_state=belief,
            opponent_model=opponent,
            horizon_rounds=HORIZON,
            scenario_count=SCENARIOS,
            advisor_mode=mode,
        )
        for mode in ("pareto_reliable_v7", "strategic_market_v9")
    }
    actual: dict[str, dict[str, dict[str, int]]] = {}
    for mode, advice in advice_by_mode.items():
        actual[mode] = {}
        for summary in advice.candidate_actions:
            candidate = summary.candidate
            actual[mode][candidate.candidate_id] = _actual_outcome(
                config,
                state,
                candidate.candidate_id,
                candidate.action.model_dump(mode="json"),
            )
    v7 = advice_by_mode["pareto_reliable_v7"]
    v9 = advice_by_mode["strategic_market_v9"]
    v7_planner = str(v7.planner_recommended_candidate_id)
    v9_planner = str(v9.planner_recommended_candidate_id)
    v7_best = max(
        actual["pareto_reliable_v7"],
        key=lambda key: (actual["pareto_reliable_v7"][key]["enterprise_value_cents"], key),
    )
    research_only = set(v9.research_only_candidate_ids or ())
    operational_v9 = {
        key: value
        for key, value in actual["strategic_market_v9"].items()
        if key not in research_only
    }
    v9_best = max(
        operational_v9,
        key=lambda key: (actual["strategic_market_v9"][key]["enterprise_value_cents"], key),
    )
    baseline = actual["strategic_market_v9"]["maintain"]
    v7_effective = v7.recommended_candidate_id or "maintain"
    v9_effective = v9.recommended_candidate_id or "maintain"
    return {
        "split": split,
        "seed": seed,
        "window": window,
        "persona_id": persona_id,
        "state_hash": state.state_hash,
        "v7_advice_hash": v7.advice_hash,
        "v9_advice_hash": v9.advice_hash,
        "v7_planner_candidate_id": v7_planner,
        "v9_planner_candidate_id": v9_planner,
        "v7_effective_candidate_id": v7.recommended_candidate_id,
        "v9_effective_candidate_id": v9.recommended_candidate_id,
        "v7_best_actual_candidate_id": v7_best,
        "v9_best_actual_candidate_id": v9_best,
        "v7_planner_regret_cents": (
            actual["strategic_market_v9"][v9_best]["enterprise_value_cents"]
            - actual["pareto_reliable_v7"][v7_planner]["enterprise_value_cents"]
        ),
        "v9_planner_regret_cents": (
            actual["strategic_market_v9"][v9_best]["enterprise_value_cents"]
            - actual["strategic_market_v9"][v9_planner]["enterprise_value_cents"]
        ),
        "expanded_best_gain_cents": (
            actual["strategic_market_v9"][v9_best]["enterprise_value_cents"]
            - actual["pareto_reliable_v7"][v7_best]["enterprise_value_cents"]
        ),
        "v9_planner_minus_v7_planner_ev_cents": (
            actual["strategic_market_v9"][v9_planner]["enterprise_value_cents"]
            - actual["pareto_reliable_v7"][v7_planner]["enterprise_value_cents"]
        ),
        "v7_deployed_minus_baseline_ev_cents": (
            actual["pareto_reliable_v7"][v7_effective]["enterprise_value_cents"]
            - baseline["enterprise_value_cents"]
        ),
        "v9_deployed_minus_baseline_ev_cents": (
            actual["strategic_market_v9"][v9_effective]["enterprise_value_cents"]
            - baseline["enterprise_value_cents"]
        ),
        "v9_candidate_ids": sorted(actual["strategic_market_v9"]),
        "v9_planner_outcome": actual["strategic_market_v9"][v9_planner],
        "baseline_outcome": baseline,
        "uses_hidden_opponent_state": bool(
            v9.uses_hidden_opponent_state
            or v9.uses_authoritative_hidden_market_state
        ),
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(int(row[field]) for row in rows) // len(rows)


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    strategic_labels = {
        "threshold_project_contribution",
        "mutual_aid_request",
        "mutual_aid_offer",
        "price_coordination_honor",
        "price_coordination_undercut",
    }
    return {
        "row_count": len(rows),
        "mean_v7_planner_regret_cents": _mean(rows, "v7_planner_regret_cents"),
        "mean_v9_planner_regret_cents": _mean(rows, "v9_planner_regret_cents"),
        "mean_v9_minus_v7_planner_ev_cents": _mean(
            rows, "v9_planner_minus_v7_planner_ev_cents"
        ),
        "mean_expanded_best_gain_cents": _mean(rows, "expanded_best_gain_cents"),
        "mean_v9_deployed_minus_baseline_ev_cents": _mean(
            rows, "v9_deployed_minus_baseline_ev_cents"
        ),
        "worst_v9_deployed_minus_baseline_ev_cents": min(
            int(row["v9_deployed_minus_baseline_ev_cents"]) for row in rows
        ),
        "v9_release_count": sum(
            row["v9_effective_candidate_id"] is not None for row in rows
        ),
        "strategic_planner_choice_count": sum(
            str(row["v9_planner_candidate_id"]) in strategic_labels for row in rows
        ),
        "expanded_opportunity_count": sum(
            int(row["expanded_best_gain_cents"]) > 0 for row in rows
        ),
        "hidden_state_leak_count": sum(
            bool(row["uses_hidden_opponent_state"]) for row in rows
        ),
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    advisor = PublicMarketRolloutAdvisor(config)
    rows = [
        _row(config, advisor, personas, seed, split, window, persona_id)
        for split, seeds in (
            ("development", DEVELOPMENT_SEEDS),
            ("holdout", HOLDOUT_SEEDS),
        )
        for seed in seeds
        for window in WINDOWS
        for persona_id in PERSONAS
    ]
    development = [row for row in rows if row["split"] == "development"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    development_metrics = _metrics(development)
    holdout_metrics = _metrics(holdout)
    gates = {
        "development_holdout_disjoint": not (
            set(DEVELOPMENT_SEEDS) & set(HOLDOUT_SEEDS)
        ),
        "at_least_ten_common_seeds": (
            len(DEVELOPMENT_SEEDS) + len(HOLDOUT_SEEDS) >= 10
        ),
        "final_market_candidates_present": all(
            "price_coordination_honor" in row["v9_candidate_ids"]
            and "price_coordination_undercut" in row["v9_candidate_ids"]
            and (
                "mutual_aid_offer" in row["v9_candidate_ids"]
                or "mutual_aid_request" in row["v9_candidate_ids"]
            )
            for row in rows
        ) and all(
            "threshold_project_contribution" in row["v9_candidate_ids"]
            for row in rows
            if row["window"] == "pivotal_project"
        ),
        "holdout_hidden_state_leak_zero": (
            holdout_metrics["hidden_state_leak_count"] == 0
        ),
        "holdout_expanded_action_has_value": (
            holdout_metrics["expanded_opportunity_count"] > 0
            and holdout_metrics["mean_expanded_best_gain_cents"] >= 0
        ),
        "holdout_planner_regret_noninferior": (
            holdout_metrics["mean_v9_planner_regret_cents"]
            <= holdout_metrics["mean_v7_planner_regret_cents"]
        ),
        "holdout_planner_value_noninferior": (
            holdout_metrics["mean_v9_minus_v7_planner_ev_cents"] >= 0
        ),
        "holdout_deployed_fail_closed": (
            holdout_metrics["mean_v9_deployed_minus_baseline_ev_cents"] >= 0
            and holdout_metrics["worst_v9_deployed_minus_baseline_ev_cents"] >= 0
        ),
        "holdout_actionable_release": holdout_metrics["v9_release_count"] > 0,
    }
    summary = {
        "result_schema_version": "final-market-advisor-v9-validation-v1.0.0",
        "config_id": config.config_id,
        "config_sha256": config.config_sha256,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "windows": list(WINDOWS),
        "personas": list(PERSONAS),
        "horizon_rounds": HORIZON,
        "scenario_count": SCENARIOS,
        "provider_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "development_metrics": development_metrics,
        "holdout_metrics": holdout_metrics,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    summary["result_hash"] = sha256_hash(summary)
    output.mkdir(parents=True, exist_ok=True)
    (output / "rows.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
