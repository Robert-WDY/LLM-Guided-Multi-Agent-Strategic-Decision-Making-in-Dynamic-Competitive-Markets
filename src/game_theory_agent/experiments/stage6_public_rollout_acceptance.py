"""Zero-LLM Stage 6.1 public-rollout policy acceptance experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

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
from game_theory_agent.opponent import OpponentModelLedger
from game_theory_agent.strategic_reliability import (
    AuthoritativeMarketRolloutEvaluator,
    PublicMarketRolloutAdvisor,
    generate_public_overlay_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v4.yaml"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "runs"
    / "stage6-public-rollout-v1"
    / "summary.json"
)
COMPANIES = ("company_A", "company_B", "company_C", "company_D")


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty unique list")
    return seeds


def _enterprise_value(state: MarketState, company_id: str, config) -> int:
    return next(
        int(item["value_cents"])
        for item in build_terminal_rankings(state, config)["composite"]
        if item["company_id"] == company_id
    )


def _actual_action(advice, state: MarketState, company_id: str) -> CompanyAction:
    payload = advice.recommended_action
    return CompanyAction(
        action_id=(
            f"public-rollout-policy:{state.episode_id}:{state.round}:{company_id}"
        ),
        episode_id=state.episode_id,
        agent_id=company_id,
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
        strategy_summary=(
            f"公开预测市场建议：{advice.recommended_candidate_id}"
        ),
    )


def _run_rule_baseline(config, *, seed: int, persona_id: str) -> dict[str, Any]:
    env = MarketEnv(config)
    state = env.reset(
        COMPANIES,
        episode_id=f"stage6-public-baseline-{seed}-{persona_id}",
        episode_seed=seed,
        market_model="balanced",
        max_rounds=10,
    )
    while not state.terminal:
        actions = {
            company_id: build_rule_action(config, state, company_id)
            for company_id in state.company_ids
        }
        state = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", actions
        ).state_after
    company = state.company("company_A")
    return {
        "enterprise_value_cents": _enterprise_value(state, "company_A", config),
        "cumulative_profit_cents": company.financial.cumulative_profit_cents,
        "final_market_share_ppm": company.commercial.market_share_ppm,
        "final_cash_cents": company.financial.cash_balance_cents,
        "final_state_hash": state.state_hash,
    }


def _run_public_policy(
    config,
    *,
    seed: int,
    persona_id: str,
    horizon_rounds: int,
    scenario_count: int,
) -> dict[str, Any]:
    env = MarketEnv(config)
    state = env.reset(
        COMPANIES,
        episode_id=f"stage6-public-treatment-{seed}-{persona_id}",
        episode_seed=seed,
        market_model="balanced",
        max_rounds=10,
    )
    registry = PersonaRegistry.from_market_config(config)
    profile = registry.get(persona_id)
    public_advisor = PublicMarketRolloutAdvisor(config)
    oracle = AuthoritativeMarketRolloutEvaluator(config)
    tracker = PersonaUtilityTracker(registry.evaluator(profile))
    belief_ledger = BeliefLedger(
        episode_id=state.episode_id,
        company_ids=state.company_ids,
    )
    opponent_ledger = OpponentModelLedger(
        episode_id=state.episode_id,
        company_ids=state.company_ids,
    )
    chosen_oracle_regrets: list[int] = []
    maintain_oracle_regrets: list[int] = []
    recommendations: list[str] = []
    realized_loss = 0
    all_actions_legal = True
    all_advice_public_safe = True

    while not state.terminal:
        belief, belief_hash = belief_ledger.company_view(
            observer_company_id="company_A",
            round_number=state.round,
            state_version=state.state_version,
        )
        opponent, opponent_hash = opponent_ledger.company_view(
            observer_company_id="company_A",
            round_number=state.round,
            state_version=state.state_version,
        )
        observation = ObservationBuilder().build(
            state,
            "company_A",
            "public",
            belief_state=belief.model_dump(mode="json"),
            belief_hash=belief_hash,
            belief_schema_version=belief.belief_schema_version,
        )
        observation["opponent_model_hash"] = opponent_hash
        observation["opponent_model_state"] = opponent.model_dump(mode="json")
        advice = public_advisor.advise(
            observation=observation,
            company_id="company_A",
            persona_profile=profile,
            belief_state=belief,
            opponent_model=opponent,
            horizon_rounds=min(horizon_rounds, state.rounds_remaining),
            scenario_count=scenario_count,
        )
        all_advice_public_safe = all_advice_public_safe and (
            advice.allowed_in_public_agent_context
            and not advice.uses_authoritative_hidden_market_state
            and not advice.uses_hidden_opponent_state
        )
        recommendations.append(advice.recommended_candidate_id)

        authoritative = oracle.evaluate(
            state=state,
            company_id="company_A",
            persona_profile=profile,
            opponent_model=opponent,
            belief_state=belief,
            horizon_rounds=min(horizon_rounds, state.rounds_remaining),
            scenario_count=scenario_count,
            candidates=generate_public_overlay_candidates(
                config, state, "company_A"
            ),
        )
        oracle_by_id = {
            item.candidate.candidate_id: item
            for item in authoritative.evaluations
        }
        oracle_best = oracle_by_id[authoritative.recommended_candidate_id]
        oracle_chosen = oracle_by_id[advice.recommended_candidate_id]
        oracle_maintain = oracle_by_id["maintain"]
        chosen_oracle_regrets.append(
            max(
                0,
                oracle_best.certainty_equivalent_value_cents
                - oracle_chosen.certainty_equivalent_value_cents,
            )
        )
        maintain_oracle_regrets.append(
            max(
                0,
                oracle_best.certainty_equivalent_value_cents
                - oracle_maintain.certainty_equivalent_value_cents,
            )
        )

        actions = {
            company_id: build_rule_action(config, state, company_id)
            for company_id in state.company_ids
        }
        actions["company_A"] = _actual_action(advice, state, "company_A")
        validation = env.validate_action(actions["company_A"], "company_A")
        all_actions_legal = all_actions_legal and validation.valid
        if not validation.valid or validation.action is None:
            raise ValueError(f"public rollout produced illegal action: {validation.errors}")
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
        "enterprise_value_cents": _enterprise_value(state, "company_A", config),
        "cumulative_profit_cents": company.financial.cumulative_profit_cents,
        "final_market_share_ppm": company.commercial.market_share_ppm,
        "final_cash_cents": company.financial.cash_balance_cents,
        "realized_loss_cents": realized_loss,
        "mean_chosen_oracle_regret_cents": round(mean(chosen_oracle_regrets)),
        "mean_maintain_oracle_regret_cents": round(mean(maintain_oracle_regrets)),
        "recommendations": recommendations,
        "all_actions_legal": all_actions_legal,
        "all_advice_public_safe": all_advice_public_safe,
        "final_state_hash": state.state_hash,
    }


def run(
    *,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    persona_id: str = "balanced_v1",
    horizon_rounds: int = 3,
    scenario_count: int = 3,
) -> dict[str, Any]:
    config = load_market_config(CONFIG_PATH)
    PersonaRegistry.from_market_config(config).get(persona_id)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        baseline = _run_rule_baseline(config, seed=seed, persona_id=persona_id)
        treatment = _run_public_policy(
            config,
            seed=seed,
            persona_id=persona_id,
            horizon_rounds=horizon_rounds,
            scenario_count=scenario_count,
        )
        rows.append(
            {
                "seed": seed,
                "baseline": baseline,
                "public_rollout": treatment,
                "enterprise_value_delta_cents": (
                    treatment["enterprise_value_cents"]
                    - baseline["enterprise_value_cents"]
                ),
                "cumulative_profit_delta_cents": (
                    treatment["cumulative_profit_cents"]
                    - baseline["cumulative_profit_cents"]
                ),
            }
        )
    mean_ev_delta = round(mean(row["enterprise_value_delta_cents"] for row in rows))
    mean_profit_delta = round(
        mean(row["cumulative_profit_delta_cents"] for row in rows)
    )
    mean_chosen_regret = round(
        mean(
            row["public_rollout"]["mean_chosen_oracle_regret_cents"]
            for row in rows
        )
    )
    mean_maintain_regret = round(
        mean(
            row["public_rollout"]["mean_maintain_oracle_regret_cents"]
            for row in rows
        )
    )
    engineering_checks = {
        "real_llm_calls_zero": True,
        "real_llm_tokens_zero": True,
        "real_llm_estimated_cost_zero": True,
        "all_recommended_actions_legal": all(
            row["public_rollout"]["all_actions_legal"] for row in rows
        ),
        "all_advice_public_context_safe": all(
            row["public_rollout"]["all_advice_public_safe"] for row in rows
        ),
        "all_rounds_completed": all(
            len(row["public_rollout"]["recommendations"]) == 10 for row in rows
        ),
    }
    promotion_gates = {
        "mean_authoritative_regret_not_above_maintain": (
            mean_chosen_regret <= mean_maintain_regret
        ),
        "mean_enterprise_value_not_below_rule_baseline": mean_ev_delta >= 0,
        "worst_seed_enterprise_value_not_below_rule_baseline": min(
            row["enterprise_value_delta_cents"] for row in rows
        )
        >= 0,
    }
    return {
        "experiment_schema_version": "stage6-public-rollout-acceptance-v1.0.0",
        "evidence_level": "deterministic_engineering_and_rule_policy_benchmark",
        "seeds": list(seeds),
        "persona_id": persona_id,
        "rounds": 10,
        "horizon_rounds": horizon_rounds,
        "scenario_count": scenario_count,
        "real_model_usage": {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_cost": 0,
            "currency": "CNY",
        },
        "rows": rows,
        "aggregate": {
            "mean_enterprise_value_delta_cents": mean_ev_delta,
            "mean_cumulative_profit_delta_cents": mean_profit_delta,
            "mean_chosen_oracle_regret_cents": mean_chosen_regret,
            "mean_maintain_oracle_regret_cents": mean_maintain_regret,
            "advisor_regret_reduction_cents": (
                mean_maintain_regret - mean_chosen_regret
            ),
        },
        "engineering_checks": engineering_checks,
        "engineering_passed": all(engineering_checks.values()),
        "promotion_gates": promotion_gates,
        "promotion_recommended": all(promotion_gates.values()),
        "conclusion_limits": [
            "Rule baseline is not a Persona-only LLM baseline",
            "five seeds are an engineering gate, not statistical evidence",
            "authoritative regret is an offline label and never enters Agent context",
            "no real model was called and no LLM behavioral claim is made",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--persona-id", default="balanced_v1")
    parser.add_argument("--horizon-rounds", type=int, default=3)
    parser.add_argument("--scenario-count", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(
        seeds=parse_seeds(args.seeds),
        persona_id=args.persona_id,
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
                "promotion_recommended": summary["promotion_recommended"],
                "real_model_usage": summary["real_model_usage"],
                "aggregate": summary["aggregate"],
                "promotion_gates": summary["promotion_gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
