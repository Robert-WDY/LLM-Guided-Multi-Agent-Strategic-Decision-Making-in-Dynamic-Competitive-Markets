from pathlib import Path

from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.market import CompanyAction, MarketEnv, Persona
from game_theory_agent.market.config import load_market_config
from game_theory_agent.strategic_reliability.rollout import (
    AuthoritativeMarketRolloutEvaluator,
    generate_candidate_actions,
)


ROOT = Path(__file__).resolve().parents[1]


def test_public_v10_forecast_preserves_published_supply_and_welfare_after_settlement():
    from game_theory_agent.experiments.final_market_advisor_v9_validation import _build_window
    from game_theory_agent.strategic_reliability.public_rollout import build_public_forecast_state, generate_public_overlay_candidates
    config = load_market_config(ROOT / "configs/market_v10_multi_objective.yaml")
    state, belief, opponent, observation = _build_window(config, 120801, "pivotal_project")
    registry = PersonaRegistry.from_market_config(config)
    forecast, _ = build_public_forecast_state(config=config, observation=observation, company_id="company_A", persona_profile=registry.get("public_service"), belief_state=belief, opponent_model=opponent)
    MarketEnv(config).load_state(forecast)
    assert forecast.supply_chain.to_dict() == observation["public_state"]["market"]["supply_chain"]
    assert forecast.welfare_accounting.to_dict() == observation["public_state"]["market"]["welfare_accounting"]
    candidates = {c.candidate_id:c for c in generate_public_overlay_candidates(config, forecast, "company_A")}
    assert candidates["sourcing_diverse"].action.primary_supplier_share_ppm == 500000
    assert candidates["sourcing_stable"].action.primary_supplier_id == "resilient_supplier"


def test_social_objective_profiles_are_executable_and_market_backed() -> None:
    config = load_market_config(ROOT / "configs/market_v10_multi_objective.yaml")
    registry = PersonaRegistry.from_market_config(config)
    env = MarketEnv(config)
    before = env.reset(
        episode_id="multi-objective-test",
        episode_seed=101,
        market_model="balanced",
        max_rounds=20,
        personas={"company_A": Persona.PUBLIC_SERVICE},
    )
    actions = {
        company_id: CompanyAction(
            action_id=f"multi:{company_id}",
            episode_id=before.episode_id,
            agent_id=company_id,
            round=before.round,
            state_version=before.state_version,
            price_cents=9_000,
            primary_supplier_id="economy_supplier",
            backup_supplier_id="resilient_supplier",
            primary_supplier_share_ppm=500_000,
        )
        for company_id in before.company_ids
    }
    after = env.step(
        f"{before.episode_id}:{before.round}:{before.state_version}", actions
    ).state_after
    assessment = registry.evaluator(registry.get("public_service")).evaluate(
        before, after, "company_A"
    )
    assert assessment.social_welfare_available
    assert assessment.component_scores_ppm["social_welfare"] != 0


def test_authoritative_rollout_contains_supply_chain_candidates() -> None:
    config = load_market_config(ROOT / "configs/market_v10_multi_objective.yaml")
    registry = PersonaRegistry.from_market_config(config)
    env = MarketEnv(config)
    state = env.reset(
        episode_id="multi-objective-rollout",
        episode_seed=102,
        market_model="balanced",
        max_rounds=20,
        personas={"company_A": Persona.RESILIENCE_STEWARD},
    )
    ids = {item.candidate_id for item in generate_candidate_actions(config, state, "company_A")}
    assert {"sourcing_stable", "sourcing_diverse"} <= ids
    plan = AuthoritativeMarketRolloutEvaluator(config).evaluate(
        state=state,
        company_id="company_A",
        persona_profile=registry.get("resilience_steward"),
        horizon_rounds=2,
        scenario_count=2,
    )
    assert plan.persona_id == "resilience_steward"
