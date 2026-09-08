from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.belief import BeliefLedger
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.opponent import OpponentModelLedger
from game_theory_agent.experiments.final_market_advisor_v9_validation import (
    FOCAL,
    REAL_CORE_WINDOWS,
    _build_window,
)
from game_theory_agent.strategic_reliability import (
    PublicMarketRolloutAdvisor,
    PublicStrategicAdvice,
    build_public_forecast_state,
)


ROOT = Path(__file__).resolve().parents[1]
COMPANIES = ("company_A", "company_B", "company_C", "company_D")


def _inputs(seed: int = 42):
    config = load_market_config(ROOT / "configs" / "market_v6_final.yaml")
    state = MarketEnv(config).reset(
        COMPANIES,
        episode_id=f"v9-test-{seed}",
        episode_seed=seed,
        market_model="balanced",
        max_rounds=10,
        cooperation_mode="combined_v1",
    )
    belief, belief_hash = BeliefLedger(
        episode_id=state.episode_id,
        company_ids=state.company_ids,
    ).company_view(
        observer_company_id="company_A",
        round_number=state.round,
        state_version=state.state_version,
    )
    opponent, opponent_hash = OpponentModelLedger(
        episode_id=state.episode_id,
        company_ids=state.company_ids,
    ).company_view(
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
    observation["opponent_model_state"] = opponent.model_dump(mode="json")
    observation["opponent_model_hash"] = opponent_hash
    profile = PersonaRegistry.from_market_config(config).get("balanced_v1")
    return config, state, observation, belief, opponent, profile


def test_v9_public_forecast_preserves_final_market_protocols():
    config, state, observation, belief, opponent, profile = _inputs()
    forecast, record = build_public_forecast_state(
        config=config,
        observation=observation,
        company_id="company_A",
        persona_profile=profile,
        belief_state=belief,
        opponent_model=opponent,
    )

    assert forecast.strategic_market == state.strategic_market
    assert forecast.shared_resilience == state.shared_resilience
    assert not record.uses_authoritative_hidden_market_state


def test_v9_candidates_cover_final_market_and_are_tamper_evident():
    config, _state, observation, belief, opponent, profile = _inputs()
    advisor = PublicMarketRolloutAdvisor(config)
    advice = advisor.advise(
        observation=observation,
        company_id="company_A",
        persona_profile=profile,
        belief_state=belief,
        opponent_model=opponent,
        horizon_rounds=2,
        scenario_count=2,
        advisor_mode="strategic_market_v9",
    )
    repeated = advisor.advise(
        observation=observation,
        company_id="company_A",
        persona_profile=profile,
        belief_state=belief,
        opponent_model=opponent,
        horizon_rounds=2,
        scenario_count=2,
        advisor_mode="strategic_market_v9",
    )

    assert advice == repeated
    assert advice.advice_schema_version == "public-strategic-market-advice-v9.0.0"
    assert advice.allowed_in_public_agent_context
    assert not advice.uses_hidden_opponent_state
    ids = {item.candidate.candidate_id for item in advice.candidate_actions}
    assert "threshold_project_contribution" in ids
    assert "mutual_aid_offer" in ids or "mutual_aid_request" in ids
    assert "price_coordination_honor" in ids
    assert "price_coordination_undercut" in ids
    assert set(advice.research_only_candidate_ids or ()) == {
        "price_coordination_honor",
        "price_coordination_undercut",
    }
    executable_ids = {
        item["candidate_id"]
        for item in advice.pareto_decision["candidate_assessments"]
    }
    assert not executable_ids.intersection(advice.research_only_candidate_ids or ())
    assert advice.final_market_gate_policy[
        "price_coordination_is_research_only"
    ]

    forged = deepcopy(advice.model_dump(mode="json"))
    coordination = next(
        item
        for item in forged["candidate_actions"]
        if item["candidate"]["candidate_id"] == "price_coordination_honor"
    )
    coordination["candidate"]["action"]["price_cents"] += 1
    with pytest.raises(ValidationError):
        PublicStrategicAdvice.model_validate(forged)


def test_real_core_windows_are_replay_stable_and_mechanically_eligible():
    config = load_market_config(ROOT / "configs" / "market_v6_final.yaml")
    for seed in (95101, 95102, 95104):
        for window in REAL_CORE_WINDOWS:
            first = _build_window(config, seed, window)[0]
            repeated = _build_window(config, seed, window)[0]
            assert first.state_hash == repeated.state_hash
            assert FOCAL in first.strategic_market.active_company_ids
            if window == "distress":
                assert first.company(FOCAL).financial.cash_balance_cents == 4_000_000
            if window == "price_war":
                assert min(
                    first.company(company_id).commercial.price_cents
                    for company_id in ("company_B", "company_C")
                ) < first.company(FOCAL).commercial.price_cents
