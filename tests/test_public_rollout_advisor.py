from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.agents.context import DecisionContextBuilder
from game_theory_agent.agents.memory import EpisodeMemory
from game_theory_agent.advisor import verify_advisor_replay
from game_theory_agent.api import SESSIONS, agent_app, app
from game_theory_agent.belief import BeliefLedger
from game_theory_agent.experiments.stage6_public_rollout_acceptance import run
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import FinancialState, MarketEnv
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.information import ObservationSnapshot
from game_theory_agent.model_clients import MockModelClient
from game_theory_agent.opponent import OpponentModelLedger
from game_theory_agent.strategic_reliability import (
    PROMOTION_EVIDENCE_SHA256,
    PublicMarketRolloutAdvisor,
    PublicStrategicAdvice,
    build_public_forecast_state,
    generate_public_overlay_candidates,
)


COMPANIES = ("company_A", "company_B", "company_C", "company_D")


def _inputs(config, *, seed: int = 42, episode_id: str = "public-rollout-test"):
    state = MarketEnv(config).reset(
        COMPANIES,
        episode_id=episode_id,
        episode_seed=seed,
        market_model="balanced",
        max_rounds=10,
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
    observation["opponent_model_hash"] = opponent_hash
    observation["opponent_model_state"] = opponent.model_dump(mode="json")
    return state, observation, belief, opponent


def test_public_forecast_estimates_hidden_opponent_fields(config):
    state, observation, belief, opponent = _inputs(config)
    profile = PersonaRegistry.from_market_config(config).get("balanced_v1")
    forecast, record = build_public_forecast_state(
        config=config,
        observation=observation,
        company_id="company_A",
        persona_profile=profile,
        belief_state=belief,
        opponent_model=opponent,
    )
    assert forecast.company("company_A") == state.company("company_A")
    assert forecast.company("company_B").persona.value == "none"
    assert forecast.company("company_B").risk.active_incident is None
    assert record.uses_authoritative_hidden_market_state is False
    assert record.hidden_opponent_fields_are_estimated is True
    assert "state_hash" not in record.public_decision_input_hash


def test_public_forecast_uses_common_random_numbers_across_treatments(config):
    _state, observation, belief, opponent = _inputs(config)
    registry = PersonaRegistry.from_market_config(config)
    bare_observation = deepcopy(observation)
    bare_observation.pop("belief_state", None)
    bare_observation.pop("belief_hash", None)
    bare_observation.pop("belief_schema_version", None)
    bare_observation.pop("opponent_model_state", None)
    bare_observation.pop("opponent_model_hash", None)

    baseline, baseline_record = build_public_forecast_state(
        config=config,
        observation=bare_observation,
        company_id="company_A",
        persona_profile=registry.get("profit_myopic"),
    )
    treatment, treatment_record = build_public_forecast_state(
        config=config,
        observation=observation,
        company_id="company_A",
        persona_profile=registry.get("aggressive_v1_extreme"),
        belief_state=belief,
        opponent_model=opponent,
    )

    assert baseline_record.public_decision_input_hash != (
        treatment_record.public_decision_input_hash
    )
    assert baseline_record.forecast_seed == treatment_record.forecast_seed
    assert baseline.state_hash == treatment.state_hash


def test_public_candidates_preserve_operating_baseline(config):
    state, _observation, _belief, _opponent = _inputs(config)
    baseline = build_rule_action(config, state, "company_A")
    candidates = generate_public_overlay_candidates(config, state, "company_A")
    maintain = next(item for item in candidates if item.candidate_id == "maintain")
    assert maintain.action.price_cents == baseline.price_cents
    assert (
        maintain.action.advertising_budget_cents
        == baseline.advertising_budget_cents
    )
    assert maintain.action.service_budget_cents == baseline.service_budget_cents
    price_up = next(
        item for item in candidates if item.candidate_id == "price_increase"
    )
    assert (
        price_up.action.advertising_budget_cents
        == baseline.advertising_budget_cents
    )
    assert price_up.action.service_budget_cents == baseline.service_budget_cents


def test_hidden_opponent_mutation_cannot_change_public_advice(config):
    state, observation, belief, opponent = _inputs(config)
    company_b = state.company("company_B")
    hidden_b = replace(
        company_b,
        financial=FinancialState(
            **{
                **company_b.financial.to_dict(),
                "cash_balance_cents": 1,
                "cumulative_profit_cents": -17_000_000,
            }
        ),
    )
    hidden_state = replace(
        state,
        companies=tuple(
            hidden_b if item.company_id == "company_B" else item
            for item in state.companies
        ),
        state_hash="",
    )
    hidden_state = replace(
        hidden_state, state_hash=state_hash(hidden_state.to_dict())
    )
    hidden_observation = ObservationBuilder().build(
        hidden_state,
        "company_A",
        "public",
        belief_state=belief.model_dump(mode="json"),
        belief_hash="same-public-belief",
        belief_schema_version=belief.belief_schema_version,
    )
    hidden_observation["opponent_model_state"] = opponent.model_dump(mode="json")
    advisor = PublicMarketRolloutAdvisor(config)
    profile = PersonaRegistry.from_market_config(config).get("balanced_v1")
    first = advisor.advise(
        observation=observation,
        company_id="company_A",
        persona_profile=profile,
        belief_state=belief,
        opponent_model=opponent,
        horizon_rounds=2,
        scenario_count=2,
    )
    second = advisor.advise(
        observation=hidden_observation,
        company_id="company_A",
        persona_profile=profile,
        belief_state=belief,
        opponent_model=opponent,
        horizon_rounds=2,
        scenario_count=2,
    )
    assert first == second


def test_public_advice_is_deterministic_non_binding_and_tamper_evident(config):
    _state, observation, belief, opponent = _inputs(config)
    profile = PersonaRegistry.from_market_config(config).get("risk_guarded_v1")
    advisor = PublicMarketRolloutAdvisor(config)
    first = advisor.advise(
        observation=observation,
        company_id="company_A",
        persona_profile=profile,
        belief_state=belief,
        opponent_model=opponent,
        horizon_rounds=2,
        scenario_count=2,
    )
    repeated = advisor.advise(
        observation=observation,
        company_id="company_A",
        persona_profile=profile,
        belief_state=belief,
        opponent_model=opponent,
        horizon_rounds=2,
        scenario_count=2,
    )
    assert first == repeated
    assert first.allowed_in_public_agent_context
    assert not first.uses_authoritative_hidden_market_state
    assert not first.uses_hidden_opponent_state
    assert first.recommendation_is_non_binding
    assert not first.claims_nash_equilibrium
    forged = deepcopy(first.model_dump(mode="json"))
    forged["recommended_action"]["price_cents"] += 1
    with pytest.raises(ValidationError):
        PublicStrategicAdvice.model_validate(forged)


def test_promoted_pareto_advice_is_public_audited_and_tamper_evident(config):
    _state, observation, belief, opponent = _inputs(config)
    profile = PersonaRegistry.from_market_config(config).get(
        "aggressive_v1_extreme"
    )
    advice = PublicMarketRolloutAdvisor(config).advise(
        observation=observation,
        company_id="company_A",
        persona_profile=profile,
        belief_state=belief,
        opponent_model=opponent,
        horizon_rounds=2,
        scenario_count=2,
        advisor_mode="pareto_rollout_v4",
    )

    assert advice.advisor_mode == "pareto_rollout_v4"
    assert advice.advice_schema_version == "public-pareto-advice-v4.0.0"
    assert advice.allowed_in_public_agent_context
    assert advice.promotion_evidence_sha256 == PROMOTION_EVIDENCE_SHA256
    assert advice.pareto_decision is not None
    assert advice.selection_situation in {
        "protect_lead",
        "catch_up_early",
        "catch_up_late",
    }
    assert not advice.uses_authoritative_hidden_market_state
    assert not advice.uses_hidden_opponent_state

    forged = deepcopy(advice.model_dump(mode="json"))
    forged["pareto_decision"]["recommended_candidate_id"] = "forged"
    with pytest.raises(ValidationError):
        PublicStrategicAdvice.model_validate(forged)


def test_reliable_pareto_advice_has_conservative_coverage_and_audited_gate(config):
    _state, observation, belief, opponent = _inputs(config)
    profile = PersonaRegistry.from_market_config(config).get("risk_guarded_v1")
    advisor = PublicMarketRolloutAdvisor(config)
    advice = advisor.advise(
        observation=observation,
        company_id="company_A",
        persona_profile=profile,
        belief_state=belief,
        opponent_model=opponent,
        horizon_rounds=2,
        scenario_count=2,
        advisor_mode="pareto_reliable_v5",
    )
    repeated = advisor.advise(
        observation=observation,
        company_id="company_A",
        persona_profile=profile,
        belief_state=belief,
        opponent_model=opponent,
        horizon_rounds=2,
        scenario_count=2,
        advisor_mode="pareto_reliable_v5",
    )

    assert advice == repeated
    assert advice.advice_schema_version == "public-pareto-reliable-advice-v5.0.0"
    assert advice.planner_recommended_candidate_id is not None
    assert advice.reliability_gate is not None
    candidate_ids = {
        item.candidate.candidate_id for item in advice.candidate_actions
    }
    # Equivalent conservative actions are deliberately de-duplicated.  The
    # risk buffer remains distinct, while status/profit recovery may equal the
    # ordinary maintain action in a clean initial state.
    assert "risk_buffer" in candidate_ids
    assert candidate_ids.intersection({"status_quo", "profit_recovery", "maintain"})
    gate = advice.reliability_gate
    safe_ids = set(gate["safe_candidate_ids"])
    excluded_ids = {
        item["candidate_id"] for item in gate["excluded_candidates"]
    }
    assert advice.recommended_candidate_id == gate["effective_candidate_id"]
    assert advice.recommended_candidate_id in safe_ids
    assert safe_ids.isdisjoint(excluded_ids)
    assert gate["uses_only_public_and_own_private_inputs"]
    assert not gate["uses_authoritative_hidden_market_state"]

    forged = deepcopy(advice.model_dump(mode="json"))
    forged["reliability_gate"]["advisor_confidence_ppm"] += 1
    with pytest.raises(ValidationError):
        PublicStrategicAdvice.model_validate(forged)


def test_public_advisor_rejects_perfect_information_observation(config):
    state, _observation, belief, opponent = _inputs(config)
    perfect = ObservationBuilder().build(state, "company_A", "perfect")
    with pytest.raises(ValueError, match="public-information"):
        PublicMarketRolloutAdvisor(config).advise(
            observation=perfect,
            company_id="company_A",
            persona_profile=PersonaRegistry.from_market_config(config).get(
                "balanced_v1"
            ),
            belief_state=belief,
            opponent_model=opponent,
            horizon_rounds=1,
            scenario_count=1,
        )


def test_public_rollout_v3_api_observation(monkeypatch):
    controller_token = "public-rollout-controller"
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", controller_token)
    SESSIONS.clear()
    controller = TestClient(app)
    gateway = TestClient(agent_app)
    created = controller.post(
        "/api/episodes",
        headers={"X-Controller-Token": controller_token},
        json={
            "episode_id": "public-rollout-api",
            "episode_seed": 91,
            "max_rounds": 5,
            "information_mode": "public",
            "belief_mode": "public_action_v1",
            "opponent_model_mode": "public_strategy_v1",
            "advisor_mode": "public_rollout_v3",
            "agent_configs": {
                "company_A": {"persona": {"persona_id": "risk_guarded_v1"}}
            },
        },
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["manifest"]["advisor_schema_version"] == (
        "public-strategic-advice-v3.0.0"
    )
    observation = gateway.get(
        "/v1/episodes/public-rollout-api/companies/company_A/observation",
        headers={"X-Agent-Token": body["agent_tokens"]["company_A"]},
    )
    assert observation.status_code == 200, observation.text
    observation_payload = observation.json()
    advice = PublicStrategicAdvice.model_validate(
        observation_payload["game_theory_advice"]
    )
    assert advice.persona_id == "risk_guarded_v1"
    assert advice.allowed_in_public_agent_context
    assert advice.scenario_count == 5
    context = DecisionContextBuilder().build(
        observation_payload, "company_A", EpisodeMemory()
    )
    generation = asyncio.run(
        MockModelClient(honor_game_theory_advice=True).generate_decision(context)
    )
    requested = generation.parsed_output["requested_action"]
    assert requested["price_cents"] == advice.recommended_action["price_cents"]
    assert requested["advertising_budget_cents"] == advice.recommended_action[
        "advertising_budget_cents"
    ]
    assert requested["service_budget_cents"] == advice.recommended_action[
        "service_budget_cents"
    ]
    snapshot = ObservationSnapshot.from_observation(
        observation_payload, "company_A"
    )
    event = SimpleNamespace(
        communication_phase=None,
        traces=[SimpleNamespace(information_snapshot=snapshot)],
    )
    replayed = verify_advisor_replay(
        [event], SimpleNamespace(advisor_mode="public_rollout_v3")
    )
    assert replayed == (advice,)


def test_public_rollout_v3_api_rejects_perfect_information(monkeypatch):
    controller_token = "public-rollout-controller-invalid"
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", controller_token)
    SESSIONS.clear()
    response = TestClient(app).post(
        "/api/episodes",
        headers={"X-Controller-Token": controller_token},
        json={
            "episode_id": "public-rollout-invalid",
            "information_mode": "perfect",
            "belief_mode": "public_action_v1",
            "opponent_model_mode": "public_strategy_v1",
            "advisor_mode": "public_rollout_v3",
        },
    )
    assert response.status_code == 422
    assert "public information" in response.text


def test_pareto_rollout_v4_api_and_replay(monkeypatch):
    controller_token = "pareto-rollout-controller"
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", controller_token)
    SESSIONS.clear()
    controller = TestClient(app)
    gateway = TestClient(agent_app)
    created = controller.post(
        "/api/episodes",
        headers={"X-Controller-Token": controller_token},
        json={
            "episode_id": "pareto-rollout-api",
            "episode_seed": 92,
            "max_rounds": 5,
            "information_mode": "public",
            "belief_mode": "public_action_v1",
            "opponent_model_mode": "public_strategy_v1",
            "advisor_mode": "pareto_rollout_v4",
            "agent_configs": {
                "company_A": {
                    "persona": {"persona_id": "risk_guarded_v1"}
                }
            },
        },
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["manifest"]["advisor_schema_version"] == (
        "public-pareto-advice-v4.0.0"
    )
    observation = gateway.get(
        "/v1/episodes/pareto-rollout-api/companies/company_A/observation",
        headers={"X-Agent-Token": body["agent_tokens"]["company_A"]},
    )
    assert observation.status_code == 200, observation.text
    payload = observation.json()
    advice = PublicStrategicAdvice.model_validate(payload["game_theory_advice"])
    snapshot = ObservationSnapshot.from_observation(payload, "company_A")
    event = SimpleNamespace(
        communication_phase=None,
        traces=[SimpleNamespace(information_snapshot=snapshot)],
    )
    replayed = verify_advisor_replay(
        [event], SimpleNamespace(advisor_mode="pareto_rollout_v4")
    )
    assert replayed == (advice,)


def test_pareto_reliable_v5_api_and_replay(monkeypatch):
    controller_token = "pareto-reliable-controller"
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", controller_token)
    SESSIONS.clear()
    controller = TestClient(app)
    gateway = TestClient(agent_app)
    created = controller.post(
        "/api/episodes",
        headers={"X-Controller-Token": controller_token},
        json={
            "episode_id": "pareto-reliable-api",
            "episode_seed": 93,
            "max_rounds": 5,
            "information_mode": "public",
            "belief_mode": "public_action_v1",
            "opponent_model_mode": "public_strategy_v1",
            "advisor_mode": "pareto_reliable_v5",
            "agent_configs": {
                "company_A": {"persona": {"persona_id": "risk_guarded_v1"}}
            },
        },
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["manifest"]["advisor_schema_version"] == (
        "public-pareto-reliable-advice-v5.0.0"
    )
    observation = gateway.get(
        "/v1/episodes/pareto-reliable-api/companies/company_A/observation",
        headers={"X-Agent-Token": body["agent_tokens"]["company_A"]},
    )
    assert observation.status_code == 200, observation.text
    payload = observation.json()
    advice = PublicStrategicAdvice.model_validate(payload["game_theory_advice"])
    assert advice.advisor_mode == "pareto_reliable_v5"
    assert advice.reliability_gate is not None
    snapshot = ObservationSnapshot.from_observation(payload, "company_A")
    event = SimpleNamespace(
        communication_phase=None,
        traces=[SimpleNamespace(information_snapshot=snapshot)],
    )
    replayed = verify_advisor_replay(
        [event], SimpleNamespace(advisor_mode="pareto_reliable_v5")
    )
    assert replayed == (advice,)


def test_public_rollout_rule_acceptance_is_zero_llm_and_passes_small_gate():
    summary = run(seeds=(1, 2), horizon_rounds=2, scenario_count=2)
    assert summary["real_model_usage"]["calls"] == 0
    assert summary["real_model_usage"]["prompt_tokens"] == 0
    assert summary["engineering_passed"]
    assert summary["promotion_recommended"]
