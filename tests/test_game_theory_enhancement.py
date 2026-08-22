from __future__ import annotations

import asyncio
from types import SimpleNamespace
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from game_theory_agent.advisor import BayesianStrategyAdvisor
from game_theory_agent.agents import DecisionContextBuilder, EpisodeMemory
from game_theory_agent.belief import (
    BeliefState,
    OpponentPriceBelief,
    PriceDirectionDistribution,
)
from game_theory_agent.model_clients import MockModelClient
from game_theory_agent.api import SESSIONS, agent_app, app
from game_theory_agent.opponent import (
    OpponentModelState,
    PublicStrategyEvidence,
    build_strategy_model,
    compute_opponent_model_hash,
)
from game_theory_agent.repeated_game import RepeatedGameStrategist
from game_theory_agent.utility_inference import OpponentUtilityInferer
from game_theory_agent.utility_inference import (
    UtilityInferenceReplayMismatchError,
    verify_utility_inference_replay,
)
from game_theory_agent.experiments.stage51_real_game_theory import (
    build_plan,
    parse_seeds,
)


def _evidence(kind: str) -> list[PublicStrategyEvidence]:
    rows: list[PublicStrategyEvidence] = []
    for round_number in range(1, 11):
        if kind == "growth":
            direction, settled, share, reputation, contribution = (
                "price_cut",
                9_000,
                20_000,
                -1_000,
                0,
            )
        elif kind == "profit":
            direction, settled, share, reputation, contribution = (
                "price_raise",
                10_500,
                0,
                0,
                0,
            )
        elif kind == "defensive":
            direction, settled, share, reputation, contribution = (
                "maintain",
                10_000,
                -10_000,
                10_000,
                0,
            )
        else:
            direction, settled, share, reputation, contribution = (
                "maintain",
                10_000,
                0,
                10_000,
                100_000,
            )
        rows.append(
            PublicStrategyEvidence(
                evidence_id=f"{kind}:{round_number}",
                episode_id="strategic-test",
                settled_round=round_number,
                target_company_id="company_B",
                previous_price_cents=10_000,
                settled_price_cents=settled,
                price_direction=direction,
                market_share_delta_ppm=share,
                public_sales_orders=4_000 if kind == "growth" else 3_000,
                reputation_delta_ppm=reputation,
                public_shared_resilience_contribution_cents=contribution,
            )
        )
    return rows


def _belief(kind: str) -> BeliefState:
    probabilities = {
        "growth": (800_000, 150_000, 50_000),
        "profit": (50_000, 150_000, 800_000),
        "defensive": (100_000, 800_000, 100_000),
        "cooperative": (200_000, 600_000, 200_000),
    }[kind]
    counts = {
        "growth": {"price_cut": 10, "maintain": 0, "price_raise": 0},
        "profit": {"price_cut": 0, "maintain": 0, "price_raise": 10},
        "defensive": {"price_cut": 0, "maintain": 10, "price_raise": 0},
        "cooperative": {"price_cut": 0, "maintain": 10, "price_raise": 0},
    }[kind]
    return BeliefState(
        episode_id="strategic-test",
        observer_company_id="company_A",
        prediction_target_round=11,
        state_version=10,
        public_evidence_through_round=10,
        opponent_beliefs={
            "company_B": OpponentPriceBelief(
                opponent_company_id="company_B",
                prediction_target_round=11,
                evidence_count=10,
                latest_evidence_round=10,
                latest_observed_direction=max(counts, key=counts.get),
                observed_counts=counts,
                next_price_direction=PriceDirectionDistribution(
                    price_cut_ppm=probabilities[0],
                    maintain_ppm=probabilities[1],
                    price_raise_ppm=probabilities[2],
                ),
            )
        },
    )


def _strategic_inputs(kind: str):
    strategy = build_strategy_model("company_B", _evidence(kind))
    model = OpponentModelState(
        episode_id="strategic-test",
        observer_company_id="company_A",
        prediction_target_round=11,
        state_version=10,
        public_evidence_through_round=10,
        opponent_models={"company_B": strategy},
    )
    utility, utility_hash = OpponentUtilityInferer().infer(model)
    advice = BayesianStrategyAdvisor().advise(
        belief_state=_belief(kind),
        opponent_model=model,
        utility_inference=utility,
        own_company={
            "commercial": {
                "price_cents": 10_000,
                "market_share_ppm": 250_000,
            },
            "operations": {"actual_unit_cost_cents": 7_000},
        },
        action_constraints={
            "bounds": {"price_cents": {"min": 8_000, "max": 12_000}}
        },
    )
    return model, utility, utility_hash, advice


def test_public_strategy_model_recovers_four_synthetic_types_deterministically():
    for kind in ("growth", "profit", "defensive", "cooperative"):
        first = build_strategy_model("company_B", _evidence(kind))
        second = build_strategy_model("company_B", _evidence(kind))
        assert first == second
        assert first.strategy_distribution.top_strategy == kind
        state = OpponentModelState(
            episode_id="strategic-test",
            observer_company_id="company_A",
            prediction_target_round=11,
            state_version=10,
            public_evidence_through_round=10,
            opponent_models={"company_B": first},
        )
        assert compute_opponent_model_hash(state) == compute_opponent_model_hash(
            state.model_dump(mode="json")
        )


def test_utility_inference_is_public_bound_and_tamper_rejected():
    model, utility, utility_hash, _ = _strategic_inputs("growth")
    inferred = utility.opponent_utilities["company_B"]
    assert inferred.growth.mean_ppm > inferred.cash_preservation.mean_ppm
    assert utility.opponent_model_hash == compute_opponent_model_hash(model)
    assert utility_hash.startswith("sha256:")
    tampered = utility.model_dump(mode="json")
    tampered["uses_hidden_persona"] = True
    with pytest.raises(ValidationError):
        type(utility).model_validate(tampered)


def test_advisor_is_approximate_non_binding_and_adapts_to_opponent_type():
    aggressive = _strategic_inputs("growth")[3]
    conservative = _strategic_inputs("defensive")[3]
    assert aggressive.recommended_action == "aggressive_price_cut"
    assert conservative.recommended_action == "maintain"
    assert aggressive.recommended_action != conservative.recommended_action
    assert aggressive.approximate_best_response
    assert not aggressive.claims_nash_equilibrium
    assert not aggressive.uses_hidden_opponent_state
    best = max(
        aggressive.candidate_actions,
        key=lambda item: (
            item.expected_utility_proxy,
            item.worst_case_utility_proxy,
            -item.strategic_risk_ppm,
            -abs(item.price_cents - aggressive.current_price_cents),
        ),
    )
    maintain = next(
        item for item in aggressive.candidate_actions
        if item.action_label == "maintain"
    )
    assert best.expected_utility_proxy - maintain.expected_utility_proxy > 0


def test_repeated_game_strategies_change_after_fulfillment_and_betrayal():
    strategist = RepeatedGameStrategist()
    trusted, _ = strategist.build(
        episode_id="repeated-test",
        observer_company_id="company_A",
        round_number=5,
        cooperation_view={
            "cooperation_memory": {
                "company_B": {
                    "credibility_ppm": 800_000,
                    "accepted_by_opponent": 3,
                    "fulfilled_by_opponent": 3,
                    "partial_betrayals_by_opponent": 0,
                    "betrayed_by_opponent": 0,
                }
            }
        },
    )
    betrayed, _ = strategist.build(
        episode_id="repeated-test",
        observer_company_id="company_A",
        round_number=5,
        cooperation_view={
            "cooperation_memory": {
                "company_B": {
                    "credibility_ppm": 200_000,
                    "accepted_by_opponent": 3,
                    "fulfilled_by_opponent": 1,
                    "partial_betrayals_by_opponent": 0,
                    "betrayed_by_opponent": 2,
                }
            }
        },
    )
    assert trusted.opponent_strategies["company_B"].recommended_stance == "cooperate"
    assert betrayed.opponent_strategies["company_B"].recommended_stance == "permanent_refusal"
    assert (
        trusted.opponent_strategies["company_B"].contribution_multiplier_ppm
        > betrayed.opponent_strategies["company_B"].contribution_multiplier_ppm
    )


def test_mock_policy_can_adopt_non_binding_advisor_without_mutating_context(
    monkeypatch,
):
    token = "game-theory-test-controller"
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", token)
    SESSIONS.clear()
    created = TestClient(app).post(
        "/api/episodes",
        headers={"X-Controller-Token": token},
        json={
            "episode_id": "strategic-mock-integration",
            "episode_seed": 20260821,
            "company_ids": ["company_A", "company_B"],
            "max_rounds": 5,
            "information_mode": "public",
            "belief_mode": "public_action_v1",
            "opponent_model_mode": "public_strategy_v1",
            "utility_inference_mode": "strategy_utility_v1",
            "advisor_mode": "bayesian_strategy_v2",
        },
    )
    assert created.status_code == 201, created.text
    credentials = created.json()["agent_tokens"]
    observation = TestClient(agent_app).get(
        "/v1/episodes/strategic-mock-integration/companies/company_A/observation",
        headers={"X-Agent-Token": credentials["company_A"]},
    ).json()
    original_price = observation["own_company"]["commercial"]["price_cents"]
    recommendation = observation["game_theory_advice"][
        "recommended_price_cents"
    ]
    context = DecisionContextBuilder().build(
        observation, "company_A", EpisodeMemory()
    )
    generation = asyncio.run(
        MockModelClient(honor_game_theory_advice=True).generate_decision(context)
    )
    assert generation.parsed_output["requested_action"]["price_cents"] == recommendation
    assert observation["own_company"]["commercial"]["price_cents"] == original_price


def test_opponent_model_only_episode_still_protects_company_private_view(
    monkeypatch,
):
    token = "opponent-only-test-controller"
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", token)
    SESSIONS.clear()
    created = TestClient(app).post(
        "/api/episodes",
        headers={"X-Controller-Token": token},
        json={
            "episode_id": "opponent-only-auth",
            "episode_seed": 20260821,
            "company_ids": ["company_A", "company_B"],
            "max_rounds": 5,
            "information_mode": "public",
            "opponent_model_mode": "public_strategy_v1",
        },
    )
    assert created.status_code == 201, created.text
    credentials = created.json()["agent_tokens"]
    endpoint = (
        "/v1/episodes/opponent-only-auth/companies/company_A/observation"
    )
    assert TestClient(agent_app).get(endpoint).status_code == 401
    assert TestClient(agent_app).get(
        endpoint,
        headers={"X-Agent-Token": credentials["company_B"]},
    ).status_code == 401
    authorized = TestClient(agent_app).get(
        endpoint,
        headers={"X-Agent-Token": credentials["company_A"]},
    )
    assert authorized.status_code == 200
    assert authorized.json()["private_state"]["company_id"] == "company_A"


def test_stage51_plan_counterbalances_condition_order():
    plan = build_plan(parse_seeds("1001,1002"))
    first = [item["condition"] for item in plan if item["seed"] == 1001]
    second = [item["condition"] for item in plan if item["seed"] == 1002]
    assert first == [
        "A_persona_only",
        "B_action_belief",
        "C_opponent_model",
        "D_utility_advisor",
    ]
    assert second == [
        "B_action_belief",
        "C_opponent_model",
        "D_utility_advisor",
        "A_persona_only",
    ]


def test_utility_replay_distinguishes_disabled_treatment_from_deleted_output():
    snapshot = SimpleNamespace(
        observation={"opponent_model_state": {"present": True}},
        company_id="company_A",
    )
    event = SimpleNamespace(
        communication_phase=None,
        traces=[SimpleNamespace(information_snapshot=snapshot)],
    )
    assert verify_utility_inference_replay(
        [event], SimpleNamespace(utility_inference_mode="off")
    ) == ()
    with pytest.raises(
        UtilityInferenceReplayMismatchError,
        match="missing utility inference",
    ):
        verify_utility_inference_replay(
            [event],
            SimpleNamespace(utility_inference_mode="strategy_utility_v1"),
        )
