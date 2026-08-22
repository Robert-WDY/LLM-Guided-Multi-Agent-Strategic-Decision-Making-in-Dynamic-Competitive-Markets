import json

from fastapi.testclient import TestClient

from game_theory_agent.agents import (
    DecisionContextBuilder,
    EpisodeMemory,
    build_diagnostic_flags,
    load_persona_registry,
)
from game_theory_agent.api import SESSIONS, agent_app, app
from game_theory_agent.economics import decision_support_metrics
from game_theory_agent.experiments.persona_intervention_matrix import build_matrix


def test_versioned_extreme_and_moderate_personas_coexist():
    registry = load_persona_registry()

    assert (
        registry.get("aggressive").utility_weights_ppm
        == registry.get("aggressive_v1_extreme").utility_weights_ppm
    )
    assert (
        registry.get("conservative").utility_weights_ppm
        == registry.get("conservative_v1_extreme").utility_weights_ppm
    )
    disciplined = registry.get("disciplined_growth_v1")
    guarded = registry.get("risk_guarded_v1")
    assert disciplined.utility_weights_ppm.profit == 380_000
    assert disciplined.utility_weights_ppm.share == 150_000
    assert guarded.utility_weights_ppm.resilience == 160_000
    assert guarded.traits_ppm.risk_aversion == 700_000


def test_economic_decision_support_is_bounded_to_current_information(
    config, initial_state
):
    support = decision_support_metrics(config, initial_state, "company_A")

    assert support["metrics_schema_version"] == "decision-support-v1.1.0"
    assert support["information_boundary"] == {
        "uses_future_rng": False,
        "uses_episode_seed": False,
        "uses_only_current_state_and_public_config": True,
    }
    assert set(support["expected_demand_orders"]) >= {
        "expected",
        "low",
        "high",
        "method",
    }
    assert support["expected_demand_orders"]["low"] <= (
        support["expected_demand_orders"]["expected"]
    ) <= support["expected_demand_orders"]["high"]
    loss = support["expected_incident_loss_cents"]
    assert 0 <= loss["low"] <= loss["mean"] <= loss["high"]
    assert "recommended_resilience_target_ppm" not in support


def test_context_support_and_semantics_are_ablatable():
    SESSIONS.clear()
    created = TestClient(app).post(
        "/api/episodes",
        json={
            "episode_id": "persona-ablation-context",
            "episode_seed": 42,
            "company_ids": ["company_A", "company_B"],
            "max_rounds": 5,
        },
    )
    assert created.status_code == 201, created.text
    observation = TestClient(agent_app).get(
        "/v1/episodes/persona-ablation-context/companies/company_A/observation"
    ).json()
    legacy = DecisionContextBuilder(
        decision_support_version="legacy_v1",
        persona_semantics_version="legacy_v1",
    ).build(observation, "company_A", EpisodeMemory())
    economic = DecisionContextBuilder(
        decision_support_version="economic_v2",
        persona_semantics_version="economic_v2",
    ).build(observation, "company_A", EpisodeMemory())

    assert legacy.context_schema_version == "decision-context-v1.14.0"
    assert legacy.decision_support["metrics_schema_version"] == (
        "decision-support-v1.0.0"
    )
    assert "expected_incident_loss_cents" not in legacy.decision_support
    assert "expected_incident_loss_cents" in economic.decision_support


def test_diagnostics_are_non_enforcing_and_explain_evidence():
    registry = load_persona_registry()
    flags = build_diagnostic_flags(
        decision_support={
            "safe_discretionary_budget_cents": 5_000_000,
            "growth_spend_efficiency_ppm": -140_000,
            "growth_efficiency_evidence": {
                "last_growth_spend_cents": 3_000_000
            },
            "forecast_capacity_gap_orders": {
                "expected": 700,
                "low": 400,
                "high": 900,
            },
            "expected_demand_orders": {
                "expected": 4_200,
                "low": 3_900,
                "high": 4_500,
            },
            "capacity_investment_payback_rounds": {
                "expected": 3,
                "low": 2,
                "high": 5,
            },
            "expected_incident_loss_cents": {
                "mean": 1_800_000,
                "low": 900_000,
                "high": 3_100_000,
            },
            "current_resilience_coverage_ppm": 300_000,
            "resilience_marginal_loss_reduction_cents_per_1000000": 620_000,
        },
        rolling_summary={
            "action_frequency": {"high_advertising_rounds": 2}
        },
        persona_profile=registry.get("risk_guarded_v1"),
        rounds_remaining=8,
    )

    assert {flag["type"] for flag in flags} == {
        "low_growth_efficiency",
        "positive_return_capacity_gap",
        "material_uncovered_risk",
    }
    assert all(flag["enforcement"] == "none" for flag in flags)
    assert all(flag["required_response"] for flag in flags)


def test_intervention_matrix_blocks_by_seed_and_rotates_persona_positions():
    rows = build_matrix(
        conditions=("A_balanced_legacy", "E_moderate_semantics"),
        seed_split="development",
    )

    balanced = [row for row in rows if row["condition"] == "A_balanced_legacy"]
    moderate = [row for row in rows if row["condition"] == "E_moderate_semantics"]
    assert len(balanced) == 10
    assert len(moderate) == 40
    assert {row["rotation_index"] for row in moderate} == {0, 1, 2, 3}
    assert {row["seed"] for row in moderate} == set(range(101, 111))
    assert all(row["primary_experiment_unit"] == "seed" for row in rows)


def test_matrix_plan_declares_decisions_are_not_independent_samples():
    # The serialized wording is part of the research contract and protects
    # against treating 1,600 nested decisions as 1,600 experimental units.
    rows = build_matrix(
        conditions=("F_moderate_diagnostics",), seed_split="validation"
    )
    payload = json.dumps(rows, ensure_ascii=False)
    assert '"primary_experiment_unit": "seed"' in payload
    assert len(rows) == 80
