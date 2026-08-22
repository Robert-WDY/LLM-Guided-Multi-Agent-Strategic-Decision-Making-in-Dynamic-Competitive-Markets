from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from game_theory_agent.agents import load_persona_registry
from game_theory_agent.experiments.persona_multiround import _run_episode
from game_theory_agent.experiments.persona_pilot import (
    _ablation_profile,
    build_scenarios,
    run,
)
from game_theory_agent.experiments.persona_pilot_counterfactual import analyze
from game_theory_agent.market import load_market_config
from game_theory_agent.model_clients import MockModelClient


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "market_v4.yaml"


def test_profit_myopic_utility_is_current_value_focused():
    profile = load_persona_registry().get("profit_myopic")

    assert profile.utility_weights_ppm.profit == 900_000
    assert profile.utility_weights_ppm.cash == 100_000
    assert profile.utility_weights_ppm.share == 0
    assert profile.utility_weights_ppm.growth == 0


def test_ablation_profiles_change_only_selected_contract_field():
    registry = load_persona_registry()
    target = registry.get("aggressive")
    baseline = registry.get("none")

    objective = _ablation_profile(target, baseline, "objective_only")
    weights = _ablation_profile(target, baseline, "weights_only")
    traits = _ablation_profile(target, baseline, "traits_only")

    assert objective.objective == target.objective
    assert objective.utility_weights_ppm == baseline.utility_weights_ppm
    assert objective.traits_ppm == baseline.traits_ppm
    assert weights.utility_weights_ppm == target.utility_weights_ppm
    assert weights.traits_ppm == baseline.traits_ppm
    assert traits.traits_ppm == target.traits_ppm
    assert traits.utility_weights_ppm == baseline.utility_weights_ppm


def test_multi_seed_scenarios_have_distinct_frozen_contexts():
    config = load_market_config(CONFIG_PATH)
    scenarios = build_scenarios(
        config,
        scenario_classes=("normal", "financial_stress"),
        market_seeds=(41, 42),
    )

    assert {item.scenario_id for item in scenarios} == {
        "normal_seed_41",
        "financial_stress_seed_41",
        "normal_seed_42",
        "financial_stress_seed_42",
    }
    assert len({item.state.state_hash for item in scenarios}) == 4


def test_state_response_scenarios_expose_capacity_and_incident_conditions():
    config = load_market_config(CONFIG_PATH)
    scenarios = {
        item.scenario_id: item
        for item in build_scenarios(
            config,
            scenario_classes=("capacity_pressure", "active_incident"),
            market_seeds=(42,),
        )
    }

    capacity = scenarios["capacity_pressure"].observation["own_company"]
    incident = scenarios["active_incident"].observation["action_constraints"]
    assert capacity["operations"]["capacity_utilization_ppm"] == 980_000
    assert capacity["brand"]["last_attempted_unfulfilled_rate_ppm"] == 200_000
    assert incident["active_incident"]["severity"] == "high"
    assert incident["max_useful_repair_budget_cents"] == 1_000_000


def test_p1_mock_artifacts_support_all_sample_regret(tmp_path):
    output = tmp_path / "pilot"
    args = argparse.Namespace(
        provider="mock",
        model=None,
        personas="none,aggressive",
        ablations="full",
        scenarios="normal",
        market_seeds="42",
        temperature=0.0,
        top_p=1.0,
        repetitions=2,
        timeout=5.0,
        output=output,
    )

    assert asyncio.run(run(args)) == 0
    result = analyze(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["sampling"]["temperature"] == 0.0
    assert result["aggregate_research_personas"]["decision_count"] == 2
    assert result["aggregate_research_personas"]["strict_optimal_rate"] == 1.0


def test_p2_mock_episode_reaches_terminal_with_memory_utility():
    config = load_market_config(CONFIG_PATH)
    registry = load_persona_registry()

    rows, summary = asyncio.run(
        _run_episode(
            provider="mock",
            client=MockModelClient(),
            config=config,
            registry=registry,
            persona_id="balanced",
            seed=42,
            rounds=5,
            market_model="balanced",
            timeout=5.0,
        )
    )

    assert len(rows) == 5
    assert summary["rounds"] == 5
    assert summary["fallback_count"] == 0
    assert summary["terminal_enterprise_value_cents"] is not None
    assert rows[-1]["cumulative_discounted_utility_ppm"] != 0
