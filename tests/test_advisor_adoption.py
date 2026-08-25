from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from game_theory_agent.advisor import (
    AdvisorAdoptionTrace,
    build_advisor_adoption_trace,
)


def _public_candidate(candidate_id: str, resilience: int, changed: list[str]):
    return {
        "candidate": {
            "candidate_id": candidate_id,
            "action": {
                "price_cents": 10_000,
                "advertising_budget_cents": 0,
                "service_budget_cents": 0,
                "capacity_investment_cents": 0,
                "resilience_budget_cents": resilience,
                "shared_resilience_contribution_cents": 0,
                "incident_response_mode": "wait",
                "repair_budget_cents": 0,
            },
            "changed_dimensions": changed,
        },
        "certainty_equivalent_value_cents": resilience,
    }


def test_pareto_adoption_distinguishes_target_acceptance_from_exact_action():
    advice = {
        "advisor_mode": "pareto_rollout_v4",
        "recommended_candidate_id": "increase_resilience",
        "recommended_action": _public_candidate(
            "increase_resilience", 1_000, ["resilience"]
        )["candidate"]["action"],
        "candidate_actions": [
            _public_candidate("maintain", 0, []),
            _public_candidate("increase_resilience", 1_000, ["resilience"]),
        ],
        "pareto_decision": {
            "candidate_assessments": [
                {
                    "candidate_id": "maintain",
                    "eligible": True,
                    "pareto_frontier": True,
                },
                {
                    "candidate_id": "increase_resilience",
                    "eligible": True,
                    "pareto_frontier": True,
                },
            ]
        },
    }
    requested = {
        "price_cents": 10_000,
        "resilience_budget_cents": 500,
        "incident_response": {"mode": "wait", "repair_budget_cents": 0},
    }
    trace = build_advisor_adoption_trace(
        advice=advice,
        llm_requested_action=requested,
        final_action={**requested, "resilience_budget_cents": 400},
        planner_output={
            "strategy_summary": "接受抗冲击方向，但控制投入规模",
            "situation_summary": "现金需要保留",
        },
    )

    assert trace is not None
    assert trace.accepted
    assert trace.adoption_status == "accepted_target"
    assert trace.target_alignment_ppm == 1_000_000
    assert trace.action_alignment_ppm < 1_000_000
    assert trace.chosen_candidate_id is None
    assert "控制投入规模" in trace.agent_reason

    forged = deepcopy(trace.model_dump(mode="json"))
    forged["accepted"] = False
    with pytest.raises(ValidationError):
        AdvisorAdoptionTrace.model_validate(forged)


def test_bayesian_price_adoption_records_candidate_rank():
    advice = {
        "advisor_mode": "bayesian_strategy_v2",
        "current_price_cents": 10_000,
        "recommended_action": "price_cut",
        "recommended_price_cents": 9_500,
        "candidate_actions": [
            {
                "action_label": "maintain",
                "price_cents": 10_000,
                "expected_utility_proxy": 10,
                "worst_case_utility_proxy": 5,
            },
            {
                "action_label": "price_cut",
                "price_cents": 9_500,
                "expected_utility_proxy": 20,
                "worst_case_utility_proxy": 8,
            },
        ],
    }
    trace = build_advisor_adoption_trace(
        advice=advice,
        llm_requested_action={"price_cents": 9_500},
        final_action={"price_cents": 9_500},
        planner_output={"strategy_summary": "接受降价建议"},
    )

    assert trace is not None
    assert trace.accepted
    assert trace.adoption_status == "exact_action"
    assert trace.advisor_rank == 1
    assert trace.chosen_candidate_id == "price_cut"
    assert trace.chosen_rank == 1


def test_v7_abstention_records_unavailable_and_keeps_agent_action():
    requested = {
        "price_cents": 10_600,
        "advertising_budget_cents": 500_000,
        "service_budget_cents": 500_000,
        "resilience_budget_cents": 500_000,
    }
    trace = build_advisor_adoption_trace(
        advice={
            "advisor_mode": "pareto_reliable_v7",
            "execution_disposition": "defer_to_agent",
            "recommended_candidate_id": None,
            "recommended_action": None,
            "candidate_actions": [
                _public_candidate("maintain", 0, []),
                _public_candidate("risk_buffer", 1_000_000, ["resilience"]),
            ],
            "pareto_decision": {"candidate_assessments": []},
        },
        llm_requested_action=requested,
        final_action=requested,
        planner_output={"strategy_summary": "保留自主经营计划"},
    )

    assert trace is not None
    assert trace.trace_schema_version == "advisor-adoption-trace-v2.0.0"
    assert trace.execution_disposition == "defer_to_agent"
    assert trace.adoption_status == "unavailable"
    assert not trace.accepted
    assert trace.advisor_candidate_id is None
    assert trace.advisor_action == {}
    assert trace.llm_requested_action["price_cents"] == 10_600
    assert trace.final_action["price_cents"] == 10_600

    forged = deepcopy(trace.model_dump(mode="json"))
    forged["advisor_action"] = {"price_cents": 9_800}
    with pytest.raises(ValidationError):
        AdvisorAdoptionTrace.model_validate(forged)
