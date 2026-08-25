"""Strict contracts for Stage 6 strategic-decision reliability.

The first Stage 6 implementation deliberately separates an authoritative
counterfactual oracle from Agent-visible advice.  The oracle may load a full
``MarketState`` in order to measure true market outcomes, but its output is
not allowed to enter an incomplete-information Agent context.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.market.protocols import sha256_hash


CandidateLabel = Literal[
    "maintain",
    "status_quo",
    "profit_recovery",
    "risk_buffer",
    "price_cut_small",
    "price_cut_large",
    "price_increase",
    "increase_advertising",
    "increase_service",
    "increase_capacity",
    "increase_resilience",
    "shared_resilience_contribution",
    "repair_incident",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CandidateEconomicAction(StrictModel):
    price_cents: int = Field(gt=0)
    advertising_budget_cents: int = Field(default=0, ge=0)
    service_budget_cents: int = Field(default=0, ge=0)
    capacity_investment_cents: int = Field(default=0, ge=0)
    resilience_budget_cents: int = Field(default=0, ge=0)
    shared_resilience_contribution_cents: int | None = Field(default=None, ge=0)
    incident_response_mode: Literal[
        "wait", "partial_repair", "full_repair"
    ] = "wait"
    repair_budget_cents: int = Field(default=0, ge=0)


class StrategicActionCandidate(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=120)
    label: CandidateLabel
    action: CandidateEconomicAction
    changed_dimensions: list[str] = Field(min_length=1, max_length=4)
    rationale: str = Field(min_length=1, max_length=400)


class RolloutScenarioOutcome(StrictModel):
    scenario_index: int = Field(ge=0)
    scenario_seed: int = Field(ge=0, lt=1 << 64)
    final_state_hash: str
    enterprise_value_cents: int
    enterprise_value_delta_cents: int
    best_opponent_enterprise_value_cents: int | None = None
    competitive_margin_cents: int | None = None
    final_rank: int | None = Field(default=None, ge=1, le=8)
    cumulative_profit_delta_cents: int
    expected_loss_cents: int = Field(ge=0)
    risk_adjusted_value_cents: int
    final_market_share_ppm: int = Field(ge=0, le=1_000_000)
    mean_discounted_persona_utility_ppm: int
    persona_aligned_value_cents: int

    @model_validator(mode="after")
    def validate_competitive_labels(self) -> "RolloutScenarioOutcome":
        labels = (
            self.best_opponent_enterprise_value_cents,
            self.competitive_margin_cents,
            self.final_rank,
        )
        if any(item is not None for item in labels) and any(
            item is None for item in labels
        ):
            raise ValueError("competitive outcome labels must be all present")
        if (
            self.best_opponent_enterprise_value_cents is not None
            and self.competitive_margin_cents
            != self.enterprise_value_cents
            - self.best_opponent_enterprise_value_cents
        ):
            raise ValueError("competitive margin does not match enterprise values")
        return self


class RolloutCandidateEvaluation(StrictModel):
    candidate: StrategicActionCandidate
    outcomes: list[RolloutScenarioOutcome] = Field(min_length=1)
    expected_enterprise_value_cents: int
    expected_enterprise_value_delta_cents: int
    expected_competitive_margin_cents: int | None = None
    worst_case_competitive_margin_cents: int | None = None
    expected_final_rank_milli: int | None = Field(default=None, ge=1_000, le=8_000)
    expected_cumulative_profit_delta_cents: int
    expected_loss_cents: int = Field(ge=0)
    expected_risk_adjusted_value_cents: int
    worst_case_risk_adjusted_value_cents: int
    expected_persona_utility_ppm: int
    expected_persona_aligned_value_cents: int
    worst_case_persona_aligned_value_cents: int
    certainty_equivalent_value_cents: int


class StrategicReliabilityPlan(StrictModel):
    plan_schema_version: Literal[
        "strategic-reliability-plan-v1.0.0"
    ] = "strategic-reliability-plan-v1.0.0"
    evaluator_version: Literal[
        "authoritative-market-rollout-v1.0.0"
    ] = "authoritative-market-rollout-v1.0.0"
    episode_id: str
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    state_hash: str
    company_id: str
    persona_id: str
    persona_profile_hash: str
    horizon_rounds: int = Field(ge=1, le=20)
    scenario_count: int = Field(ge=1, le=100)
    evaluations: list[RolloutCandidateEvaluation] = Field(min_length=2)
    recommended_candidate_id: str
    baseline_candidate_id: str
    expected_gain_over_baseline_cents: int
    baseline_regret_cents: int = Field(ge=0)
    recommendation_is_non_binding: Literal[True] = True
    uses_authoritative_hidden_market_state: Literal[True] = True
    allowed_in_public_agent_context: Literal[False] = False
    claims_nash_equilibrium: Literal[False] = False
    limitations: list[str] = Field(min_length=1)
    plan_hash: str

    @model_validator(mode="after")
    def validate_ranking_and_hash(self) -> "StrategicReliabilityPlan":
        by_id = {
            item.candidate.candidate_id: item for item in self.evaluations
        }
        if len(by_id) != len(self.evaluations):
            raise ValueError("strategic candidates must have unique ids")
        if self.baseline_candidate_id not in by_id:
            raise ValueError("baseline candidate is missing")
        ranked = max(
            self.evaluations,
            key=lambda item: (
                item.certainty_equivalent_value_cents,
                item.expected_enterprise_value_cents,
                item.worst_case_risk_adjusted_value_cents,
                item.candidate.candidate_id,
            ),
        )
        if self.recommended_candidate_id != ranked.candidate.candidate_id:
            raise ValueError("recommended candidate is not top ranked")
        baseline = by_id[self.baseline_candidate_id]
        expected_gain = (
            ranked.certainty_equivalent_value_cents
            - baseline.certainty_equivalent_value_cents
        )
        if self.expected_gain_over_baseline_cents != expected_gain:
            raise ValueError("baseline gain does not match candidate ranking")
        if self.baseline_regret_cents != max(0, expected_gain):
            raise ValueError("baseline regret does not match candidate ranking")
        if self.plan_hash != compute_reliability_plan_hash(self):
            raise ValueError("strategic reliability plan hash mismatch")
        return self


def compute_reliability_plan_hash(
    plan: StrategicReliabilityPlan | Mapping[str, Any],
) -> str:
    payload = deepcopy(
        plan.model_dump(mode="json")
        if isinstance(plan, StrategicReliabilityPlan)
        else dict(plan)
    )
    payload.pop("plan_hash", None)
    # Competitive labels were added as optional Stage 6.3 fields.  Omitting
    # only their null defaults preserves hashes of pre-6.3 stored plans while
    # keeping all older nullable economic-action fields in the hash.
    for evaluation in payload.get("evaluations", []):
        for key in (
            "expected_competitive_margin_cents",
            "worst_case_competitive_margin_cents",
            "expected_final_rank_milli",
        ):
            if evaluation.get(key) is None:
                evaluation.pop(key, None)
        for outcome in evaluation.get("outcomes", []):
            for key in (
                "best_opponent_enterprise_value_cents",
                "competitive_margin_cents",
                "final_rank",
            ):
                if outcome.get(key) is None:
                    outcome.pop(key, None)
    return sha256_hash(
        {
            "hash_protocol_version": "strategic-reliability-hash-v1.0.0",
            "plan_schema_version": payload.get("plan_schema_version"),
            "plan": payload,
        }
    )
