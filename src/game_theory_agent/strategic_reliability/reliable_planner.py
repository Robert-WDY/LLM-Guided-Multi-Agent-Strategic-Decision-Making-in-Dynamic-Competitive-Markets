"""Public-only reliability gate layered on the promoted Pareto v4 planner."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import Field, model_validator

from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.opponent import (
    OpponentModelState,
    compute_opponent_model_hash,
)

from .contracts import StrategicReliabilityPlan, StrictModel
from .pareto_planner import ParetoPlannerDecision


PPM = 1_000_000
MIN_OPPONENT_CONFIDENCE_PPM = 700_000


class ExcludedCandidate(StrictModel):
    candidate_id: str
    reason_codes: list[str] = Field(min_length=1)


class ParetoReliabilityGate(StrictModel):
    gate_schema_version: Literal[
        "pareto-reliability-gate-v1.0.0",
        "pareto-reliability-gate-v2.0.0",
    ] = (
        "pareto-reliability-gate-v1.0.0"
    )
    planner_decision_hash: str
    reliability_input_hash: str
    planner_candidate_id: str
    effective_candidate_id: str
    fallback_candidate_id: str
    safe_candidate_ids: list[str] = Field(min_length=1)
    excluded_candidates: list[ExcludedCandidate]
    opponent_model_confidence_ppm: int = Field(ge=0, le=PPM)
    top_two_value_gap_cents: int = Field(ge=0)
    forecast_uncertainty_cents: int = Field(ge=0)
    advisor_confidence_ppm: int = Field(ge=0, le=PPM)
    selected_passes_fallback_value_floor: bool
    selected_passes_fallback_worst_floor: bool
    selected_passes_fallback_persona_floor: bool
    should_abstain: bool
    abstain_reason_codes: list[str]
    uses_only_public_and_own_private_inputs: Literal[True] = True
    uses_authoritative_hidden_market_state: Literal[False] = False
    allowed_in_agent_context: Literal[True] = True
    marginal_investment_plan_hash: str | None = None
    gate_hash: str

    @model_validator(mode="after")
    def validate_gate(self) -> "ParetoReliabilityGate":
        if self.effective_candidate_id != (
            self.fallback_candidate_id
            if self.should_abstain
            else self.planner_candidate_id
        ):
            raise ValueError("reliability effective candidate mismatch")
        if self.should_abstain != bool(self.abstain_reason_codes):
            raise ValueError("reliability abstain reasons mismatch")
        if self.effective_candidate_id not in self.safe_candidate_ids:
            raise ValueError("reliability effective candidate must be gate-safe")
        excluded_ids = {item.candidate_id for item in self.excluded_candidates}
        if excluded_ids.intersection(self.safe_candidate_ids):
            raise ValueError("safe and excluded reliability candidates must be disjoint")
        if (
            self.gate_schema_version == "pareto-reliability-gate-v1.0.0"
        ) != (self.marginal_investment_plan_hash is None):
            raise ValueError("reliability gate marginal-plan binding mismatch")
        if self.gate_hash != compute_reliability_gate_hash(self):
            raise ValueError("reliability gate hash mismatch")
        return self


def compute_reliability_gate_hash(
    gate: ParetoReliabilityGate | Mapping[str, Any],
) -> str:
    payload = (
        gate.model_dump(mode="json")
        if isinstance(gate, ParetoReliabilityGate)
        else dict(gate)
    )
    payload.pop("gate_hash", None)
    if (
        payload.get("gate_schema_version")
        == "pareto-reliability-gate-v1.0.0"
        and payload.get("marginal_investment_plan_hash") is None
    ):
        payload.pop("marginal_investment_plan_hash", None)
    return sha256_hash(
        {
            "hash_protocol_version": "pareto-reliability-gate-hash-v1.0.0",
            "gate": payload,
        }
    )


def build_reliability_gate(
    *,
    plan: StrategicReliabilityPlan,
    decision: ParetoPlannerDecision,
    fallback_candidate_id: str,
    public_decision_input_hash: str,
    observation: Mapping[str, Any],
    opponent_model: OpponentModelState,
    marginal_investment_plan_hash: str | None = None,
) -> ParetoReliabilityGate:
    evaluations = {
        item.candidate.candidate_id: item for item in plan.evaluations
    }
    if fallback_candidate_id not in evaluations:
        raise ValueError("reliability fallback candidate is missing")
    planner_id = decision.recommended_candidate_id
    planner = evaluations[planner_id]
    fallback = evaluations[fallback_candidate_id]
    safe_ids = sorted(
        item.candidate_id
        for item in decision.candidate_assessments
        if item.eligible
    )
    if fallback_candidate_id not in safe_ids:
        # This set describes candidates safe for *execution after abstention*,
        # not only candidates accepted by the forecast-driven Pareto filter.
        # Requiring the fallback to pass that same forecast would make the
        # reliability gate circular precisely when its forecast is unreliable.
        safe_ids.append(fallback_candidate_id)
        safe_ids.sort()
    excluded: list[ExcludedCandidate] = []
    for item in decision.candidate_assessments:
        reasons: list[str] = []
        if not item.passes_expected_value_floor:
            reasons.append("expected_value_below_floor")
        if not item.passes_competitive_margin_floor:
            reasons.append("competitive_margin_below_floor")
        if not item.passes_worst_case_value_floor:
            reasons.append("worst_case_below_floor")
        if not item.passes_persona_value_floor:
            reasons.append("persona_value_below_floor")
        if reasons and item.candidate_id not in safe_ids:
            excluded.append(
                ExcludedCandidate(
                    candidate_id=item.candidate_id,
                    reason_codes=reasons,
                )
            )

    profiles = list(opponent_model.opponent_models.values())
    opponent_confidence = (
        sum(item.confidence_ppm for item in profiles) // len(profiles)
        if profiles
        else 0
    )
    eligible_values = sorted(
        (
            evaluations[candidate_id].certainty_equivalent_value_cents
            for candidate_id in safe_ids
        ),
        reverse=True,
    )
    top_gap = (
        eligible_values[0] - eligible_values[1]
        if len(eligible_values) > 1
        else 0
    )
    scenario_values = [
        item.enterprise_value_cents for item in planner.outcomes
    ]
    uncertainty = max(scenario_values) - min(scenario_values)
    passes_value = (
        planner.expected_enterprise_value_cents
        >= fallback.expected_enterprise_value_cents
    )
    passes_worst = (
        planner.worst_case_risk_adjusted_value_cents
        >= fallback.worst_case_risk_adjusted_value_cents
    )
    passes_persona = (
        planner.certainty_equivalent_value_cents
        >= fallback.certainty_equivalent_value_cents
    )
    reasons: list[str] = []
    if opponent_confidence < MIN_OPPONENT_CONFIDENCE_PPM:
        reasons.append("opponent_model_confidence_below_gate")
    if not passes_value:
        reasons.append("planner_below_fallback_expected_value")
    if not passes_worst:
        reasons.append("planner_below_fallback_worst_case")
    if not passes_persona:
        reasons.append("planner_below_fallback_persona_value")
    if len(eligible_values) > 1 and top_gap * 2 <= uncertainty:
        reasons.append("top_candidates_overlap_forecast_uncertainty")

    should_abstain = bool(reasons)
    effective_id = fallback_candidate_id if should_abstain else planner_id
    gap_confidence = min(
        PPM,
        top_gap * PPM // max(1, uncertainty),
    )
    advisor_confidence = min(opponent_confidence, gap_confidence)
    reliability_input_hash = sha256_hash(
        {
            "protocol": (
                "pareto-reliability-public-input-v2.0.0"
                if marginal_investment_plan_hash is not None
                else "pareto-reliability-public-input-v1.0.0"
            ),
            "public_decision_input_hash": public_decision_input_hash,
            "planner_decision_hash": decision.decision_hash,
            "fallback_candidate_id": fallback_candidate_id,
            "decision_support": observation.get("decision_support", {}),
            "active_market_events": observation.get("active_market_events", []),
            "risk_signals": observation.get("risk_signals", []),
            "opponent_model_hash": compute_opponent_model_hash(opponent_model),
            **(
                {"marginal_investment_plan_hash": marginal_investment_plan_hash}
                if marginal_investment_plan_hash is not None
                else {}
            ),
        }
    )
    payload: dict[str, Any] = {
        "gate_schema_version": (
            "pareto-reliability-gate-v2.0.0"
            if marginal_investment_plan_hash is not None
            else "pareto-reliability-gate-v1.0.0"
        ),
        "planner_decision_hash": decision.decision_hash,
        "reliability_input_hash": reliability_input_hash,
        "planner_candidate_id": planner_id,
        "effective_candidate_id": effective_id,
        "fallback_candidate_id": fallback_candidate_id,
        "safe_candidate_ids": safe_ids,
        "excluded_candidates": [item.model_dump(mode="json") for item in excluded],
        "opponent_model_confidence_ppm": opponent_confidence,
        "top_two_value_gap_cents": max(0, top_gap),
        "forecast_uncertainty_cents": uncertainty,
        "advisor_confidence_ppm": advisor_confidence,
        "selected_passes_fallback_value_floor": passes_value,
        "selected_passes_fallback_worst_floor": passes_worst,
        "selected_passes_fallback_persona_floor": passes_persona,
        "should_abstain": should_abstain,
        "abstain_reason_codes": reasons,
        "uses_only_public_and_own_private_inputs": True,
        "uses_authoritative_hidden_market_state": False,
        "allowed_in_agent_context": True,
        **(
            {"marginal_investment_plan_hash": marginal_investment_plan_hash}
            if marginal_investment_plan_hash is not None
            else {}
        ),
        "gate_hash": "pending",
    }
    payload["gate_hash"] = compute_reliability_gate_hash(payload)
    return ParetoReliabilityGate.model_validate(payload)
