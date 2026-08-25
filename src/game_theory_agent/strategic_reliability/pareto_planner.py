"""Constrained Pareto selection for Stage 6.4 strategic reliability."""

from __future__ import annotations

from typing import Literal, Mapping

from pydantic import Field, model_validator

from game_theory_agent.market.protocols import sha256_hash

from .contracts import RolloutCandidateEvaluation, StrategicReliabilityPlan, StrictModel


PPM = 1_000_000
TrailingPriority = Literal["value", "competition"]
DecisionSituation = Literal["protect_lead", "catch_up_early", "catch_up_late"]


class ParetoPlannerSpec(StrictModel):
    spec_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    early_trailing_priority: TrailingPriority = "value"
    late_round_threshold: int = Field(default=2, ge=1, le=10)
    late_expected_value_sacrifice_ppm: int = Field(default=0, ge=0, le=100_000)
    worst_case_value_sacrifice_ppm: int = Field(default=0, ge=0, le=50_000)
    minimum_competitive_improvement_cents: int = Field(default=0, ge=0)


PROMOTED_PARETO_SPEC = ParetoPlannerSpec(
    spec_id="strict_value",
    early_trailing_priority="value",
)
PROMOTION_EVIDENCE_SHA256 = (
    "66540760D0C250FADD2DF023124BAC7CACE83116379B0D6D9426CDA0239228EE"
)


class ParetoSituation(StrictModel):
    current_rank: int = Field(ge=1, le=8)
    current_competitive_margin_cents: int
    rounds_remaining: int = Field(ge=1, le=20)
    situation: DecisionSituation

    @model_validator(mode="after")
    def validate_situation(self) -> "ParetoSituation":
        leading = self.current_rank == 1 or self.current_competitive_margin_cents >= 0
        if leading and self.situation != "protect_lead":
            raise ValueError("leading company must protect the lead")
        if not leading and self.situation == "protect_lead":
            raise ValueError("trailing company cannot use protect-lead mode")
        return self


class ParetoCandidateAssessment(StrictModel):
    candidate_id: str
    expected_enterprise_value_cents: int
    expected_competitive_margin_cents: int
    worst_case_risk_adjusted_value_cents: int
    persona_certainty_equivalent_cents: int
    expected_final_rank_milli: int = Field(ge=1_000, le=8_000)
    expected_value_floor_cents: int
    competitive_margin_floor_cents: int
    worst_case_value_floor_cents: int
    persona_value_floor_cents: int
    passes_expected_value_floor: bool
    passes_competitive_margin_floor: bool
    passes_worst_case_value_floor: bool
    passes_persona_value_floor: bool
    eligible: bool
    dominated: bool
    pareto_frontier: bool


class ParetoPlannerDecision(StrictModel):
    decision_schema_version: Literal[
        "pareto-planner-decision-v1.0.0"
    ] = "pareto-planner-decision-v1.0.0"
    source_plan_hash: str
    spec: ParetoPlannerSpec
    situation: ParetoSituation
    baseline_candidate_id: Literal["maintain"] = "maintain"
    candidate_assessments: list[ParetoCandidateAssessment] = Field(min_length=2)
    recommended_candidate_id: str
    fallback_to_baseline: bool
    recommendation_is_non_binding: Literal[True] = True
    experimental_only: Literal[True] = True
    allowed_in_agent_context: Literal[False] = False
    decision_hash: str

    @model_validator(mode="after")
    def validate_decision(self) -> "ParetoPlannerDecision":
        by_id = {item.candidate_id: item for item in self.candidate_assessments}
        if len(by_id) != len(self.candidate_assessments):
            raise ValueError("Pareto candidate assessments must be unique")
        if self.baseline_candidate_id not in by_id:
            raise ValueError("Pareto baseline is missing")
        selected = by_id.get(self.recommended_candidate_id)
        if selected is None or not selected.pareto_frontier:
            raise ValueError("Pareto recommendation must be on the safe frontier")
        if self.fallback_to_baseline != (
            self.recommended_candidate_id == self.baseline_candidate_id
        ):
            raise ValueError("Pareto baseline fallback flag mismatch")
        if self.decision_hash != compute_pareto_decision_hash(self):
            raise ValueError("Pareto planner decision hash mismatch")
        return self


def compute_pareto_decision_hash(
    decision: ParetoPlannerDecision | Mapping[str, object],
) -> str:
    payload = (
        decision.model_dump(mode="json")
        if isinstance(decision, ParetoPlannerDecision)
        else dict(decision)
    )
    payload.pop("decision_hash", None)
    return sha256_hash(
        {
            "hash_protocol_version": "pareto-planner-hash-v1.0.0",
            "decision_schema_version": payload.get("decision_schema_version"),
            "decision": payload,
        }
    )


def _situation(
    *,
    current_rank: int,
    current_competitive_margin_cents: int,
    rounds_remaining: int,
    late_round_threshold: int,
) -> ParetoSituation:
    leading = current_rank == 1 or current_competitive_margin_cents >= 0
    if leading:
        mode: DecisionSituation = "protect_lead"
    elif rounds_remaining <= late_round_threshold:
        mode = "catch_up_late"
    else:
        mode = "catch_up_early"
    return ParetoSituation(
        current_rank=current_rank,
        current_competitive_margin_cents=current_competitive_margin_cents,
        rounds_remaining=rounds_remaining,
        situation=mode,
    )


def _required_competitive_margin(
    baseline_margin: int, minimum_improvement_cents: int
) -> int:
    return baseline_margin + minimum_improvement_cents


def _dominates(
    left: RolloutCandidateEvaluation,
    right: RolloutCandidateEvaluation,
) -> bool:
    if (
        left.expected_competitive_margin_cents is None
        or right.expected_competitive_margin_cents is None
    ):
        raise ValueError("Pareto comparison requires competitive labels")
    left_values = (
        left.expected_enterprise_value_cents,
        left.expected_competitive_margin_cents,
        left.worst_case_risk_adjusted_value_cents,
        left.certainty_equivalent_value_cents,
    )
    right_values = (
        right.expected_enterprise_value_cents,
        right.expected_competitive_margin_cents,
        right.worst_case_risk_adjusted_value_cents,
        right.certainty_equivalent_value_cents,
    )
    return all(a >= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a > b for a, b in zip(left_values, right_values, strict=True)
    )


def select_pareto_decision(
    plan: StrategicReliabilityPlan,
    spec: ParetoPlannerSpec,
    *,
    current_rank: int,
    current_competitive_margin_cents: int,
    rounds_remaining: int,
) -> ParetoPlannerDecision:
    situation = _situation(
        current_rank=current_rank,
        current_competitive_margin_cents=current_competitive_margin_cents,
        rounds_remaining=rounds_remaining,
        late_round_threshold=spec.late_round_threshold,
    )
    baseline = next(
        item for item in plan.evaluations if item.candidate.candidate_id == "maintain"
    )
    if (
        baseline.expected_competitive_margin_cents is None
        or baseline.expected_final_rank_milli is None
    ):
        raise ValueError("Pareto planner requires Stage 6.3 rollout labels")

    late_budget_ppm = (
        spec.late_expected_value_sacrifice_ppm
        if situation.situation == "catch_up_late"
        else 0
    )
    expected_budget = (
        abs(baseline.expected_enterprise_value_cents) * late_budget_ppm // PPM
    )
    worst_budget = (
        abs(baseline.worst_case_risk_adjusted_value_cents)
        * spec.worst_case_value_sacrifice_ppm
        // PPM
    )
    expected_floor = baseline.expected_enterprise_value_cents - expected_budget
    worst_floor = baseline.worst_case_risk_adjusted_value_cents - worst_budget
    persona_floor = baseline.certainty_equivalent_value_cents - expected_budget
    competitive_floor = _required_competitive_margin(
        baseline.expected_competitive_margin_cents,
        spec.minimum_competitive_improvement_cents,
    )

    eligibility: dict[str, bool] = {}
    for item in plan.evaluations:
        if (
            item.expected_competitive_margin_cents is None
            or item.expected_final_rank_milli is None
        ):
            raise ValueError("Pareto planner requires complete competitive labels")
        eligibility[item.candidate.candidate_id] = all(
            (
                item.expected_enterprise_value_cents >= expected_floor,
                item.expected_competitive_margin_cents >= competitive_floor,
                item.worst_case_risk_adjusted_value_cents >= worst_floor,
                item.certainty_equivalent_value_cents >= persona_floor,
            )
        )
    # The legal Rule operating baseline is always a safe fallback even when a
    # positive competition-improvement threshold excludes it.
    eligibility["maintain"] = True
    eligible = [
        item
        for item in plan.evaluations
        if eligibility[item.candidate.candidate_id]
    ]
    dominated = {
        item.candidate.candidate_id: any(
            other.candidate.candidate_id != item.candidate.candidate_id
            and _dominates(other, item)
            for other in eligible
        )
        for item in eligible
    }
    frontier = [
        item
        for item in eligible
        if not dominated[item.candidate.candidate_id]
    ]
    if not frontier:
        frontier = [baseline]

    def selection_key(item: RolloutCandidateEvaluation) -> tuple[int, int, int, str]:
        margin = item.expected_competitive_margin_cents
        if margin is None:
            raise ValueError("Pareto selection requires a competitive margin")
        if situation.situation == "protect_lead":
            return (
                item.certainty_equivalent_value_cents,
                margin,
                item.expected_enterprise_value_cents,
                item.candidate.candidate_id,
            )
        if situation.situation == "catch_up_late" or (
            spec.early_trailing_priority == "competition"
        ):
            return (
                margin,
                item.worst_case_risk_adjusted_value_cents,
                item.expected_enterprise_value_cents,
                item.candidate.candidate_id,
            )
        return (
            item.expected_enterprise_value_cents,
            margin,
            item.worst_case_risk_adjusted_value_cents,
            item.candidate.candidate_id,
        )

    selected = max(frontier, key=selection_key)

    assessments: list[ParetoCandidateAssessment] = []
    frontier_ids = {item.candidate.candidate_id for item in frontier}
    for item in plan.evaluations:
        candidate_id = item.candidate.candidate_id
        competitive_margin = item.expected_competitive_margin_cents
        expected_rank = item.expected_final_rank_milli
        if competitive_margin is None or expected_rank is None:
            raise ValueError("Pareto assessment requires complete competitive labels")
        is_eligible = eligibility[candidate_id]
        assessments.append(
            ParetoCandidateAssessment(
                candidate_id=candidate_id,
                expected_enterprise_value_cents=(
                    item.expected_enterprise_value_cents
                ),
                expected_competitive_margin_cents=competitive_margin,
                worst_case_risk_adjusted_value_cents=(
                    item.worst_case_risk_adjusted_value_cents
                ),
                persona_certainty_equivalent_cents=(
                    item.certainty_equivalent_value_cents
                ),
                expected_final_rank_milli=expected_rank,
                expected_value_floor_cents=expected_floor,
                competitive_margin_floor_cents=competitive_floor,
                worst_case_value_floor_cents=worst_floor,
                persona_value_floor_cents=persona_floor,
                passes_expected_value_floor=(
                    item.expected_enterprise_value_cents >= expected_floor
                ),
                passes_competitive_margin_floor=(
                    competitive_margin >= competitive_floor
                    or candidate_id == "maintain"
                ),
                passes_worst_case_value_floor=(
                    item.worst_case_risk_adjusted_value_cents >= worst_floor
                ),
                passes_persona_value_floor=(
                    item.certainty_equivalent_value_cents >= persona_floor
                ),
                eligible=is_eligible,
                dominated=bool(is_eligible and dominated.get(candidate_id, False)),
                pareto_frontier=candidate_id in frontier_ids,
            )
        )
    payload = {
        "decision_schema_version": "pareto-planner-decision-v1.0.0",
        "source_plan_hash": plan.plan_hash,
        "spec": spec.model_dump(mode="json"),
        "situation": situation.model_dump(mode="json"),
        "baseline_candidate_id": "maintain",
        "candidate_assessments": [item.model_dump(mode="json") for item in assessments],
        "recommended_candidate_id": selected.candidate.candidate_id,
        "fallback_to_baseline": selected.candidate.candidate_id == "maintain",
        "recommendation_is_non_binding": True,
        "experimental_only": True,
        "allowed_in_agent_context": False,
        "decision_hash": "pending",
    }
    payload["decision_hash"] = compute_pareto_decision_hash(payload)
    return ParetoPlannerDecision.model_validate(payload)
