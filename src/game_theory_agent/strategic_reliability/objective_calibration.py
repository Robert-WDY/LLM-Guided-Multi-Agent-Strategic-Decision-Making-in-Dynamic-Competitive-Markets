"""Offline objective calibration over authoritative/public-forecast rollouts.

This module deliberately reranks an already evaluated candidate set.  It does
not change MarketEnv, generate new actions, or enter Agent observations.
"""

from __future__ import annotations

from typing import Literal, Mapping

from pydantic import Field, model_validator

from game_theory_agent.market.protocols import sha256_hash

from .contracts import (
    RolloutCandidateEvaluation,
    StrategicReliabilityPlan,
    StrictModel,
)


PPM = 1_000_000
StrategicObjectiveMode = Literal[
    "persona_aligned",
    "absolute_value",
    "competitive_hybrid",
]


def _risk_adjusted_certainty_equivalent(
    expected: int, worst: int, risk_aversion_ppm: int
) -> int:
    return expected - risk_aversion_ppm * (expected - worst) // PPM


class ObjectiveCalibrationSpec(StrictModel):
    mode: StrategicObjectiveMode
    competitive_weight_ppm: int = Field(default=0, ge=0, le=PPM)
    risk_aversion_ppm: int = Field(ge=0, le=PPM)

    @model_validator(mode="after")
    def validate_mode_weight(self) -> "ObjectiveCalibrationSpec":
        if self.mode == "competitive_hybrid":
            if self.competitive_weight_ppm <= 0:
                raise ValueError("competitive hybrid requires a positive weight")
        elif self.competitive_weight_ppm != 0:
            raise ValueError("only competitive hybrid accepts a competition weight")
        return self


class ObjectiveCandidateScore(StrictModel):
    candidate_id: str
    persona_aligned_certainty_equivalent_cents: int
    absolute_value_certainty_equivalent_cents: int
    competitive_margin_certainty_equivalent_cents: int
    objective_score_cents: int
    expected_enterprise_value_cents: int
    expected_competitive_margin_cents: int
    worst_case_competitive_margin_cents: int
    expected_final_rank_milli: int = Field(ge=1_000, le=8_000)


class ObjectiveCalibrationDecision(StrictModel):
    decision_schema_version: Literal[
        "objective-calibration-decision-v1.0.0"
    ] = "objective-calibration-decision-v1.0.0"
    source_plan_hash: str
    objective_spec: ObjectiveCalibrationSpec
    candidate_scores: list[ObjectiveCandidateScore] = Field(min_length=2)
    recommended_candidate_id: str
    experimental_only: Literal[True] = True
    allowed_in_agent_context: Literal[False] = False
    decision_hash: str

    @model_validator(mode="after")
    def validate_ranking_and_hash(self) -> "ObjectiveCalibrationDecision":
        by_id = {item.candidate_id: item for item in self.candidate_scores}
        if len(by_id) != len(self.candidate_scores):
            raise ValueError("objective calibration candidates must be unique")
        ranked = max(
            self.candidate_scores,
            key=lambda item: (
                item.objective_score_cents,
                item.expected_enterprise_value_cents,
                item.expected_competitive_margin_cents,
                item.candidate_id,
            ),
        )
        if self.recommended_candidate_id != ranked.candidate_id:
            raise ValueError("objective calibration recommendation is not top ranked")
        if self.decision_hash != compute_objective_decision_hash(self):
            raise ValueError("objective calibration decision hash mismatch")
        return self


def compute_objective_decision_hash(
    decision: ObjectiveCalibrationDecision | Mapping[str, object],
) -> str:
    payload = (
        decision.model_dump(mode="json")
        if isinstance(decision, ObjectiveCalibrationDecision)
        else dict(decision)
    )
    payload.pop("decision_hash", None)
    return sha256_hash(
        {
            "hash_protocol_version": "objective-calibration-hash-v1.0.0",
            "decision_schema_version": payload.get("decision_schema_version"),
            "decision": payload,
        }
    )


def _score_candidate(
    evaluation: RolloutCandidateEvaluation,
    spec: ObjectiveCalibrationSpec,
) -> ObjectiveCandidateScore:
    if (
        evaluation.expected_competitive_margin_cents is None
        or evaluation.worst_case_competitive_margin_cents is None
        or evaluation.expected_final_rank_milli is None
    ):
        raise ValueError("rollout plan does not contain Stage 6.3 competitive labels")
    absolute_ce = _risk_adjusted_certainty_equivalent(
        evaluation.expected_risk_adjusted_value_cents,
        evaluation.worst_case_risk_adjusted_value_cents,
        spec.risk_aversion_ppm,
    )
    competitive_ce = _risk_adjusted_certainty_equivalent(
        evaluation.expected_competitive_margin_cents,
        evaluation.worst_case_competitive_margin_cents,
        spec.risk_aversion_ppm,
    )
    persona_ce = evaluation.certainty_equivalent_value_cents
    if spec.mode == "persona_aligned":
        objective_score = persona_ce
    elif spec.mode == "absolute_value":
        objective_score = absolute_ce
    else:
        competition = spec.competitive_weight_ppm
        objective_score = (
            (PPM - competition) * persona_ce + competition * competitive_ce
        ) // PPM
    return ObjectiveCandidateScore(
        candidate_id=evaluation.candidate.candidate_id,
        persona_aligned_certainty_equivalent_cents=persona_ce,
        absolute_value_certainty_equivalent_cents=absolute_ce,
        competitive_margin_certainty_equivalent_cents=competitive_ce,
        objective_score_cents=objective_score,
        expected_enterprise_value_cents=(
            evaluation.expected_enterprise_value_cents
        ),
        expected_competitive_margin_cents=(
            evaluation.expected_competitive_margin_cents
        ),
        worst_case_competitive_margin_cents=(
            evaluation.worst_case_competitive_margin_cents
        ),
        expected_final_rank_milli=evaluation.expected_final_rank_milli,
    )


def calibrate_objective_decision(
    plan: StrategicReliabilityPlan,
    spec: ObjectiveCalibrationSpec,
) -> ObjectiveCalibrationDecision:
    scores = [_score_candidate(item, spec) for item in plan.evaluations]
    ranked = max(
        scores,
        key=lambda item: (
            item.objective_score_cents,
            item.expected_enterprise_value_cents,
            item.expected_competitive_margin_cents,
            item.candidate_id,
        ),
    )
    payload = {
        "decision_schema_version": "objective-calibration-decision-v1.0.0",
        "source_plan_hash": plan.plan_hash,
        "objective_spec": spec.model_dump(mode="json"),
        "candidate_scores": [item.model_dump(mode="json") for item in scores],
        "recommended_candidate_id": ranked.candidate_id,
        "experimental_only": True,
        "allowed_in_agent_context": False,
        "decision_hash": "pending",
    }
    payload["decision_hash"] = compute_objective_decision_hash(payload)
    return ObjectiveCalibrationDecision.model_validate(payload)
