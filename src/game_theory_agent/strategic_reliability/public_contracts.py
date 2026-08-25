"""Agent-visible contracts for public-information market rollouts.

Unlike the authoritative Stage 6 oracle, these records are derived only from
the observing company's legal view.  They may enter an incomplete-information
Agent context, but remain non-binding finite-horizon forecasts.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import Field, model_validator

from game_theory_agent.market.protocols import sha256_hash

from .contracts import StrictModel, StrategicActionCandidate


class PublicForecastStateRecord(StrictModel):
    forecast_schema_version: Literal["public-forecast-state-v1.0.0"] = (
        "public-forecast-state-v1.0.0"
    )
    episode_id: str
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    company_id: str
    public_decision_input_hash: str
    forecast_state_hash: str
    forecast_seed: int = Field(ge=0, lt=1 << 64)
    uses_public_state: Literal[True] = True
    uses_own_private_state: Literal[True] = True
    uses_belief_state: bool
    uses_public_opponent_model: bool
    uses_authoritative_hidden_market_state: Literal[False] = False
    hidden_opponent_fields_are_estimated: Literal[True] = True
    assumptions: list[str] = Field(min_length=1)


class PublicRolloutCandidateSummary(StrictModel):
    candidate: StrategicActionCandidate
    expected_enterprise_value_delta_cents: int
    expected_cumulative_profit_delta_cents: int
    expected_loss_cents: int = Field(ge=0)
    worst_case_risk_adjusted_value_cents: int
    expected_persona_utility_ppm: int
    certainty_equivalent_value_cents: int


class PublicStrategicAdvice(StrictModel):
    advice_schema_version: Literal[
        "public-strategic-advice-v3.0.0",
        "public-pareto-advice-v4.0.0",
        "public-pareto-reliable-advice-v5.0.0",
        "public-pareto-marginal-advice-v6.0.0",
        "public-pareto-abstention-advice-v7.0.0",
    ] = (
        "public-strategic-advice-v3.0.0"
    )
    advisor_mode: Literal[
        "public_rollout_v3",
        "pareto_rollout_v4",
        "pareto_reliable_v5",
        "pareto_reliable_v6",
        "pareto_reliable_v7",
    ] = (
        "public_rollout_v3"
    )
    advisor_model_version: Literal[
        "public-observation-market-rollout-v1.0.0",
        "public-pareto-market-rollout-v1.0.0",
        "public-pareto-reliable-market-rollout-v1.0.0",
        "public-pareto-marginal-market-rollout-v2.0.0",
        "public-pareto-abstention-market-rollout-v3.0.0",
    ] = "public-observation-market-rollout-v1.0.0"
    episode_id: str
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    company_id: str
    persona_id: str
    persona_profile_hash: str
    public_decision_input_hash: str
    forecast_state_hash: str
    belief_hash: str | None = None
    opponent_model_hash: str | None = None
    horizon_rounds: int = Field(ge=1, le=20)
    scenario_count: int = Field(ge=1, le=100)
    candidate_actions: list[PublicRolloutCandidateSummary] = Field(min_length=2)
    recommended_candidate_id: str | None = None
    recommended_action: dict[str, Any] | None = None
    expected_gain_over_baseline_cents: int
    baseline_regret_cents: int = Field(ge=0)
    recommendation_reason: str
    recommendation_is_non_binding: Literal[True] = True
    approximate_best_response: Literal[True] = True
    claims_nash_equilibrium: Literal[False] = False
    uses_authoritative_hidden_market_state: Literal[False] = False
    uses_hidden_opponent_state: Literal[False] = False
    hidden_opponent_fields_are_estimated: Literal[True] = True
    allowed_in_public_agent_context: Literal[True] = True
    pareto_decision: dict[str, Any] | None = None
    selection_situation: str | None = None
    promotion_evidence_sha256: str | None = None
    planner_recommended_candidate_id: str | None = None
    safe_candidate_ids: list[str] | None = None
    excluded_candidates: list[dict[str, Any]] | None = None
    reliability_gate: dict[str, Any] | None = None
    investment_marginal_plan: dict[str, Any] | None = None
    execution_disposition: Literal["recommend", "defer_to_agent"] | None = None
    withheld_candidate_id: str | None = None
    limitations: list[str] = Field(min_length=1)
    advice_hash: str

    @model_validator(mode="after")
    def validate_ranking_and_hash(self) -> "PublicStrategicAdvice":
        by_id = {
            item.candidate.candidate_id: item for item in self.candidate_actions
        }
        if len(by_id) != len(self.candidate_actions):
            raise ValueError("public rollout candidates must have unique ids")
        if "maintain" not in by_id:
            raise ValueError("public rollout baseline is missing")
        if self.advisor_mode == "public_rollout_v3":
            if (
                self.advice_schema_version
                != "public-strategic-advice-v3.0.0"
                or self.advisor_model_version
                != "public-observation-market-rollout-v1.0.0"
                or self.pareto_decision is not None
                or self.selection_situation is not None
                or self.promotion_evidence_sha256 is not None
                or self.planner_recommended_candidate_id is not None
                or self.safe_candidate_ids is not None
                or self.excluded_candidates is not None
                or self.reliability_gate is not None
                or self.investment_marginal_plan is not None
                or self.execution_disposition is not None
                or self.withheld_candidate_id is not None
            ):
                raise ValueError("public_rollout_v3 contract fields are inconsistent")
            selected = max(
                self.candidate_actions,
                key=lambda item: (
                    item.certainty_equivalent_value_cents,
                    item.expected_enterprise_value_delta_cents,
                    item.worst_case_risk_adjusted_value_cents,
                    item.candidate.candidate_id,
                ),
            )
        else:
            from .pareto_planner import (
                PROMOTION_EVIDENCE_SHA256,
                ParetoPlannerDecision,
            )
            from .reliable_planner import (
                ParetoAbstentionGate,
                ParetoReliabilityGate,
            )
            from .marginal_investment import MarginalInvestmentPlan

            is_v5 = self.advisor_mode == "pareto_reliable_v5"
            is_v6 = self.advisor_mode == "pareto_reliable_v6"
            is_v7 = self.advisor_mode == "pareto_reliable_v7"
            expected_schema = (
                "public-pareto-abstention-advice-v7.0.0"
                if is_v7
                else "public-pareto-marginal-advice-v6.0.0"
                if is_v6
                else
                "public-pareto-reliable-advice-v5.0.0"
                if is_v5
                else "public-pareto-advice-v4.0.0"
            )
            expected_model = (
                "public-pareto-abstention-market-rollout-v3.0.0"
                if is_v7
                else "public-pareto-marginal-market-rollout-v2.0.0"
                if is_v6
                else
                "public-pareto-reliable-market-rollout-v1.0.0"
                if is_v5
                else "public-pareto-market-rollout-v1.0.0"
            )
            if (
                self.advice_schema_version != expected_schema
                or self.advisor_model_version != expected_model
                or self.pareto_decision is None
                or self.promotion_evidence_sha256 != PROMOTION_EVIDENCE_SHA256
            ):
                raise ValueError("Pareto advice contract fields are inconsistent")
            decision = ParetoPlannerDecision.model_validate(self.pareto_decision)
            if self.selection_situation != decision.situation.situation:
                raise ValueError("Pareto selection situation mismatch")
            if is_v5 or is_v6 or is_v7:
                if (
                    self.reliability_gate is None
                    or self.safe_candidate_ids is None
                    or self.excluded_candidates is None
                    or self.planner_recommended_candidate_id
                    != decision.recommended_candidate_id
                ):
                    raise ValueError("reliable Pareto fields are missing")
                gate = (
                    ParetoAbstentionGate.model_validate(self.reliability_gate)
                    if is_v7
                    else ParetoReliabilityGate.model_validate(
                        self.reliability_gate
                    )
                )
                if (
                    gate.planner_decision_hash != decision.decision_hash
                    or gate.planner_candidate_id
                    != self.planner_recommended_candidate_id
                    or gate.effective_candidate_id
                    != self.recommended_candidate_id
                    or gate.safe_candidate_ids != self.safe_candidate_ids
                    or [
                        item.model_dump(mode="json")
                        for item in gate.excluded_candidates
                    ]
                    != self.excluded_candidates
                ):
                    raise ValueError("reliable Pareto gate binding mismatch")
                if is_v7:
                    if (
                        self.execution_disposition
                        != gate.execution_disposition
                        or self.withheld_candidate_id
                        != gate.withheld_candidate_id
                    ):
                        raise ValueError("v7 abstention disposition mismatch")
                elif (
                    self.execution_disposition is not None
                    or self.withheld_candidate_id is not None
                ):
                    raise ValueError("legacy advice cannot carry v7 disposition")
                if is_v6 or is_v7:
                    if self.investment_marginal_plan is None:
                        raise ValueError("marginal investment plan is missing")
                    marginal = MarginalInvestmentPlan.model_validate(
                        self.investment_marginal_plan
                    )
                    status_quo = by_id.get("status_quo")
                    if (
                        gate.gate_schema_version
                        != (
                            "pareto-reliability-gate-v3.0.0"
                            if is_v7
                            else "pareto-reliability-gate-v2.0.0"
                        )
                        or gate.marginal_investment_plan_hash != marginal.plan_hash
                        or marginal.public_decision_input_hash
                        != self.public_decision_input_hash
                        or status_quo is None
                        or status_quo.candidate.action != marginal.selected_action
                    ):
                        raise ValueError("marginal plan binding mismatch")
                elif self.investment_marginal_plan is not None:
                    raise ValueError("v5 cannot carry a marginal investment plan")
            else:
                if (
                    self.recommended_candidate_id
                    != decision.recommended_candidate_id
                    or self.planner_recommended_candidate_id is not None
                    or self.safe_candidate_ids is not None
                    or self.excluded_candidates is not None
                    or self.reliability_gate is not None
                    or self.investment_marginal_plan is not None
                    or self.execution_disposition is not None
                    or self.withheld_candidate_id is not None
                ):
                    raise ValueError("Pareto v4 recommendation mismatch")
            selected = (
                by_id[self.recommended_candidate_id]
                if self.recommended_candidate_id is not None
                else None
            )
        if selected is None:
            if (
                self.advisor_mode != "pareto_reliable_v7"
                or self.execution_disposition != "defer_to_agent"
                or self.recommended_candidate_id is not None
                or self.recommended_action is not None
            ):
                raise ValueError("only v7 abstention may omit a recommendation")
        else:
            if self.recommended_candidate_id != selected.candidate.candidate_id:
                raise ValueError("public rollout recommendation is not selected")
            if self.recommended_action != selected.candidate.action.model_dump(mode="json"):
                raise ValueError("recommended action does not match recommended candidate")
        baseline = by_id["maintain"]
        gain = (
            0
            if selected is None
            else selected.certainty_equivalent_value_cents
            - baseline.certainty_equivalent_value_cents
        )
        if self.expected_gain_over_baseline_cents != gain:
            raise ValueError("public rollout baseline gain mismatch")
        if self.baseline_regret_cents != max(0, gain):
            raise ValueError("public rollout baseline regret mismatch")
        if self.advice_hash != compute_public_advice_hash(self):
            raise ValueError("public strategic advice hash mismatch")
        return self


def compute_public_advice_hash(
    advice: PublicStrategicAdvice | Mapping[str, Any],
) -> str:
    payload = (
        advice.model_dump(mode="json")
        if isinstance(advice, PublicStrategicAdvice)
        else dict(advice)
    )
    payload.pop("advice_hash", None)
    if payload.get("advisor_mode") == "public_rollout_v3":
        for field in (
            "pareto_decision",
            "selection_situation",
            "promotion_evidence_sha256",
        ):
            if payload.get(field) is None:
                payload.pop(field, None)
    if payload.get("advisor_mode") in {"public_rollout_v3", "pareto_rollout_v4"}:
        for field in (
            "planner_recommended_candidate_id",
            "safe_candidate_ids",
            "excluded_candidates",
            "reliability_gate",
            "investment_marginal_plan",
        ):
            if payload.get(field) is None:
                payload.pop(field, None)
    if payload.get("advisor_mode") == "pareto_reliable_v5":
        if payload.get("investment_marginal_plan") is None:
            payload.pop("investment_marginal_plan", None)
    if payload.get("advisor_mode") != "pareto_reliable_v7":
        for field in ("execution_disposition", "withheld_candidate_id"):
            if payload.get(field) is None:
                payload.pop(field, None)
    return sha256_hash(
        {
            "hash_protocol_version": "public-strategic-advice-hash-v1.0.0",
            "advice_schema_version": payload.get("advice_schema_version"),
            "advice": payload,
        }
    )
