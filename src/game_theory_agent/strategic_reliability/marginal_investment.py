"""Public-only marginal investment screen for reliable strategic advice.

The screen deliberately starts from a zero-discretionary-spend action and
tests advertising, service, capacity, resilience, cooperation, and repair one
at a time.  Individually useful items are then added to a portfolio only when
their combined forecast still clears value, downside, and Persona floors.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import Field, model_validator

from game_theory_agent.agents.personas import PersonaProfile
from game_theory_agent.belief import BeliefState
from game_theory_agent.market import (
    CompanyAction,
    IncidentResponse,
    IncidentResponseMode,
    MarketConfig,
    MarketEnv,
    MarketState,
)
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.opponent import OpponentModelState

from .contracts import (
    CandidateEconomicAction,
    StrategicActionCandidate,
    StrictModel,
)
from .rollout import AuthoritativeMarketRolloutEvaluator, generate_candidate_actions


class InvestmentMarginalAssessment(StrictModel):
    assessment_schema_version: Literal[
        "investment-marginal-assessment-v1.0.0"
    ] = "investment-marginal-assessment-v1.0.0"
    dimension: str
    probe_candidate_id: str
    probe_action: CandidateEconomicAction
    budget_cents: int = Field(ge=0)
    marginal_expected_enterprise_value_cents: int
    marginal_expected_profit_cents: int
    marginal_expected_loss_cents: int
    marginal_worst_case_risk_adjusted_value_cents: int
    marginal_certainty_equivalent_value_cents: int
    passes_expected_value_floor: bool
    passes_worst_case_floor: bool
    passes_persona_floor: bool
    individually_eligible: bool
    selected_in_portfolio: bool
    reason_codes: list[str]
    assessment_hash: str

    @model_validator(mode="after")
    def validate_assessment(self) -> "InvestmentMarginalAssessment":
        eligible = (
            self.passes_expected_value_floor
            and self.passes_worst_case_floor
            and self.passes_persona_floor
        )
        if self.individually_eligible != eligible:
            raise ValueError("marginal investment eligibility mismatch")
        if self.selected_in_portfolio and not self.individually_eligible:
            raise ValueError("ineligible investment cannot enter portfolio")
        if self.assessment_hash != compute_marginal_assessment_hash(self):
            raise ValueError("marginal investment assessment hash mismatch")
        return self


class MarginalInvestmentPlan(StrictModel):
    plan_schema_version: Literal[
        "marginal-investment-plan-v1.0.0"
    ] = "marginal-investment-plan-v1.0.0"
    evaluator_version: Literal[
        "public-forecast-marginal-rollout-v1.0.0"
    ] = "public-forecast-marginal-rollout-v1.0.0"
    episode_id: str
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    company_id: str
    public_decision_input_hash: str
    horizon_rounds: int = Field(ge=1, le=20)
    scenario_count: int = Field(ge=1, le=100)
    baseline_action: CandidateEconomicAction
    assessments: list[InvestmentMarginalAssessment]
    selected_dimensions: list[str]
    selected_action: CandidateEconomicAction
    uses_only_public_and_own_private_inputs: Literal[True] = True
    uses_authoritative_hidden_market_state: Literal[False] = False
    plan_hash: str

    @model_validator(mode="after")
    def validate_plan(self) -> "MarginalInvestmentPlan":
        selected = sorted(
            item.dimension for item in self.assessments if item.selected_in_portfolio
        )
        if sorted(self.selected_dimensions) != selected:
            raise ValueError("marginal investment selected dimensions mismatch")
        if self.plan_hash != compute_marginal_investment_plan_hash(self):
            raise ValueError("marginal investment plan hash mismatch")
        return self


def compute_marginal_assessment_hash(
    assessment: InvestmentMarginalAssessment | Mapping[str, Any],
) -> str:
    payload = (
        assessment.model_dump(mode="json")
        if isinstance(assessment, InvestmentMarginalAssessment)
        else dict(assessment)
    )
    payload.pop("assessment_hash", None)
    return sha256_hash(
        {
            "hash_protocol_version": "investment-marginal-assessment-hash-v1.0.0",
            "assessment": payload,
        }
    )


def compute_marginal_investment_plan_hash(
    plan: MarginalInvestmentPlan | Mapping[str, Any],
) -> str:
    payload = (
        plan.model_dump(mode="json")
        if isinstance(plan, MarginalInvestmentPlan)
        else dict(plan)
    )
    payload.pop("plan_hash", None)
    return sha256_hash(
        {
            "hash_protocol_version": "marginal-investment-plan-hash-v1.0.0",
            "plan": payload,
        }
    )


def _candidate_action(action: CompanyAction) -> CandidateEconomicAction:
    return CandidateEconomicAction(
        price_cents=action.price_cents,
        advertising_budget_cents=action.advertising_budget_cents,
        service_budget_cents=action.service_budget_cents,
        capacity_investment_cents=action.capacity_investment_cents,
        resilience_budget_cents=action.resilience_budget_cents,
        shared_resilience_contribution_cents=(
            action.shared_resilience_contribution_cents
        ),
        incident_response_mode=action.incident_response.mode.value,
        repair_budget_cents=action.incident_response.repair_budget_cents,
    )


def _baseline_candidate(state: MarketState, company_id: str) -> StrategicActionCandidate:
    shared = 0 if state.shared_resilience is not None else None
    return StrategicActionCandidate(
        candidate_id="maintain",
        label="maintain",
        action=CandidateEconomicAction(
            price_cents=state.company(company_id).commercial.price_cents,
            shared_resilience_contribution_cents=shared,
        ),
        changed_dimensions=["price"],
        rationale="保持当前价格，所有可选投入归零，作为边际收益比较基准。",
    )


def _combine_action(
    baseline: CandidateEconomicAction,
    additions: list[CandidateEconomicAction],
) -> CandidateEconomicAction:
    payload = baseline.model_dump(mode="json")
    for item in additions:
        payload["advertising_budget_cents"] += item.advertising_budget_cents
        payload["service_budget_cents"] += item.service_budget_cents
        payload["capacity_investment_cents"] += item.capacity_investment_cents
        payload["resilience_budget_cents"] += item.resilience_budget_cents
        if payload["shared_resilience_contribution_cents"] is not None:
            payload["shared_resilience_contribution_cents"] += (
                item.shared_resilience_contribution_cents or 0
            )
        if item.incident_response_mode != "wait":
            payload["incident_response_mode"] = item.incident_response_mode
            payload["repair_budget_cents"] = item.repair_budget_cents
    return CandidateEconomicAction.model_validate(payload)


def _validate_portfolio_action(
    config: MarketConfig,
    state: MarketState,
    company_id: str,
    action: CandidateEconomicAction,
) -> CandidateEconomicAction | None:
    env = MarketEnv(config)
    env.load_state(state)
    raw = CompanyAction(
        action_id=(
            f"marginal-portfolio:{state.episode_id}:{state.round}:{company_id}"
        ),
        episode_id=state.episode_id,
        agent_id=company_id,
        round=state.round,
        state_version=state.state_version,
        price_cents=action.price_cents,
        advertising_budget_cents=action.advertising_budget_cents,
        service_budget_cents=action.service_budget_cents,
        capacity_investment_cents=action.capacity_investment_cents,
        resilience_budget_cents=action.resilience_budget_cents,
        shared_resilience_contribution_cents=(
            action.shared_resilience_contribution_cents
        ),
        incident_response=IncidentResponse(
            IncidentResponseMode(action.incident_response_mode),
            action.repair_budget_cents,
        ),
        strategy_summary="公开边际投入筛选后的经营基线",
    )
    validated = env.validate_action(raw, company_id)
    return _candidate_action(validated.action) if validated.valid and validated.action else None


def _portfolio_candidate(action: CandidateEconomicAction) -> StrategicActionCandidate:
    return StrategicActionCandidate(
        candidate_id="marginal_portfolio",
        label="status_quo",
        action=action,
        changed_dimensions=["marginally_screened_portfolio"],
        rationale="只保留通过单项与组合价值、最坏情景和人格效用门槛的投入。",
    )


def build_marginal_investment_plan(
    *,
    config: MarketConfig,
    state: MarketState,
    company_id: str,
    persona_profile: PersonaProfile,
    public_decision_input_hash: str,
    opponent_model: OpponentModelState | None,
    belief_state: BeliefState | None,
    horizon_rounds: int,
    scenario_count: int,
) -> MarginalInvestmentPlan:
    """Build a deterministic marginal screen from the legal forecast state."""

    evaluator = AuthoritativeMarketRolloutEvaluator(config)
    baseline = _baseline_candidate(state, company_id)
    templates = generate_candidate_actions(config, state, company_id)
    probe_labels = {
        "increase_advertising",
        "increase_service",
        "increase_capacity",
        "increase_resilience",
        "shared_resilience_contribution",
        "repair_incident",
    }
    probes = [item for item in templates if item.label in probe_labels]
    initial = evaluator.evaluate(
        state=state,
        company_id=company_id,
        persona_profile=persona_profile,
        opponent_model=opponent_model,
        belief_state=belief_state,
        horizon_rounds=horizon_rounds,
        scenario_count=scenario_count,
        candidates=(baseline, *probes),
    )
    by_id = {item.candidate.candidate_id: item for item in initial.evaluations}
    base = by_id["maintain"]
    raw: dict[str, dict[str, Any]] = {}
    eligible: list[tuple[str, CandidateEconomicAction, int, int]] = []
    for probe in probes:
        result = by_id[probe.candidate_id]
        ev = result.expected_enterprise_value_cents - base.expected_enterprise_value_cents
        profit = (
            result.expected_cumulative_profit_delta_cents
            - base.expected_cumulative_profit_delta_cents
        )
        loss = result.expected_loss_cents - base.expected_loss_cents
        worst = (
            result.worst_case_risk_adjusted_value_cents
            - base.worst_case_risk_adjusted_value_cents
        )
        ce = (
            result.certainty_equivalent_value_cents
            - base.certainty_equivalent_value_cents
        )
        passes_ev = ev > 0
        passes_worst = worst >= 0
        passes_persona = ce > 0
        reasons: list[str] = []
        if not passes_ev:
            reasons.append("expected_enterprise_value_not_positive")
        if not passes_worst:
            reasons.append("worst_case_value_below_baseline")
        if not passes_persona:
            reasons.append("persona_certainty_equivalent_not_positive")
        dimension = probe.changed_dimensions[0]
        budget = sum(
            (
                probe.action.advertising_budget_cents,
                probe.action.service_budget_cents,
                probe.action.capacity_investment_cents,
                probe.action.resilience_budget_cents,
                probe.action.shared_resilience_contribution_cents or 0,
                probe.action.repair_budget_cents,
            )
        )
        raw[dimension] = {
            "assessment_schema_version": (
                "investment-marginal-assessment-v1.0.0"
            ),
            "dimension": dimension,
            "probe_candidate_id": probe.candidate_id,
            "probe_action": probe.action.model_dump(mode="json"),
            "budget_cents": budget,
            "marginal_expected_enterprise_value_cents": ev,
            "marginal_expected_profit_cents": profit,
            "marginal_expected_loss_cents": loss,
            "marginal_worst_case_risk_adjusted_value_cents": worst,
            "marginal_certainty_equivalent_value_cents": ce,
            "passes_expected_value_floor": passes_ev,
            "passes_worst_case_floor": passes_worst,
            "passes_persona_floor": passes_persona,
            "individually_eligible": passes_ev and passes_worst and passes_persona,
            "selected_in_portfolio": False,
            "reason_codes": reasons,
            "assessment_hash": "pending",
        }
        if passes_ev and passes_worst and passes_persona:
            eligible.append((dimension, probe.action, ce, ev))

    selected: list[tuple[str, CandidateEconomicAction]] = []
    # A deterministic greedy interaction check prevents individually attractive
    # items from becoming an over-investment bundle.
    for dimension, action, _ce, _ev in sorted(
        eligible, key=lambda item: (-item[2], -item[3], item[0])
    ):
        tentative = _combine_action(
            baseline.action, [item[1] for item in selected] + [action]
        )
        valid = _validate_portfolio_action(config, state, company_id, tentative)
        if valid is None:
            raw[dimension]["reason_codes"].append("portfolio_action_illegal")
            continue
        combined = evaluator.evaluate(
            state=state,
            company_id=company_id,
            persona_profile=persona_profile,
            opponent_model=opponent_model,
            belief_state=belief_state,
            horizon_rounds=horizon_rounds,
            scenario_count=scenario_count,
            candidates=(baseline, _portfolio_candidate(valid)),
        )
        combined_by_id = {
            item.candidate.candidate_id: item for item in combined.evaluations
        }
        combined_base = combined_by_id["maintain"]
        portfolio = combined_by_id["marginal_portfolio"]
        portfolio_passes = (
            portfolio.expected_enterprise_value_cents
            > combined_base.expected_enterprise_value_cents
            and portfolio.worst_case_risk_adjusted_value_cents
            >= combined_base.worst_case_risk_adjusted_value_cents
            and portfolio.certainty_equivalent_value_cents
            > combined_base.certainty_equivalent_value_cents
        )
        if portfolio_passes:
            selected.append((dimension, action))
            raw[dimension]["selected_in_portfolio"] = True
        else:
            raw[dimension]["reason_codes"].append(
                "portfolio_interaction_floor_failed"
            )

    selected_action = _combine_action(baseline.action, [item[1] for item in selected])
    selected_action = _validate_portfolio_action(
        config, state, company_id, selected_action
    ) or baseline.action
    assessments: list[InvestmentMarginalAssessment] = []
    for dimension in sorted(raw):
        payload = raw[dimension]
        if payload["selected_in_portfolio"]:
            payload["reason_codes"].append("selected_after_portfolio_check")
        elif payload["individually_eligible"] and not payload["reason_codes"]:
            payload["reason_codes"].append("not_selected_after_portfolio_check")
        payload["assessment_hash"] = compute_marginal_assessment_hash(payload)
        assessments.append(InvestmentMarginalAssessment.model_validate(payload))

    plan_payload: dict[str, Any] = {
        "plan_schema_version": "marginal-investment-plan-v1.0.0",
        "evaluator_version": "public-forecast-marginal-rollout-v1.0.0",
        "episode_id": state.episode_id,
        "round": state.round,
        "state_version": state.state_version,
        "company_id": company_id,
        "public_decision_input_hash": public_decision_input_hash,
        "horizon_rounds": min(horizon_rounds, state.rounds_remaining),
        "scenario_count": min(scenario_count, 100),
        "baseline_action": baseline.action.model_dump(mode="json"),
        "assessments": [item.model_dump(mode="json") for item in assessments],
        "selected_dimensions": [item[0] for item in selected],
        "selected_action": selected_action.model_dump(mode="json"),
        "uses_only_public_and_own_private_inputs": True,
        "uses_authoritative_hidden_market_state": False,
        "plan_hash": "pending",
    }
    plan_payload["plan_hash"] = compute_marginal_investment_plan_hash(plan_payload)
    return MarginalInvestmentPlan.model_validate(plan_payload)
