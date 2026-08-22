"""Versioned, provider-neutral contracts for strategic Agents."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.agents.personas import PersonaProfile
from game_theory_agent.interaction.contracts import (
    CommunicationMode,
    CommunicationSubmission,
    CommunicationView,
)


OutcomeDirection = Literal["down", "stable", "up"]


class IncidentIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["wait", "partial_repair", "full_repair"] = "wait"
    repair_budget_cents: int = Field(default=0, ge=0)


class AgentRequestedAction(BaseModel):
    """Economic intent only; trusted protocol fields never come from a model."""

    model_config = ConfigDict(extra="forbid")

    price_cents: int
    advertising_budget_cents: int = Field(default=0, ge=0)
    service_budget_cents: int = Field(default=0, ge=0)
    capacity_investment_cents: int = Field(default=0, ge=0)
    resilience_budget_cents: int = Field(default=0, ge=0)
    shared_resilience_contribution_cents: int | None = Field(default=None, ge=0)
    incident_response: IncidentIntent = Field(default_factory=IncidentIntent)
    strategy_summary: str = Field(default="", max_length=500)


class ExpectedOutcome(BaseModel):
    """Forecast directions; these are not success criteria."""

    model_config = ConfigDict(extra="forbid")

    profit: OutcomeDirection = "stable"
    market_share: OutcomeDirection = "stable"
    capacity: OutcomeDirection = "stable"
    risk_exposure: OutcomeDirection = "stable"


class SuccessCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_round_profit_cents: int = 0
    minimum_cash_reserve_cents: int = Field(default=0, ge=0)
    maximum_fixed_spend_cents: int | None = Field(default=None, ge=0)
    minimum_market_share_ppm: int | None = Field(default=None, ge=0, le=1_000_000)


class StrategyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=1, max_length=300)
    situation_summary: str = Field(min_length=1, max_length=1200)
    key_factors: list[str] = Field(default_factory=list, max_length=12)
    strategy_summary: str = Field(min_length=1, max_length=500)
    expected_outcome: ExpectedOutcome = Field(default_factory=ExpectedOutcome)
    success_criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)


MessageDisposition = Literal["accepted", "rejected", "ignored"]


class MessageResponse(BaseModel):
    """An auditable decision-stage response to one visible message."""

    model_config = ConfigDict(extra="forbid")

    message_id: str = Field(min_length=1)
    disposition: MessageDisposition
    rationale: str = Field(min_length=1, max_length=500)


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "agent-decision-v1.0.0", "agent-decision-v1.1.0"
    ] = "agent-decision-v1.1.0"
    plan: StrategyPlan
    requested_action: AgentRequestedAction
    confidence_ppm: int = Field(default=500_000, ge=0, le=1_000_000)
    message_responses: list[MessageResponse] = Field(
        default_factory=list, max_length=16
    )

    @model_validator(mode="after")
    def reject_duplicate_message_responses(self) -> "AgentDecision":
        message_ids = [item.message_id for item in self.message_responses]
        if len(message_ids) != len(set(message_ids)):
            raise ValueError("message_responses cannot reference one message twice")
        return self


class DecisionMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    round: int
    state_version: int
    state_hash: str
    observation_hash: str
    belief_hash: str | None = None
    opponent_model_hash: str | None = None
    utility_inference_hash: str | None = None
    game_theory_advice_hash: str | None = None
    repeated_game_strategy_hash: str | None = None
    rounds_remaining: int
    information_mode: Literal["perfect", "public"]


class AgentIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_id: str
    persona: str
    objective: str


class CommunicationContext(BaseModel):
    """Prompt-ready state for the one-shot, pre-decision communication wave."""

    model_config = ConfigDict(extra="forbid")

    context_schema_version: Literal[
        "communication-context-v1.0.0",
        "communication-context-v1.1.0",
        "communication-context-v1.2.0",
        "communication-context-v1.3.0",
        "communication-context-v1.4.0",
        "communication-context-v1.5.0",
        "communication-context-v1.6.0",
    ] = (
        "communication-context-v1.6.0"
    )
    context_mode: Literal["full", "state_only"] = "full"
    cooperation_history_mode: Literal["full", "none"] = "full"
    communication_mode: CommunicationMode
    meta: DecisionMeta
    identity: AgentIdentity
    persona_profile: PersonaProfile
    market: dict[str, Any]
    shared_resilience: dict[str, Any] | None = None
    cooperation: dict[str, Any] | None = None
    belief_state: dict[str, Any] | None = None
    market_regime: dict[str, Any]
    decision_support: dict[str, Any]
    own_company: dict[str, Any]
    competitors: list[dict[str, Any]]
    risk_signals: list[dict[str, Any]] = Field(default_factory=list)
    active_market_events: list[dict[str, Any]] = Field(default_factory=list)
    recent_communication_views: list[CommunicationView] = Field(
        default_factory=list, max_length=3
    )
    recent_rounds: list[dict[str, Any]] = Field(default_factory=list)
    rolling_summary: dict[str, Any] = Field(default_factory=dict)
    current_plan: dict[str, Any] | None = None
    eligible_recipient_company_ids: list[str] = Field(default_factory=list)
    message_limits: dict[str, Any]
    action_claim_constraints: dict[str, Any]
    same_wave_messages_visible: bool = False
    messages_are_non_binding: bool = True

    @property
    def episode_id(self) -> str:
        return self.meta.episode_id

    @property
    def company_id(self) -> str:
        return self.identity.company_id

    @property
    def round(self) -> int:
        return self.meta.round

    @property
    def state_version(self) -> int:
        return self.meta.state_version

    @property
    def state_hash(self) -> str:
        return self.meta.state_hash


class DecisionContext(BaseModel):
    """Prompt-ready view; it intentionally excludes trusted execution fields."""

    model_config = ConfigDict(extra="forbid")

    context_schema_version: Literal[
        "decision-context-v1.5.0",
        "decision-context-v1.6.0",
        "decision-context-v1.7.0",
        "decision-context-v1.8.0",
        "decision-context-v1.9.0",
        "decision-context-v1.10.0",
        "decision-context-v1.11.0",
        "decision-context-v1.12.0",
        "decision-context-v1.13.0",
        "decision-context-v1.14.0",
    ] = (
        "decision-context-v1.14.0"
    )
    context_mode: Literal["full", "state_only"] = "full"
    cooperation_history_mode: Literal["full", "none"] = "full"
    decision_support_version: Literal["legacy_v1", "economic_v2"] = "economic_v2"
    persona_semantics_version: Literal["legacy_v1", "economic_v2"] = "economic_v2"
    diagnostic_mode: Literal["off", "observe"] = "off"
    meta: DecisionMeta
    identity: AgentIdentity
    persona_profile: PersonaProfile
    market: dict[str, Any]
    shared_resilience: dict[str, Any] | None = None
    cooperation: dict[str, Any] | None = None
    belief_state: dict[str, Any] | None = None
    opponent_model_state: dict[str, Any] | None = None
    utility_inference_state: dict[str, Any] | None = None
    game_theory_advice: dict[str, Any] | None = None
    repeated_game_strategy: dict[str, Any] | None = None
    market_regime: dict[str, Any]
    decision_support: dict[str, Any]
    diagnostic_flags: list[dict[str, Any]] = Field(default_factory=list)
    own_company: dict[str, Any]
    competitors: list[dict[str, Any]]
    risk_signals: list[dict[str, Any]]
    active_market_events: list[dict[str, Any]]
    recent_rounds: list[dict[str, Any]]
    rolling_summary: dict[str, Any]
    critical_events: list[dict[str, Any]]
    current_plan: dict[str, Any] | None = None
    action_constraints: dict[str, Any]
    communication_view: CommunicationView | None = None
    recent_communication_views: list[CommunicationView] = Field(
        default_factory=list, max_length=3
    )

    @property
    def episode_id(self) -> str:
        return self.meta.episode_id

    @property
    def company_id(self) -> str:
        return self.identity.company_id

    @property
    def persona(self) -> str:
        return self.identity.persona

    @property
    def objective(self) -> str:
        return self.identity.objective

    @property
    def round(self) -> int:
        return self.meta.round

    @property
    def state_version(self) -> int:
        return self.meta.state_version

    @property
    def state_hash(self) -> str:
        return self.meta.state_hash

    @property
    def information_mode(self) -> str:
        return self.meta.information_mode

    @property
    def current_company(self) -> dict[str, Any]:
        return self.own_company


class ModelGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    prompt_version: str | None = None
    parsed_output: dict[str, Any]
    raw_response: str = ""
    latency_ms: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    retry_count: int = Field(default=0, ge=0)


CommunicationSilenceReason = Literal[
    "not_silent",
    "intentional",
    "communication_disabled",
    "model_timeout",
    "invalid_model_output",
    "model_error",
    "unsupported_client",
]


class AgentCommunicationResult(BaseModel):
    """Auditable communication output; failures always degrade to silence."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    success: bool
    agent_id: str
    company_id: str
    context: CommunicationContext
    submission: CommunicationSubmission = Field(
        default_factory=CommunicationSubmission
    )
    is_silence: bool
    silence_reason: CommunicationSilenceReason
    model_name: str | None = None
    prompt_version: str | None = None
    raw_response: str = ""
    latency_ms: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    retry_count: int = Field(default=0, ge=0)
    error_code: str | None = None
    error_message: str | None = None
    fallback_to_silence: bool = False

    @model_validator(mode="after")
    def validate_silence_audit(self) -> "AgentCommunicationResult":
        empty = not self.submission.messages
        if self.is_silence != empty:
            raise ValueError("is_silence must match the submitted message list")
        if empty and self.silence_reason == "not_silent":
            raise ValueError("an empty submission must record a silence reason")
        if not empty and self.silence_reason != "not_silent":
            raise ValueError("a non-empty submission cannot record silence")
        return self


class MessageReferenceValidationError(ValueError):
    """A decision referenced a message outside its company-scoped view."""


def validate_decision_message_references(
    decision: AgentDecision,
    communication_view: CommunicationView | None,
) -> None:
    """Reject decision responses that cite messages the Agent could not see."""

    referenced = {item.message_id for item in decision.message_responses}
    if not referenced:
        return
    visible = (
        {item.message_id for item in communication_view.visible_messages}
        if communication_view is not None
        else set()
    )
    hidden = sorted(referenced - visible)
    if hidden:
        raise MessageReferenceValidationError(
            "message_responses reference non-visible message IDs: "
            + ", ".join(hidden)
        )


class AgentDecisionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    success: bool
    agent_id: str
    company_id: str
    context: DecisionContext
    decision: AgentDecision | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    raw_response: str = ""
    latency_ms: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    retry_count: int = Field(default=0, ge=0)
    error_code: str | None = None
    error_message: str | None = None
    fallback_required: bool = False


class MetricChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cash_delta_cents: int
    previous_round_profit_cents: int | None
    profit_change_vs_previous_round_cents: int | None
    market_share_delta_ppm: int
    sales_delta_orders: int
    base_capacity_delta_orders: int
    capacity_investment_cents: int
    capacity_utilization_delta_ppm: int
    awareness_delta_ppm: int
    service_delta_ppm: int
    reputation_delta_ppm: int
    resilience_delta_ppm: int
    stockout_orders: int
    stockout_rate_delta_ppm: int


class CompanyOutcomeSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cash_balance_cents: int
    round_profit_cents: int
    market_share_ppm: int
    sales_orders: int
    price_cents: int
    base_capacity_orders: int
    capacity_utilization_ppm: int
    stockout_rate_ppm: int
    reputation_ppm: int
    resilience_ppm: int
    active_incident: dict[str, Any] | None


class MarketOutcomeSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    realized_demand_orders: int
    average_paid_price_cents: int
    no_purchase_orders: int
    lost_after_stockout_orders: int


class MarketChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")

    realized_demand_delta_orders: int
    average_paid_price_delta_cents: int
    no_purchase_delta_orders: int
    lost_after_stockout_delta_orders: int


class ObservedOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_before: CompanyOutcomeSnapshot
    company_after: CompanyOutcomeSnapshot
    company_changes: MetricChanges
    market_before: MarketOutcomeSnapshot
    market_after: MarketOutcomeSnapshot
    market_changes: MarketChanges
    round_market_conditions: dict[str, int]
    next_round_market_conditions: dict[str, int]
    exogenous_events: list[dict[str, Any]]


class ExpectationAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actual_directions: dict[str, str]
    observed_directions: dict[str, str]
    comparison_basis: dict[str, str]
    matches: dict[str, bool | None]
    mismatches: list[str]
    causal_claim: Literal[
        "unavailable_without_counterfactual",
        "controlled_same_seed_counterfactual",
    ] = (
        "unavailable_without_counterfactual"
    )


class GoalAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criteria: SuccessCriteria
    checks: dict[str, bool | None]
    achieved: bool
    violations: list[str]


class ResultAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_schema_version: Literal["result-analysis-v1.3.0"] = (
        "result-analysis-v1.3.0"
    )
    company_id: str
    settled_round: int
    observed_outcome: ObservedOutcome
    expectation_assessment: ExpectationAssessment
    goal_assessment: GoalAssessment
    counterfactual_analysis: dict[str, Any] | None = None
    resolution_adjustments: list[dict[str, Any]]
    successful_effects: list[str]
    next_round_attention: list[str]
    summary: str

    @property
    def changes(self) -> MetricChanges:
        return self.observed_outcome.company_changes

    @property
    def expectation_matches(self) -> dict[str, bool | None]:
        return self.expectation_assessment.matches

    @property
    def unexpected_effects(self) -> list[str]:
        return self.expectation_assessment.mismatches
