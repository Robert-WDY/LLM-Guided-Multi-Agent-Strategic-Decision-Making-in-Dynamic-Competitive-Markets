"""Typed inputs and outputs for the single-agent decision runtime."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.agents.personas import (
    PersonaProfile as VersionedPersonaProfile,
    PersonaTraits,
    PersonaUtilityWeights,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PersonaProfile(StrictModel):
    persona_id: str = Field(default="growth", min_length=1, max_length=80)
    label: str = Field(default="进取型经营者", min_length=1, max_length=120)
    risk_tolerance: float = Field(default=0.78, ge=0, le=1)
    competitive_orientation: float = Field(default=0.84, ge=0, le=1)
    investment_horizon: float = Field(default=0.67, ge=0, le=1)
    cooperation_orientation: float = Field(default=0.35, ge=0, le=1)
    uncertainty_tolerance: float = Field(default=0.62, ge=0, le=1)
    communication_style: Literal["concise", "balanced", "detailed"] = "balanced"


class PersonaTraceManifest(StrictModel):
    """可审计且可验证的版本化 Persona 快照。"""

    profile_schema_version: Literal["persona-profile-v1.0.0"] = "persona-profile-v1.0.0"
    persona_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    catalog_version: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=120)
    objective: str = Field(min_length=1, max_length=500)
    utility_weights_ppm: PersonaUtilityWeights
    traits_ppm: PersonaTraits
    social_welfare_enabled: bool = False
    cooperation_enabled: bool = False
    profile_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def profile_hash_matches_manifest(self) -> "PersonaTraceManifest":
        profile = VersionedPersonaProfile.model_validate(
            self.model_dump(mode="json", exclude={"profile_hash"})
        )
        if self.profile_hash != profile.profile_hash:
            raise ValueError("persona profile_hash does not match manifest")
        return self


DEFAULT_SYSTEM_PROMPT = (
    "你是生鲜市场经营模拟中的公司决策代理。只返回指定的 JSON 决策，"
    "仅使用输入中提供的证据，并简洁说明依据、权衡和预期结果。"
    "不要输出隐藏推理过程，也不要修改约束或身份字段。"
)
DEFAULT_USER_PROMPT_TEMPLATE = "请根据以下本轮决策输入生成候选方案，并只返回符合 Schema 的 JSON：\n{{decision_input}}"

# System prompt 模板与数值人格预设是两套独立概念，名称不得混用。
PROMPT_PRESETS: tuple[dict[str, str], ...] = (
    {
        "preset_id": "prompt_breakthrough",
        "label": "竞争突破",
        "description": "主动寻找扩大市场份额和竞争优势的机会。",
        "system_prompt": DEFAULT_SYSTEM_PROMPT
        + " 采用竞争突破风格，在约束允许时主动争取份额和增长机会，同时说明现金代价。",
    },
    {
        "preset_id": "prompt_synthesis",
        "label": "综合研判",
        "description": "综合比较利润、份额、现金和运营约束后决策。",
        "system_prompt": DEFAULT_SYSTEM_PROMPT
        + " 采用综合研判风格，在短期收益、市场位置和持续经营之间明确取舍。",
    },
    {
        "preset_id": "prompt_guardian",
        "label": "风险审慎",
        "description": "优先控制现金、事故和经营下行风险。",
        "system_prompt": DEFAULT_SYSTEM_PROMPT
        + " 采用风险审慎风格，优先识别现金、产能、事故和需求不确定性并保留安全边际。",
    },
    {
        "preset_id": "prompt_cooperation",
        "label": "协同长期",
        "description": "兼顾长期能力建设、服务质量和市场稳定。",
        "system_prompt": DEFAULT_SYSTEM_PROMPT
        + " 采用协同长期风格，兼顾长期能力、服务质量、韧性和市场稳定，避免短视竞争。",
    },
)

DEFAULT_PROMPT_PRESET_BY_COMPANY: dict[str, str] = {
    "company_A": "prompt_breakthrough",
    "company_B": "prompt_synthesis",
    "company_C": "prompt_guardian",
    "company_D": "prompt_cooperation",
}


class PromptTemplate(StrictModel):
    """Editable provider messages with one required dynamic decision-input slot."""

    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT, min_length=1, max_length=12000)
    user_prompt_template: str = Field(
        default=DEFAULT_USER_PROMPT_TEMPLATE,
        min_length=1,
        max_length=24000,
    )

    @model_validator(mode="after")
    def decision_input_slot_is_required(self) -> "PromptTemplate":
        if self.user_prompt_template.count("{{decision_input}}") != 1:
            raise ValueError("user_prompt_template must contain {{decision_input}} exactly once")
        return self


class IncidentResponse(StrictModel):
    mode: Literal["wait", "partial_repair", "full_repair"] = "wait"
    repair_budget_cents: int = Field(default=0, ge=0)


class EconomicAction(StrictModel):
    price_cents: int = Field(gt=0)
    advertising_budget_cents: int = Field(default=0, ge=0)
    service_budget_cents: int = Field(default=0, ge=0)
    capacity_investment_cents: int = Field(default=0, ge=0)
    resilience_budget_cents: int = Field(default=0, ge=0)
    shared_resilience_contribution_cents: int = Field(default=0, ge=0)
    incident_response: IncidentResponse = Field(default_factory=IncidentResponse)
    strategy_summary: str = Field(default="", max_length=240)


class PersonaInfluence(StrictModel):
    trait_key: Literal[
        "risk_tolerance",
        "competitive_orientation",
        "investment_horizon",
        "cooperation_orientation",
        "uncertainty_tolerance",
        "communication_style",
    ]
    direction: Literal["increase", "decrease", "neutral"]
    affected_choice: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=1, max_length=300)


class DecisionCandidate(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=80)
    action: EconomicAction
    evidence_paths: list[str] = Field(default_factory=list, max_length=12)
    tradeoffs: list[str] = Field(default_factory=list, max_length=4)
    risk_flags: list[str] = Field(default_factory=list, max_length=4)
    expected_outcome: str = Field(default="", max_length=240)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_persona_influences(cls, value: Any) -> Any:
        """Read old traces without exposing Persona as a current AI output field."""

        if isinstance(value, dict) and "persona_influences" in value:
            value = {key: item for key, item in value.items() if key != "persona_influences"}
        return value


class DecisionProposal(StrictModel):
    candidates: list[DecisionCandidate] = Field(min_length=3, max_length=3)
    selected_candidate_id: str = Field(min_length=1, max_length=80)
    selection_reason_codes: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def selected_candidate_must_exist(self) -> "DecisionProposal":
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate ids must be unique")
        if self.selected_candidate_id not in ids:
            raise ValueError("selected candidate id does not exist")
        return self

    @property
    def selected_candidate(self) -> DecisionCandidate:
        return next(
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == self.selected_candidate_id
        )


class SnapshotKey(StrictModel):
    episode_id: str = Field(min_length=1)
    company_id: str = Field(min_length=1)
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    state_hash: str = Field(min_length=1)


class RoundFeedback(StrictModel):
    settled_round: int = Field(ge=1)
    own_action: dict[str, Any]
    own_result: dict[str, int]
    market: dict[str, Any]
    active_events_during_round: list[dict[str, Any]] = Field(default_factory=list)
    resolved_signal_outcomes: list[dict[str, Any]] = Field(default_factory=list)


class EpisodeMemoryView(StrictModel):
    history_limit: int = Field(default=2, ge=1, le=5)
    recent_feedback: list[RoundFeedback] = Field(default_factory=list, max_length=5)
    previous_selected_candidate_id: str | None = None
    previous_expected_outcome: str | None = None
    diagnostic_codes: list[str] = Field(default_factory=list, max_length=12)


class StrategyReflection(StrictModel):
    source: Literal["none", "deterministic"]
    lesson_codes: list[str] = Field(default_factory=list, max_length=12)
    adjustments: list[str] = Field(default_factory=list, max_length=12)
    evidence_paths: list[str] = Field(default_factory=list, max_length=12)
    summary: str = Field(default="", max_length=500)


class DecisionContext(StrictModel):
    snapshot_key: SnapshotKey
    observation: dict[str, Any]
    action_contract: dict[str, Any]
    memory: EpisodeMemoryView
    reflection: StrategyReflection


class IntentDraft(StrictModel):
    snapshot_key: SnapshotKey
    agent_id: str = Field(min_length=1, max_length=120)
    action: EconomicAction
    rationale: str = Field(default="", max_length=500)
    expected_outcome: str = Field(default="", max_length=500)


class PromptAudit(StrictModel):
    """Exact provider messages that are safe to expose for decision auditing."""

    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)


DecisionStatus = Literal[
    "running",
    "accepted",
    "no_intent",
    "terminal",
    "stale",
    "submission_unknown",
]


class DecisionTrace(StrictModel):
    trace_version: Literal[
        "single-agent-trace-v1.0.0",
        "single-agent-trace-v1.1.0",
        "single-agent-trace-v1.2.0",
    ] = "single-agent-trace-v1.2.0"
    episode_id: str
    company_id: str
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    status: DecisionStatus
    model_id: str
    persona: PersonaProfile | None = None
    persona_manifest: PersonaTraceManifest | None = None
    candidates: list[DecisionCandidate] = Field(default_factory=list, max_length=4)
    selected_candidate_id: str | None = None
    selection_reason_codes: list[str] = Field(default_factory=list, max_length=8)
    validation_errors: list[str] = Field(default_factory=list, max_length=20)
    repair_attempts: int = Field(default=0, ge=0, le=1)
    provider_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = Field(default=0, ge=0)
    provider_finish_reason: str | None = None
    provider_error_category: str | None = None
    provider_usage_available: bool = False
    intent_receipt: dict[str, Any] | None = None
    memory_view: EpisodeMemoryView | None = None
    strategy_reflection: StrategyReflection | None = None
    prepared_intent: IntentDraft | None = None
    prompt_audit: PromptAudit | None = None
    error_code: str | None = None


class RoundDecisionResult(StrictModel):
    status: DecisionStatus
    episode_id: str
    company_id: str
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    intent_id: str | None = None
    trace: DecisionTrace
