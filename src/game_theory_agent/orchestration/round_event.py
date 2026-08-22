"""Auditable Agent reasoning metadata around an authoritative market transition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from game_theory_agent.agents.contracts import ResultAnalysis
from game_theory_agent.agents.personas import PersonaUtilityAssessment
from game_theory_agent.cooperation import CooperationRoundRecord
from game_theory_agent.interaction.contracts import (
    CommunicationClosure,
    CommunicationMode,
    CommunicationSubmission,
    CommunicationView,
)
from game_theory_agent.information import ObservationSnapshot


class CommunicationGenerationTrace(BaseModel):
    """One company's auditable attempt to generate its round message(s)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    company_id: str
    agent_id: str
    agent_type: Literal["model", "mock", "random", "rule"]
    generation_status: Literal[
        "submitted",
        "silent",
        "fallback",
        "invalid",
        "not_applicable",
        "disabled",
    ]
    observation_hash: str | None = None
    information_snapshot: ObservationSnapshot | None = None
    communication_context: dict[str, Any] | None = None
    submission: CommunicationSubmission | None = None
    accepted_message_ids: list[str] = Field(default_factory=list)
    is_silence: bool | None = None
    silence_reason: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    raw_model_output: str = ""
    latency_ms: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    retry_count: int = Field(default=0, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    validation_errors: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None


class CommunicationViewRecord(BaseModel):
    """Compact public audit index for one closed company-scoped view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    company_id: str
    view_digest: str
    visible_message_ids: list[str] = Field(default_factory=list)
    own_message_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_view(cls, view: CommunicationView) -> "CommunicationViewRecord":
        return cls(
            company_id=view.company_id,
            view_digest=view.view_digest,
            visible_message_ids=[
                message.message_id for message in view.visible_messages
            ],
            own_message_ids=list(view.own_message_ids),
        )


class CommunicationPhaseRecord(BaseModel):
    """Closed communication barrier attached to one authoritative round event."""

    model_config = ConfigDict(extra="forbid")

    phase_schema_version: Literal[
        "communication-phase-v1.0.0",
        "communication-phase-v1.1.0",
    ] = (
        "communication-phase-v1.1.0"
    )
    mode: CommunicationMode
    status: Literal["closed"] = "closed"
    closed: Literal[True] = True
    closure: CommunicationClosure
    company_views: dict[str, CommunicationViewRecord]
    generation_traces: list[CommunicationGenerationTrace] = Field(
        default_factory=list
    )

    @classmethod
    def from_closure(
        cls,
        closure: CommunicationClosure,
        *,
        generation_traces: list[CommunicationGenerationTrace] | None = None,
    ) -> "CommunicationPhaseRecord":
        return cls(
            mode=closure.mode,
            closure=closure,
            company_views={
                company_id: CommunicationViewRecord.from_view(view)
                for company_id, view in closure.views.items()
            },
            generation_traces=list(generation_traces or ()),
        )


class AgentRoundTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    company_id: str
    agent_id: str
    agent_type: Literal["model", "mock", "random", "rule"]
    decision_status: Literal["submitted", "fallback"]
    observation_hash: str | None = None
    observation: dict[str, Any] | None = None
    information_snapshot: ObservationSnapshot | None = None
    decision_context: dict[str, Any] | None = None
    communication_view: CommunicationView | None = None
    persona: str | None = None
    persona_catalog_version: str | None = None
    persona_profile_hash: str | None = None
    persona_utility: PersonaUtilityAssessment | None = None
    belief_before: dict[str, Any] | None = None
    opponent_model: dict[str, Any] | None = None
    utility_inference: dict[str, Any] | None = None
    advisor_output: dict[str, Any] | None = None
    repeated_game_strategy: dict[str, Any] | None = None
    chosen_action: dict[str, Any] | None = None
    counterfactual_results: dict[str, Any] | None = None
    belief_after: dict[str, Any] | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    planner_output: dict[str, Any] | None = None
    raw_model_output: str = ""
    message_responses: list[dict[str, Any]] = Field(default_factory=list)
    requested_action: dict[str, Any] | None = None
    intent_id: str | None = None
    final_action: dict[str, Any]
    resolution_source: str
    resolution_adjustments: list[dict[str, Any]] = Field(default_factory=list)
    result_analysis: ResultAnalysis
    latency_ms: int = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    retry_count: int = Field(default=0, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    validation_errors: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None


class RoundEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_schema_version: Literal[
        "agent-round-event-v1.0.0",
        "agent-round-event-v1.1.0",
        "agent-round-event-v1.2.0",
        "agent-round-event-v1.3.0",
        "agent-round-event-v1.4.0",
        "agent-round-event-v1.5.0",
        "agent-round-event-v1.6.0",
        "agent-round-event-v1.7.0",
        "agent-round-event-v1.8.0",
        "agent-round-event-v1.9.0",
    ] = "agent-round-event-v1.9.0"
    event_id: str
    episode_id: str
    settled_round: int
    state_before_hash: str
    state_after_hash: str
    joint_action_hash: str
    state_before: dict[str, Any] = Field(default_factory=dict)
    state_after: dict[str, Any] = Field(default_factory=dict)
    joint_action: dict[str, dict[str, Any]] = Field(default_factory=dict)
    phases: list[str] = Field(default_factory=list)
    random_draw_summary: dict[str, int] = Field(default_factory=dict)
    step_result: dict[str, Any]
    traces: list[AgentRoundTrace]
    communication_phase: CommunicationPhaseRecord | None = None
    cooperation_round: CooperationRoundRecord | None = None


class JsonlRoundEventLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: RoundEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json())
            handle.write("\n")

    def read_all(self) -> tuple[RoundEvent, ...]:
        if not self.path.exists():
            return ()
        events: list[RoundEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    events.append(RoundEvent.model_validate_json(line))
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid round event at {self.path}:{line_number}: {exc}"
                    ) from exc
        return tuple(events)
