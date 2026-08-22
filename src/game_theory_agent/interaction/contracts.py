"""Versioned contracts for the simultaneous communication phase."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.cooperation.contracts import (
    ProposalResponseDraft,
    SharedResilienceProposal,
    SharedResilienceProposalDraft,
)
from game_theory_agent.market.protocols import sha256_hash


CommunicationMode = Literal["off", "public_only", "public_private"]
CommunicationStatus = Literal["open", "closed"]
MessageChannel = Literal["public", "private"]
SpeechAct = Literal[
    "statement",
    "proposal",
    "promise",
    "threat",
    "question",
    "response",
    "other",
]


class PartialActionClaim(BaseModel):
    """A non-binding, machine-readable claim about a possible market action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    price_cents: int | None = Field(default=None, ge=0)
    advertising_budget_cents: int | None = Field(default=None, ge=0)
    service_budget_cents: int | None = Field(default=None, ge=0)
    capacity_investment_cents: int | None = Field(default=None, ge=0)
    resilience_budget_cents: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_at_least_one_claim(self) -> "PartialActionClaim":
        if not any(
            getattr(self, field_name) is not None
            for field_name in type(self).model_fields
        ):
            raise ValueError("a partial action claim must set at least one field")
        return self


class MessageDraft(BaseModel):
    """Model-authored content; identity and round fields are controller-owned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    channel: MessageChannel
    recipients: list[str] = Field(default_factory=list, max_length=1)
    speech_act: SpeechAct = "statement"
    content: str = Field(min_length=1, max_length=500)
    own_action_claim: PartialActionClaim | None = None
    requested_peer_action: PartialActionClaim | None = None
    cooperation_proposal: SharedResilienceProposalDraft | None = None
    cooperation_response: ProposalResponseDraft | None = None

    @model_validator(mode="after")
    def validate_channel_shape(self) -> "MessageDraft":
        if self.channel == "public" and self.recipients:
            raise ValueError("public messages cannot name recipients")
        if self.channel == "private" and len(self.recipients) != 1:
            raise ValueError("private messages require exactly one recipient")
        if self.cooperation_proposal is not None:
            if self.channel != "private" or self.speech_act != "proposal":
                raise ValueError(
                    "a cooperation proposal must be a private proposal message"
                )
        if self.cooperation_response is not None:
            if self.channel != "private" or self.speech_act != "response":
                raise ValueError(
                    "a cooperation response must be a private response message"
                )
        if (
            self.cooperation_proposal is not None
            and self.cooperation_response is not None
        ):
            raise ValueError(
                "one message cannot contain both a proposal and a response"
            )
        return self


class CommunicationSubmission(BaseModel):
    """At most one public and one private message from one company per round."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["communication-submission-v1.0.0"] = (
        "communication-submission-v1.0.0"
    )
    messages: list[MessageDraft] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def enforce_channel_quotas(self) -> "CommunicationSubmission":
        channels = [message.channel for message in self.messages]
        if channels.count("public") > 1:
            raise ValueError("only one public message is allowed per round")
        if channels.count("private") > 1:
            raise ValueError("only one private message is allowed per round")
        return self


class DeliveredMessage(BaseModel):
    """Authoritative immutable message created from a model draft."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_schema_version: Literal["delivered-message-v1.0.0"] = (
        "delivered-message-v1.0.0"
    )
    message_id: str
    episode_id: str
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    state_hash: str
    sender_company_id: str
    channel: MessageChannel
    recipients: list[str] = Field(default_factory=list, max_length=1)
    speech_act: SpeechAct
    content: str = Field(min_length=1, max_length=500)
    own_action_claim: PartialActionClaim | None = None
    requested_peer_action: PartialActionClaim | None = None
    cooperation_proposal: SharedResilienceProposal | None = None
    cooperation_response: ProposalResponseDraft | None = None


class CommunicationView(BaseModel):
    """The closed transcript slice that one company is permitted to see."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    view_schema_version: Literal["communication-view-v1.0.0"] = (
        "communication-view-v1.0.0"
    )
    mode: CommunicationMode
    status: CommunicationStatus
    episode_id: str
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    state_hash: str
    company_id: str
    visible_messages: list[DeliveredMessage] = Field(default_factory=list)
    own_message_ids: list[str] = Field(default_factory=list)
    view_digest: str
    messages_are_non_binding: Literal[True] = True
    opponent_content_is_untrusted: Literal[True] = True


class CommunicationClosure(BaseModel):
    """Internal audit result; only per-company views go to individual Agents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    closure_schema_version: Literal["communication-closure-v1.0.0"] = (
        "communication-closure-v1.0.0"
    )
    mode: CommunicationMode
    episode_id: str
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    state_hash: str
    transcript_hash: str
    all_messages: list[DeliveredMessage] = Field(default_factory=list)
    submitted_company_ids: list[str] = Field(default_factory=list)
    silent_company_ids: list[str] = Field(default_factory=list)
    views: dict[str, CommunicationView]


def compute_communication_view_digest(view: CommunicationView) -> str:
    """Recompute the canonical digest instead of trusting a supplied value."""

    return sha256_hash(
        {
            "protocol": "communication-view-v1.0.0",
            "mode": view.mode,
            "episode_id": view.episode_id,
            "round": view.round,
            "state_version": view.state_version,
            "state_hash": view.state_hash,
            "company_id": view.company_id,
            "visible_messages": [
                message.model_dump(mode="json")
                for message in view.visible_messages
            ],
        }
    )


def validate_communication_view_digest(view: CommunicationView) -> None:
    """Reject a view whose body no longer matches its controller digest."""

    expected = compute_communication_view_digest(view)
    if view.view_digest != expected:
        raise ValueError("communication_view digest does not match its contents")
    visible_ids = [message.message_id for message in view.visible_messages]
    if len(visible_ids) != len(set(visible_ids)):
        raise ValueError("communication_view contains duplicate message ids")
    expected_own_ids = [
        message.message_id
        for message in view.visible_messages
        if message.sender_company_id == view.company_id
    ]
    if view.own_message_ids != expected_own_ids:
        raise ValueError(
            "communication_view own_message_ids do not match visible messages"
        )
