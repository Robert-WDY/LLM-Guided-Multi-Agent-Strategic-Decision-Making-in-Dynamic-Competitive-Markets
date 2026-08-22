"""Deterministic, simultaneous one-shot communication round."""

from __future__ import annotations

from game_theory_agent.interaction.contracts import (
    CommunicationClosure,
    CommunicationMode,
    CommunicationSubmission,
    CommunicationView,
    DeliveredMessage,
    compute_communication_view_digest,
)
from game_theory_agent.cooperation.contracts import SharedResilienceProposal
from game_theory_agent.market.protocols import sha256_hash


class CommunicationError(RuntimeError):
    """Base class for communication protocol failures."""


class CommunicationConflictError(CommunicationError):
    """A participant tried to replace an accepted submission."""


class CommunicationStateError(CommunicationError):
    """An operation is not valid in the current communication phase."""


class CommunicationValidationError(CommunicationError):
    """A submission violates participant or visibility rules."""


class CommunicationRoundLedger:
    """Collect submissions without exposing any until an atomic close.

    All participants submit against one frozen market state. A repeated identical
    submission is idempotent; a different second submission is a conflict. The
    close operation is deterministic and returns a separately hashed view for
    each company, so a private message never appears in another Agent's context.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        round_number: int,
        state_version: int,
        state_hash: str,
        company_ids: list[str] | tuple[str, ...],
        mode: CommunicationMode,
    ) -> None:
        ordered_company_ids = tuple(sorted(company_ids))
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        if round_number < 1:
            raise ValueError("round_number must be positive")
        if state_version < 0:
            raise ValueError("state_version must not be negative")
        if not state_hash:
            raise ValueError("state_hash must not be empty")
        if not ordered_company_ids or len(set(ordered_company_ids)) != len(
            ordered_company_ids
        ):
            raise ValueError("company_ids must be non-empty and unique")
        if any(not company_id for company_id in ordered_company_ids):
            raise ValueError("company_ids must not contain empty values")
        if mode not in {"off", "public_only", "public_private"}:
            raise ValueError("unsupported communication mode")
        self.episode_id = episode_id
        self.round_number = round_number
        self.state_version = state_version
        self.state_hash = state_hash
        self.company_ids = ordered_company_ids
        self.mode = mode
        self._submissions: dict[str, CommunicationSubmission] = {}
        self._delivered: dict[str, tuple[DeliveredMessage, ...]] = {}
        self._closure: CommunicationClosure | None = None

    @property
    def status(self) -> str:
        return "closed" if self._closure is not None else "open"

    def submit(
        self,
        sender_company_id: str,
        submission: CommunicationSubmission,
    ) -> tuple[DeliveredMessage, ...]:
        if self.mode == "off":
            raise CommunicationStateError("communication is disabled")
        if self._closure is not None:
            raise CommunicationStateError("communication is already closed")
        if sender_company_id not in self.company_ids:
            raise CommunicationValidationError("sender is not an episode company")
        prior = self._submissions.get(sender_company_id)
        if prior is not None:
            if prior == submission:
                return tuple(
                    message.model_copy(deep=True)
                    for message in self._delivered[sender_company_id]
                )
            raise CommunicationConflictError(
                "a different submission was already accepted for this sender"
            )

        delivered: list[DeliveredMessage] = []
        for message_index, draft in enumerate(submission.messages):
            if self.mode == "public_only" and draft.channel == "private":
                raise CommunicationValidationError(
                    "private messages are disabled in public_only mode"
                )
            if draft.channel == "private":
                recipient = draft.recipients[0]
                if recipient not in self.company_ids:
                    raise CommunicationValidationError(
                        "private recipient is not an episode company"
                    )
                if recipient == sender_company_id:
                    raise CommunicationValidationError(
                        "a company cannot send a private message to itself"
                    )
            message_payload = {
                "episode_id": self.episode_id,
                "round": self.round_number,
                "state_version": self.state_version,
                "state_hash": self.state_hash,
                "sender_company_id": sender_company_id,
                "message_index": message_index,
                "draft": draft.model_dump(mode="json"),
            }
            message_id = sha256_hash(message_payload)
            cooperation_proposal = (
                SharedResilienceProposal.create(
                    source_message_id=message_id,
                    episode_id=self.episode_id,
                    created_round=self.round_number,
                    state_version=self.state_version,
                    sender_company_id=sender_company_id,
                    receiver_company_ids=list(draft.recipients),
                    draft=draft.cooperation_proposal,
                )
                if draft.cooperation_proposal is not None
                else None
            )
            delivered.append(
                DeliveredMessage(
                    message_id=message_id,
                    episode_id=self.episode_id,
                    round=self.round_number,
                    state_version=self.state_version,
                    state_hash=self.state_hash,
                    sender_company_id=sender_company_id,
                    channel=draft.channel,
                    recipients=list(draft.recipients),
                    speech_act=draft.speech_act,
                    content=draft.content,
                    own_action_claim=draft.own_action_claim,
                    requested_peer_action=draft.requested_peer_action,
                    cooperation_proposal=cooperation_proposal,
                    cooperation_response=draft.cooperation_response,
                )
            )
        accepted = tuple(delivered)
        self._submissions[sender_company_id] = submission.model_copy(deep=True)
        self._delivered[sender_company_id] = accepted
        return tuple(message.model_copy(deep=True) for message in accepted)

    def close(self) -> CommunicationClosure:
        if self._closure is not None:
            return self._closure.model_copy(deep=True)

        all_messages = [
            message
            for company_id in self.company_ids
            for message in self._delivered.get(company_id, ())
        ]
        audit_payload = {
            "protocol": "simultaneous-one-shot-v1.0.0",
            "mode": self.mode,
            "episode_id": self.episode_id,
            "round": self.round_number,
            "state_version": self.state_version,
            "state_hash": self.state_hash,
            "messages": [
                message.model_dump(mode="json") for message in all_messages
            ],
        }
        transcript_hash = sha256_hash(audit_payload)
        views: dict[str, CommunicationView] = {}
        for company_id in self.company_ids:
            visible = [
                message
                for message in all_messages
                if self._is_visible(message, company_id)
            ]
            own_message_ids = [
                message.message_id
                for message in visible
                if message.sender_company_id == company_id
            ]
            provisional_view = CommunicationView(
                mode=self.mode,
                status="closed",
                episode_id=self.episode_id,
                round=self.round_number,
                state_version=self.state_version,
                state_hash=self.state_hash,
                company_id=company_id,
                visible_messages=visible,
                own_message_ids=own_message_ids,
                view_digest="pending",
            )
            views[company_id] = provisional_view.model_copy(
                update={
                    "view_digest": compute_communication_view_digest(
                        provisional_view
                    )
                },
                deep=True,
            )

        submitted = [
            company_id
            for company_id in self.company_ids
            if company_id in self._submissions
        ]
        silent = [
            company_id
            for company_id in self.company_ids
            if company_id not in self._submissions
            or not self._submissions[company_id].messages
        ]
        self._closure = CommunicationClosure(
            mode=self.mode,
            episode_id=self.episode_id,
            round=self.round_number,
            state_version=self.state_version,
            state_hash=self.state_hash,
            transcript_hash=transcript_hash,
            all_messages=all_messages,
            submitted_company_ids=submitted,
            silent_company_ids=silent,
            views=views,
        )
        return self._closure.model_copy(deep=True)

    @staticmethod
    def _is_visible(message: DeliveredMessage, company_id: str) -> bool:
        return (
            message.channel == "public"
            or message.sender_company_id == company_id
            or company_id in message.recipients
        )
