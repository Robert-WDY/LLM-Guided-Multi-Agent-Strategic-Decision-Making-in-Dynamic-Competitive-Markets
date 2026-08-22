"""Versioned contracts for the Shared Resilience Cooperation MVP."""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.market.protocols import sha256_hash


CooperationMode = Literal["off", "shared_resilience_v1"]
ProposalResponseDisposition = Literal["accept", "reject"]
CommitmentVerificationStatus = Literal[
    "fulfilled", "partial_betrayal", "betrayed"
]


def apply_cooperation_history_mode(
    raw: dict[str, Any] | None,
    *,
    round_number: int,
    history_mode: Literal["full", "none"],
) -> dict[str, Any] | None:
    """Apply the registered causal-history ablation deterministically."""

    if raw is None:
        return None
    cooperation = copy.deepcopy(raw)
    if history_mode == "full":
        return cooperation
    cooperation["commitment_history"] = []
    cooperation["responses"] = []
    cooperation["cooperation_memory"] = {
        company_id: {
            **record,
            "proposals_received": 0,
            "proposals_sent": 0,
            "accepted_by_self": 0,
            "accepted_by_opponent": 0,
            "commitments_by_opponent": 0,
            "fulfilled_by_opponent": 0,
            "partial_betrayals_by_opponent": 0,
            "betrayed_by_opponent": 0,
            "promised_by_opponent_cents": 0,
            "fulfilled_by_opponent_cents": 0,
            "credibility_ppm": 500_000,
            "history_is_neutralized": True,
        }
        for company_id, record in cooperation.get(
            "cooperation_memory", {}
        ).items()
    }
    active_proposal_ids = {
        item["proposal_id"]
        for item in cooperation.get("active_commitments", [])
    }
    pending_ids = {
        item["proposal_id"]
        for item in cooperation.get("pending_proposals_received", [])
    }
    visible_current_ids = active_proposal_ids | pending_ids
    for field_name in ("proposals_sent", "proposals_received"):
        cooperation[field_name] = [
            item
            for item in cooperation.get(field_name, [])
            if item.get("proposal_id") in visible_current_ids
            or int(item.get("target_round", -1)) >= round_number
        ]
    for company_id, record in cooperation.get("public_credibility", {}).items():
        cooperation["public_credibility"][company_id] = {
            **record,
            "verified_commitment_count": 0,
            "fulfilled_count": 0,
            "partial_betrayal_count": 0,
            "betrayed_count": 0,
            "total_promised_contribution_cents": 0,
            "total_actual_capped_contribution_cents": 0,
            "credibility_ppm": 500_000,
        }
    return cooperation


class SharedResilienceProposalDraft(BaseModel):
    """Model-authored proposal payload; Controller owns identity and IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_type: Literal["shared_resilience"] = "shared_resilience"
    target_round: int = Field(ge=2)
    requested_contribution_cents: int = Field(gt=0)


class SharedResilienceProposal(BaseModel):
    """Authoritative private proposal derived from one delivered message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_schema_version: Literal["shared-resilience-proposal-v1.0.0"] = (
        "shared-resilience-proposal-v1.0.0"
    )
    proposal_id: str
    source_message_id: str
    episode_id: str
    created_round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    sender_company_id: str
    receiver_company_ids: tuple[str, ...] = Field(min_length=1, max_length=1)
    proposal_type: Literal["shared_resilience"] = "shared_resilience"
    target_round: int = Field(ge=2)
    requested_contribution_cents: int = Field(gt=0)

    @classmethod
    def create(
        cls,
        *,
        source_message_id: str,
        episode_id: str,
        created_round: int,
        state_version: int,
        sender_company_id: str,
        receiver_company_ids: list[str],
        draft: SharedResilienceProposalDraft,
    ) -> "SharedResilienceProposal":
        payload = {
            "protocol": "shared-resilience-proposal-v1.0.0",
            "source_message_id": source_message_id,
            "episode_id": episode_id,
            "created_round": created_round,
            "state_version": state_version,
            "sender_company_id": sender_company_id,
            "receiver_company_ids": list(receiver_company_ids),
            "draft": draft.model_dump(mode="json"),
        }
        return cls(
            proposal_id="proposal:" + sha256_hash(payload).removeprefix("sha256:"),
            source_message_id=source_message_id,
            episode_id=episode_id,
            created_round=created_round,
            state_version=state_version,
            sender_company_id=sender_company_id,
            receiver_company_ids=tuple(receiver_company_ids),
            target_round=draft.target_round,
            requested_contribution_cents=draft.requested_contribution_cents,
        )


class ProposalResponseDraft(BaseModel):
    """Model-authored response to a proposal visible in an earlier round."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str = Field(min_length=1)
    response: ProposalResponseDisposition


class ProposalResponse(BaseModel):
    """Controller-attributed response validated against the proposal ledger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    response_schema_version: Literal["proposal-response-v1.0.0"] = (
        "proposal-response-v1.0.0"
    )
    response_id: str
    source_message_id: str
    episode_id: str
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    proposal_id: str
    company_id: str
    response: ProposalResponseDisposition

    @classmethod
    def create(
        cls,
        *,
        source_message_id: str,
        episode_id: str,
        round_number: int,
        state_version: int,
        company_id: str,
        draft: ProposalResponseDraft,
    ) -> "ProposalResponse":
        payload = {
            "protocol": "proposal-response-v1.0.0",
            "source_message_id": source_message_id,
            "episode_id": episode_id,
            "round": round_number,
            "state_version": state_version,
            "company_id": company_id,
            "draft": draft.model_dump(mode="json"),
        }
        return cls(
            response_id="response:" + sha256_hash(payload).removeprefix("sha256:"),
            source_message_id=source_message_id,
            episode_id=episode_id,
            round=round_number,
            state_version=state_version,
            proposal_id=draft.proposal_id,
            company_id=company_id,
            response=draft.response,
        )


class Commitment(BaseModel):
    """Non-binding contribution promise created only from an acceptance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    commitment_schema_version: Literal["commitment-v1.0.0"] = (
        "commitment-v1.0.0"
    )
    commitment_id: str
    proposal_id: str
    response_id: str
    company_id: str
    created_round: int = Field(ge=1)
    target_round: int = Field(ge=2)
    promised_contribution_cents: int = Field(gt=0)
    binding: Literal[False] = False

    @classmethod
    def create(
        cls,
        proposal: SharedResilienceProposal,
        response: ProposalResponse,
    ) -> "Commitment":
        payload = {
            "protocol": "commitment-v1.0.0",
            "proposal_id": proposal.proposal_id,
            "response_id": response.response_id,
            "company_id": response.company_id,
            "created_round": response.round,
            "target_round": proposal.target_round,
            "promised_contribution_cents": (
                proposal.requested_contribution_cents
            ),
        }
        return cls(
            commitment_id="commitment:"
            + sha256_hash(payload).removeprefix("sha256:"),
            proposal_id=proposal.proposal_id,
            response_id=response.response_id,
            company_id=response.company_id,
            created_round=response.round,
            target_round=proposal.target_round,
            promised_contribution_cents=(
                proposal.requested_contribution_cents
            ),
        )


class CommitmentVerification(BaseModel):
    """Deterministic comparison between a promise and settled contribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verification_schema_version: Literal[
        "commitment-verification-v1.0.0"
    ] = "commitment-verification-v1.0.0"
    commitment_id: str
    proposal_id: str
    company_id: str
    target_round: int
    promised_contribution_cents: int
    actual_contribution_cents: int = Field(ge=0)
    fulfillment_ratio_ppm: int = Field(ge=0, le=1_000_000)
    status: CommitmentVerificationStatus

    @model_validator(mode="after")
    def validate_verification_math(self) -> "CommitmentVerification":
        promised = self.promised_contribution_cents
        if promised <= 0:
            raise ValueError("promised contribution must be positive")
        expected_ratio = min(
            1_000_000,
            self.actual_contribution_cents * 1_000_000 // promised,
        )
        expected_status: CommitmentVerificationStatus
        if self.actual_contribution_cents >= promised:
            expected_status = "fulfilled"
        elif self.actual_contribution_cents == 0:
            expected_status = "betrayed"
        else:
            expected_status = "partial_betrayal"
        if (
            self.fulfillment_ratio_ppm != expected_ratio
            or self.status != expected_status
        ):
            raise ValueError("commitment verification math is inconsistent")
        return self

    @classmethod
    def verify(
        cls, commitment: Commitment, actual_contribution_cents: int
    ) -> "CommitmentVerification":
        actual = max(0, int(actual_contribution_cents))
        promised = commitment.promised_contribution_cents
        ratio = min(1_000_000, actual * 1_000_000 // promised)
        status: CommitmentVerificationStatus
        if actual >= promised:
            status = "fulfilled"
        elif actual == 0:
            status = "betrayed"
        else:
            status = "partial_betrayal"
        return cls(
            commitment_id=commitment.commitment_id,
            proposal_id=commitment.proposal_id,
            company_id=commitment.company_id,
            target_round=commitment.target_round,
            promised_contribution_cents=promised,
            actual_contribution_cents=actual,
            fulfillment_ratio_ppm=ratio,
            status=status,
        )


class CredibilityRecord(BaseModel):
    """Controller-computed amount-weighted fulfillment history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    credibility_schema_version: Literal["credibility-v1.0.0"] = (
        "credibility-v1.0.0"
    )
    company_id: str
    verified_commitment_count: int = Field(ge=0)
    fulfilled_count: int = Field(ge=0)
    partial_betrayal_count: int = Field(ge=0)
    betrayed_count: int = Field(ge=0)
    total_promised_contribution_cents: int = Field(ge=0)
    total_actual_capped_contribution_cents: int = Field(ge=0)
    credibility_ppm: int = Field(ge=0, le=1_000_000)

    @classmethod
    def from_verifications(
        cls,
        company_id: str,
        verifications: list[CommitmentVerification],
    ) -> "CredibilityRecord":
        own = [item for item in verifications if item.company_id == company_id]
        promised = sum(item.promised_contribution_cents for item in own)
        actual = sum(
            min(
                item.actual_contribution_cents,
                item.promised_contribution_cents,
            )
            for item in own
        )
        # One synthetic prior commitment at 50% prevents 0/0 and keeps new
        # opponents neutral until real behavior is observed.
        prior_promised = 1_000_000
        prior_actual = 500_000
        score = (actual + prior_actual) * 1_000_000 // (
            promised + prior_promised
        )
        return cls(
            company_id=company_id,
            verified_commitment_count=len(own),
            fulfilled_count=sum(item.status == "fulfilled" for item in own),
            partial_betrayal_count=sum(
                item.status == "partial_betrayal" for item in own
            ),
            betrayed_count=sum(item.status == "betrayed" for item in own),
            total_promised_contribution_cents=promised,
            total_actual_capped_contribution_cents=actual,
            credibility_ppm=score,
        )


class CooperativeBenefitAttribution(BaseModel):
    """Company-level realized value of inherited public resilience.

    The counterfactual removes the public stock available at the start of this
    round while keeping actions and component RNG fixed.  The current
    contribution is still paid in both branches because it creates protection
    for a later round; this avoids crediting a contribution with its own
    same-round benefit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    attribution_schema_version: Literal[
        "cooperative-benefit-attribution-v1.0.0"
    ] = "cooperative-benefit-attribution-v1.0.0"
    company_id: str
    current_contribution_cost_cents: int = Field(ge=0)
    latest_source_contribution_cents: int = Field(ge=0)
    public_protection_received_ppm: int = Field(ge=0, le=1_000_000)
    actual_round_profit_cents: int
    no_public_protection_round_profit_cents: int
    counterfactual_profit_delta_cents: int
    realized_avoided_loss_cents: int = Field(ge=0)
    public_protection_opportunity_cost_cents: int = Field(ge=0)
    net_cooperative_cash_flow_cents: int
    individual_cooperative_roi_ppm: int | None = None
    free_rider_advantage_cents: int = Field(ge=0)
    avoided_next_incident: bool = False
    counterfactual_scope: Literal[
        "current_round_public_stock_vs_zero_stock"
    ] = "current_round_public_stock_vs_zero_stock"
    contribution_benefits_begin_next_round: Literal[True] = True

    @model_validator(mode="after")
    def validate_attribution_math(self) -> "CooperativeBenefitAttribution":
        delta = (
            self.actual_round_profit_cents
            - self.no_public_protection_round_profit_cents
        )
        avoided = max(0, delta)
        opportunity = max(0, -delta)
        expected_roi = (
            delta * 1_000_000 // self.latest_source_contribution_cents
            if self.latest_source_contribution_cents > 0
            else None
        )
        expected_free_rider = (
            avoided
            if self.latest_source_contribution_cents == 0
            and self.public_protection_received_ppm > 0
            else 0
        )
        if (
            self.counterfactual_profit_delta_cents != delta
            or self.realized_avoided_loss_cents != avoided
            or self.public_protection_opportunity_cost_cents != opportunity
            or self.net_cooperative_cash_flow_cents
            != delta - self.latest_source_contribution_cents
            or self.individual_cooperative_roi_ppm != expected_roi
            or self.free_rider_advantage_cents != expected_free_rider
        ):
            raise ValueError("cooperative benefit attribution math is inconsistent")
        return self

    @classmethod
    def from_counterfactual(
        cls,
        *,
        company_id: str,
        current_contribution_cost_cents: int,
        latest_source_contribution_cents: int,
        public_protection_received_ppm: int,
        actual_round_profit_cents: int,
        no_public_protection_round_profit_cents: int,
        avoided_next_incident: bool,
    ) -> "CooperativeBenefitAttribution":
        current_contribution = max(0, int(current_contribution_cost_cents))
        source_contribution = max(0, int(latest_source_contribution_cents))
        actual = int(actual_round_profit_cents)
        counterfactual = int(no_public_protection_round_profit_cents)
        delta = actual - counterfactual
        avoided = max(0, delta)
        return cls(
            company_id=company_id,
            current_contribution_cost_cents=current_contribution,
            latest_source_contribution_cents=source_contribution,
            public_protection_received_ppm=public_protection_received_ppm,
            actual_round_profit_cents=actual,
            no_public_protection_round_profit_cents=counterfactual,
            counterfactual_profit_delta_cents=delta,
            realized_avoided_loss_cents=avoided,
            public_protection_opportunity_cost_cents=max(0, -delta),
            net_cooperative_cash_flow_cents=delta - source_contribution,
            individual_cooperative_roi_ppm=(
                delta * 1_000_000 // source_contribution
                if source_contribution
                else None
            ),
            free_rider_advantage_cents=(
                avoided
                if source_contribution == 0
                and public_protection_received_ppm > 0
                else 0
            ),
            avoided_next_incident=avoided_next_incident,
        )
class CooperationCloseRecord(BaseModel):
    """Immutable cooperation result produced at Communication Close."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    close_schema_version: Literal["cooperation-close-v1.0.0"] = (
        "cooperation-close-v1.0.0"
    )
    mode: CooperationMode
    episode_id: str
    round: int
    state_version: int
    state_hash: str
    communication_transcript_hash: str
    proposals_created: tuple[SharedResilienceProposal, ...] = Field(
        default_factory=tuple
    )
    responses_recorded: tuple[ProposalResponse, ...] = Field(default_factory=tuple)
    commitments_created: tuple[Commitment, ...] = Field(default_factory=tuple)
    active_commitments: tuple[Commitment, ...] = Field(default_factory=tuple)
    credibility_before: dict[str, CredibilityRecord] = Field(
        default_factory=dict
    )
    close_hash: str

    @model_validator(mode="after")
    def validate_close_hash(self) -> "CooperationCloseRecord":
        if self.close_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"close_hash"})
        ):
            raise ValueError("cooperation close hash is inconsistent")
        return self


class CooperationRoundRecord(BaseModel):
    """Full close→action→verification audit record for one settled round."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    round_schema_version: Literal[
        "cooperation-round-v1.0.0", "cooperation-round-v1.1.0"
    ] = "cooperation-round-v1.1.0"
    close: CooperationCloseRecord
    contribution_by_company_cents: dict[str, int]
    total_contribution_cents: int = Field(ge=0)
    industry_resilience_before_ppm: int = Field(ge=0, le=1_000_000)
    public_protection_applied_ppm: int = Field(ge=0, le=1_000_000)
    industry_resilience_after_ppm: int = Field(ge=0, le=1_000_000)
    commitments_due: tuple[Commitment, ...] = Field(default_factory=tuple)
    verifications: tuple[CommitmentVerification, ...] = Field(default_factory=tuple)
    credibility_after: dict[str, CredibilityRecord] = Field(
        default_factory=dict
    )
    benefit_attribution_by_company: dict[
        str, CooperativeBenefitAttribution
    ] = Field(default_factory=dict)
    round_hash: str

    @model_validator(mode="after")
    def validate_contribution_total(self) -> "CooperationRoundRecord":
        if any(value < 0 for value in self.contribution_by_company_cents.values()):
            raise ValueError("company contributions cannot be negative")
        if self.total_contribution_cents != sum(
            self.contribution_by_company_cents.values()
        ):
            raise ValueError("total contribution does not match company values")
        if self.benefit_attribution_by_company:
            if set(self.benefit_attribution_by_company) != set(
                self.contribution_by_company_cents
            ):
                raise ValueError("benefit attribution must cover every company")
            for company_id, attribution in (
                self.benefit_attribution_by_company.items()
            ):
                if (
                    attribution.company_id != company_id
                    or attribution.current_contribution_cost_cents
                    != self.contribution_by_company_cents[company_id]
                ):
                    raise ValueError("benefit attribution company binding mismatch")
        if self.round_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"round_hash"})
        ):
            raise ValueError("cooperation round hash is inconsistent")
        return self


def canonical_hash(value: BaseModel | dict[str, Any]) -> str:
    payload = (
        value.model_dump(mode="json")
        if isinstance(value, BaseModel)
        else dict(value)
    )
    return sha256_hash(payload)


__all__ = [
    "Commitment",
    "CommitmentVerification",
    "CooperativeBenefitAttribution",
    "CooperationCloseRecord",
    "CooperationMode",
    "CooperationRoundRecord",
    "CredibilityRecord",
    "ProposalResponse",
    "ProposalResponseDraft",
    "SharedResilienceProposal",
    "SharedResilienceProposalDraft",
    "apply_cooperation_history_mode",
    "canonical_hash",
]
