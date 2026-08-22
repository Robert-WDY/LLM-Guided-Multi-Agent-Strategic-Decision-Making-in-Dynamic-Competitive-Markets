"""Deterministic proposal, commitment, verification, and credibility ledger."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from game_theory_agent.cooperation.contracts import (
    Commitment,
    CommitmentVerification,
    CooperationCloseRecord,
    CooperationMode,
    CooperationRoundRecord,
    CooperativeBenefitAttribution,
    CredibilityRecord,
    ProposalResponse,
    SharedResilienceProposal,
    canonical_hash,
)


class CooperationProtocolError(ValueError):
    """Structured cooperation data violated the MVP protocol."""


class CooperationLedger:
    """Episode-scoped authority outside MarketState for non-binding promises."""

    def __init__(
        self,
        *,
        mode: CooperationMode,
        episode_id: str,
        company_ids: Iterable[str],
        max_rounds: int,
        max_contribution_cents: int = 4_000_000,
    ) -> None:
        self.mode = mode
        self.episode_id = episode_id
        self.company_ids = tuple(sorted(company_ids))
        self.max_rounds = int(max_rounds)
        self.max_contribution_cents = int(max_contribution_cents)
        self._proposals: dict[str, SharedResilienceProposal] = {}
        self._responses: dict[str, ProposalResponse] = {}
        self._response_by_proposal_company: dict[tuple[str, str], str] = {}
        self._commitments: dict[str, Commitment] = {}
        self._verifications: dict[str, CommitmentVerification] = {}
        self._closes: dict[int, CooperationCloseRecord] = {}
        self._rounds: dict[int, CooperationRoundRecord] = {}

    @property
    def proposals(self) -> tuple[SharedResilienceProposal, ...]:
        return tuple(self._proposals[key] for key in sorted(self._proposals))

    @property
    def commitments(self) -> tuple[Commitment, ...]:
        return tuple(self._commitments[key] for key in sorted(self._commitments))

    @property
    def verifications(self) -> tuple[CommitmentVerification, ...]:
        return tuple(
            self._verifications[key] for key in sorted(self._verifications)
        )

    def credibility(self) -> dict[str, CredibilityRecord]:
        verified = list(self.verifications)
        return {
            company_id: CredibilityRecord.from_verifications(
                company_id, verified
            )
            for company_id in self.company_ids
        }

    def has_closed_round(self, round_number: int) -> bool:
        return int(round_number) in self._closes

    def cooperation_memory(self, company_id: str) -> dict[str, dict[str, Any]]:
        """Build a company-scoped, opponent-indexed summary from the ledger."""

        if company_id not in self.company_ids:
            raise CooperationProtocolError("unknown company")
        credibility = self.credibility()
        memory: dict[str, dict[str, Any]] = {}
        for opponent_id in self.company_ids:
            if opponent_id == company_id:
                continue
            received = [
                item
                for item in self.proposals
                if item.sender_company_id == opponent_id
                and company_id in item.receiver_company_ids
            ]
            sent = [
                item
                for item in self.proposals
                if item.sender_company_id == company_id
                and opponent_id in item.receiver_company_ids
            ]
            received_ids = {item.proposal_id for item in received}
            sent_ids = {item.proposal_id for item in sent}
            accepted_by_self = [
                item
                for item in self._responses.values()
                if item.proposal_id in received_ids
                and item.company_id == company_id
                and item.response == "accept"
            ]
            accepted_by_opponent = [
                item
                for item in self._responses.values()
                if item.proposal_id in sent_ids
                and item.company_id == opponent_id
                and item.response == "accept"
            ]
            opponent_commitments = [
                item
                for item in self.commitments
                if item.company_id == opponent_id
                and item.proposal_id in sent_ids
            ]
            commitment_ids = {
                item.commitment_id for item in opponent_commitments
            }
            opponent_verifications = [
                item
                for item in self.verifications
                if item.commitment_id in commitment_ids
            ]
            record = credibility[opponent_id]
            memory[opponent_id] = {
                "memory_schema_version": "cooperation-memory-v1.0.0",
                "opponent_company_id": opponent_id,
                "proposals_received": len(received),
                "proposals_sent": len(sent),
                "accepted_by_self": len(accepted_by_self),
                "accepted_by_opponent": len(accepted_by_opponent),
                "commitments_by_opponent": len(opponent_commitments),
                "fulfilled_by_opponent": sum(
                    item.status == "fulfilled"
                    for item in opponent_verifications
                ),
                "partial_betrayals_by_opponent": sum(
                    item.status == "partial_betrayal"
                    for item in opponent_verifications
                ),
                "betrayed_by_opponent": sum(
                    item.status == "betrayed"
                    for item in opponent_verifications
                ),
                "promised_by_opponent_cents": sum(
                    item.promised_contribution_cents
                    for item in opponent_verifications
                ),
                "fulfilled_by_opponent_cents": sum(
                    min(
                        item.actual_contribution_cents,
                        item.promised_contribution_cents,
                    )
                    for item in opponent_verifications
                ),
                "credibility_ppm": record.credibility_ppm,
                "history_is_neutralized": False,
            }
        return memory

    def validate_submission(
        self,
        *,
        sender_company_id: str,
        round_number: int,
        submission: Any,
    ) -> None:
        """Fail closed before a bad structured message can poison a close."""

        structured = [
            message
            for message in submission.messages
            if message.cooperation_proposal is not None
            or message.cooperation_response is not None
        ]
        if self.mode == "off" and structured:
            raise CooperationProtocolError(
                "cooperation payload is disabled for this episode"
            )
        for message in structured:
            proposal = message.cooperation_proposal
            if proposal is not None:
                if proposal.target_round <= round_number:
                    raise CooperationProtocolError(
                        "proposal target must be later than its creation round"
                    )
                if proposal.target_round > self.max_rounds:
                    raise CooperationProtocolError(
                        "proposal target exceeds the episode horizon"
                    )
                if (
                    proposal.requested_contribution_cents
                    > self.max_contribution_cents
                ):
                    raise CooperationProtocolError(
                        "requested contribution exceeds the action bound"
                    )
            response = message.cooperation_response
            if response is None:
                continue
            known = self._proposals.get(response.proposal_id)
            if known is None:
                raise CooperationProtocolError(
                    "response references an unknown earlier proposal"
                )
            if known.created_round >= round_number:
                raise CooperationProtocolError(
                    "same-wave proposal responses are not allowed"
                )
            if round_number > known.target_round:
                raise CooperationProtocolError(
                    "response arrived after the proposal target round"
                )
            if sender_company_id not in known.receiver_company_ids:
                raise CooperationProtocolError(
                    "response sender is not a proposal receiver"
                )
            if list(message.recipients or []) != [known.sender_company_id]:
                raise CooperationProtocolError(
                    "proposal response must be private to the proposer"
                )
            if (response.proposal_id, sender_company_id) in (
                self._response_by_proposal_company
            ):
                raise CooperationProtocolError(
                    "company already responded to this proposal"
                )

    def company_view(
        self,
        company_id: str,
        *,
        round_number: int,
        include_memory: bool = True,
    ) -> dict[str, Any]:
        """Return trusted, company-scoped cooperation state for one decision."""

        if company_id not in self.company_ids:
            raise CooperationProtocolError("unknown company")
        sent = [
            item
            for item in self.proposals
            if item.sender_company_id == company_id
        ]
        received = [
            item
            for item in self.proposals
            if company_id in item.receiver_company_ids
        ]
        visible_proposal_ids = {
            item.proposal_id for item in (*sent, *received)
        }
        responses = [
            item
            for item in self._responses.values()
            if item.proposal_id in visible_proposal_ids
        ]
        responded = {
            (item.proposal_id, item.company_id) for item in responses
        }
        pending_received = [
            item
            for item in received
            if item.target_round >= round_number
            and (item.proposal_id, company_id) not in responded
        ]
        visible_commitments = [
            item
            for item in self.commitments
            if item.proposal_id in visible_proposal_ids
        ]
        visible_verifications = [
            item
            for item in self.verifications
            if item.proposal_id in visible_proposal_ids
            or item.company_id == company_id
        ]
        view = {
            "cooperation_view_schema_version": (
                "cooperation-view-v1.1.0"
                if include_memory
                else "cooperation-view-v1.0.0"
            ),
            "mode": self.mode,
            "round": round_number,
            "proposals_sent": [item.model_dump(mode="json") for item in sent],
            "proposals_received": [
                item.model_dump(mode="json") for item in received
            ],
            "pending_proposals_received": [
                item.model_dump(mode="json") for item in pending_received
            ],
            "responses": [item.model_dump(mode="json") for item in responses],
            "active_commitments": [
                item.model_dump(mode="json")
                for item in visible_commitments
                if item.target_round >= round_number
                and item.commitment_id not in self._verifications
            ],
            "commitment_history": [
                item.model_dump(mode="json") for item in visible_verifications
            ],
            "public_credibility": {
                key: value.model_dump(mode="json")
                for key, value in self.credibility().items()
            },
            "commitments_are_non_binding": True,
        }
        if include_memory:
            view["cooperation_memory"] = self.cooperation_memory(company_id)
        return view

    def close_round(self, closure: Any) -> CooperationCloseRecord:
        """Derive authoritative proposals/responses without touching market."""

        prior = self._closes.get(int(closure.round))
        if prior is not None:
            if prior.communication_transcript_hash != closure.transcript_hash:
                raise CooperationProtocolError(
                    "round was already closed with another transcript"
                )
            return prior.model_copy(deep=True)
        if closure.episode_id != self.episode_id:
            raise CooperationProtocolError("closure belongs to another episode")

        proposals_created: list[SharedResilienceProposal] = []
        responses_recorded: list[ProposalResponse] = []
        commitments_created: list[Commitment] = []
        structured_messages = [
            message
            for message in closure.all_messages
            if message.cooperation_proposal is not None
            or message.cooperation_response is not None
        ]
        if self.mode == "off" and structured_messages:
            raise CooperationProtocolError(
                "cooperation payload is disabled for this episode"
            )

        for message in closure.all_messages:
            proposal = message.cooperation_proposal
            if proposal is None:
                continue
            if proposal.proposal_id in self._proposals:
                raise CooperationProtocolError("duplicate proposal id")
            if proposal.created_round != closure.round:
                raise CooperationProtocolError("proposal round mismatch")
            if proposal.target_round <= closure.round:
                raise CooperationProtocolError(
                    "proposal target must allow a later response wave"
                )
            if proposal.target_round > self.max_rounds:
                raise CooperationProtocolError(
                    "proposal target exceeds the episode horizon"
                )
            if (
                proposal.requested_contribution_cents
                > self.max_contribution_cents
            ):
                raise CooperationProtocolError(
                    "requested contribution exceeds the action bound"
                )
            receiver = proposal.receiver_company_ids[0]
            if receiver not in self.company_ids or receiver == proposal.sender_company_id:
                raise CooperationProtocolError("invalid proposal receiver")
            self._proposals[proposal.proposal_id] = proposal
            proposals_created.append(proposal)

        for message in closure.all_messages:
            draft = message.cooperation_response
            if draft is None:
                continue
            proposal = self._proposals.get(draft.proposal_id)
            if proposal is None:
                raise CooperationProtocolError("response references unknown proposal")
            if proposal.created_round >= closure.round:
                raise CooperationProtocolError(
                    "same-wave proposal responses are not allowed"
                )
            if closure.round > proposal.target_round:
                raise CooperationProtocolError("response arrived after target round")
            if message.sender_company_id not in proposal.receiver_company_ids:
                raise CooperationProtocolError(
                    "response sender is not a proposal receiver"
                )
            if list(message.recipients) != [proposal.sender_company_id]:
                raise CooperationProtocolError(
                    "proposal response must be private to the proposer"
                )
            response_key = (proposal.proposal_id, message.sender_company_id)
            if response_key in self._response_by_proposal_company:
                raise CooperationProtocolError(
                    "company already responded to this proposal"
                )
            response = ProposalResponse.create(
                source_message_id=message.message_id,
                episode_id=closure.episode_id,
                round_number=closure.round,
                state_version=closure.state_version,
                company_id=message.sender_company_id,
                draft=draft,
            )
            self._responses[response.response_id] = response
            self._response_by_proposal_company[response_key] = response.response_id
            responses_recorded.append(response)
            if response.response == "accept":
                commitment = Commitment.create(proposal, response)
                self._commitments[commitment.commitment_id] = commitment
                commitments_created.append(commitment)

        active = [
            commitment
            for commitment in self.commitments
            if commitment.target_round >= closure.round
            and commitment.commitment_id not in self._verifications
        ]
        credibility_before = self.credibility()
        payload = {
            "close_schema_version": "cooperation-close-v1.0.0",
            "mode": self.mode,
            "episode_id": closure.episode_id,
            "round": closure.round,
            "state_version": closure.state_version,
            "state_hash": closure.state_hash,
            "communication_transcript_hash": closure.transcript_hash,
            "proposals_created": [
                item.model_dump(mode="json") for item in proposals_created
            ],
            "responses_recorded": [
                item.model_dump(mode="json") for item in responses_recorded
            ],
            "commitments_created": [
                item.model_dump(mode="json") for item in commitments_created
            ],
            "active_commitments": [
                item.model_dump(mode="json") for item in active
            ],
            "credibility_before": {
                key: value.model_dump(mode="json")
                for key, value in credibility_before.items()
            },
        }
        record = CooperationCloseRecord(
            **payload,
            close_hash=canonical_hash(payload),
        )
        self._closes[closure.round] = record
        return record.model_copy(deep=True)

    def settle_round(
        self,
        *,
        round_number: int,
        final_actions: Mapping[str, Mapping[str, Any]],
        industry_resilience_before_ppm: int,
        public_protection_applied_ppm: int,
        industry_resilience_after_ppm: int,
        benefit_attribution_by_company: Mapping[
            str, CooperativeBenefitAttribution | Mapping[str, Any]
        ] | None = None,
    ) -> CooperationRoundRecord:
        prior = self._rounds.get(round_number)
        if prior is not None:
            return prior.model_copy(deep=True)
        close = self._closes.get(round_number)
        if close is None:
            raise CooperationProtocolError(
                "cooperation close is required before settlement"
            )
        contributions = {
            company_id: max(
                0,
                int(
                    final_actions[company_id].get(
                        "shared_resilience_contribution_cents", 0
                    )
                    or 0
                ),
            )
            for company_id in self.company_ids
        }
        due = [
            commitment
            for commitment in self.commitments
            if commitment.target_round == round_number
            and commitment.commitment_id not in self._verifications
        ]
        verifications = [
            CommitmentVerification.verify(
                commitment,
                contributions[commitment.company_id],
            )
            for commitment in due
        ]
        for verification in verifications:
            self._verifications[verification.commitment_id] = verification
        credibility_after = self.credibility()
        attributions = {
            company_id: (
                value
                if isinstance(value, CooperativeBenefitAttribution)
                else CooperativeBenefitAttribution.model_validate(value)
            )
            for company_id, value in (
                benefit_attribution_by_company or {}
            ).items()
        }
        payload = {
            "round_schema_version": "cooperation-round-v1.1.0",
            "close": close.model_dump(mode="json"),
            "contribution_by_company_cents": contributions,
            "total_contribution_cents": sum(contributions.values()),
            "industry_resilience_before_ppm": industry_resilience_before_ppm,
            "public_protection_applied_ppm": public_protection_applied_ppm,
            "industry_resilience_after_ppm": industry_resilience_after_ppm,
            "commitments_due": [item.model_dump(mode="json") for item in due],
            "verifications": [
                item.model_dump(mode="json") for item in verifications
            ],
            "credibility_after": {
                key: value.model_dump(mode="json")
                for key, value in credibility_after.items()
            },
            "benefit_attribution_by_company": {
                key: value.model_dump(mode="json")
                for key, value in sorted(attributions.items())
            },
        }
        record = CooperationRoundRecord(
            **payload,
            round_hash=canonical_hash(payload),
        )
        self._rounds[round_number] = record
        return record.model_copy(deep=True)


__all__ = ["CooperationLedger", "CooperationProtocolError"]
