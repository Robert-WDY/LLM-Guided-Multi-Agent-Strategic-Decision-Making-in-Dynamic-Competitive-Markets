from __future__ import annotations

import pytest

from game_theory_agent.cooperation import (
    CooperativeBenefitAttribution,
    CooperationLedger,
    CooperationProtocolError,
    ProposalResponseDraft,
    SharedResilienceProposalDraft,
)
from game_theory_agent.interaction import (
    CommunicationRoundLedger,
    CommunicationSubmission,
    MessageDraft,
)


COMPANIES = ("company_A", "company_B", "company_C", "company_D")


def _communication(round_number: int) -> CommunicationRoundLedger:
    return CommunicationRoundLedger(
        episode_id="cooperation-episode",
        round_number=round_number,
        state_version=round_number - 1,
        state_hash=f"sha256:state-{round_number}",
        company_ids=COMPANIES,
        mode="public_private",
    )


def _proposal_message(target_round: int = 2) -> MessageDraft:
    return MessageDraft(
        channel="private",
        recipients=["company_B"],
        speech_act="proposal",
        content="请在目标轮贡献共享韧性资金。",
        cooperation_proposal=SharedResilienceProposalDraft(
            target_round=target_round,
            requested_contribution_cents=1_000_000,
        ),
    )


def test_private_proposal_visibility_and_controller_attribution() -> None:
    communication = _communication(1)
    delivered = communication.submit(
        "company_A",
        CommunicationSubmission(messages=[_proposal_message()]),
    )[0]
    closure = communication.close()
    proposal = delivered.cooperation_proposal

    assert proposal is not None
    assert proposal.sender_company_id == "company_A"
    assert proposal.receiver_company_ids == ("company_B",)
    assert proposal.proposal_id.startswith("proposal:")
    assert [item.message_id for item in closure.views["company_A"].visible_messages] == [
        delivered.message_id
    ]
    assert [item.message_id for item in closure.views["company_B"].visible_messages] == [
        delivered.message_id
    ]
    assert closure.views["company_C"].visible_messages == []
    assert closure.views["company_D"].visible_messages == []


def test_acceptance_creates_commitment_and_partial_betrayal_verification() -> None:
    cooperation = CooperationLedger(
        mode="shared_resilience_v1",
        episode_id="cooperation-episode",
        company_ids=COMPANIES,
        max_rounds=5,
    )
    first = _communication(1)
    proposal_message = first.submit(
        "company_A",
        CommunicationSubmission(messages=[_proposal_message()]),
    )[0]
    first_close = cooperation.close_round(first.close())
    proposal = first_close.proposals_created[0]
    assert proposal.proposal_id == (
        proposal_message.cooperation_proposal.proposal_id
    )

    second = _communication(2)
    second.submit(
        "company_B",
        CommunicationSubmission(
            messages=[
                MessageDraft(
                    channel="private",
                    recipients=["company_A"],
                    speech_act="response",
                    content="接受该非绑定共享韧性提议。",
                    cooperation_response=ProposalResponseDraft(
                        proposal_id=proposal.proposal_id,
                        response="accept",
                    ),
                )
            ]
        ),
    )
    second_close = cooperation.close_round(second.close())

    assert len(second_close.responses_recorded) == 1
    assert len(second_close.commitments_created) == 1
    commitment = second_close.commitments_created[0]
    assert commitment.company_id == "company_B"
    assert commitment.promised_contribution_cents == 1_000_000
    assert commitment.binding is False

    actions = {
        company_id: {"shared_resilience_contribution_cents": 0}
        for company_id in COMPANIES
    }
    actions["company_B"]["shared_resilience_contribution_cents"] = 300_000
    settled = cooperation.settle_round(
        round_number=2,
        final_actions=actions,
        industry_resilience_before_ppm=0,
        public_protection_applied_ppm=0,
        industry_resilience_after_ppm=120_000,
    )

    verification = settled.verifications[0]
    assert verification.actual_contribution_cents == 300_000
    assert verification.fulfillment_ratio_ppm == 300_000
    assert verification.status == "partial_betrayal"
    assert settled.credibility_after["company_B"].credibility_ppm < 500_000
    company_a_memory = cooperation.company_view(
        "company_A", round_number=3
    )["cooperation_memory"]["company_B"]
    assert company_a_memory["proposals_sent"] == 1
    assert company_a_memory["accepted_by_opponent"] == 1
    assert company_a_memory["commitments_by_opponent"] == 1
    assert company_a_memory["partial_betrayals_by_opponent"] == 1
    assert company_a_memory["promised_by_opponent_cents"] == 1_000_000
    assert company_a_memory["fulfilled_by_opponent_cents"] == 300_000
    assert company_a_memory["credibility_ppm"] < 500_000


def test_company_benefit_attribution_enforces_counterfactual_math() -> None:
    attribution = CooperativeBenefitAttribution.from_counterfactual(
        company_id="company_A",
        current_contribution_cost_cents=0,
        latest_source_contribution_cents=1_000_000,
        public_protection_received_ppm=200_000,
        actual_round_profit_cents=2_500_000,
        no_public_protection_round_profit_cents=2_000_000,
        avoided_next_incident=True,
    )
    assert attribution.counterfactual_profit_delta_cents == 500_000
    assert attribution.realized_avoided_loss_cents == 500_000
    assert attribution.net_cooperative_cash_flow_cents == -500_000
    assert attribution.individual_cooperative_roi_ppm == 500_000
    assert attribution.free_rider_advantage_cents == 0

    free_rider = CooperativeBenefitAttribution.from_counterfactual(
        company_id="company_D",
        current_contribution_cost_cents=0,
        latest_source_contribution_cents=0,
        public_protection_received_ppm=200_000,
        actual_round_profit_cents=2_500_000,
        no_public_protection_round_profit_cents=2_000_000,
        avoided_next_incident=False,
    )
    assert free_rider.individual_cooperative_roi_ppm is None
    assert free_rider.free_rider_advantage_cents == 500_000

    with pytest.raises(ValueError, match="attribution math"):
        CooperativeBenefitAttribution.model_validate(
            {
                **attribution.model_dump(mode="json"),
                "counterfactual_profit_delta_cents": 1,
            }
        )


def test_rejection_does_not_create_commitment() -> None:
    cooperation = CooperationLedger(
        mode="shared_resilience_v1",
        episode_id="cooperation-episode",
        company_ids=COMPANIES,
        max_rounds=5,
    )
    first = _communication(1)
    first.submit(
        "company_A", CommunicationSubmission(messages=[_proposal_message()])
    )
    proposal = cooperation.close_round(first.close()).proposals_created[0]
    second = _communication(2)
    second.submit(
        "company_B",
        CommunicationSubmission(
            messages=[
                MessageDraft(
                    channel="private",
                    recipients=["company_A"],
                    speech_act="response",
                    content="拒绝。",
                    cooperation_response=ProposalResponseDraft(
                        proposal_id=proposal.proposal_id,
                        response="reject",
                    ),
                )
            ]
        ),
    )

    close = cooperation.close_round(second.close())

    assert close.responses_recorded[0].response == "reject"
    assert close.commitments_created == ()


def test_same_wave_response_and_disabled_payload_fail_closed() -> None:
    communication = _communication(1)
    proposal_message = communication.submit(
        "company_A",
        CommunicationSubmission(messages=[_proposal_message(target_round=3)]),
    )[0]
    proposal = proposal_message.cooperation_proposal
    assert proposal is not None
    communication.submit(
        "company_B",
        CommunicationSubmission(
            messages=[
                MessageDraft(
                    channel="private",
                    recipients=["company_A"],
                    speech_act="response",
                    content="试图同波次接受。",
                    cooperation_response=ProposalResponseDraft(
                        proposal_id=proposal.proposal_id,
                        response="accept",
                    ),
                )
            ]
        ),
    )
    closure = communication.close()
    enabled = CooperationLedger(
        mode="shared_resilience_v1",
        episode_id="cooperation-episode",
        company_ids=COMPANIES,
        max_rounds=5,
    )
    disabled = CooperationLedger(
        mode="off",
        episode_id="cooperation-episode",
        company_ids=COMPANIES,
        max_rounds=5,
    )

    with pytest.raises(CooperationProtocolError, match="same-wave"):
        enabled.close_round(closure)
    with pytest.raises(CooperationProtocolError, match="disabled"):
        disabled.close_round(closure)


def test_structured_cooperation_requires_private_typed_messages() -> None:
    with pytest.raises(ValueError, match="private proposal"):
        MessageDraft(
            channel="public",
            speech_act="proposal",
            content="invalid public proposal",
            cooperation_proposal=SharedResilienceProposalDraft(
                target_round=2,
                requested_contribution_cents=1,
            ),
        )
    with pytest.raises(ValueError, match="private response"):
        MessageDraft(
            channel="public",
            speech_act="response",
            content="invalid public response",
            cooperation_response=ProposalResponseDraft(
                proposal_id="proposal:1",
                response="accept",
            ),
        )
