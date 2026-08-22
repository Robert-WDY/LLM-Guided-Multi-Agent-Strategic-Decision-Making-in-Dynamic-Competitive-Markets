import pytest
from pydantic import ValidationError

from game_theory_agent.interaction import (
    CommunicationConflictError,
    CommunicationRoundLedger,
    CommunicationStateError,
    CommunicationSubmission,
    CommunicationValidationError,
    MessageDraft,
    PartialActionClaim,
)


COMPANIES = ["company_A", "company_B", "company_C", "company_D"]


def _ledger(mode: str = "public_private") -> CommunicationRoundLedger:
    return CommunicationRoundLedger(
        episode_id="interaction-test",
        round_number=2,
        state_version=1,
        state_hash="sha256:frozen-state",
        company_ids=COMPANIES,
        mode=mode,
    )


def test_message_contract_rejects_ambiguous_visibility_and_empty_claims():
    with pytest.raises(ValidationError, match="cannot name recipients"):
        MessageDraft(
            channel="public",
            recipients=["company_B"],
            content="visible to everyone",
        )
    with pytest.raises(ValidationError, match="exactly one recipient"):
        MessageDraft(channel="private", content="missing recipient")
    with pytest.raises(ValidationError, match="at least one field"):
        PartialActionClaim()
    with pytest.raises(ValidationError, match="only one public"):
        CommunicationSubmission(
            messages=[
                MessageDraft(channel="public", content="one"),
                MessageDraft(channel="public", content="two"),
            ]
        )


def test_close_creates_deterministic_company_scoped_views():
    ledger = _ledger()
    public_messages = ledger.submit(
        "company_A",
        CommunicationSubmission(
            messages=[
                MessageDraft(
                    channel="public",
                    speech_act="promise",
                    content="I intend to keep price at 10000.",
                    own_action_claim=PartialActionClaim(price_cents=10_000),
                )
            ]
        ),
    )
    private_messages = ledger.submit(
        "company_B",
        CommunicationSubmission(
            messages=[
                MessageDraft(
                    channel="private",
                    recipients=["company_C"],
                    speech_act="proposal",
                    content="Consider a price of 10500.",
                    requested_peer_action=PartialActionClaim(price_cents=10_500),
                )
            ]
        ),
    )

    first = ledger.close()
    second = ledger.close()

    assert first == second
    assert first.transcript_hash.startswith("sha256:")
    assert len(first.all_messages) == 2
    assert first.silent_company_ids == ["company_C", "company_D"]
    public_id = public_messages[0].message_id
    private_id = private_messages[0].message_id
    assert all(
        public_id in {message.message_id for message in view.visible_messages}
        for view in first.views.values()
    )
    assert private_id in {
        message.message_id
        for message in first.views["company_B"].visible_messages
    }
    assert private_id in {
        message.message_id
        for message in first.views["company_C"].visible_messages
    }
    assert private_id not in {
        message.message_id
        for message in first.views["company_A"].visible_messages
    }
    assert private_id not in {
        message.message_id
        for message in first.views["company_D"].visible_messages
    }
    assert len({view.view_digest for view in first.views.values()}) == 4


def test_submission_is_idempotent_but_cannot_be_replaced_or_added_after_close():
    ledger = _ledger()
    submission = CommunicationSubmission(
        messages=[MessageDraft(channel="public", content="first")]
    )

    first = ledger.submit("company_A", submission)
    repeated = ledger.submit("company_A", submission)

    assert first == repeated
    with pytest.raises(CommunicationConflictError):
        ledger.submit(
            "company_A",
            CommunicationSubmission(
                messages=[MessageDraft(channel="public", content="replacement")]
            ),
        )
    ledger.close()
    with pytest.raises(CommunicationStateError, match="already closed"):
        ledger.submit("company_B", CommunicationSubmission())


def test_mode_and_participant_boundaries_are_enforced():
    public_only = _ledger("public_only")
    with pytest.raises(CommunicationValidationError, match="disabled"):
        public_only.submit(
            "company_A",
            CommunicationSubmission(
                messages=[
                    MessageDraft(
                        channel="private",
                        recipients=["company_B"],
                        content="secret",
                    )
                ]
            ),
        )
    with pytest.raises(CommunicationValidationError, match="recipient"):
        _ledger().submit(
            "company_A",
            CommunicationSubmission(
                messages=[
                    MessageDraft(
                        channel="private",
                        recipients=["unknown"],
                        content="secret",
                    )
                ]
            ),
        )
    with pytest.raises(CommunicationValidationError, match="itself"):
        _ledger().submit(
            "company_A",
            CommunicationSubmission(
                messages=[
                    MessageDraft(
                        channel="private",
                        recipients=["company_A"],
                        content="self message",
                    )
                ]
            ),
        )
    with pytest.raises(CommunicationStateError, match="disabled"):
        _ledger("off").submit("company_A", CommunicationSubmission())


def test_off_mode_closes_to_empty_non_leaking_views():
    closure = _ledger("off").close()

    assert closure.all_messages == []
    assert closure.submitted_company_ids == []
    assert closure.silent_company_ids == COMPANIES
    for company_id, view in closure.views.items():
        assert view.company_id == company_id
        assert view.mode == "off"
        assert view.status == "closed"
        assert view.visible_messages == []
        assert view.messages_are_non_binding is True
        assert view.opponent_content_is_untrusted is True


def test_all_directed_private_pairs_have_exact_sender_recipient_visibility():
    for sender in COMPANIES:
        for recipient in COMPANIES:
            if sender == recipient:
                continue
            ledger = _ledger()
            message = ledger.submit(
                sender,
                CommunicationSubmission(
                    messages=[
                        MessageDraft(
                            channel="private",
                            recipients=[recipient],
                            content=f"secret {sender} to {recipient}",
                        )
                    ]
                ),
            )[0]
            closure = ledger.close()
            visible_to = {
                company_id
                for company_id, view in closure.views.items()
                if message.message_id
                in {item.message_id for item in view.visible_messages}
            }
            assert visible_to == {sender, recipient}


def test_network_arrival_order_does_not_change_canonical_close_hash():
    a_submission = CommunicationSubmission(
        messages=[MessageDraft(channel="public", content="from A")]
    )
    b_submission = CommunicationSubmission(
        messages=[MessageDraft(channel="public", content="from B")]
    )
    first = _ledger()
    first.submit("company_A", a_submission)
    first.submit("company_B", b_submission)
    second = _ledger()
    second.submit("company_B", b_submission)
    second.submit("company_A", a_submission)

    assert first.close() == second.close()


def test_mutating_returned_models_cannot_change_authoritative_visibility():
    ledger = _ledger()
    returned = ledger.submit(
        "company_A",
        CommunicationSubmission(
            messages=[
                MessageDraft(
                    channel="private",
                    recipients=["company_B"],
                    content="secret",
                )
            ]
        ),
    )
    returned[0].recipients.append("company_C")
    first = ledger.close()
    first.all_messages[0].recipients.append("company_D")

    authoritative = ledger.close()
    secret_id = authoritative.all_messages[0].message_id
    assert secret_id not in {
        item.message_id
        for item in authoritative.views["company_C"].visible_messages
    }
    assert secret_id not in {
        item.message_id
        for item in authoritative.views["company_D"].visible_messages
    }
