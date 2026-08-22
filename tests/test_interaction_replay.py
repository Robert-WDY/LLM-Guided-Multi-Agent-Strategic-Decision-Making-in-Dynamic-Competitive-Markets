from __future__ import annotations

from copy import deepcopy

import pytest

from game_theory_agent.interaction import (
    CommunicationRoundLedger,
    CommunicationSubmission,
    InteractionReplayMismatchError,
    MessageDraft,
    verify_interaction_replay,
)
from game_theory_agent.orchestration import (
    AgentRoundTrace,
    CommunicationGenerationTrace,
    CommunicationPhaseRecord,
    RoundEvent,
)


COMPANIES = ["company_A", "company_B", "company_C"]


def _communication_context(
    company_id: str,
    *,
    round_number: int = 1,
    state_version: int = 0,
    state_hash: str = "sha256:frozen-market-state",
    recent_views: list[dict] | None = None,
) -> dict:
    return {
        "communication_mode": "public_private",
        "meta": {
            "episode_id": "interaction-replay",
            "round": round_number,
            "state_version": state_version,
            "state_hash": state_hash,
        },
        "identity": {"company_id": company_id},
        "recent_communication_views": list(recent_views or []),
    }


def _phase() -> CommunicationPhaseRecord:
    ledger = CommunicationRoundLedger(
        episode_id="interaction-replay",
        round_number=1,
        state_version=0,
        state_hash="sha256:frozen-market-state",
        company_ids=COMPANIES,
        mode="public_private",
    )
    submission_a = CommunicationSubmission(
        messages=[
            MessageDraft(channel="public", content="My public statement."),
            MessageDraft(
                channel="private",
                recipients=["company_B"],
                content="A private proposal.",
            ),
        ]
    )
    accepted_a = ledger.submit("company_A", submission_a)
    submission_b = CommunicationSubmission()
    accepted_b = ledger.submit("company_B", submission_b)
    closure = ledger.close()
    return CommunicationPhaseRecord.from_closure(
        closure,
        generation_traces=[
            CommunicationGenerationTrace(
                company_id="company_A",
                agent_id="agent-A",
                agent_type="mock",
                generation_status="submitted",
                communication_context=_communication_context("company_A"),
                submission=submission_a,
                accepted_message_ids=[item.message_id for item in accepted_a],
                is_silence=False,
            ),
            CommunicationGenerationTrace(
                company_id="company_B",
                agent_id="agent-B",
                agent_type="mock",
                generation_status="silent",
                communication_context=_communication_context("company_B"),
                submission=submission_b,
                accepted_message_ids=[item.message_id for item in accepted_b],
                is_silence=True,
                silence_reason="agent chose not to send a message",
            ),
            CommunicationGenerationTrace(
                company_id="company_C",
                agent_id="agent-C",
                agent_type="mock",
                generation_status="fallback",
                communication_context=_communication_context("company_C"),
                is_silence=True,
                silence_reason="no communication runtime",
            ),
        ],
    )


def _event(phase: CommunicationPhaseRecord) -> RoundEvent:
    traces = [
        AgentRoundTrace.model_construct(
            company_id=company_id,
            agent_id=f"agent-{company_id[-1]}",
            agent_type="mock",
            communication_view=phase.closure.views[company_id],
            decision_context={
                "communication_view": phase.closure.views[
                    company_id
                ].model_dump(mode="json"),
                "recent_communication_views": [],
            },
        )
        for company_id in COMPANIES
    ]
    return RoundEvent.model_construct(
        event_schema_version="agent-round-event-v1.4.0",
        event_id="interaction-replay:round-01",
        episode_id="interaction-replay",
        settled_round=1,
        state_before_hash="sha256:frozen-market-state",
        state_before={"state_version": 0},
        traces=traces,
        communication_phase=phase,
    )


def _mutate_phase(
    phase: CommunicationPhaseRecord,
    mutate,
) -> CommunicationPhaseRecord:
    payload = deepcopy(phase.model_dump(mode="json"))
    mutate(payload)
    return CommunicationPhaseRecord.model_validate(payload)


def test_replay_rebuilds_canonical_transcript_visibility_and_context_binding():
    phase = _phase()

    verified = verify_interaction_replay(_event(phase))

    assert verified == (phase.closure,)
    public_id, private_id = [
        message.message_id for message in verified[0].all_messages
    ]
    assert [
        message.message_id
        for message in verified[0].views["company_A"].visible_messages
    ] == [public_id, private_id]
    assert [
        message.message_id
        for message in verified[0].views["company_B"].visible_messages
    ] == [public_id, private_id]
    assert [
        message.message_id
        for message in verified[0].views["company_C"].visible_messages
    ] == [public_id]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["closure"]["all_messages"][0].__setitem__(
            "content", "tampered body"
        ),
        lambda data: data["closure"]["all_messages"][1].__setitem__(
            "recipients", ["company_A"]
        ),
        lambda data: data["closure"]["all_messages"][0].__setitem__(
            "sender_company_id", "company_C"
        ),
        lambda data: data["closure"]["all_messages"].reverse(),
        lambda data: data["closure"]["all_messages"][0].__setitem__(
            "message_id", "sha256:tampered"
        ),
        lambda data: data["closure"].__setitem__(
            "transcript_hash", "sha256:tampered"
        ),
        lambda data: data["closure"]["views"]["company_A"].__setitem__(
            "view_digest", "sha256:tampered"
        ),
        lambda data: data["company_views"]["company_A"].__setitem__(
            "view_digest", "sha256:tampered"
        ),
        lambda data: data["generation_traces"][0][
            "communication_context"
        ].__setitem__("state_hash", "sha256:tampered"),
    ],
    ids=[
        "body",
        "recipient",
        "sender",
        "order",
        "message-id",
        "transcript-hash",
        "closure-view-hash",
        "phase-view-hash",
        "generation-context",
    ],
)
def test_replay_rejects_transcript_recipient_and_hash_tampering(mutate):
    tampered = _mutate_phase(_phase(), mutate)

    with pytest.raises(
        InteractionReplayMismatchError,
        match="interaction replay mismatch",
    ):
        verify_interaction_replay(tampered)


def test_replay_rejects_decision_context_or_trace_view_from_another_company():
    phase = _phase()
    event = _event(phase)
    event.traces[0].communication_view = phase.closure.views["company_B"]

    with pytest.raises(
        InteractionReplayMismatchError,
        match="trace view binding mismatch for company_A",
    ):
        verify_interaction_replay(event)


def test_replay_rejects_response_to_message_outside_company_view():
    phase = _phase()
    event = _event(phase)
    private_id = phase.closure.all_messages[1].message_id
    event.traces[2].message_responses = [
        {"message_id": private_id, "disposition": "accepted"}
    ]

    with pytest.raises(
        InteractionReplayMismatchError,
        match="message response references a hidden message for company_C",
    ):
        verify_interaction_replay(event)


def test_v13_round_event_without_communication_phase_fails_closed():
    event = _event(_phase())
    event.communication_phase = None

    with pytest.raises(
        InteractionReplayMismatchError,
        match="missing its communication phase",
    ):
        verify_interaction_replay(event)


def test_strict_event_replay_rejects_missing_generation_trace():
    phase = _mutate_phase(
        _phase(), lambda data: data["generation_traces"].pop()
    )

    with pytest.raises(
        InteractionReplayMismatchError,
        match="generation trace companies do not match communication views",
    ):
        verify_interaction_replay(_event(phase))


def test_strict_event_replay_rejects_missing_generation_context_field():
    phase = _mutate_phase(
        _phase(),
        lambda data: data["generation_traces"][0][
            "communication_context"
        ]["meta"].pop("state_hash"),
    )

    with pytest.raises(
        InteractionReplayMismatchError,
        match="generation context state_hash missing for company_A",
    ):
        verify_interaction_replay(_event(phase))


def test_strict_event_replay_rejects_missing_generation_context():
    phase = _mutate_phase(
        _phase(),
        lambda data: data["generation_traces"][0].__setitem__(
            "communication_context", None
        ),
    )

    with pytest.raises(
        InteractionReplayMismatchError,
        match="generation context missing for company_A",
    ):
        verify_interaction_replay(_event(phase))


def test_standalone_phase_replay_keeps_minimal_trace_compatibility():
    phase = _mutate_phase(
        _phase(), lambda data: data.__setitem__("generation_traces", [])
    )

    assert verify_interaction_replay(phase) == (phase.closure,)


def _second_round_phase(
    first_phase: CommunicationPhaseRecord,
) -> CommunicationPhaseRecord:
    state_hash = "sha256:frozen-market-state-round-2"
    ledger = CommunicationRoundLedger(
        episode_id="interaction-replay",
        round_number=2,
        state_version=1,
        state_hash=state_hash,
        company_ids=COMPANIES,
        mode="public_private",
    )
    submissions = {
        company_id: CommunicationSubmission() for company_id in COMPANIES
    }
    accepted = {
        company_id: ledger.submit(company_id, submission)
        for company_id, submission in submissions.items()
    }
    closure = ledger.close()
    return CommunicationPhaseRecord.from_closure(
        closure,
        generation_traces=[
            CommunicationGenerationTrace(
                company_id=company_id,
                agent_id=f"agent-{company_id[-1]}",
                agent_type="mock",
                generation_status="silent",
                communication_context=_communication_context(
                    company_id,
                    round_number=2,
                    state_version=1,
                    state_hash=state_hash,
                    recent_views=[
                        first_phase.closure.views[company_id].model_dump(
                            mode="json"
                        )
                    ],
                ),
                submission=submissions[company_id],
                accepted_message_ids=[
                    item.message_id for item in accepted[company_id]
                ],
                is_silence=True,
                silence_reason="agent chose not to send a message",
            )
            for company_id in COMPANIES
        ],
    )


def _second_round_event(
    phase: CommunicationPhaseRecord,
    first_phase: CommunicationPhaseRecord,
) -> RoundEvent:
    traces = [
        AgentRoundTrace.model_construct(
            company_id=company_id,
            agent_id=f"agent-{company_id[-1]}",
            agent_type="mock",
            communication_view=phase.closure.views[company_id],
            decision_context={
                "communication_view": phase.closure.views[
                    company_id
                ].model_dump(mode="json"),
                "recent_communication_views": [
                    first_phase.closure.views[company_id].model_dump(
                        mode="json"
                    )
                ],
            },
        )
        for company_id in COMPANIES
    ]
    return RoundEvent.model_construct(
        event_schema_version="agent-round-event-v1.4.0",
        event_id="interaction-replay:round-02",
        episode_id="interaction-replay",
        settled_round=2,
        state_before_hash="sha256:frozen-market-state-round-2",
        state_before={"state_version": 1},
        traces=traces,
        communication_phase=phase,
    )


@pytest.mark.parametrize("source", ["generation", "decision"])
def test_replay_rejects_authority_inconsistent_private_history(source):
    first_phase = _phase()
    first_event = _event(first_phase)
    second_phase = _second_round_phase(first_phase)
    second_event = _second_round_event(second_phase, first_phase)
    injected_private_view = first_phase.closure.views[
        "company_B"
    ].model_dump(mode="json")
    if source == "generation":
        second_phase.generation_traces[2].communication_context[
            "recent_communication_views"
        ] = [injected_private_view]
    else:
        second_event.traces[2].decision_context[
            "recent_communication_views"
        ] = [injected_private_view]

    with pytest.raises(
        InteractionReplayMismatchError,
        match="communication history mismatch for company_C",
    ):
        verify_interaction_replay([first_event, second_event])


def test_old_round_event_versions_remain_readable_without_communication():
    for version in (
        "agent-round-event-v1.0.0",
        "agent-round-event-v1.1.0",
        "agent-round-event-v1.2.0",
    ):
        event = RoundEvent(
            event_schema_version=version,
            event_id=f"legacy:{version}",
            episode_id="legacy",
            settled_round=1,
            state_before_hash="sha256:before",
            state_after_hash="sha256:after",
            joint_action_hash="sha256:actions",
            step_result={},
            traces=[],
        )
        loaded = RoundEvent.model_validate_json(event.model_dump_json())

        assert loaded.event_schema_version == version
        assert loaded.communication_phase is None
        assert verify_interaction_replay(loaded) == ()


def test_v13_round_event_json_round_trip_preserves_communication_phase():
    phase = _phase()
    event = RoundEvent(
        event_id="interaction-replay:round-01",
        episode_id="interaction-replay",
        settled_round=1,
        state_before_hash="sha256:frozen-market-state",
        state_after_hash="sha256:after-market-step",
        joint_action_hash="sha256:joint-action",
        state_before={"state_version": 0},
        step_result={},
        traces=[],
        communication_phase=phase,
    )

    loaded = RoundEvent.model_validate_json(event.model_dump_json())

    assert loaded.event_schema_version == "agent-round-event-v1.9.0"
    assert loaded.communication_phase == phase
    assert verify_interaction_replay(
        loaded, require_trace_binding=False
    ) == (phase.closure,)
