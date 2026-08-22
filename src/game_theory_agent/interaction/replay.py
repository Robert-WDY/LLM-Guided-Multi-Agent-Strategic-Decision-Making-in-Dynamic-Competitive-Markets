"""Hash- and visibility-verified replay for closed communication phases."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from game_theory_agent.interaction.contracts import (
    CommunicationClosure,
    CommunicationSubmission,
    CommunicationView,
    MessageDraft,
)
from game_theory_agent.interaction.round import CommunicationRoundLedger


class InteractionReplayMismatchError(RuntimeError):
    """A recorded communication phase cannot be reproduced exactly."""


def _fail(detail: str) -> None:
    raise InteractionReplayMismatchError(
        f"interaction replay mismatch: {detail}"
    )


def _draft_from_delivered(message: Any) -> MessageDraft:
    return MessageDraft(
        channel=message.channel,
        recipients=list(message.recipients),
        speech_act=message.speech_act,
        content=message.content,
        own_action_claim=message.own_action_claim,
        requested_peer_action=message.requested_peer_action,
        cooperation_proposal=(
            {
                "proposal_type": message.cooperation_proposal.proposal_type,
                "target_round": message.cooperation_proposal.target_round,
                "requested_contribution_cents": (
                    message.cooperation_proposal.requested_contribution_cents
                ),
            }
            if message.cooperation_proposal is not None
            else None
        ),
        cooperation_response=message.cooperation_response,
    )


def _submission_from_closure(
    closure: CommunicationClosure,
    company_id: str,
) -> CommunicationSubmission:
    return CommunicationSubmission(
        messages=[
            _draft_from_delivered(message)
            for message in closure.all_messages
            if message.sender_company_id == company_id
        ]
    )


def _verify_generation_context(
    context: dict[str, Any],
    *,
    company_id: str,
    closure: CommunicationClosure,
    require_complete: bool = False,
) -> None:
    if not isinstance(context, dict):
        _fail(f"generation context is not an object for {company_id}")
    meta = context.get("meta")
    identity = context.get("identity")
    if meta is not None and not isinstance(meta, dict):
        _fail(f"generation context meta is not an object for {company_id}")
    if identity is not None and not isinstance(identity, dict):
        _fail(
            f"generation context identity is not an object for {company_id}"
        )
    meta = meta or {}
    identity = identity or {}
    bindings = {
        "episode_id": (
            (
                (context["episode_id"],)
                if "episode_id" in context
                else ()
            )
            + ((meta["episode_id"],) if "episode_id" in meta else ()),
            closure.episode_id,
        ),
        "round": (
            ((context["round"],) if "round" in context else ())
            + ((meta["round"],) if "round" in meta else ()),
            closure.round,
        ),
        "state_version": (
            (
                (context["state_version"],)
                if "state_version" in context
                else ()
            )
            + (
                (meta["state_version"],)
                if "state_version" in meta
                else ()
            ),
            closure.state_version,
        ),
        "state_hash": (
            ((context["state_hash"],) if "state_hash" in context else ())
            + ((meta["state_hash"],) if "state_hash" in meta else ()),
            closure.state_hash,
        ),
        "company_id": (
            ((context["company_id"],) if "company_id" in context else ())
            + (
                (identity["company_id"],)
                if "company_id" in identity
                else ()
            ),
            company_id,
        ),
        "communication_mode": (
            (
                (context["communication_mode"],)
                if "communication_mode" in context
                else ()
            ),
            closure.mode,
        ),
    }
    for field_name, (actual_values, expected) in bindings.items():
        if require_complete and not actual_values:
            _fail(
                f"generation context {field_name} missing for {company_id}"
            )
        if any(actual != expected for actual in actual_values):
            _fail(
                f"generation context {field_name} mismatch for {company_id}"
            )


def rebuild_communication_closure(phase: Any) -> CommunicationClosure:
    """Re-run the deterministic barrier from its recorded submissions.

    Generation traces are the preferred source because they bind the authored
    draft to the delivered transcript. Older/minimal phase records can still be
    replayed by recovering drafts from the delivered messages themselves.
    """

    closure = phase.closure
    if not isinstance(closure, CommunicationClosure):
        closure = CommunicationClosure.model_validate(closure)

    company_ids = tuple(sorted(closure.views))
    if not company_ids:
        _fail("closure has no company views")
    if phase.mode != closure.mode:
        _fail("phase mode does not match closure mode")
    if getattr(phase, "status", None) != "closed" or not getattr(
        phase, "closed", False
    ):
        _fail("communication phase is not closed")

    generation_by_company: dict[str, Any] = {}
    for trace in phase.generation_traces:
        if trace.company_id in generation_by_company:
            _fail(f"duplicate generation trace for {trace.company_id}")
        if trace.company_id not in company_ids:
            _fail(f"generation trace has unknown company {trace.company_id}")
        generation_by_company[trace.company_id] = trace
        if trace.communication_context is not None:
            _verify_generation_context(
                trace.communication_context,
                company_id=trace.company_id,
                closure=closure,
            )

    submitted_ids = list(closure.submitted_company_ids)
    if len(set(submitted_ids)) != len(submitted_ids):
        _fail("closure has duplicate submitted company ids")
    if any(company_id not in company_ids for company_id in submitted_ids):
        _fail("closure has an unknown submitted company")

    for company_id, trace in generation_by_company.items():
        has_submission = trace.submission is not None
        if has_submission != (company_id in submitted_ids):
            _fail(
                f"generation submission binding mismatch for {company_id}"
            )
        if not has_submission and trace.accepted_message_ids:
            _fail(f"unsubmitted generation has accepted ids for {company_id}")
        if has_submission:
            accepted_is_silence = not trace.submission.messages
            if (
                trace.is_silence is not None
                and trace.is_silence != accepted_is_silence
            ):
                _fail(f"silence marker mismatch for {company_id}")
        if trace.generation_status == "submitted" and not has_submission:
            _fail(f"submitted generation has no submission for {company_id}")
        if trace.generation_status == "silent" and has_submission:
            if trace.submission.messages:
                _fail(f"silent generation contains messages for {company_id}")

    ledger = CommunicationRoundLedger(
        episode_id=closure.episode_id,
        round_number=closure.round,
        state_version=closure.state_version,
        state_hash=closure.state_hash,
        company_ids=company_ids,
        mode=closure.mode,
    )
    for company_id in company_ids:
        if company_id not in submitted_ids:
            continue
        generation = generation_by_company.get(company_id)
        submission = (
            generation.submission
            if generation is not None and generation.submission is not None
            else _submission_from_closure(closure, company_id)
        )
        try:
            accepted = ledger.submit(company_id, submission)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise InteractionReplayMismatchError(
                "interaction replay mismatch: "
                f"submission reconstruction failed for {company_id}: {exc}"
            ) from exc
        if generation is not None:
            accepted_ids = [message.message_id for message in accepted]
            if list(generation.accepted_message_ids) != accepted_ids:
                _fail(f"accepted message ids mismatch for {company_id}")
    try:
        return ledger.close()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise InteractionReplayMismatchError(
            f"interaction replay mismatch: barrier reconstruction failed: {exc}"
        ) from exc


def _verify_closure(
    recorded: CommunicationClosure,
    rebuilt: CommunicationClosure,
) -> None:
    if recorded.mode != rebuilt.mode:
        _fail("closure mode mismatch")
    for field_name in (
        "episode_id",
        "round",
        "state_version",
        "state_hash",
    ):
        if getattr(recorded, field_name) != getattr(rebuilt, field_name):
            _fail(f"closure {field_name} mismatch")

    recorded_ids = [message.message_id for message in recorded.all_messages]
    rebuilt_ids = [message.message_id for message in rebuilt.all_messages]
    if recorded_ids != rebuilt_ids:
        _fail("canonical message ids mismatch")
    if recorded.all_messages != rebuilt.all_messages:
        _fail("canonical message bodies or recipients mismatch")
    if recorded.transcript_hash != rebuilt.transcript_hash:
        _fail("transcript hash mismatch")
    if recorded.submitted_company_ids != rebuilt.submitted_company_ids:
        _fail("submitted company ids mismatch")
    if recorded.silent_company_ids != rebuilt.silent_company_ids:
        _fail("silent company ids mismatch")
    if set(recorded.views) != set(rebuilt.views):
        _fail("company view set mismatch")
    for company_id, rebuilt_view in rebuilt.views.items():
        recorded_view = recorded.views[company_id]
        if recorded_view.view_digest != rebuilt_view.view_digest:
            _fail(f"view hash mismatch for {company_id}")
        if recorded_view != rebuilt_view:
            _fail(f"visible transcript mismatch for {company_id}")


def _verify_view_records(phase: Any, rebuilt: CommunicationClosure) -> None:
    records = phase.company_views
    if set(records) != set(rebuilt.views):
        _fail("phase company view record set mismatch")
    for company_id, view in rebuilt.views.items():
        record = records[company_id]
        expected_visible_ids = [
            message.message_id for message in view.visible_messages
        ]
        if record.company_id != company_id:
            _fail(f"company view record identity mismatch for {company_id}")
        if record.view_digest != view.view_digest:
            _fail(f"phase view hash mismatch for {company_id}")
        if list(record.visible_message_ids) != expected_visible_ids:
            _fail(f"phase visible message ids mismatch for {company_id}")
        if list(record.own_message_ids) != list(view.own_message_ids):
            _fail(f"phase own message ids mismatch for {company_id}")


def _verify_strict_generation_binding(
    phase: Any,
    rebuilt: CommunicationClosure,
) -> None:
    generation_by_company = {
        trace.company_id: trace for trace in phase.generation_traces
    }
    if set(generation_by_company) != set(rebuilt.views):
        _fail("generation trace companies do not match communication views")
    if rebuilt.mode == "off":
        return
    for company_id, trace in generation_by_company.items():
        if trace.agent_type == "rule":
            continue
        context = trace.communication_context
        if context is None:
            _fail(f"generation context missing for {company_id}")
        _verify_generation_context(
            context,
            company_id=company_id,
            closure=rebuilt,
            require_complete=True,
        )


def _verify_generation_decision_identity(event: Any, phase: Any) -> None:
    decision_by_company = {trace.company_id: trace for trace in event.traces}
    for generation in phase.generation_traces:
        decision = decision_by_company.get(generation.company_id)
        if decision is None:
            _fail(
                "generation trace has no decision trace for "
                f"{generation.company_id}"
            )
        if generation.agent_id != decision.agent_id:
            _fail(
                "generation and decision agent identity mismatch for "
                f"{generation.company_id}"
            )
        if generation.agent_type == "rule":
            expected_status = (
                "disabled" if phase.mode == "off" else "not_applicable"
            )
            if (
                generation.agent_id != "controller-rule"
                or generation.generation_status != expected_status
                or decision.agent_type != "rule"
            ):
                _fail(
                    "rule generation exemption is inconsistent for "
                    f"{generation.company_id}"
                )


def _coerce_view(value: Any, *, company_id: str, source: str) -> CommunicationView:
    try:
        return (
            value
            if isinstance(value, CommunicationView)
            else CommunicationView.model_validate(value)
        )
    except (TypeError, ValueError) as exc:
        raise InteractionReplayMismatchError(
            "interaction replay mismatch: "
            f"invalid {source} communication view for {company_id}: {exc}"
        ) from exc


def _verify_event_binding(
    event: Any,
    rebuilt: CommunicationClosure,
    *,
    require_trace_binding: bool,
) -> None:
    if event.episode_id != rebuilt.episode_id:
        _fail("round event episode id does not match communication closure")
    if event.settled_round != rebuilt.round:
        _fail("round event number does not match communication closure")
    if event.state_before_hash != rebuilt.state_hash:
        _fail("communication was not bound to the pre-decision market state")
    state_before = getattr(event, "state_before", None)
    if state_before:
        if state_before.get("state_version") != rebuilt.state_version:
            _fail("communication state version does not match round event")

    traces: dict[str, Any] = {}
    for trace in event.traces:
        if trace.company_id in traces:
            _fail(f"duplicate decision trace for {trace.company_id}")
        traces[trace.company_id] = trace
    if require_trace_binding and set(traces) != set(rebuilt.views):
        _fail("decision trace companies do not match communication views")

    for company_id, trace in traces.items():
        if company_id not in rebuilt.views:
            _fail(f"decision trace has no communication view for {company_id}")
        expected = rebuilt.views[company_id]
        bound_views: list[tuple[str, Any]] = []
        trace_view = getattr(trace, "communication_view", None)
        if trace_view is not None:
            bound_views.append(("trace", trace_view))
        decision_context = getattr(trace, "decision_context", None)
        if decision_context is not None:
            if hasattr(decision_context, "model_dump"):
                decision_context = decision_context.model_dump(mode="python")
            context_view = decision_context.get("communication_view")
            if context_view is not None:
                bound_views.append(("decision context", context_view))
        if require_trace_binding and not bound_views:
            _fail(f"decision trace is not bound to a view for {company_id}")
        for source, value in bound_views:
            bound = _coerce_view(
                value,
                company_id=company_id,
                source=source,
            )
            if bound != expected:
                _fail(f"{source} view binding mismatch for {company_id}")
        visible_ids = {
            message.message_id for message in expected.visible_messages
        }
        response_ids: list[str] = []
        for response in getattr(trace, "message_responses", ()):
            message_id = response.get("message_id")
            if not isinstance(message_id, str) or not message_id:
                _fail(f"invalid message response audit for {company_id}")
            response_ids.append(message_id)
        if len(response_ids) != len(set(response_ids)):
            _fail(f"duplicate message response for {company_id}")
        hidden_response_ids = sorted(set(response_ids) - visible_ids)
        if hidden_response_ids:
            _fail(
                f"message response references a hidden message for {company_id}"
            )


def _coerce_history(
    value: Any,
    *,
    company_id: str,
    source: str,
) -> list[CommunicationView]:
    if value is None:
        return []
    if not isinstance(value, list):
        _fail(f"{source} communication history is not a list for {company_id}")
    if len(value) > 3:
        _fail(f"{source} communication history is too long for {company_id}")
    return [
        _coerce_view(item, company_id=company_id, source=source)
        for item in value
    ]


def _verify_history_binding(
    event: Any,
    phase: Any,
    expected_history: dict[str, list[CommunicationView]],
) -> None:
    for generation in phase.generation_traces:
        context = generation.communication_context
        if context is None:
            continue
        if generation.company_id not in expected_history:
            _fail(
                "generation context history has unknown company "
                f"{generation.company_id}"
            )
        expected = (
            []
            if context.get("context_mode") == "state_only"
            else expected_history[generation.company_id]
        )
        actual = _coerce_history(
            context.get("recent_communication_views", []),
            company_id=generation.company_id,
            source="generation context",
        )
        if actual != expected:
            _fail(
                "generation context communication history mismatch for "
                f"{generation.company_id}"
            )
    for trace in event.traces:
        context = trace.decision_context
        if context is None:
            continue
        if hasattr(context, "model_dump"):
            context = context.model_dump(mode="python")
        if not isinstance(context, dict):
            _fail(f"decision context is not an object for {trace.company_id}")
        if trace.company_id not in expected_history:
            _fail(
                "decision context history has unknown company "
                f"{trace.company_id}"
            )
        expected = (
            []
            if context.get("context_mode") == "state_only"
            else expected_history[trace.company_id]
        )
        actual = _coerce_history(
            context.get("recent_communication_views", []),
            company_id=trace.company_id,
            source="decision context",
        )
        if actual != expected:
            _fail(
                "decision context communication history mismatch for "
                f"{trace.company_id}"
            )


def _normalise_records(records: Any) -> list[tuple[Any | None, Any]]:
    if hasattr(records, "communication_phase"):
        candidates = [records]
    elif hasattr(records, "closure") and hasattr(records, "company_views"):
        candidates = [records]
    elif isinstance(records, Sequence) and not isinstance(
        records, (str, bytes, bytearray)
    ):
        candidates = list(records)
    else:
        raise TypeError(
            "records must be a RoundEvent, CommunicationPhaseRecord, or sequence"
        )

    normalised: list[tuple[Any | None, Any]] = []
    for candidate in candidates:
        if hasattr(candidate, "communication_phase"):
            phase = candidate.communication_phase
            if phase is not None:
                normalised.append((candidate, phase))
            else:
                version = getattr(candidate, "event_schema_version", None)
                if version not in {
                    "agent-round-event-v1.0.0",
                    "agent-round-event-v1.1.0",
                    "agent-round-event-v1.2.0",
                }:
                    _fail(
                        "v1.3+ round event is missing its communication phase"
                    )
        elif hasattr(candidate, "closure") and hasattr(candidate, "company_views"):
            normalised.append((None, candidate))
        else:
            raise TypeError("sequence contains an unsupported replay record")
    return normalised


def verify_interaction_replay(
    records: Any,
    *,
    require_trace_binding: bool = True,
) -> tuple[CommunicationClosure, ...]:
    """Rebuild and verify recorded messages, visibility, hashes, and bindings.

    Old RoundEvent v1.0-v1.2 entries with no communication phase are skipped, so
    their original market replay behavior remains unchanged. For a v1.3 event,
    every decision trace must bind its authorized ``communication_view`` either
    directly or inside its serialized DecisionContext unless strict binding is
    explicitly disabled for a standalone engineering audit.
    """

    verified: list[CommunicationClosure] = []
    history_by_episode: dict[str, dict[str, list[CommunicationView]]] = {}
    last_round_by_episode: dict[str, int] = {}
    history_verifiable: dict[str, bool] = {}
    for event, phase in _normalise_records(records):
        rebuilt = rebuild_communication_closure(phase)
        recorded = phase.closure
        if not isinstance(recorded, CommunicationClosure):
            recorded = CommunicationClosure.model_validate(recorded)
        _verify_closure(recorded, rebuilt)
        _verify_view_records(phase, rebuilt)
        if event is not None:
            if require_trace_binding:
                _verify_strict_generation_binding(phase, rebuilt)
                _verify_generation_decision_identity(event, phase)
            _verify_event_binding(
                event,
                rebuilt,
                require_trace_binding=require_trace_binding,
            )
            episode_id = rebuilt.episode_id
            if episode_id not in last_round_by_episode:
                last_round_by_episode[episode_id] = rebuilt.round - 1
                history_verifiable[episode_id] = rebuilt.round == 1
                history_by_episode[episode_id] = {
                    company_id: [] for company_id in rebuilt.views
                }
                if require_trace_binding and rebuilt.round != 1:
                    _fail(
                        "strict communication history replay must start at round 1"
                    )
            if rebuilt.round != last_round_by_episode[episode_id] + 1:
                _fail("communication round sequence is not contiguous")
            event_version = getattr(event, "event_schema_version", "")
            if (
                history_verifiable[episode_id]
                and event_version
                in {
                    "agent-round-event-v1.4.0",
                    "agent-round-event-v1.5.0",
                    "agent-round-event-v1.6.0",
                    "agent-round-event-v1.7.0",
                    "agent-round-event-v1.8.0",
                    "agent-round-event-v1.9.0",
                }
            ):
                episode_history = history_by_episode[episode_id]
                if set(episode_history) != set(rebuilt.views):
                    _fail("communication history company set changed")
                expected_history = {
                    company_id: (
                        list(episode_history[company_id][-3:])
                        if rebuilt.mode != "off"
                        else []
                    )
                    for company_id in rebuilt.views
                }
                _verify_history_binding(event, phase, expected_history)
                if rebuilt.mode != "off":
                    for company_id, view in rebuilt.views.items():
                        episode_history[company_id].append(view)
            last_round_by_episode[episode_id] = rebuilt.round
        verified.append(rebuilt)
    return tuple(verified)


__all__ = [
    "InteractionReplayMismatchError",
    "rebuild_communication_closure",
    "verify_interaction_replay",
]
