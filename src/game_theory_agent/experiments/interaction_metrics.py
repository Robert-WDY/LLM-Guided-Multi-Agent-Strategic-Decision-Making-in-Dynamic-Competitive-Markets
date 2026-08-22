"""Descriptive Interaction MVP metrics with no required research direction."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def compute_interaction_metrics(events: Iterable[Any]) -> dict[str, Any]:
    """Summarize messages, responses, cost, and exact structured-claim alignment."""

    event_list = list(events)
    messages: list[Any] = []
    generation_traces: list[Any] = []
    decision_traces: list[Any] = []
    communication_rounds = 0
    for event in event_list:
        decision_traces.extend(event.traces)
        phase = getattr(event, "communication_phase", None)
        if phase is None:
            continue
        communication_rounds += 1
        messages.extend(phase.closure.all_messages)
        generation_traces.extend(phase.generation_traces)

    channel_counts = Counter(message.channel for message in messages)
    speech_act_counts = Counter(message.speech_act for message in messages)
    generation_status_counts = Counter(
        trace.generation_status for trace in generation_traces
    )
    silence_reason_counts = Counter(
        trace.silence_reason
        for trace in generation_traces
        if getattr(trace, "silence_reason", None)
    )
    dispositions: Counter[str] = Counter()
    response_count = 0
    for trace in decision_traces:
        responses = getattr(trace, "message_responses", ()) or ()
        for response in responses:
            disposition = (
                response.get("disposition")
                if isinstance(response, dict)
                else response.disposition
            )
            dispositions[str(disposition)] += 1
            response_count += 1

    claim_fields = 0
    aligned_claim_fields = 0
    deviated_claim_fields = 0
    requested_peer_claim_fields = 0
    aligned_requested_peer_claim_fields = 0
    deviated_requested_peer_claim_fields = 0
    for event in event_list:
        phase = getattr(event, "communication_phase", None)
        if phase is None:
            continue
        for message in phase.closure.all_messages:
            claim = message.own_action_claim
            if claim is not None:
                action = event.joint_action.get(message.sender_company_id, {})
                for field_name in type(claim).model_fields:
                    claimed_value = getattr(claim, field_name)
                    if claimed_value is None:
                        continue
                    claim_fields += 1
                    if action.get(field_name) == claimed_value:
                        aligned_claim_fields += 1
                    else:
                        deviated_claim_fields += 1
            peer_claim = message.requested_peer_action
            if peer_claim is None:
                continue
            targets = (
                list(message.recipients)
                if message.channel == "private"
                else [
                    company_id
                    for company_id in event.joint_action
                    if company_id != message.sender_company_id
                ]
            )
            for target_company_id in targets:
                target_action = event.joint_action.get(target_company_id, {})
                for field_name in type(peer_claim).model_fields:
                    claimed_value = getattr(peer_claim, field_name)
                    if claimed_value is None:
                        continue
                    requested_peer_claim_fields += 1
                    if target_action.get(field_name) == claimed_value:
                        aligned_requested_peer_claim_fields += 1
                    else:
                        deviated_requested_peer_claim_fields += 1

    total_input_tokens = sum(
        int(trace.input_tokens or 0) for trace in generation_traces
    )
    total_output_tokens = sum(
        int(trace.output_tokens or 0) for trace in generation_traces
    )
    total_latency_ms = sum(trace.latency_ms for trace in generation_traces)
    message_count = len(messages)
    return {
        "metrics_schema_version": "interaction-metrics-v1.1.0",
        "classification": {
            "engineering_health": [
                "communication_rounds",
                "generation_status_counts",
                "communication_input_tokens",
                "communication_output_tokens",
                "communication_latency_ms",
            ],
            "research_no_required_direction": [
                "message_count",
                "channel_counts",
                "speech_act_counts",
                "decision_message_response_count",
                "message_disposition_counts",
                "structured_claim_alignment_ppm",
                "requested_peer_claim_alignment_ppm",
            ],
        },
        "communication_rounds": communication_rounds,
        "message_count": message_count,
        "messages_per_round_milli": (
            message_count * 1000 // communication_rounds
            if communication_rounds
            else 0
        ),
        "channel_counts": dict(sorted(channel_counts.items())),
        "speech_act_counts": dict(sorted(speech_act_counts.items())),
        "generation_status_counts": dict(
            sorted(generation_status_counts.items())
        ),
        "silence_reason_counts": dict(sorted(silence_reason_counts.items())),
        "decision_message_response_count": response_count,
        "message_disposition_counts": dict(sorted(dispositions.items())),
        "structured_claim_field_count": claim_fields,
        "structured_claim_aligned_field_count": aligned_claim_fields,
        "structured_claim_deviated_field_count": deviated_claim_fields,
        "structured_claim_alignment_ppm": (
            aligned_claim_fields * 1_000_000 // claim_fields
            if claim_fields
            else None
        ),
        "requested_peer_claim_field_count": requested_peer_claim_fields,
        "requested_peer_claim_aligned_field_count": (
            aligned_requested_peer_claim_fields
        ),
        "requested_peer_claim_deviated_field_count": (
            deviated_requested_peer_claim_fields
        ),
        "requested_peer_claim_alignment_ppm": (
            aligned_requested_peer_claim_fields * 1_000_000
            // requested_peer_claim_fields
            if requested_peer_claim_fields
            else None
        ),
        "communication_input_tokens": total_input_tokens,
        "communication_output_tokens": total_output_tokens,
        "communication_latency_ms": total_latency_ms,
    }


__all__ = ["compute_interaction_metrics"]
