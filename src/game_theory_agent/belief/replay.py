"""Belief replay and calibration from authoritative round events."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from game_theory_agent.belief.contracts import (
    BELIEF_HASH_PROTOCOL_VERSION,
    BELIEF_SCHEMA_VERSION,
    SIGNAL_BELIEF_SCHEMA_VERSION,
    BeliefState,
    PriceDirection,
    compute_belief_hash,
)
from game_theory_agent.belief.ledger import BeliefLedger, classify_price_direction
from game_theory_agent.information import ObservationSnapshot
from game_theory_agent.market.models import MarketState


class BeliefReplayMismatchError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise BeliefReplayMismatchError(f"belief replay mismatch: {message}")


def _event_snapshots(event: Any) -> list[ObservationSnapshot]:
    snapshots: list[ObservationSnapshot] = []
    phase = getattr(event, "communication_phase", None)
    if phase is not None:
        for trace in phase.generation_traces:
            if trace.information_snapshot is not None:
                snapshots.append(trace.information_snapshot)
    for trace in event.traces:
        if trace.information_snapshot is not None:
            snapshots.append(trace.information_snapshot)
    return snapshots


def verify_belief_replay(
    events: Sequence[Any], manifest: Any | None = None
) -> tuple[BeliefState, ...]:
    if not events:
        return ()
    first_state = MarketState.from_dict(events[0].state_before)
    mode = getattr(manifest, "belief_mode", None) if manifest is not None else None
    if mode is None:
        sample = _event_snapshots(events[0])
        mode = (
            (
                "public_action_signal_v2"
                if sample
                and sample[0].belief_schema_version
                == SIGNAL_BELIEF_SCHEMA_VERSION
                else "public_action_v1"
            )
            if sample and sample[0].belief_schema_version != "none"
            else "off"
        )
    if mode == "off":
        for event in events:
            for snapshot in _event_snapshots(event):
                if snapshot.belief_schema_version != "none":
                    _fail("off treatment contains an enabled belief state")
        return ()
    if mode not in {"public_action_v1", "public_action_signal_v2"}:
        _fail(f"unsupported belief mode {mode}")
    ledger = BeliefLedger(
        episode_id=first_state.episode_id,
        company_ids=first_state.company_ids,
        mode=mode,
    )
    verified: list[BeliefState] = []
    expected_round = first_state.round
    for event in events:
        state_before = MarketState.from_dict(event.state_before)
        if state_before.round != expected_round:
            _fail("events must start at round 1 and be consecutive")
        by_company: dict[str, list[BeliefState]] = {}
        for snapshot in _event_snapshots(event):
            expected_schema = (
                SIGNAL_BELIEF_SCHEMA_VERSION
                if mode == "public_action_signal_v2"
                else BELIEF_SCHEMA_VERSION
            )
            if snapshot.belief_schema_version != expected_schema:
                _fail(f"missing enabled belief for {snapshot.company_id}")
            raw = snapshot.observation.get("belief_state")
            if not isinstance(raw, dict):
                _fail(f"missing belief_state for {snapshot.company_id}")
            recorded = BeliefState.model_validate(raw)
            raw_view = snapshot.observation.get("communication_view")
            visible_messages = (
                raw_view.get("visible_messages", [])
                if isinstance(raw_view, dict)
                and raw_view.get("status") == "closed"
                else ()
            )
            expected, expected_hash = ledger.company_view(
                observer_company_id=snapshot.company_id,
                round_number=state_before.round,
                state_version=state_before.state_version,
                visible_messages=visible_messages,
                public_prices={
                    item.company_id: item.commercial.price_cents
                    for item in state_before.companies
                },
            )
            if recorded != expected:
                _fail(f"belief state differs for {snapshot.company_id}")
            if snapshot.belief_hash != expected_hash:
                _fail(f"belief hash differs for {snapshot.company_id}")
            if compute_belief_hash(recorded) != expected_hash:
                _fail(f"belief content hash differs for {snapshot.company_id}")
            by_company.setdefault(snapshot.company_id, []).append(recorded)
        strict = event.event_schema_version in {
            "agent-round-event-v1.8.0",
            "agent-round-event-v1.9.0",
        }
        required_companies = {
            trace.company_id
            for trace in event.traces
            if getattr(trace, "observation", None) is not None
            or getattr(trace, "decision_context", None) is not None
            or getattr(trace, "information_snapshot", None) is not None
        }
        if strict and set(by_company) != required_companies:
            _fail("strict event is missing one or more company belief snapshots")
        # Return one final (decision-time) view per company for compatibility;
        # every pre-close and post-close snapshot above was still verified.
        verified.extend(beliefs[-1] for beliefs in by_company.values())
        phase = getattr(event, "communication_phase", None)
        messages = (
            phase.closure.all_messages if phase is not None else ()
        )
        ledger.update_after_settlement(
            state_before,
            event.joint_action,
            communication_messages=messages,
        )
        expected_round += 1
    return tuple(verified)


def compute_belief_calibration(events: Sequence[Any]) -> dict[str, Any]:
    """Score only pre-action predictions against the later settled public action."""

    records: list[tuple[int, bool, float, float]] = []
    for event in events:
        state_before = MarketState.from_dict(event.state_before)
        snapshots = _event_snapshots(event)
        by_observer: dict[str, ObservationSnapshot] = {}
        for snapshot in snapshots:
            by_observer.setdefault(snapshot.company_id, snapshot)
        for observer, snapshot in by_observer.items():
            raw = snapshot.observation.get("belief_state")
            if not isinstance(raw, dict):
                continue
            state = BeliefState.model_validate(raw)
            for opponent, belief in state.opponent_beliefs.items():
                actual_price = int(event.joint_action[opponent]["price_cents"])
                prior_price = state_before.company(opponent).commercial.price_cents
                actual: PriceDirection = classify_price_direction(
                    prior_price, actual_price
                )
                distribution = belief.next_price_direction
                probability_ppm = distribution.probability_ppm(actual)
                probabilities = {
                    "price_cut": distribution.price_cut_ppm / 1_000_000,
                    "maintain": distribution.maintain_ppm / 1_000_000,
                    "price_raise": distribution.price_raise_ppm / 1_000_000,
                }
                brier = sum(
                    (probability - (1.0 if direction == actual else 0.0)) ** 2
                    for direction, probability in probabilities.items()
                )
                records.append(
                    (
                        probability_ppm,
                        distribution.top_direction == actual,
                        brier,
                        -math.log(max(probability_ppm, 1) / 1_000_000),
                    )
                )
    if not records:
        return {
            "metrics_schema_version": "belief-calibration-v1.0.0",
            "prediction_count": 0,
            "top1_accuracy_ppm": None,
            "mean_brier_score": None,
            "mean_log_loss": None,
        }
    return {
        "metrics_schema_version": "belief-calibration-v1.0.0",
        "prediction_count": len(records),
        "top1_accuracy_ppm": round(
            1_000_000 * sum(int(item[1]) for item in records) / len(records)
        ),
        "mean_brier_score": sum(item[2] for item in records) / len(records),
        "mean_log_loss": sum(item[3] for item in records) / len(records),
        "belief_hash_protocol_version": BELIEF_HASH_PROTOCOL_VERSION,
    }
