"""Replay public evidence into company-scoped opponent models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from game_theory_agent.information import ObservationSnapshot
from game_theory_agent.market.models import MarketState
from game_theory_agent.opponent.schema import (
    OPPONENT_MODEL_SCHEMA_VERSION,
    OpponentModelState,
    compute_opponent_model_hash,
)
from game_theory_agent.opponent.updater import OpponentModelLedger


class OpponentModelReplayMismatchError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise OpponentModelReplayMismatchError(
        f"opponent model replay mismatch: {message}"
    )


def _snapshots(event: Any) -> list[ObservationSnapshot]:
    result: list[ObservationSnapshot] = []
    phase = getattr(event, "communication_phase", None)
    if phase is not None:
        result.extend(
            trace.information_snapshot
            for trace in phase.generation_traces
            if trace.information_snapshot is not None
        )
    result.extend(
        trace.information_snapshot
        for trace in event.traces
        if trace.information_snapshot is not None
    )
    return result


def verify_opponent_model_replay(
    events: Sequence[Any], manifest: Any | None = None
) -> tuple[OpponentModelState, ...]:
    if not events:
        return ()
    mode = getattr(manifest, "opponent_model_mode", None)
    if mode is None:
        mode = (
            "public_strategy_v1"
            if any(
                snapshot.observation.get("opponent_model_state") is not None
                for snapshot in _snapshots(events[0])
            )
            else "off"
        )
    if mode == "off":
        if any(
            snapshot.observation.get("opponent_model_state") is not None
            for event in events
            for snapshot in _snapshots(event)
        ):
            _fail("off treatment contains opponent model state")
        return ()
    if mode != "public_strategy_v1":
        _fail(f"unsupported mode {mode}")
    first = MarketState.from_dict(events[0].state_before)
    ledger = OpponentModelLedger(
        episode_id=first.episode_id, company_ids=first.company_ids
    )
    verified: list[OpponentModelState] = []
    for event in events:
        before = MarketState.from_dict(event.state_before)
        by_company: dict[str, OpponentModelState] = {}
        for snapshot in _snapshots(event):
            raw = snapshot.observation.get("opponent_model_state")
            recorded_hash = snapshot.observation.get("opponent_model_hash")
            if not isinstance(raw, dict):
                _fail(f"missing state for {snapshot.company_id}")
            recorded = OpponentModelState.model_validate(raw)
            if (
                recorded.opponent_model_schema_version
                != OPPONENT_MODEL_SCHEMA_VERSION
            ):
                _fail("opponent model schema mismatch")
            expected, expected_hash = ledger.company_view(
                observer_company_id=snapshot.company_id,
                round_number=before.round,
                state_version=before.state_version,
            )
            if recorded != expected or recorded_hash != expected_hash:
                _fail(f"model differs for {snapshot.company_id}")
            if compute_opponent_model_hash(recorded) != expected_hash:
                _fail(f"model hash differs for {snapshot.company_id}")
            prior = by_company.get(snapshot.company_id)
            if prior is not None and prior != recorded:
                _fail("same-round opponent model snapshots differ")
            by_company[snapshot.company_id] = recorded
        verified.extend(by_company.values())
        raw_after = getattr(event, "state_after", None)
        if not isinstance(raw_after, dict):
            raw_after = event.step_result.get("state_after")
        after = MarketState.from_dict(raw_after)
        ledger.update_after_settlement(before, after, event.joint_action)
    return tuple(verified)
