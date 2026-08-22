"""Replay utility inference from recorded opponent model inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from game_theory_agent.utility_inference.contracts import UtilityInferenceState
from game_theory_agent.utility_inference.inference import OpponentUtilityInferer


class UtilityInferenceReplayMismatchError(RuntimeError):
    pass


def verify_utility_inference_replay(
    events: Sequence[Any],
    manifest: Any | None = None,
) -> tuple[UtilityInferenceState, ...]:
    expected_mode = getattr(manifest, "utility_inference_mode", None)
    verified: list[UtilityInferenceState] = []
    for event in events:
        snapshots = []
        phase = getattr(event, "communication_phase", None)
        if phase is not None:
            snapshots.extend(
                trace.information_snapshot
                for trace in phase.generation_traces
                if trace.information_snapshot is not None
            )
        snapshots.extend(
            trace.information_snapshot
            for trace in event.traces
            if trace.information_snapshot is not None
        )
        by_company: dict[str, UtilityInferenceState] = {}
        for snapshot in snapshots:
            observation = snapshot.observation
            raw = observation.get("utility_inference_state")
            raw_model = observation.get("opponent_model_state")
            if raw is None:
                if expected_mode == "strategy_utility_v1":
                    raise UtilityInferenceReplayMismatchError(
                        "enabled treatment is missing utility inference"
                    )
                continue
            if expected_mode == "off":
                raise UtilityInferenceReplayMismatchError(
                    "off treatment contains utility inference"
                )
            if not isinstance(raw, dict) or not isinstance(raw_model, dict):
                raise UtilityInferenceReplayMismatchError(
                    "utility inference is missing its opponent model"
                )
            expected, expected_hash = OpponentUtilityInferer().infer(raw_model)
            recorded = UtilityInferenceState.model_validate(raw)
            if (
                recorded != expected
                or observation.get("utility_inference_hash") != expected_hash
            ):
                raise UtilityInferenceReplayMismatchError(
                    f"utility inference differs for {snapshot.company_id}"
                )
            prior = by_company.get(snapshot.company_id)
            if prior is not None and prior != recorded:
                raise UtilityInferenceReplayMismatchError(
                    "same-round utility snapshots differ"
                )
            by_company[snapshot.company_id] = recorded
        verified.extend(by_company.values())
    return tuple(verified)
