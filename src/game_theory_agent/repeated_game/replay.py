"""Replay repeated-game advice from company-scoped cooperation memory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from game_theory_agent.repeated_game.contracts import RepeatedGameStrategyState
from game_theory_agent.repeated_game.strategy import RepeatedGameStrategist


class RepeatedGameReplayMismatchError(RuntimeError):
    pass


def verify_repeated_game_replay(
    events: Sequence[Any],
    manifest: Any | None = None,
) -> tuple[RepeatedGameStrategyState, ...]:
    expected_mode = getattr(manifest, "repeated_game_mode", None)
    verified: list[RepeatedGameStrategyState] = []
    for event in events:
        for trace in event.traces:
            snapshot = trace.information_snapshot
            if snapshot is None:
                continue
            observation = snapshot.observation
            raw = observation.get("repeated_game_strategy")
            if raw is None:
                if expected_mode not in {None, "off"}:
                    raise RepeatedGameReplayMismatchError(
                        "enabled treatment is missing repeated-game strategy"
                    )
                continue
            if expected_mode == "off":
                raise RepeatedGameReplayMismatchError(
                    "off treatment contains repeated-game strategy"
                )
            cooperation = observation.get("cooperation")
            if not isinstance(cooperation, dict):
                raise RepeatedGameReplayMismatchError(
                    "repeated-game strategy lacks cooperation memory"
                )
            expected, expected_hash = RepeatedGameStrategist().build(
                episode_id=snapshot.episode_id,
                observer_company_id=snapshot.company_id,
                round_number=snapshot.round,
                cooperation_view=cooperation,
            )
            recorded = RepeatedGameStrategyState.model_validate(raw)
            if (
                recorded != expected
                or observation.get("repeated_game_strategy_hash")
                != expected_hash
            ):
                raise RepeatedGameReplayMismatchError(
                    f"strategy differs for {snapshot.company_id}"
                )
            verified.append(recorded)
    return tuple(verified)
