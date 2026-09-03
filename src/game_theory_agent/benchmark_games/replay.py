"""Tamper-evident replay for canonical Cournot events."""

from __future__ import annotations

from collections.abc import Sequence

from game_theory_agent.benchmark_games.cournot import (
    CournotEnvironment,
    CournotRoundEvent,
)


def replay_cournot_events(events: Sequence[CournotRoundEvent]) -> bool:
    previous_hash: str | None = None
    episode_id: str | None = None
    for expected_round, event in enumerate(events, start=1):
        if episode_id is None:
            episode_id = event.episode_id
        if event.episode_id != episode_id or event.round != expected_round:
            return False
        if event.previous_event_hash != previous_hash:
            return False
        rebuilt_outcome = CournotEnvironment(event.config).settle(event.actions)
        if rebuilt_outcome != event.outcome:
            return False
        rebuilt_event = CournotRoundEvent.create(
            episode_id=event.episode_id,
            round_number=event.round,
            config=event.config,
            actions=event.actions,
            outcome=rebuilt_outcome,
            previous_event_hash=event.previous_event_hash,
        )
        if rebuilt_event.event_hash != event.event_hash:
            return False
        previous_hash = event.event_hash
    return True
