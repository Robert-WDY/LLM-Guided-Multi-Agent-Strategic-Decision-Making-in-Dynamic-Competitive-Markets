"""Canonical games with analytic ground truth for strategic validation."""

from game_theory_agent.benchmark_games.cournot import (
    CournotAction,
    CournotAdvice,
    CournotConfig,
    CournotEnvironment,
    CournotOutcome,
    CournotRoundEvent,
    exact_best_response,
)
from game_theory_agent.benchmark_games.replay import replay_cournot_events

__all__ = [
    "CournotAction",
    "CournotAdvice",
    "CournotConfig",
    "CournotEnvironment",
    "CournotOutcome",
    "CournotRoundEvent",
    "exact_best_response",
    "replay_cournot_events",
]
