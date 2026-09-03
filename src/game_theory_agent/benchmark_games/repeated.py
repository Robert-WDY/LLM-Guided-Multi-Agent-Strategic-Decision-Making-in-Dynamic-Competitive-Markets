"""Reusable public-history policies for repeated Cournot experiments."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from game_theory_agent.benchmark_games.cournot import PPM, exact_best_response


RuleOpponent = Literal["fixed_nash", "adaptive_best_response"]


def rule_opponent_quantity(
    opponent: RuleOpponent,
    *,
    focal_history: list[int],
) -> int:
    if opponent == "fixed_nash":
        return 8
    if opponent == "adaptive_best_response":
        return exact_best_response(focal_history[-1] if focal_history else 8)
    raise ValueError(f"unsupported rule opponent: {opponent}")


def forecast_opponent_belief(
    opponent: str,
    *,
    focal_history: list[int],
    opponent_history: list[int],
) -> dict[int, int]:
    """Build a public-history-only belief that sums exactly to one million."""

    if opponent == "fixed_nash":
        return {8: PPM}
    if opponent == "adaptive_best_response":
        return {
            exact_best_response(focal_history[-1] if focal_history else 8): PPM
        }
    if opponent != "deepseek_llm":
        raise ValueError(f"unsupported opponent: {opponent}")
    recent = opponent_history[-3:] or [8]
    counts = Counter(recent)
    quantities = sorted(counts)
    belief = {
        quantity: counts[quantity] * PPM // len(recent) for quantity in quantities
    }
    remainder = PPM - sum(belief.values())
    belief[quantities[0]] += remainder
    return belief
