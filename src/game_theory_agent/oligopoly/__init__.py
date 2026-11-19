"""Repeated logit-Bertrand pricing game with analytic game-theory benchmarks.

This package is the v21 research line: it measures how far LLM pricing agents
move away from the one-shot Nash equilibrium toward the joint-monopoly outcome
(the collusion index Delta of Calvano et al., 2020), and how personas,
cheap-talk communication and a game-theory advisor change that distance and
social welfare.
"""

from game_theory_agent.oligopoly.economics import LogitMarket

__all__ = ["LogitMarket"]
