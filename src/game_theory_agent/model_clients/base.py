"""The narrow boundary between Agent orchestration and model providers."""

from __future__ import annotations

from typing import Protocol

from game_theory_agent.agents.contracts import (
    CommunicationContext,
    DecisionContext,
    ModelGeneration,
)


class ModelClient(Protocol):
    async def generate_communication(
        self, context: CommunicationContext
    ) -> ModelGeneration:
        """Return a structured cheap-talk draft without sending it."""

    async def generate_decision(self, context: DecisionContext) -> ModelGeneration:
        """Return structured model output without modifying market state."""
