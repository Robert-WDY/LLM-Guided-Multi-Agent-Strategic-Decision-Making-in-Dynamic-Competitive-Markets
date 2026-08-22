"""Multi-Agent coordination without granting Agents execution authority."""

from game_theory_agent.orchestration.clients import (
    AgentGatewayClient,
    ControllerClient,
    HttpAgentGatewayClient,
    HttpControllerClient,
)
from game_theory_agent.orchestration.coordinator import (
    CoordinatedRound,
    RoundCoordinator,
    StaleRoundError,
)
from game_theory_agent.orchestration.round_event import (
    AgentRoundTrace,
    CommunicationGenerationTrace,
    CommunicationPhaseRecord,
    CommunicationViewRecord,
    JsonlRoundEventLogger,
    RoundEvent,
)
from game_theory_agent.interaction.replay import (
    InteractionReplayMismatchError,
    rebuild_communication_closure,
    verify_interaction_replay,
)

__all__ = [
    "AgentGatewayClient",
    "AgentRoundTrace",
    "CommunicationGenerationTrace",
    "CommunicationPhaseRecord",
    "CommunicationViewRecord",
    "ControllerClient",
    "CoordinatedRound",
    "HttpAgentGatewayClient",
    "HttpControllerClient",
    "JsonlRoundEventLogger",
    "InteractionReplayMismatchError",
    "RoundCoordinator",
    "RoundEvent",
    "StaleRoundError",
    "rebuild_communication_closure",
    "verify_interaction_replay",
]
