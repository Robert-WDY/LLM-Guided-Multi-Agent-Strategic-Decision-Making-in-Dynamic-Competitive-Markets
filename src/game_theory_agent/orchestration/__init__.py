"""Multi-Agent coordination without granting Agents execution authority."""

from game_theory_agent.orchestration.clients import (
    AgentGatewayClient,
    ControllerClient,
    HttpAgentGatewayClient,
    HttpControllerClient,
)
from game_theory_agent.orchestration.roster import (
    COMPANY_IDS,
    DEFAULT_COMPANIES,
    parse_company_list,
    require_subset,
    validate_company_roster,
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
    "COMPANY_IDS",
    "ControllerClient",
    "DEFAULT_COMPANIES",
    "CoordinatedRound",
    "HttpAgentGatewayClient",
    "HttpControllerClient",
    "JsonlRoundEventLogger",
    "InteractionReplayMismatchError",
    "RoundCoordinator",
    "RoundEvent",
    "StaleRoundError",
    "parse_company_list",
    "require_subset",
    "validate_company_roster",
    "rebuild_communication_closure",
    "verify_interaction_replay",
]
