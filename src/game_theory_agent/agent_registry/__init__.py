"""Immutable Agent version registry public API."""

from game_theory_agent.agent_registry.contracts import (
    ActionStackSpec,
    AgentBehaviorSpec,
    AgentInstanceBinding,
    AgentLifecycleRecord,
    AgentLineage,
    AgentVersionManifest,
    ArtifactRef,
    CheckpointRef,
    CompatibilitySpec,
    ContextStackSpec,
    PersonaBinding,
    PromptStackSpec,
    ResolvedAgentVersion,
    RuntimeSpec,
    SourceRevision,
    StrategicStackSpec,
)
from game_theory_agent.agent_registry.registry import (
    AgentRegistry,
    AgentRegistryError,
    AgentRegistryIntegrityError,
    AgentVersionNotFoundError,
    ImmutableAgentVersionError,
)
from game_theory_agent.agent_registry.replay import replay_agent_attribution

__all__ = [
    "ActionStackSpec",
    "AgentBehaviorSpec",
    "AgentInstanceBinding",
    "AgentLifecycleRecord",
    "AgentLineage",
    "AgentRegistry",
    "AgentRegistryError",
    "AgentRegistryIntegrityError",
    "AgentVersionManifest",
    "AgentVersionNotFoundError",
    "ArtifactRef",
    "CheckpointRef",
    "CompatibilitySpec",
    "ContextStackSpec",
    "ImmutableAgentVersionError",
    "PersonaBinding",
    "PromptStackSpec",
    "ResolvedAgentVersion",
    "RuntimeSpec",
    "SourceRevision",
    "StrategicStackSpec",
    "replay_agent_attribution",
]
