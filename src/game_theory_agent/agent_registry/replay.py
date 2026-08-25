"""Agent-version attribution replay independent of economic replay."""

from __future__ import annotations

from collections.abc import Iterable

from game_theory_agent.agent_registry.contracts import AgentInstanceBinding
from game_theory_agent.agent_registry.registry import (
    AgentRegistry,
    AgentRegistryIntegrityError,
)


def replay_agent_attribution(
    *,
    episode_id: str,
    company_ids: Iterable[str],
    roster: dict[str, AgentInstanceBinding],
    events: Iterable[object],
    registry: AgentRegistry,
) -> None:
    """Fail closed if a round trace is not attributable to the bound version."""

    expected_companies = set(company_ids)
    if set(roster) != expected_companies:
        raise AgentRegistryIntegrityError("agent roster does not match episode companies")
    for company_id, binding in roster.items():
        if binding.episode_id != episode_id or binding.company_id != company_id:
            raise AgentRegistryIntegrityError("agent instance binding scope mismatch")
        manifest = registry.get(binding.agent_version_id)
        if manifest.manifest_hash != binding.registry_manifest_hash:
            raise AgentRegistryIntegrityError("bound registry manifest hash mismatch")
    for event in events:
        event_episode = str(getattr(event, "episode_id"))
        if event_episode != episode_id:
            raise AgentRegistryIntegrityError("round event belongs to another episode")
        traces = list(getattr(event, "traces"))
        by_company = {str(trace.company_id): trace for trace in traces}
        if set(by_company) != expected_companies:
            raise AgentRegistryIntegrityError("round traces do not match agent roster")
        for company_id, trace in by_company.items():
            binding = roster[company_id]
            actual = (
                getattr(trace, "agent_family_id", None),
                getattr(trace, "agent_version_id", None),
                getattr(trace, "agent_instance_id", None),
                getattr(trace, "registry_manifest_hash", None),
                getattr(trace, "behavior_spec_hash", None),
                getattr(trace, "prompt_bundle_hash", None),
                getattr(trace, "source_bundle_hash", None),
            )
            expected = (
                binding.family_id,
                binding.agent_version_id,
                binding.agent_instance_id,
                binding.registry_manifest_hash,
                binding.behavior_spec_hash,
                binding.prompt_bundle_hash,
                binding.source_bundle_hash,
            )
            if actual != expected:
                raise AgentRegistryIntegrityError(
                    f"agent attribution mismatch for {company_id}"
                )
