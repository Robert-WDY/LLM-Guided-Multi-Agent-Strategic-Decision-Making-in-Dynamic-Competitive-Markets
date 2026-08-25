from __future__ import annotations

import pytest

from game_theory_agent.agent_registry import AgentRegistry
from game_theory_agent.agent_registry.bootstrap import (
    CurrentAgentVersionBuilder,
    current_source_bundle,
)
from game_theory_agent.agent_registry.runtime_factory import (
    AgentRuntimeCompatibilityError,
    RegisteredAgentRuntimeFactory,
    common_episode_settings,
)
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.market import load_market_config


def _components(tmp_path):
    config = load_market_config("configs/market_v4.yaml")
    personas = PersonaRegistry.from_market_config(config)
    registry = AgentRegistry(tmp_path / "registry")
    builder = CurrentAgentVersionBuilder(registry, config, personas)
    return config, personas, registry, builder


def test_current_source_bundle_is_byte_stable() -> None:
    assert current_source_bundle() == current_source_bundle()


def test_registered_version_constructs_exact_runtime(tmp_path) -> None:
    config, personas, registry, builder = _components(tmp_path)
    manifest = builder.register(
        provider="mock",
        model="mock-balanced-v1",
        model_revision="mock-policy-v1.1.0",
        persona_id="risk_guarded_v1",
    )
    runtime = RegisteredAgentRuntimeFactory(
        registry, config, personas
    ).build(
        agent_version_id=manifest.agent_version_id,
        episode_id="runtime-factory-episode",
        company_id="company_A",
        agent_id="registered-agent-A",
    )

    assert runtime.persona_profile.persona_id == "risk_guarded_v1"
    assert runtime.version_binding is not None
    assert runtime.version_binding.agent_version_id == manifest.agent_version_id
    assert runtime.version_binding.registry_manifest_hash == manifest.manifest_hash
    assert runtime.memory.snapshot()["recent_rounds"] == []


def test_runtime_factory_fails_when_checkout_differs_from_registered_source(
    tmp_path, monkeypatch
) -> None:
    config, personas, registry, builder = _components(tmp_path)
    manifest = builder.register(
        provider="mock",
        model="mock-balanced-v1",
        model_revision="mock-policy-v1.1.0",
        persona_id="balanced_v1",
    )
    monkeypatch.setattr(
        "game_theory_agent.agent_registry.runtime_factory.current_source_bundle",
        lambda _root: b"changed source",
    )

    with pytest.raises(AgentRuntimeCompatibilityError, match="source differs"):
        RegisteredAgentRuntimeFactory(registry, config, personas).build(
            agent_version_id=manifest.agent_version_id,
            episode_id="source-mismatch-episode",
            company_id="company_A",
            agent_id="registered-agent-A",
        )


def test_common_episode_settings_reject_mixed_strategic_stacks(tmp_path) -> None:
    _config, _personas, _registry, builder = _components(tmp_path)
    enabled = builder.register(
        provider="mock",
        model="mock-balanced-v1",
        model_revision="mock-policy-v1.1.0",
        persona_id="balanced_v1",
        advisor_mode="pareto_reliable_v5",
    )
    disabled = builder.register(
        provider="mock",
        model="mock-balanced-v1",
        model_revision="mock-policy-v1.1.0",
        persona_id="balanced_v1",
        advisor_mode="off",
    )

    with pytest.raises(AgentRuntimeCompatibilityError, match="share controller modes"):
        common_episode_settings([enabled, disabled])
