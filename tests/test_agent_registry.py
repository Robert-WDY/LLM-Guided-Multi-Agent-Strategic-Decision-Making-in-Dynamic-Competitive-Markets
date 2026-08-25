from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from game_theory_agent.agent_registry import (
    ActionStackSpec,
    AgentBehaviorSpec,
    AgentInstanceBinding,
    AgentLineage,
    AgentRegistry,
    AgentRegistryIntegrityError,
    AgentVersionManifest,
    CompatibilitySpec,
    ContextStackSpec,
    PersonaBinding,
    PromptStackSpec,
    RuntimeSpec,
    SourceRevision,
    StrategicStackSpec,
    replay_agent_attribution,
)
from game_theory_agent.agents.runtime import AgentRuntime
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.market.replay import EpisodeManifest
from game_theory_agent.model_clients.mock import MockModelClient
from game_theory_agent.orchestration import RoundCoordinator


def _behavior(
    registry: AgentRegistry,
    *,
    family_id: str = "strategic-planner",
    system_prompt: bytes = b"You are a strategic company.",
    advisor_version: str = "pareto-reliable-v1",
) -> AgentBehaviorSpec:
    system = registry.put_artifact(system_prompt, media_type="text/plain")
    decision = registry.put_artifact(b"Return one legal action.", media_type="text/plain")
    source_bundle = registry.put_artifact(b"source archive", media_type="application/zip")
    dependency_lock = registry.put_artifact(b"dependencies", media_type="text/plain")
    return AgentBehaviorSpec(
        family_id=family_id,
        runtime=RuntimeSpec(
            agent_kind="mock",
            provider="mock",
            model="mock-strategic-v1",
            model_revision="local-deterministic-v1",
            seed_mode="deterministic_rule",
            request_schema_version="decision-context-v1",
            response_schema_version="agent-decision-v1",
        ),
        persona=PersonaBinding(
            persona_id="balanced",
            catalog_version="persona-profile-v1",
            profile_hash=sha256_hash({"persona": "balanced"}),
            planning_mode="utility_planner",
            planner_version="persona-planner-v1",
        ),
        prompts=PromptStackSpec(
            prompt_schema_version="prompt-stack-v1",
            system_prompt=system,
            decision_template=decision,
        ),
        context=ContextStackSpec(
            observation_schema_version="agent-observation-v1.8.0",
            observation_builder_version="observation-builder-v2",
            visibility_policy_version="visibility-public-v2",
            decision_context_version="decision-context-v1.15.0",
            context_builder_version="context-builder-v3",
            decision_support_version="economic-v2",
            memory_policy_version="episode-memory-v1",
            memory_scope="episode",
            critical_event_selector_version="critical-event-v1",
            communication_context_version="communication-context-v1",
        ),
        strategic_stack=StrategicStackSpec(
            belief_mode="public_action_v1",
            belief_updater_version="dirichlet-public-price-v1",
            opponent_model_mode="public_strategy_v1",
            opponent_model_version="public-strategy-rule-bayes-v1",
            utility_inference_mode="strategy_utility_v1",
            utility_inference_version="strategy-mixture-utility-v1",
            advisor_mode="pareto_reliable_v5",
            advisor_version=advisor_version,
            repeated_game_mode="off",
            repeated_game_version="none",
            cooperation_mode="off",
            cooperation_version="none",
        ),
        action_stack=ActionStackSpec(
            candidate_generator_version="public-overlay-candidates-v2",
            adoption_contract_version="safe-menu-adoption-v1",
            action_schema_version="company-action-v4",
            parser_version="pydantic-parser-v1",
            resolver_version="action-resolution-policy-v1",
            safety_policy_version="market-guardrails-v4",
            fallback_policy_version="rule-fallback-v1",
        ),
        source=SourceRevision(
            code_commit="test-commit",
            source_bundle=source_bundle,
            dependency_lock=dependency_lock,
        ),
        compatibility=CompatibilitySpec(
            environment_versions=("market-env-v4.2.0",),
            market_config_hashes=(sha256_hash({"market": "v4"}),),
            observation_schema_versions=("agent-observation-v1.8.0",),
            event_schema_versions=("agent-round-event-v1.11.0",),
            information_modes=("perfect", "public"),
        ),
    )


def _registered(registry: AgentRegistry) -> AgentVersionManifest:
    registry.register_family("strategic-planner", description="test family")
    manifest = AgentVersionManifest.create(
        behavior_spec=_behavior(registry),
        human_version="v1",
        reproducibility_tier="A_deterministic",
        created_at="2026-01-01T00:00:00+00:00",
    )
    return registry.register_version(manifest)


def test_behavior_identity_is_deterministic_and_sensitive(tmp_path) -> None:
    registry = AgentRegistry(tmp_path / "registry")
    first = _behavior(registry)
    same = _behavior(registry)
    prompt_changed = _behavior(registry, system_prompt=b"You are a strategic company!")
    advisor_changed = _behavior(registry, advisor_version="pareto-reliable-v2")

    assert first.agent_version_id == same.agent_version_id
    assert prompt_changed.agent_version_id != first.agent_version_id
    assert advisor_changed.agent_version_id != first.agent_version_id


def test_registry_is_immutable_but_lifecycle_is_mutable(tmp_path) -> None:
    registry = AgentRegistry(tmp_path / "registry")
    original = _registered(registry)
    duplicate = AgentVersionManifest.create(
        behavior_spec=original.behavior_spec,
        human_version="a-label-that-does-not-change-behavior",
        reproducibility_tier="A_deterministic",
        created_at="2030-01-01T00:00:00+00:00",
    )

    assert registry.register_version(duplicate) == original
    assert registry.update_lifecycle(
        original.agent_version_id,
        display_name="candidate one",
        tags=("baseline", "pareto"),
        rating_milli=1_532_000,
        games_played=20,
    ).rating_milli == 1_532_000
    assert registry.get(original.agent_version_id).agent_version_id == (
        original.agent_version_id
    )
    registry.promote(original.agent_version_id, "candidate")
    registry.promote(original.agent_version_id, "validated")
    assert registry.promote(original.agent_version_id, "active").state == "active"


def test_lineage_creates_new_version_without_mutating_parent(tmp_path) -> None:
    registry = AgentRegistry(tmp_path / "registry")
    parent = _registered(registry)
    child_behavior = _behavior(registry, advisor_version="pareto-reliable-v2")
    child = registry.register_version(
        AgentVersionManifest.create(
            behavior_spec=child_behavior,
            human_version="v2",
            reproducibility_tier="A_deterministic",
            lineage=AgentLineage(
                parent_version_ids=(parent.agent_version_id,),
                mutation_operator="advisor_upgrade",
                training_run_id="training-001",
                reason="replace one strategic component",
            ),
            created_at="2026-01-02T00:00:00+00:00",
        )
    )

    assert child.agent_version_id != parent.agent_version_id
    assert registry.get(parent.agent_version_id) == parent
    assert child.lineage.parent_version_ids == (parent.agent_version_id,)


def test_registry_detects_manifest_and_artifact_tampering(tmp_path) -> None:
    registry = AgentRegistry(tmp_path / "registry")
    manifest = _registered(registry)
    manifest_path = registry._manifest_path(manifest.agent_version_id)
    manifest_path.write_text(manifest_path.read_text("utf-8") + " ", "utf-8")
    with pytest.raises(AgentRegistryIntegrityError):
        registry.get(manifest.agent_version_id)


def test_registry_detects_artifact_tampering(tmp_path) -> None:
    registry = AgentRegistry(tmp_path / "registry")
    manifest = _registered(registry)
    artifact = manifest.behavior_spec.prompts.system_prompt
    registry._artifact_path(artifact.sha256).write_bytes(b"changed")

    with pytest.raises(AgentRegistryIntegrityError):
        registry.resolve(manifest.agent_version_id)


def test_episode_roster_and_trace_replay_are_exact(tmp_path) -> None:
    registry = AgentRegistry(tmp_path / "registry")
    manifest = _registered(registry)
    config = load_market_config("configs/market_v4.yaml")
    env = MarketEnv(config)
    state = env.reset(
        ("company_A", "company_B"),
        episode_id="versioned-episode",
        episode_seed=7,
        max_rounds=5,
    )
    roster = {
        company_id: AgentInstanceBinding.create(
            episode_id=state.episode_id,
            company_id=company_id,
            agent_id=f"agent-{company_id}",
            manifest=manifest,
        )
        for company_id in state.company_ids
    }
    episode_manifest = EpisodeManifest.create(
        env,
        state,
        agent_versioning_mode="immutable_v1",
        agent_roster={
            key: value.model_dump(mode="json") for key, value in roster.items()
        },
    )
    episode_manifest.verify_agent_roster_hash()
    event = SimpleNamespace(
        episode_id=state.episode_id,
        traces=[
            SimpleNamespace(
                company_id=company_id,
                agent_family_id=binding.family_id,
                agent_version_id=binding.agent_version_id,
                agent_instance_id=binding.agent_instance_id,
                registry_manifest_hash=binding.registry_manifest_hash,
                behavior_spec_hash=binding.behavior_spec_hash,
                prompt_bundle_hash=binding.prompt_bundle_hash,
                source_bundle_hash=binding.source_bundle_hash,
            )
            for company_id, binding in roster.items()
        ],
    )
    replay_agent_attribution(
        episode_id=state.episode_id,
        company_ids=state.company_ids,
        roster=roster,
        events=(event,),
        registry=registry,
    )

    event.traces[0].agent_version_id = sha256_hash({"tampered": True})
    with pytest.raises(AgentRegistryIntegrityError):
        replay_agent_attribution(
            episode_id=state.episode_id,
            company_ids=state.company_ids,
            roster=roster,
            events=(event,),
            registry=registry,
        )


def test_runtime_rejects_binding_for_another_agent(tmp_path) -> None:
    registry = AgentRegistry(tmp_path / "registry")
    manifest = _registered(registry)
    binding = AgentInstanceBinding.create(
        episode_id="episode-one",
        company_id="company_A",
        agent_id="agent-A",
        manifest=manifest,
    )
    with pytest.raises(ValueError, match="does not match"):
        AgentRuntime(
            agent_id="agent-B",
            company_id="company_A",
            model_client=MockModelClient(),
            version_binding=binding,
        )


def test_same_version_instances_do_not_share_episode_memory(tmp_path) -> None:
    registry = AgentRegistry(tmp_path / "registry")
    manifest = _registered(registry)
    first = AgentRuntime(
        agent_id="agent-A",
        company_id="company_A",
        model_client=MockModelClient(),
        version_binding=AgentInstanceBinding.create(
            episode_id="episode-one",
            company_id="company_A",
            agent_id="agent-A",
            manifest=manifest,
        ),
    )
    second = AgentRuntime(
        agent_id="agent-A",
        company_id="company_A",
        model_client=MockModelClient(),
        version_binding=AgentInstanceBinding.create(
            episode_id="episode-two",
            company_id="company_A",
            agent_id="agent-A",
            manifest=manifest,
        ),
    )

    assert first.version_binding.agent_version_id == second.version_binding.agent_version_id
    assert first.version_binding.agent_instance_id != second.version_binding.agent_instance_id
    assert first.memory is not second.memory


def test_api_resolves_every_version_before_creating_immutable_episode(
    tmp_path, monkeypatch
) -> None:
    from game_theory_agent import api

    registry_path = tmp_path / "registry"
    registry = AgentRegistry(registry_path)
    manifest = _registered(registry)
    monkeypatch.setattr(api, "AGENT_REGISTRY_PATH", registry_path)
    api.SESSIONS.clear()
    client = TestClient(api.app)
    response = client.post(
        "/api/episodes",
        json={
            "episode_id": "immutable-api-episode",
            "episode_seed": 19,
            "company_ids": ["company_A", "company_B"],
            "agent_versioning_mode": "immutable_v1",
            "agent_version_ids": {
                "company_A": manifest.agent_version_id,
                "company_B": manifest.agent_version_id,
            },
            "agent_configs": {
                "company_A": {"agent_id": "agent-A"},
                "company_B": {"agent_id": "agent-B"},
            },
        },
    )

    assert response.status_code == 201
    manifest_payload = response.json()["manifest"]
    assert manifest_payload["agent_versioning_mode"] == "immutable_v1"
    assert set(manifest_payload["agent_roster"]) == {"company_A", "company_B"}
    assert all(
        item["agent_version_id"] == manifest.agent_version_id
        for item in manifest_payload["agent_roster"].values()
    )
    assert manifest_payload["agent_roster_hash"].startswith("sha256:")


def test_api_rejects_unresolved_or_partial_immutable_roster(tmp_path, monkeypatch) -> None:
    from game_theory_agent import api

    monkeypatch.setattr(api, "AGENT_REGISTRY_PATH", tmp_path / "empty-registry")
    api.SESSIONS.clear()
    response = TestClient(api.app).post(
        "/api/episodes",
        json={
            "episode_id": "invalid-immutable-episode",
            "episode_seed": 21,
            "company_ids": ["company_A", "company_B"],
            "agent_versioning_mode": "immutable_v1",
            "agent_version_ids": {
                "company_A": sha256_hash({"missing": "version"}),
                "company_B": sha256_hash({"missing": "version"}),
            },
        },
    )

    assert response.status_code == 422
    assert "cannot resolve immutable agent roster" in response.json()["detail"]


def test_coordinator_fails_before_decision_when_runtime_binding_differs(
    tmp_path, monkeypatch
) -> None:
    from game_theory_agent import api

    registry_path = tmp_path / "registry"
    registry = AgentRegistry(registry_path)
    manifest = _registered(registry)
    monkeypatch.setattr(api, "AGENT_REGISTRY_PATH", registry_path)
    api.SESSIONS.clear()
    client = TestClient(api.app)
    created = client.post(
        "/api/episodes",
        json={
            "episode_id": "immutable-coordinator-episode",
            "episode_seed": 23,
            "company_ids": ["company_A", "company_B"],
            "agent_versioning_mode": "immutable_v1",
            "agent_version_ids": {
                "company_A": manifest.agent_version_id,
                "company_B": manifest.agent_version_id,
            },
            "agent_configs": {
                "company_A": {"agent_id": "agent-A"},
                "company_B": {"agent_id": "agent-B"},
            },
        },
    )
    assert created.status_code == 201

    class StaticController:
        async def get_episode(self, episode_id: str):
            response = client.get(f"/api/episodes/{episode_id}/state")
            assert response.status_code == 200
            return response.json()

    coordinator = RoundCoordinator(
        StaticController(),
        SimpleNamespace(),
        {
            "company_A": AgentRuntime(
                agent_id="agent-A",
                company_id="company_A",
                model_client=MockModelClient(),
            )
        },
    )
    with pytest.raises(ValueError, match="not constructed from the bound"):
        asyncio.run(coordinator.run_round("immutable-coordinator-episode"))
