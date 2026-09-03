"""Immutable Agent version, lineage, and episode-binding contracts.

The market state remains authoritative.  These contracts identify the exact
decision policy that was allowed to observe and act in one experiment.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.market.protocols import sha256_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
ID_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,199}$"


class ImmutableModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, protected_namespaces=()
    )


class ArtifactRef(ImmutableModel):
    """Content-addressed artifact reference; local paths are never identity."""

    artifact_schema_version: Literal["agent-artifact-ref-v1.0.0"] = (
        "agent-artifact-ref-v1.0.0"
    )
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1, max_length=160)


class RuntimeSpec(ImmutableModel):
    agent_kind: Literal["llm", "rule", "random", "mock"]
    provider: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=160)
    model_revision: str = Field(min_length=1, max_length=160)
    endpoint_id: str | None = Field(default=None, max_length=300)
    temperature_ppm: int = Field(default=0, ge=0, le=2_000_000)
    top_p_ppm: int = Field(default=1_000_000, ge=0, le=1_000_000)
    max_output_tokens: int = Field(default=4096, ge=1, le=1_000_000)
    max_schema_attempts: int = Field(default=2, ge=1, le=2)
    request_timeout_ms: int = Field(default=40_000, ge=1, le=600_000)
    decision_timeout_ms: int = Field(default=45_000, ge=1, le=600_000)
    communication_timeout_ms: int = Field(default=30_000, ge=1, le=600_000)
    seed_mode: Literal["provider", "request", "unsupported", "deterministic_rule"]
    request_schema_version: str = Field(min_length=1, max_length=160)
    response_schema_version: str = Field(min_length=1, max_length=160)
    tool_schema_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    policy_parameters: tuple[
        tuple[str, int | str | bool | None], ...
    ] = ()

    @model_validator(mode="after")
    def validate_policy_parameters(self) -> "RuntimeSpec":
        names = [name for name, _ in self.policy_parameters]
        if len(names) != len(set(names)):
            raise ValueError("runtime policy parameter names must be unique")
        if names != sorted(names):
            raise ValueError("runtime policy parameters must be sorted")
        return self


class PersonaBinding(ImmutableModel):
    persona_id: str = Field(min_length=1, max_length=80)
    catalog_version: str = Field(min_length=1, max_length=160)
    profile_hash: str = Field(pattern=SHA256_PATTERN)
    planning_mode: str = Field(min_length=1, max_length=120)
    planner_version: str = Field(min_length=1, max_length=160)


class PromptStackSpec(ImmutableModel):
    prompt_schema_version: str = Field(min_length=1, max_length=160)
    system_prompt: ArtifactRef
    decision_template: ArtifactRef
    communication_template: ArtifactRef | None = None
    parser_instruction: ArtifactRef | None = None


class ContextStackSpec(ImmutableModel):
    observation_schema_version: str = Field(min_length=1, max_length=160)
    observation_builder_version: str = Field(min_length=1, max_length=160)
    visibility_policy_version: str = Field(min_length=1, max_length=160)
    decision_context_version: str = Field(min_length=1, max_length=160)
    context_builder_version: str = Field(min_length=1, max_length=160)
    decision_support_version: str = Field(min_length=1, max_length=160)
    memory_policy_version: str = Field(min_length=1, max_length=160)
    memory_scope: Literal["episode", "checkpoint"] = "episode"
    critical_event_selector_version: str = Field(min_length=1, max_length=160)
    communication_context_version: str = Field(min_length=1, max_length=160)
    context_mode: Literal["full", "state_only"] = "full"
    persona_semantics_version: Literal["legacy_v1", "economic_v2"] = "economic_v2"
    diagnostic_mode: Literal["off", "observe"] = "off"
    cooperation_history_mode: Literal["full", "none"] = "full"
    cooperation_prompt_variant: Literal[
        "explicit_options_v1", "neutral_numeric_v1"
    ] = "explicit_options_v1"


class StrategicStackSpec(ImmutableModel):
    belief_mode: str = Field(min_length=1, max_length=120)
    belief_updater_version: str = Field(min_length=1, max_length=160)
    opponent_model_mode: str = Field(min_length=1, max_length=120)
    opponent_model_version: str = Field(min_length=1, max_length=160)
    utility_inference_mode: str = Field(min_length=1, max_length=120)
    utility_inference_version: str = Field(min_length=1, max_length=160)
    advisor_mode: str = Field(min_length=1, max_length=120)
    advisor_version: str = Field(min_length=1, max_length=160)
    repeated_game_mode: str = Field(min_length=1, max_length=120)
    repeated_game_version: str = Field(min_length=1, max_length=160)
    cooperation_mode: str = Field(min_length=1, max_length=120)
    cooperation_version: str = Field(min_length=1, max_length=160)


class ActionStackSpec(ImmutableModel):
    candidate_generator_version: str = Field(min_length=1, max_length=160)
    adoption_contract_version: str = Field(min_length=1, max_length=160)
    action_schema_version: str = Field(min_length=1, max_length=160)
    parser_version: str = Field(min_length=1, max_length=160)
    resolver_version: str = Field(min_length=1, max_length=160)
    safety_policy_version: str = Field(min_length=1, max_length=160)
    fallback_policy_version: str = Field(min_length=1, max_length=160)


class SourceRevision(ImmutableModel):
    code_commit: str = Field(min_length=1, max_length=160)
    source_bundle: ArtifactRef
    dependency_lock: ArtifactRef
    container_image_digest: str | None = Field(default=None, max_length=300)


class CompatibilitySpec(ImmutableModel):
    environment_versions: tuple[str, ...] = Field(min_length=1)
    market_config_hashes: tuple[str, ...] = Field(min_length=1)
    observation_schema_versions: tuple[str, ...] = Field(min_length=1)
    event_schema_versions: tuple[str, ...] = Field(min_length=1)
    information_modes: tuple[Literal["perfect", "public"], ...] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def validate_hashes(self) -> "CompatibilitySpec":
        for value in self.market_config_hashes:
            if not value.startswith("sha256:") or len(value) != 71:
                raise ValueError("market_config_hashes must contain sha256 identifiers")
        return self


class CheckpointRef(ImmutableModel):
    checkpoint_schema_version: str = Field(min_length=1, max_length=160)
    artifact: ArtifactRef
    state_hash: str = Field(pattern=SHA256_PATTERN)
    memory_scope: Literal["checkpoint"] = "checkpoint"


class AgentBehaviorSpec(ImmutableModel):
    """Complete behavior-impacting input to an AgentVersionID."""

    behavior_schema_version: Literal["agent-behavior-spec-v1.0.0"] = (
        "agent-behavior-spec-v1.0.0"
    )
    family_id: str = Field(pattern=ID_PATTERN)
    runtime: RuntimeSpec
    persona: PersonaBinding
    prompts: PromptStackSpec
    context: ContextStackSpec
    strategic_stack: StrategicStackSpec
    action_stack: ActionStackSpec
    source: SourceRevision
    compatibility: CompatibilitySpec
    checkpoint: CheckpointRef | None = None

    @property
    def behavior_hash(self) -> str:
        return sha256_hash(
            {
                "hash_protocol_version": "agent-behavior-hash-v1.0.0",
                "behavior_spec": self.model_dump(mode="json"),
            }
        )

    @property
    def agent_version_id(self) -> str:
        return self.behavior_hash

    @property
    def prompt_bundle_hash(self) -> str:
        return sha256_hash(self.prompts.model_dump(mode="json"))

    def artifact_refs(self) -> tuple[ArtifactRef, ...]:
        refs = [self.prompts.system_prompt, self.prompts.decision_template]
        refs.extend(
            ref
            for ref in (
                self.prompts.communication_template,
                self.prompts.parser_instruction,
            )
            if ref is not None
        )
        if self.checkpoint is not None:
            refs.append(self.checkpoint.artifact)
        refs.extend((self.source.source_bundle, self.source.dependency_lock))
        return tuple(refs)


class AgentLineage(ImmutableModel):
    lineage_schema_version: Literal["agent-lineage-v1.0.0"] = (
        "agent-lineage-v1.0.0"
    )
    parent_version_ids: tuple[str, ...] = ()
    mutation_operator: str | None = Field(default=None, max_length=160)
    training_run_id: str | None = Field(default=None, max_length=200)
    reason: str = Field(default="initial", min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_parents(self) -> "AgentLineage":
        if len(set(self.parent_version_ids)) != len(self.parent_version_ids):
            raise ValueError("lineage parent ids must be unique")
        for version_id in self.parent_version_ids:
            if not version_id.startswith("sha256:") or len(version_id) != 71:
                raise ValueError("lineage parent ids must be sha256 identifiers")
        return self


ReproducibilityTier = Literal[
    "A_deterministic",
    "B_provider_seeded",
    "C_recorded_output",
]


class AgentVersionManifest(ImmutableModel):
    manifest_schema_version: Literal["agent-version-manifest-v1.0.0"] = (
        "agent-version-manifest-v1.0.0"
    )
    agent_version_id: str = Field(pattern=SHA256_PATTERN)
    behavior_spec_hash: str = Field(pattern=SHA256_PATTERN)
    family_id: str = Field(pattern=ID_PATTERN)
    human_version: str = Field(min_length=1, max_length=120)
    behavior_spec: AgentBehaviorSpec
    lineage: AgentLineage = AgentLineage()
    reproducibility_tier: ReproducibilityTier
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        behavior_spec: AgentBehaviorSpec,
        human_version: str,
        reproducibility_tier: ReproducibilityTier,
        lineage: AgentLineage | None = None,
        created_at: str | None = None,
    ) -> "AgentVersionManifest":
        return cls(
            agent_version_id=behavior_spec.agent_version_id,
            behavior_spec_hash=behavior_spec.behavior_hash,
            family_id=behavior_spec.family_id,
            human_version=human_version,
            behavior_spec=behavior_spec,
            lineage=lineage or AgentLineage(),
            reproducibility_tier=reproducibility_tier,
            created_at=created_at or datetime.now(UTC).isoformat(),
        )

    @model_validator(mode="after")
    def validate_identity(self) -> "AgentVersionManifest":
        if self.family_id != self.behavior_spec.family_id:
            raise ValueError("manifest family_id does not match behavior spec")
        expected = self.behavior_spec.behavior_hash
        if self.behavior_spec_hash != expected or self.agent_version_id != expected:
            raise ValueError("agent version identity does not match behavior spec")
        if self.agent_version_id in self.lineage.parent_version_ids:
            raise ValueError("agent version cannot be its own parent")
        return self

    @property
    def manifest_hash(self) -> str:
        return sha256_hash(self.model_dump(mode="json"))


LifecycleState = Literal[
    "draft",
    "candidate",
    "validated",
    "active",
    "champion",
    "historical",
    "retired",
]


class AgentLifecycleRecord(ImmutableModel):
    record_schema_version: Literal["agent-lifecycle-record-v1.0.0"] = (
        "agent-lifecycle-record-v1.0.0"
    )
    agent_version_id: str = Field(pattern=SHA256_PATTERN)
    state: LifecycleState = "draft"
    display_name: str | None = Field(default=None, max_length=160)
    tags: tuple[str, ...] = ()
    rating_milli: int | None = None
    games_played: int = Field(default=0, ge=0)
    updated_at: str


class AgentInstanceBinding(ImmutableModel):
    """Exact immutable version selected for one company in one episode."""

    binding_schema_version: Literal["agent-instance-binding-v1.0.0"] = (
        "agent-instance-binding-v1.0.0"
    )
    episode_id: str = Field(pattern=ID_PATTERN)
    company_id: str = Field(pattern=ID_PATTERN)
    agent_id: str = Field(pattern=ID_PATTERN)
    family_id: str = Field(pattern=ID_PATTERN)
    agent_version_id: str = Field(pattern=SHA256_PATTERN)
    agent_instance_id: str = Field(pattern=ID_PATTERN)
    registry_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    behavior_spec_hash: str = Field(pattern=SHA256_PATTERN)
    prompt_bundle_hash: str = Field(pattern=SHA256_PATTERN)
    source_bundle_hash: str = Field(pattern=SHA256_PATTERN)
    checkpoint_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    reproducibility_tier: ReproducibilityTier

    @classmethod
    def create(
        cls,
        *,
        episode_id: str,
        company_id: str,
        agent_id: str,
        manifest: AgentVersionManifest,
    ) -> "AgentInstanceBinding":
        checkpoint = manifest.behavior_spec.checkpoint
        return cls(
            episode_id=episode_id,
            company_id=company_id,
            agent_id=agent_id,
            family_id=manifest.family_id,
            agent_version_id=manifest.agent_version_id,
            agent_instance_id=f"{episode_id}:{company_id}",
            registry_manifest_hash=manifest.manifest_hash,
            behavior_spec_hash=manifest.behavior_spec_hash,
            prompt_bundle_hash=manifest.behavior_spec.prompt_bundle_hash,
            source_bundle_hash=manifest.behavior_spec.source.source_bundle.sha256,
            checkpoint_id=(checkpoint.state_hash if checkpoint else None),
            reproducibility_tier=manifest.reproducibility_tier,
        )


class ResolvedAgentVersion(ImmutableModel):
    manifest: AgentVersionManifest
    artifact_paths: dict[str, str]
