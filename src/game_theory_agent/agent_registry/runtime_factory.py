"""Fail-closed construction of an AgentRuntime from a registered version."""

from __future__ import annotations

from pathlib import Path

from game_theory_agent.agent_registry.bootstrap import (
    PROJECT_ROOT,
    current_dependency_lock,
    current_source_bundle,
    sha256_bytes,
)
from game_theory_agent.agent_registry.contracts import (
    AgentInstanceBinding,
    AgentVersionManifest,
)
from game_theory_agent.agent_registry.registry import (
    AgentRegistry,
    AgentRegistryIntegrityError,
)
from game_theory_agent.agents.context import DecisionContextBuilder
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.agents.runtime import AgentRuntime
from game_theory_agent.market import MarketConfig
from game_theory_agent.model_clients.deepseek import DeepSeekModelClient
from game_theory_agent.model_clients.doubao import DoubaoModelClient
from game_theory_agent.model_clients.mock import MockModelClient
from game_theory_agent.model_clients.selective import (
    FixedEconomicBaselineModelClient,
    SelectiveDecisionModelClient,
)
from game_theory_agent.strategic_reliability.cost_guard import RealModelCostGuard


class AgentRuntimeCompatibilityError(AgentRegistryIntegrityError):
    pass


class RegisteredAgentRuntimeFactory:
    """Uses current code only when it exactly matches the registered source bundle."""

    def __init__(
        self,
        registry: AgentRegistry,
        market_config: MarketConfig,
        persona_registry: PersonaRegistry,
        *,
        project_root: str | Path = PROJECT_ROOT,
        strict_source: bool = True,
        real_model_cost_guard: RealModelCostGuard | None = None,
    ) -> None:
        self.registry = registry
        self.market_config = market_config
        self.persona_registry = persona_registry
        self.project_root = Path(project_root).resolve()
        self.strict_source = strict_source
        self.real_model_cost_guard = real_model_cost_guard

    def _validate_current_checkout(self, manifest: AgentVersionManifest) -> None:
        behavior = manifest.behavior_spec
        compatibility = behavior.compatibility
        if self.market_config.environment_version not in compatibility.environment_versions:
            raise AgentRuntimeCompatibilityError("environment version is incompatible")
        if self.market_config.config_sha256 not in compatibility.market_config_hashes:
            raise AgentRuntimeCompatibilityError("market config hash is incompatible")
        if not self.strict_source:
            return
        if sha256_bytes(current_source_bundle(self.project_root)) != (
            behavior.source.source_bundle.sha256
        ):
            raise AgentRuntimeCompatibilityError(
                "current source differs from the registered source bundle"
            )
        if sha256_bytes(current_dependency_lock(self.project_root)) != (
            behavior.source.dependency_lock.sha256
        ):
            raise AgentRuntimeCompatibilityError(
                "current dependencies differ from the registered dependency lock"
            )

    def _model_client(self, manifest: AgentVersionManifest):
        runtime = manifest.behavior_spec.runtime
        params = dict(runtime.policy_parameters)
        if runtime.provider == "mock":
            return MockModelClient(model_name=runtime.model, **params)
        sampling = {
            "temperature": runtime.temperature_ppm / 1_000_000,
            "top_p": runtime.top_p_ppm / 1_000_000,
        }
        max_transport_retries = int(params.pop("max_transport_retries", 1))
        if runtime.provider == "doubao":
            client = DoubaoModelClient(
                model=runtime.model,
                base_url=runtime.endpoint_id,
                timeout_seconds=runtime.request_timeout_ms / 1000,
                max_schema_attempts=runtime.max_schema_attempts,
                **sampling,
            )
        elif runtime.provider == "deepseek":
            client = DeepSeekModelClient(
                model=runtime.model,
                base_url=runtime.endpoint_id,
                timeout_seconds=runtime.request_timeout_ms / 1000,
                max_schema_attempts=runtime.max_schema_attempts,
                max_transport_retries=max_transport_retries,
                **sampling,
            )
        else:
            raise AgentRuntimeCompatibilityError(
                f"unsupported registered provider: {runtime.provider}"
            )
        paid_rounds_csv = params.get("paid_rounds_csv")
        if paid_rounds_csv is None:
            return client
        if self.real_model_cost_guard is None:
            raise AgentRuntimeCompatibilityError(
                "registered paid-round policy requires a real-model cost guard"
            )
        paid_rounds = tuple(
            int(value)
            for value in str(paid_rounds_csv).split(",")
            if value.strip()
        )
        return SelectiveDecisionModelClient(
            client,
            FixedEconomicBaselineModelClient(),
            paid_rounds=paid_rounds,
            cost_guard=self.real_model_cost_guard,
            reserved_prompt_tokens_per_call=int(
                params.get("reserved_prompt_tokens_per_call", 32_000)
            ),
            reserved_completion_tokens_per_call=int(
                params.get("reserved_completion_tokens_per_call", 4_000)
            ),
            input_price_microunits_per_token=int(
                params.get("input_price_microunits_per_token", 2)
            ),
            output_price_microunits_per_token=int(
                params.get("output_price_microunits_per_token", 4)
            ),
        )

    def build(
        self,
        *,
        agent_version_id: str,
        episode_id: str,
        company_id: str,
        agent_id: str,
    ) -> AgentRuntime:
        resolved = self.registry.resolve(agent_version_id)
        manifest = resolved.manifest
        self._validate_current_checkout(manifest)
        profile = self.persona_registry.get(
            manifest.behavior_spec.persona.persona_id
        )
        if profile.catalog_version != manifest.behavior_spec.persona.catalog_version:
            raise AgentRuntimeCompatibilityError("persona catalog version mismatch")
        if profile.profile_hash != manifest.behavior_spec.persona.profile_hash:
            raise AgentRuntimeCompatibilityError("persona profile hash mismatch")
        context = manifest.behavior_spec.context
        context_builder = DecisionContextBuilder(
            persona_profile=profile,
            persona_registry=self.persona_registry,
            context_mode=context.context_mode,
            decision_support_version=context.decision_support_version,
            persona_semantics_version=context.persona_semantics_version,
            diagnostic_mode=context.diagnostic_mode,
            cooperation_history_mode=context.cooperation_history_mode,
        )
        binding = AgentInstanceBinding.create(
            episode_id=episode_id,
            company_id=company_id,
            agent_id=agent_id,
            manifest=manifest,
        )
        return AgentRuntime(
            agent_id=agent_id,
            company_id=company_id,
            model_client=self._model_client(manifest),
            context_builder=context_builder,
            persona_registry=self.persona_registry,
            version_binding=binding,
        )


def common_episode_settings(manifests: list[AgentVersionManifest]) -> dict[str, str]:
    if not manifests:
        raise ValueError("at least one Agent version is required")
    first = manifests[0].behavior_spec
    fields = {
        "information_mode": "public",
        "belief_mode": first.strategic_stack.belief_mode,
        "opponent_model_mode": first.strategic_stack.opponent_model_mode,
        "utility_inference_mode": first.strategic_stack.utility_inference_mode,
        "advisor_mode": first.strategic_stack.advisor_mode,
        "repeated_game_mode": first.strategic_stack.repeated_game_mode,
        "cooperation_mode": first.strategic_stack.cooperation_mode,
        "communication_mode": "off",
    }
    for manifest in manifests:
        behavior = manifest.behavior_spec
        candidate = {
            "information_mode": "public",
            "belief_mode": behavior.strategic_stack.belief_mode,
            "opponent_model_mode": behavior.strategic_stack.opponent_model_mode,
            "utility_inference_mode": behavior.strategic_stack.utility_inference_mode,
            "advisor_mode": behavior.strategic_stack.advisor_mode,
            "repeated_game_mode": behavior.strategic_stack.repeated_game_mode,
            "cooperation_mode": behavior.strategic_stack.cooperation_mode,
            "communication_mode": "off",
        }
        if candidate != fields:
            raise AgentRuntimeCompatibilityError(
                "all Agent versions in one episode must share controller modes"
            )
        if fields["information_mode"] not in behavior.compatibility.information_modes:
            raise AgentRuntimeCompatibilityError("information mode is incompatible")
    return fields
