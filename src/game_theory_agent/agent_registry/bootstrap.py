"""Build immutable manifests for the decision stacks shipped in this checkout."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from importlib import metadata
from pathlib import Path
from typing import Literal

from game_theory_agent.agent_registry.contracts import (
    ActionStackSpec,
    AgentBehaviorSpec,
    AgentVersionManifest,
    CompatibilitySpec,
    ContextStackSpec,
    PersonaBinding,
    PromptStackSpec,
    ReproducibilityTier,
    RuntimeSpec,
    SourceRevision,
    StrategicStackSpec,
)
from game_theory_agent.agent_registry.registry import AgentRegistry
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.market import MarketConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATHS = ("src/game_theory_agent", "configs", "pyproject.toml")
DIRECT_DISTRIBUTIONS = (
    "fastapi",
    "openai",
    "python-dotenv",
    "PyYAML",
    "uvicorn",
    "pydantic",
)


def _included_files(project_root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for relative in SOURCE_PATHS:
        candidate = project_root / relative
        if candidate.is_file():
            files.append(candidate)
            continue
        files.extend(
            path
            for path in candidate.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    return tuple(sorted(files, key=lambda path: path.relative_to(project_root).as_posix()))


def current_source_bundle(project_root: str | Path = PROJECT_ROOT) -> bytes:
    """Return a byte-stable ZIP of behavior-relevant source and configuration."""

    root = Path(project_root).resolve()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in _included_files(root):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return output.getvalue()


def current_dependency_lock(project_root: str | Path = PROJECT_ROOT) -> bytes:
    root = Path(project_root).resolve()
    installed: dict[str, str] = {}
    for distribution in DIRECT_DISTRIBUTIONS:
        try:
            installed[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            installed[distribution] = "not-installed"
    payload = {
        "lock_schema_version": "runtime-dependency-lock-v1.0.0",
        "requires_python": ">=3.11",
        "declared_project": (root / "pyproject.toml").read_text("utf-8"),
        "installed_direct_distributions": installed,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def current_git_commit(project_root: str | Path = PROJECT_ROOT) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(project_root),
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "unknown"


class CurrentAgentVersionBuilder:
    """Registers exact first-party Mock or remote-LLM stacks from this checkout."""

    def __init__(
        self,
        registry: AgentRegistry,
        market_config: MarketConfig,
        persona_registry: PersonaRegistry,
        *,
        project_root: str | Path = PROJECT_ROOT,
    ) -> None:
        self.registry = registry
        self.market_config = market_config
        self.persona_registry = persona_registry
        self.project_root = Path(project_root).resolve()
        self._source_ref = registry.put_artifact(
            current_source_bundle(self.project_root),
            media_type="application/zip",
        )
        self._dependency_ref = registry.put_artifact(
            current_dependency_lock(self.project_root),
            media_type="application/json",
        )

    def _prompt_stack(self, provider: str) -> PromptStackSpec:
        if provider == "mock":
            policy = self.registry.put_artifact(
                (self.project_root / "src/game_theory_agent/model_clients/mock.py").read_bytes(),
                media_type="text/x-python",
            )
            system = self.registry.put_artifact(
                b"deterministic MockModelClient policy; no remote prompt",
                media_type="text/plain",
            )
            return PromptStackSpec(
                prompt_schema_version="mock-policy-v1.1.0",
                system_prompt=system,
                decision_template=policy,
                communication_template=policy,
            )
        prompt_builder = self.registry.put_artifact(
            (self.project_root / "src/game_theory_agent/agents/prompt_builder.py").read_bytes(),
            media_type="text/x-python",
        )
        parser = self.registry.put_artifact(
            (self.project_root / "src/game_theory_agent/model_clients/json_output.py").read_bytes(),
            media_type="text/x-python",
        )
        system_payload = {
            "doubao": {
                "decision": "AgentPromptBuilder output is passed as Responses input",
                "communication": "CommunicationPromptBuilder output is passed as Responses input",
            },
            "deepseek": {
                "decision": "你是受约束的市场经营规划器。只输出合法 JSON，不得调用工具或输出额外说明。",
                "communication": "你是受约束的市场通信生成器。只输出合法 JSON；消息非绑定且不能调用工具。",
            },
        }[provider]
        system = self.registry.put_artifact(
            json.dumps(
                system_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            media_type="application/json",
        )
        return PromptStackSpec(
            prompt_schema_version="market-prompt-stack-v1.15.0",
            system_prompt=system,
            decision_template=prompt_builder,
            communication_template=prompt_builder,
            parser_instruction=parser,
        )

    @staticmethod
    def _strategic_stack(advisor_mode: str) -> StrategicStackSpec:
        enabled = advisor_mode != "off"
        return StrategicStackSpec(
            belief_mode="public_action_v1" if enabled else "off",
            belief_updater_version=(
                "dirichlet-public-price-v1.0.0" if enabled else "none"
            ),
            opponent_model_mode="public_strategy_v1" if enabled else "off",
            opponent_model_version=(
                "public-strategy-rule-bayes-v1.0.0" if enabled else "none"
            ),
            utility_inference_mode=(
                "strategy_utility_v1" if enabled else "off"
            ),
            utility_inference_version=(
                "strategy-mixture-utility-v1.0.0" if enabled else "none"
            ),
            advisor_mode=advisor_mode,
            advisor_version=(
                "public-pareto-marginal-market-rollout-v2.0.0"
                if advisor_mode == "pareto_reliable_v6"
                else "public-pareto-reliable-market-rollout-v1.0.0"
                if advisor_mode == "pareto_reliable_v5"
                else "none"
            ),
            repeated_game_mode="off",
            repeated_game_version="none",
            cooperation_mode="off",
            cooperation_version="none",
        )

    def register(
        self,
        *,
        provider: Literal["mock", "doubao", "deepseek"],
        model: str,
        model_revision: str,
        persona_id: str,
        family_id: str = "current-pareto-agent",
        human_version: str | None = None,
        advisor_mode: Literal[
            "off", "pareto_reliable_v5", "pareto_reliable_v6"
        ] = "pareto_reliable_v6",
        policy_parameters: dict[str, int | str | bool | None] | None = None,
        reproducibility_tier: ReproducibilityTier | None = None,
        max_schema_attempts: int | None = None,
    ) -> AgentVersionManifest:
        persona = self.persona_registry.get(persona_id)
        params = dict(policy_parameters or {})
        if provider == "mock" and advisor_mode != "off":
            params.setdefault("honor_game_theory_advice", True)
        sorted_params = tuple(sorted(params.items()))
        tier = reproducibility_tier or (
            "A_deterministic" if provider == "mock" else "C_recorded_output"
        )
        endpoint = {
            "mock": "local:first-party",
            "doubao": "https://ark.cn-beijing.volces.com/api/v3",
            "deepseek": "https://api.deepseek.com",
        }[provider]
        family_description = "Current checkout strategic Agent family"
        self.registry.register_family(family_id, description=family_description)
        behavior = AgentBehaviorSpec(
            family_id=family_id,
            runtime=RuntimeSpec(
                agent_kind="mock" if provider == "mock" else "llm",
                provider=provider,
                model=model,
                model_revision=model_revision,
                endpoint_id=endpoint,
                temperature_ppm=0,
                top_p_ppm=1_000_000,
                max_output_tokens=4_000,
                max_schema_attempts=(
                    max_schema_attempts
                    if max_schema_attempts is not None
                    else (1 if provider == "mock" else 2)
                ),
                request_timeout_ms=40_000,
                decision_timeout_ms=45_000,
                communication_timeout_ms=30_000,
                seed_mode=(
                    "deterministic_rule" if provider == "mock" else "unsupported"
                ),
                request_schema_version="decision-context-v1.15.0",
                response_schema_version="agent-decision-v1.5.0",
                policy_parameters=sorted_params,
            ),
            persona=PersonaBinding(
                persona_id=persona.persona_id,
                catalog_version=persona.catalog_version,
                profile_hash=persona.profile_hash,
                planning_mode="prompt-plus-persona-utility",
                planner_version="persona-planning-layer-v1.0.0",
            ),
            prompts=self._prompt_stack(provider),
            context=ContextStackSpec(
                observation_schema_version="agent-observation-v1.8.0",
                observation_builder_version="observation-builder-v2.0.0",
                visibility_policy_version="visibility-public-v2.0.0",
                decision_context_version="decision-context-v1.15.0",
                context_builder_version="decision-context-builder-v1.15.0",
                decision_support_version="economic_v2",
                memory_policy_version="episode-memory-v1.2.0",
                memory_scope="episode",
                critical_event_selector_version="critical-event-v1.0.0",
                communication_context_version="communication-context-v1.6.0",
                context_mode="full",
                persona_semantics_version="economic_v2",
                diagnostic_mode="off",
                cooperation_history_mode="full",
            ),
            strategic_stack=self._strategic_stack(advisor_mode),
            action_stack=ActionStackSpec(
                candidate_generator_version=(
                    "public-marginal-candidates-v3.0.0"
                    if advisor_mode == "pareto_reliable_v6"
                    else "public-overlay-candidates-v2.0.0"
                    if advisor_mode == "pareto_reliable_v5"
                    else "none"
                ),
                adoption_contract_version="advisor-adoption-trace-v1.0.0",
                action_schema_version="company-action-v4.0.0",
                parser_version="pydantic-agent-decision-parser-v1.0.0",
                resolver_version="action-resolution-policy-v1.0.0",
                safety_policy_version="market-action-guardrails-v4.2.0",
                fallback_policy_version="controller-rule-fallback-v3.0.0",
            ),
            source=SourceRevision(
                code_commit=current_git_commit(self.project_root),
                source_bundle=self._source_ref,
                dependency_lock=self._dependency_ref,
            ),
            compatibility=CompatibilitySpec(
                environment_versions=(self.market_config.environment_version,),
                market_config_hashes=(self.market_config.config_sha256,),
                observation_schema_versions=("agent-observation-v1.8.0",),
                event_schema_versions=("agent-round-event-v1.12.0",),
                information_modes=("public",),
            ),
        )
        return self.registry.register_version(
            AgentVersionManifest.create(
                behavior_spec=behavior,
                human_version=human_version or f"current-{provider}-{persona_id}",
                reproducibility_tier=tier,
            )
        )


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()
