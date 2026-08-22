"""Episode manifest, JSONL transition logging, and hash-verified replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from game_theory_agent.market.environment import MarketEnv
from game_theory_agent.market.exceptions import ReplayMismatchError
from game_theory_agent.market.models import CompanyAction, MarketState, StepResult


@dataclass(frozen=True, slots=True)
class EpisodeManifest:
    experiment_id: str
    episode_id: str
    episode_seed: int
    config_id: str
    config_version: str
    config_sha256: str
    environment_version: str
    state_schema_version: str
    action_schema_version: str
    event_schema_version: str
    rng_protocol_version: str
    hash_protocol_version: str
    rng_component_versions: tuple[tuple[str, str], ...]
    num_agents: int
    max_rounds: int
    information_mode: str
    initial_state: MarketState
    observation_schema_version: str = "agent-observation-v1.8.0"
    visibility_policy_version: str = "visibility-perfect-v1.0.0"
    observation_hash_protocol_version: str = "observation-view-hash-v1.0.0"
    belief_schema_version: str = "none"
    belief_mode: str = "off"
    belief_updater_version: str = "none"
    belief_hash_protocol_version: str = "none"
    opponent_model_mode: str = "off"
    opponent_model_schema_version: str = "none"
    opponent_model_updater_version: str = "none"
    opponent_model_hash_protocol_version: str = "none"
    utility_inference_mode: str = "off"
    utility_inference_schema_version: str = "none"
    utility_inference_model_version: str = "none"
    utility_inference_hash_protocol_version: str = "none"
    advisor_mode: str = "off"
    advisor_schema_version: str = "none"
    advisor_model_version: str = "none"
    repeated_game_mode: str = "off"
    repeated_game_schema_version: str = "none"
    repeated_game_hash_protocol_version: str = "none"
    communication_mode: str = "off"
    communication_protocol_version: str = "simultaneous-one-shot-v1.0.0"
    communication_waves: int = 1
    communication_max_messages_per_agent: int = 2
    cooperation_mode: str = "off"
    cooperation_protocol_version: str = "shared-resilience-v1.0.0"
    code_commit: str = "unknown"
    agent_configs: tuple[tuple[str, Mapping[str, Any]], ...] = ()
    observer_information_modes: tuple[tuple[str, str], ...] = ()

    @classmethod
    def create(
        cls,
        env: MarketEnv,
        initial_state: MarketState,
        *,
        experiment_id: str = "experiment-001",
        code_commit: str = "unknown",
        agent_configs: Mapping[str, Mapping[str, Any]] | None = None,
        information_mode: str = "perfect",
        communication_mode: str = "off",
        cooperation_mode: str = "off",
        belief_mode: str = "off",
        opponent_model_mode: str = "off",
        utility_inference_mode: str = "off",
        advisor_mode: str = "off",
        repeated_game_mode: str = "off",
        observer_information_modes: Mapping[str, str] | None = None,
    ) -> "EpisodeManifest":
        if information_mode not in {"perfect", "public"}:
            raise ValueError(f"unsupported information mode: {information_mode}")
        if communication_mode not in {"off", "public_only", "public_private"}:
            raise ValueError(
                f"unsupported communication mode: {communication_mode}"
            )
        if cooperation_mode not in {"off", "shared_resilience_v1"}:
            raise ValueError(f"unsupported cooperation mode: {cooperation_mode}")
        if belief_mode not in {
            "off", "public_action_v1", "public_action_signal_v2"
        }:
            raise ValueError(f"unsupported belief mode: {belief_mode}")
        if opponent_model_mode not in {"off", "public_strategy_v1"}:
            raise ValueError(
                f"unsupported opponent model mode: {opponent_model_mode}"
            )
        if utility_inference_mode not in {"off", "strategy_utility_v1"}:
            raise ValueError(
                f"unsupported utility inference mode: {utility_inference_mode}"
            )
        if repeated_game_mode not in {"off", "reciprocity_v1"}:
            raise ValueError(
                f"unsupported repeated game mode: {repeated_game_mode}"
            )
        if utility_inference_mode != "off" and opponent_model_mode == "off":
            raise ValueError("utility inference requires opponent modeling")
        if advisor_mode == "bayesian_strategy_v2" and (
            opponent_model_mode == "off" or utility_inference_mode == "off"
        ):
            raise ValueError(
                "bayesian_strategy_v2 requires opponent model and utility inference"
            )
        if repeated_game_mode != "off" and cooperation_mode == "off":
            raise ValueError("repeated game strategy requires cooperation")
        observer_modes = dict(observer_information_modes or {})
        unknown_observers = set(observer_modes) - set(initial_state.company_ids)
        if unknown_observers:
            raise ValueError(
                "observer information modes contain unknown companies: "
                f"{sorted(unknown_observers)}"
            )
        invalid_observer_modes = {
            company_id: mode
            for company_id, mode in observer_modes.items()
            if mode not in {"perfect", "public"}
        }
        if invalid_observer_modes:
            raise ValueError(
                "unsupported observer information modes: "
                f"{invalid_observer_modes}"
            )
        if advisor_mode not in {
            "off", "bayesian_price_v1", "bayesian_strategy_v2"
        }:
            raise ValueError(f"unsupported advisor mode: {advisor_mode}")
        if advisor_mode != "off" and belief_mode == "off":
            raise ValueError("Bayesian advisor requires an enabled belief mode")
        if belief_mode == "public_action_signal_v2" and communication_mode == "off":
            raise ValueError("signal belief requires communication")
        if cooperation_mode != "off" and communication_mode == "public_only":
            raise ValueError(
                "shared_resilience_v1 supports communication off or public_private"
            )
        config = env.config
        return cls(
            experiment_id=experiment_id,
            episode_id=initial_state.episode_id,
            episode_seed=initial_state.episode_seed,
            config_id=config.config_id,
            config_version=config.config_version,
            config_sha256=config.config_sha256,
            environment_version=config.environment_version,
            state_schema_version=config.text("schema_versions", "state"),
            action_schema_version=config.text("schema_versions", "action"),
            event_schema_version=config.text("schema_versions", "event"),
            rng_protocol_version=config.rng_protocol_version,
            hash_protocol_version=config.hash_protocol_version,
            rng_component_versions=tuple(
                sorted(
                    (str(k), str(v))
                    for k, v in config.mapping("protocols", "rng_components").items()
                )
            ),
            code_commit=code_commit,
            num_agents=len(initial_state.companies),
            max_rounds=initial_state.max_rounds,
            information_mode=information_mode,
            visibility_policy_version=(
                "visibility-perfect-v1.0.0"
                if information_mode == "perfect"
                else "visibility-public-v2.0.0"
            ),
            communication_mode=communication_mode,
            cooperation_mode=cooperation_mode,
            belief_mode=belief_mode,
            belief_schema_version=(
                "belief-state-v1.0.0"
                if belief_mode == "public_action_v1"
                else (
                    "belief-state-v2.0.0"
                    if belief_mode == "public_action_signal_v2"
                    else "none"
                )
            ),
            belief_updater_version=(
                "dirichlet-public-price-v1.0.0"
                if belief_mode == "public_action_v1"
                else (
                    "dirichlet-public-price-signal-v2.0.0"
                    if belief_mode == "public_action_signal_v2"
                    else "none"
                )
            ),
            belief_hash_protocol_version=(
                "belief-view-hash-v1.0.0"
                if belief_mode != "off"
                else "none"
            ),
            advisor_mode=advisor_mode,
            advisor_schema_version=(
                "bayesian-price-advice-v1.0.0"
                if advisor_mode == "bayesian_price_v1"
                else (
                    "bayesian-strategy-advice-v2.0.0"
                    if advisor_mode == "bayesian_strategy_v2"
                    else "none"
                )
            ),
            advisor_model_version=(
                "independent-direction-payoff-proxy-v1.0.0"
                if advisor_mode == "bayesian_price_v1"
                else (
                    "expected-strategic-response-v2.0.0"
                    if advisor_mode == "bayesian_strategy_v2"
                    else "none"
                )
            ),
            opponent_model_mode=opponent_model_mode,
            opponent_model_schema_version=(
                "opponent-model-state-v1.0.0"
                if opponent_model_mode == "public_strategy_v1"
                else "none"
            ),
            opponent_model_updater_version=(
                "public-strategy-rule-bayes-v1.0.0"
                if opponent_model_mode == "public_strategy_v1"
                else "none"
            ),
            opponent_model_hash_protocol_version=(
                "opponent-model-hash-v1.0.0"
                if opponent_model_mode == "public_strategy_v1"
                else "none"
            ),
            utility_inference_mode=utility_inference_mode,
            utility_inference_schema_version=(
                "utility-inference-state-v1.0.0"
                if utility_inference_mode == "strategy_utility_v1"
                else "none"
            ),
            utility_inference_model_version=(
                "strategy-mixture-utility-v1.0.0"
                if utility_inference_mode == "strategy_utility_v1"
                else "none"
            ),
            utility_inference_hash_protocol_version=(
                "utility-inference-hash-v1.0.0"
                if utility_inference_mode == "strategy_utility_v1"
                else "none"
            ),
            repeated_game_mode=repeated_game_mode,
            repeated_game_schema_version=(
                "repeated-game-strategy-v1.0.0"
                if repeated_game_mode == "reciprocity_v1"
                else "none"
            ),
            repeated_game_hash_protocol_version=(
                "repeated-game-strategy-hash-v1.0.0"
                if repeated_game_mode == "reciprocity_v1"
                else "none"
            ),
            agent_configs=tuple(sorted((agent_configs or {}).items())),
            observer_information_modes=tuple(sorted(observer_modes.items())),
            initial_state=initial_state,
        )

    def information_mode_for(self, company_id: str) -> str:
        """Return the auditable company-scoped information treatment."""

        return dict(self.observer_information_modes).get(
            company_id, self.information_mode
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": "episode-manifest-v1.7.0",
            "experiment_id": self.experiment_id,
            "config_id": self.config_id,
            "config_version": self.config_version,
            "config_sha256": self.config_sha256,
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "environment_version": self.environment_version,
            "state_schema_version": self.state_schema_version,
            "action_schema_version": self.action_schema_version,
            "event_schema_version": self.event_schema_version,
            "rng_protocol_version": self.rng_protocol_version,
            "hash_protocol_version": self.hash_protocol_version,
            "rng_component_versions": dict(self.rng_component_versions),
            "code_commit": self.code_commit,
            "num_agents": self.num_agents,
            "max_rounds": self.max_rounds,
            "information_mode": self.information_mode,
            "observer_information_modes": dict(
                self.observer_information_modes
            ),
            "observation_schema_version": self.observation_schema_version,
            "visibility_policy_version": self.visibility_policy_version,
            "observation_hash_protocol_version": (
                self.observation_hash_protocol_version
            ),
            "belief_schema_version": self.belief_schema_version,
            "belief_mode": self.belief_mode,
            "belief_updater_version": self.belief_updater_version,
            "belief_hash_protocol_version": self.belief_hash_protocol_version,
            "opponent_model_mode": self.opponent_model_mode,
            "opponent_model_schema_version": self.opponent_model_schema_version,
            "opponent_model_updater_version": self.opponent_model_updater_version,
            "opponent_model_hash_protocol_version": (
                self.opponent_model_hash_protocol_version
            ),
            "utility_inference_mode": self.utility_inference_mode,
            "utility_inference_schema_version": (
                self.utility_inference_schema_version
            ),
            "utility_inference_model_version": (
                self.utility_inference_model_version
            ),
            "utility_inference_hash_protocol_version": (
                self.utility_inference_hash_protocol_version
            ),
            "advisor_mode": self.advisor_mode,
            "advisor_schema_version": self.advisor_schema_version,
            "advisor_model_version": self.advisor_model_version,
            "repeated_game_mode": self.repeated_game_mode,
            "repeated_game_schema_version": self.repeated_game_schema_version,
            "repeated_game_hash_protocol_version": (
                self.repeated_game_hash_protocol_version
            ),
            "communication_mode": self.communication_mode,
            "communication_protocol_version": (
                self.communication_protocol_version
            ),
            "communication_waves": self.communication_waves,
            "communication_max_messages_per_agent": (
                self.communication_max_messages_per_agent
            ),
            "cooperation_mode": self.cooperation_mode,
            "cooperation_protocol_version": self.cooperation_protocol_version,
            "agent_configs": dict(self.agent_configs),
            "initial_state": self.initial_state.to_dict(),
            "initial_state_hash": self.initial_state.state_hash,
        }


@dataclass(frozen=True, slots=True)
class MarketTransition:
    state_before: MarketState
    joint_action: tuple[tuple[str, CompanyAction], ...]
    step_result: StepResult

    @classmethod
    def create(
        cls,
        state_before: MarketState,
        joint_action: Mapping[str, CompanyAction],
        step_result: StepResult,
    ) -> "MarketTransition":
        return cls(state_before, tuple(sorted(joint_action.items())), step_result)

    @property
    def state_after(self) -> MarketState:
        return self.step_result.state_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": f"{self.state_before.episode_id}:round-{self.step_result.settled_round:02d}",
            "step_id": self.step_result.step_id,
            "settled_round": self.step_result.settled_round,
            "state_before_hash": self.state_before.state_hash,
            "state_before": self.state_before.to_dict(),
            "final_actions": {
                company_id: action.to_dict() for company_id, action in self.joint_action
            },
            "joint_action_hash": self.step_result.joint_action_hash,
            "random_draw_summary": dict(self.step_result.random_draw_summary),
            "step_result": self.step_result.to_dict(),
            "state_after": self.state_after.to_dict(),
            "state_after_hash": self.state_after.state_hash,
            "invariant_results": list(self.step_result.invariant_results),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarketTransition":
        before = MarketState.from_dict(data["state_before"])
        after = MarketState.from_dict(data["state_after"])
        actions = tuple(
            sorted(
                (str(company_id), CompanyAction.from_dict(action))
                for company_id, action in data["final_actions"].items()
            )
        )
        step = StepResult(
            step_id=str(data["step_id"]),
            settled_round=int(data["settled_round"]),
            state_before_hash=str(data["state_before_hash"]),
            state_after=after,
            joint_action_hash=str(data["joint_action_hash"]),
            random_draw_summary=tuple(
                sorted(
                    (str(k), int(v))
                    for k, v in data.get("random_draw_summary", {}).items()
                )
            ),
            invariant_results=tuple(data.get("invariant_results", ())),
        )
        return cls(before, actions, step)


class JsonlTransitionLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, transition: MarketTransition) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(
                transition.to_dict(),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")

    def read_all(self) -> tuple[MarketTransition, ...]:
        if not self.path.exists():
            return ()
        transitions: list[MarketTransition] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    transitions.append(MarketTransition.from_dict(json.loads(line)))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Invalid transition at {self.path}:{line_number}: {exc}"
                    ) from exc
        return tuple(transitions)


def replay(
    env: MarketEnv,
    initial_state: MarketState,
    joint_actions: Iterable[Mapping[str, CompanyAction]],
) -> tuple[MarketState, ...]:
    env.load_state(initial_state)
    states = [initial_state]
    for joint_action in joint_actions:
        current = env.get_state()
        result = env.step(
            f"{current.episode_id}:{current.round}:{current.state_version}",
            joint_action,
        )
        states.append(result.state_after)
    return tuple(states)


def verify_replay(
    env: MarketEnv,
    manifest: EpisodeManifest,
    transitions: Sequence[MarketTransition],
) -> tuple[MarketState, ...]:
    if manifest.config_id != env.config.config_id:
        raise ReplayMismatchError("manifest config_id does not match environment")
    if manifest.config_sha256 != env.config.config_sha256:
        raise ReplayMismatchError("manifest config hash does not match environment")
    env.load_state(manifest.initial_state)
    states = [manifest.initial_state]
    for index, transition in enumerate(transitions, start=1):
        current = env.get_state()
        if current.state_hash != transition.state_before.state_hash:
            raise ReplayMismatchError(f"transition {index} state_before hash mismatch")
        result = env.step(
            transition.step_result.step_id,
            dict(transition.joint_action),
        )
        if result.state_after.state_hash != transition.state_after.state_hash:
            raise ReplayMismatchError(f"transition {index} state_after hash mismatch")
        states.append(result.state_after)
    return tuple(states)
