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
    code_commit: str = "unknown"
    agent_configs: tuple[tuple[str, Mapping[str, Any]], ...] = ()

    @classmethod
    def create(
        cls,
        env: MarketEnv,
        initial_state: MarketState,
        *,
        experiment_id: str = "experiment-001",
        code_commit: str = "unknown",
        agent_configs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> "EpisodeManifest":
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
            information_mode="perfect",
            agent_configs=tuple(sorted((agent_configs or {}).items())),
            initial_state=initial_state,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": "episode-manifest-v1.0.0",
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
