"""Versioned JSON checkpoints for the single-process internal research server.

Only explicitly registered domain types are decoded. Provider clients, API keys,
threads and locks are never serialized; loading a checkpoint never calls a model.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from game_theory_agent.agents.memory import EpisodeMemory
from game_theory_agent.belief import BeliefLedger
from game_theory_agent.cooperation import CooperationLedger
from game_theory_agent.interaction import CommunicationRoundLedger
from game_theory_agent.market import MarketEnv
from game_theory_agent.opponent import OpponentModelLedger

SCHEMA = "market-session-checkpoint-v1"
MODULES = (
    "market.models", "market.replay", "cooperation.contracts",
    "interaction.contracts", "belief.contracts", "opponent.schema",
    "agents.contracts", "agents.personas", "decisioning",
)
PLAIN_FIELDS = {
    BeliefLedger: {"episode_id", "company_ids", "mode", "_evidence", "_settlement_hashes", "_claim_outcomes"},
    OpponentModelLedger: {"episode_id", "company_ids", "_evidence", "_settlement_hashes"},
    CooperationLedger: {"mode", "episode_id", "company_ids", "max_rounds", "max_contribution_cents", "_proposals", "_responses", "_response_by_proposal_company", "_commitments", "_verifications", "_closes", "_rounds"},
    CommunicationRoundLedger: {"episode_id", "round_number", "state_version", "state_hash", "company_ids", "mode", "_submissions", "_delivered", "_closure"},
    EpisodeMemory: {"_recent", "_trend", "_critical_events", "_trend_window_rounds", "fallback_count", "current_plan"},
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class CheckpointError(ValueError):
    pass


class DomainCodec:
    def __init__(self) -> None:
        self.types: dict[str, type] = {}
        for name in MODULES:
            module = importlib.import_module("game_theory_agent." + name)
            for value in vars(module).values():
                if isinstance(value, type) and value.__module__ == module.__name__:
                    if is_dataclass(value) or issubclass(value, (BaseModel, Enum)):
                        self.register(value)
        for cls in PLAIN_FIELDS:
            self.register(cls)

    def register(self, cls: type) -> None:
        self.types[cls.__module__ + ":" + cls.__name__] = cls

    def encode(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return {"kind": "enum", "type": self._type(value), "data": value.value}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {"kind": "map", "data": [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple, set, frozenset, deque)):
            result = {"kind": type(value).__name__, "data": [self.encode(item) for item in value]}
            if isinstance(value, deque):
                result["maxlen"] = value.maxlen
            return result
        if isinstance(value, BaseModel):
            return {"kind": "model", "type": self._type(value), "data": value.model_dump(mode="json")}
        if is_dataclass(value):
            return {"kind": "dataclass", "type": self._type(value), "data": {f.name:self.encode(getattr(value, f.name)) for f in fields(value)}}
        if type(value) in PLAIN_FIELDS:
            if set(vars(value)) != PLAIN_FIELDS[type(value)]:
                raise CheckpointError(f"Checkpoint schema needs updating for {type(value).__name__}")
            return {"kind": "ledger", "type": self._type(value), "data": {k:self.encode(v) for k,v in vars(value).items()}}
        raise CheckpointError(f"Unsupported checkpoint value: {type(value).__name__}")

    def _type(self, value: Any) -> str:
        key = type(value).__module__ + ":" + type(value).__name__
        if key not in self.types:
            raise CheckpointError(f"Unregistered checkpoint type: {key}")
        return key

    def decode(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        kind, data = value.get("kind"), value.get("data")
        if kind == "map":
            return {self.decode(k):self.decode(v) for k,v in data}
        if kind in {"list", "tuple", "set", "frozenset", "deque"}:
            items = [self.decode(item) for item in data]
            return deque(items, maxlen=value["maxlen"]) if kind == "deque" else {"list":list,"tuple":tuple,"set":set,"frozenset":frozenset}[kind](items)
        cls = self.types.get(value.get("type", ""))
        if cls is None:
            raise CheckpointError("Unknown checkpoint type")
        if kind == "enum" and issubclass(cls, Enum):
            return cls(data)
        if kind == "model" and issubclass(cls, BaseModel):
            return cls.model_validate(data)
        if kind == "dataclass" and is_dataclass(cls):
            return cls(**{k:self.decode(v) for k,v in data.items()})
        if kind == "ledger" and cls in PLAIN_FIELDS and set(data) == PLAIN_FIELDS[cls]:
            instance = cls.__new__(cls)
            instance.__dict__.update({k:self.decode(v) for k,v in data.items()})
            return instance
        raise CheckpointError("Invalid checkpoint object")


CODEC = DomainCodec()


def runtime_state(runtime: Any) -> dict[str, Any]:
    tracker = runtime._utility_tracker
    return {
        "memory": runtime.memory,
        "utility_episode_id": runtime._utility_episode_id,
        "utility": None if tracker is None else {
            "profile_id": tracker.evaluator.profile.persona_id,
            "discount_multiplier_ppm": tracker.discount_multiplier_ppm,
            "cumulative_discounted_utility_ppm": tracker.cumulative_discounted_utility_ppm,
        },
    }


def checkpoint(session: Any) -> dict[str, Any]:
    excluded = {"env", "lock", "agent_runtimes", "restored_runtime_states"}
    values = {f.name:getattr(session, f.name) for f in fields(session) if f.name not in excluded}
    runtime_states = dict(session.restored_runtime_states)
    runtime_states.update({key:runtime_state(runtime) for key,runtime in session.agent_runtimes.items()})
    state = session.env.get_state()
    return {
        "schema": SCHEMA,
        "config_sha256": session.env.config.config_sha256,
        "episode_id": state.episode_id,
        "round": state.round,
        "terminal": state.terminal,
        "state_hash": state.state_hash,
        "recovery_required": bool(session.coordinator_active_run_id or session.recovery_required),
        "saved_at": datetime.now(UTC).isoformat(),
        "session": CODEC.encode(values),
        "market": CODEC.encode({"state":state,"step_cache":session.env._step_cache,"action_registry":session.env._action_registry}),
        "runtime_states": CODEC.encode(runtime_states),
    }


def restore(data: dict[str, Any], config: Any, session_type: type) -> Any:
    if data.get("schema") != SCHEMA or data.get("config_sha256") != config.config_sha256:
        raise CheckpointError("Checkpoint version/config differs from this server; use its frozen release")
    market = CODEC.decode(data["market"])
    env = MarketEnv(config)
    env.load_state(market["state"])
    if env.get_state().state_hash != data["state_hash"] or env.get_state().episode_id != data["episode_id"]:
        raise CheckpointError("Checkpoint market identity mismatch")
    env._step_cache = market["step_cache"]
    env._action_registry = market["action_registry"]
    values = CODEC.decode(data["session"])
    session = session_type(env=env, **values)
    session.restored_runtime_states = CODEC.decode(data["runtime_states"])
    if session.coordinator_active_run_id:
        session.interrupted_run_ids.add(session.coordinator_active_run_id)
        session.coordinator_active_run_id = None
        session.recovery_required = True
    return session


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=FULL")
                db.execute("CREATE TABLE IF NOT EXISTS checkpoints (episode_id TEXT PRIMARY KEY, document TEXT NOT NULL, sha256 TEXT NOT NULL, saved_at TEXT NOT NULL)")
                yield db
        finally:
            db.close()

    def save(self, session: Any) -> dict[str, Any]:
        data = checkpoint(session)
        document = _json(data)
        checksum = hashlib.sha256(document.encode("utf-8")).hexdigest()
        with self._connect() as db:
            db.execute("INSERT INTO checkpoints VALUES (?,?,?,?) ON CONFLICT(episode_id) DO UPDATE SET document=excluded.document, sha256=excluded.sha256, saved_at=excluded.saved_at",(data["episode_id"],document,checksum,data["saved_at"]))
        return self.metadata(data, checksum)

    @staticmethod
    def metadata(data: dict[str, Any], checksum: str) -> dict[str, Any]:
        return {**{k:data[k] for k in ("episode_id","round","terminal","state_hash","config_sha256","saved_at","recovery_required")},"checkpoint_sha256":checksum}

    def load(self, episode_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT document, sha256 FROM checkpoints WHERE episode_id=?",(episode_id,)).fetchone()
        if row is None:
            return None
        if hashlib.sha256(row[0].encode("utf-8")).hexdigest() != row[1]:
            raise CheckpointError("Checkpoint checksum mismatch")
        return json.loads(row[0])

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT document, sha256 FROM checkpoints ORDER BY saved_at DESC LIMIT 200").fetchall()
        result = []
        for document, checksum in rows:
            if hashlib.sha256(document.encode("utf-8")).hexdigest() != checksum:
                raise CheckpointError("Checkpoint checksum mismatch")
            result.append(self.metadata(json.loads(document),checksum))
        return result
