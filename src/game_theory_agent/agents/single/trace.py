"""Append-only decision traces and local LangGraph checkpoint storage."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .models import DecisionTrace


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_CHECKPOINT_MODEL_ALLOWLIST = [
    ("game_theory_agent.agents.single.gateway", "GatewaySnapshot"),
    ("game_theory_agent.agents.single.models", "DecisionContext"),
    ("game_theory_agent.agents.single.models", "DecisionProposal"),
    ("game_theory_agent.agents.single.models", "DecisionTrace"),
    ("game_theory_agent.agents.single.models", "IntentDraft"),
    ("game_theory_agent.agents.single.models", "PersonaTraceManifest"),
    ("game_theory_agent.agents.single.models", "PromptAudit"),
    ("game_theory_agent.agents.single.models", "PromptTemplate"),
    ("game_theory_agent.agents.single.models", "StrategyReflection"),
    ("game_theory_agent.agents.single.provider", "ProviderResult"),
]


class JsonlTraceStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def append(self, trace: DecisionTrace) -> None:
        payload = json.dumps(
            trace.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        path = self._path(trace.episode_id)
        with self._lock, path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")

    def read_episode(self, episode_id: str) -> list[DecisionTrace]:
        path = self._path(episode_id)
        if not path.exists():
            return []
        with self._lock, path.open("r", encoding="utf-8") as stream:
            return [
                DecisionTrace.model_validate_json(line)
                for line in stream
                if line.strip()
            ]

    def read_company_before_round(
        self,
        episode_id: str,
        company_id: str,
        round_number: int,
        limit: int = 5,
    ) -> list[DecisionTrace]:
        traces = [
            trace
            for trace in self.read_episode(episode_id)
            if trace.company_id == company_id and trace.round < round_number
        ]
        traces.sort(key=lambda item: item.round)
        bounded_limit = max(0, limit)
        if bounded_limit == 0:
            return []
        return traces[-bounded_limit:]

    def _path(self, episode_id: str) -> Path:
        safe_name = _SAFE_NAME.sub("_", episode_id).strip("._") or "episode"
        return self.root / f"{safe_name}.jsonl"


def open_sqlite_checkpointer(path: str | Path) -> SqliteSaver:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    serializer = JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_MODEL_ALLOWLIST)
    return SqliteSaver(connection, serde=serializer)
