"""Filesystem object store plus SQLite index for immutable Agent versions."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from game_theory_agent.agent_registry.contracts import (
    AgentLifecycleRecord,
    AgentVersionManifest,
    ArtifactRef,
    LifecycleState,
    ResolvedAgentVersion,
)
from game_theory_agent.market.protocols import canonical_json


class AgentRegistryError(RuntimeError):
    pass


class AgentVersionNotFoundError(AgentRegistryError):
    pass


class AgentRegistryIntegrityError(AgentRegistryError):
    pass


class ImmutableAgentVersionError(AgentRegistryError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hex_id(identifier: str) -> str:
    if not identifier.startswith("sha256:") or len(identifier) != 71:
        raise ValueError("expected sha256 identifier")
    return identifier[7:]


class AgentRegistry:
    """Registers immutable manifests and keeps mutable lifecycle data separate."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.objects_root = self.root / "objects" / "sha256"
        self.versions_root = self.root / "versions" / "sha256"
        self.database_path = self.root / "registry.sqlite3"
        self.objects_root.mkdir(parents=True, exist_ok=True)
        self.versions_root.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_families (
                    family_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_versions (
                    agent_version_id TEXT PRIMARY KEY,
                    family_id TEXT NOT NULL REFERENCES agent_families(family_id),
                    behavior_spec_hash TEXT NOT NULL,
                    manifest_hash TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_lineage (
                    child_version_id TEXT NOT NULL REFERENCES agent_versions(agent_version_id),
                    parent_version_id TEXT NOT NULL REFERENCES agent_versions(agent_version_id),
                    mutation_operator TEXT,
                    training_run_id TEXT,
                    reason TEXT NOT NULL,
                    PRIMARY KEY (child_version_id, parent_version_id)
                );
                CREATE TABLE IF NOT EXISTS agent_lifecycle (
                    agent_version_id TEXT PRIMARY KEY REFERENCES agent_versions(agent_version_id),
                    state TEXT NOT NULL,
                    display_name TEXT,
                    tags_json TEXT NOT NULL,
                    rating_milli INTEGER,
                    games_played INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS agent_versions_no_update
                BEFORE UPDATE ON agent_versions
                BEGIN SELECT RAISE(ABORT, 'agent_versions are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS agent_versions_no_delete
                BEFORE DELETE ON agent_versions
                BEGIN SELECT RAISE(ABORT, 'agent_versions are immutable'); END;
                """
            )

    def register_family(self, family_id: str, *, description: str = "") -> None:
        if not family_id or len(family_id) > 200:
            raise ValueError("family_id must be non-empty and at most 200 characters")
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT description FROM agent_families WHERE family_id = ?",
                (family_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["description"]) != description:
                    raise ImmutableAgentVersionError(
                        "family already exists with a different description"
                    )
                return
            connection.execute(
                "INSERT INTO agent_families VALUES (?, ?, ?)",
                (family_id, description, _now()),
            )

    def _artifact_path(self, artifact_id: str) -> Path:
        digest = _hex_id(artifact_id)
        return self.objects_root / digest[:2] / digest[2:]

    def _manifest_path(self, version_id: str) -> Path:
        digest = _hex_id(version_id)
        return self.versions_root / digest[:2] / digest[2:] / "manifest.json"

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def put_artifact(self, data: bytes, *, media_type: str) -> ArtifactRef:
        artifact_id = "sha256:" + hashlib.sha256(data).hexdigest()
        path = self._artifact_path(artifact_id)
        if path.exists():
            if path.read_bytes() != data:
                raise AgentRegistryIntegrityError("artifact hash collision or corruption")
        else:
            self._atomic_write(path, data)
        return ArtifactRef(
            sha256=artifact_id,
            size_bytes=len(data),
            media_type=media_type,
        )

    def _verify_artifact(self, ref: ArtifactRef) -> Path:
        path = self._artifact_path(ref.sha256)
        if not path.exists():
            raise AgentRegistryIntegrityError(f"missing artifact: {ref.sha256}")
        data = path.read_bytes()
        actual = "sha256:" + hashlib.sha256(data).hexdigest()
        if actual != ref.sha256 or len(data) != ref.size_bytes:
            raise AgentRegistryIntegrityError(f"corrupt artifact: {ref.sha256}")
        return path

    def register_version(self, manifest: AgentVersionManifest) -> AgentVersionManifest:
        """Register once; an existing version is returned without being mutated."""

        for ref in manifest.behavior_spec.artifact_refs():
            self._verify_artifact(ref)
        with self._connect() as connection:
            family = connection.execute(
                "SELECT 1 FROM agent_families WHERE family_id = ?",
                (manifest.family_id,),
            ).fetchone()
            if family is None:
                raise AgentRegistryError(
                    f"unknown agent family: {manifest.family_id}; register it first"
                )
            existing = connection.execute(
                "SELECT manifest_path FROM agent_versions WHERE agent_version_id = ?",
                (manifest.agent_version_id,),
            ).fetchone()
            if existing is not None:
                stored = self.get(manifest.agent_version_id)
                if stored.behavior_spec != manifest.behavior_spec:
                    raise ImmutableAgentVersionError(
                        "existing version id has different behavior content"
                    )
                return stored
            missing_parents = [
                parent
                for parent in manifest.lineage.parent_version_ids
                if connection.execute(
                    "SELECT 1 FROM agent_versions WHERE agent_version_id = ?",
                    (parent,),
                ).fetchone()
                is None
            ]
            if missing_parents:
                raise AgentRegistryError(
                    f"lineage parents are not registered: {missing_parents}"
                )
            path = self._manifest_path(manifest.agent_version_id)
            payload = canonical_json(manifest.model_dump(mode="json")).encode("utf-8")
            if path.exists():
                raise AgentRegistryIntegrityError(
                    "unindexed manifest already exists at immutable version path"
                )
            self._atomic_write(path, payload)
            relative = path.relative_to(self.root).as_posix()
            try:
                connection.execute(
                    "INSERT INTO agent_versions VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        manifest.agent_version_id,
                        manifest.family_id,
                        manifest.behavior_spec_hash,
                        manifest.manifest_hash,
                        relative,
                        manifest.created_at,
                    ),
                )
                for parent in manifest.lineage.parent_version_ids:
                    connection.execute(
                        "INSERT INTO agent_lineage VALUES (?, ?, ?, ?, ?)",
                        (
                            manifest.agent_version_id,
                            parent,
                            manifest.lineage.mutation_operator,
                            manifest.lineage.training_run_id,
                            manifest.lineage.reason,
                        ),
                    )
                connection.execute(
                    "INSERT INTO agent_lifecycle VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        manifest.agent_version_id,
                        "draft",
                        None,
                        "[]",
                        None,
                        0,
                        _now(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise AgentRegistryError(str(exc)) from exc
        return manifest

    def get(self, agent_version_id: str) -> AgentVersionManifest:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_versions WHERE agent_version_id = ?",
                (agent_version_id,),
            ).fetchone()
        if row is None:
            raise AgentVersionNotFoundError(agent_version_id)
        path = (self.root / str(row["manifest_path"])).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise AgentRegistryIntegrityError("manifest path escapes registry") from exc
        if not path.exists():
            raise AgentRegistryIntegrityError("registered manifest file is missing")
        raw_payload = path.read_bytes()
        try:
            manifest = AgentVersionManifest.model_validate_json(raw_payload)
        except ValueError as exc:
            raise AgentRegistryIntegrityError(f"invalid manifest: {exc}") from exc
        canonical_payload = canonical_json(
            manifest.model_dump(mode="json")
        ).encode("utf-8")
        if raw_payload != canonical_payload:
            raise AgentRegistryIntegrityError("manifest bytes are not canonical or were changed")
        if manifest.agent_version_id != agent_version_id:
            raise AgentRegistryIntegrityError("manifest version id mismatch")
        if manifest.manifest_hash != str(row["manifest_hash"]):
            raise AgentRegistryIntegrityError("manifest hash mismatch")
        if manifest.behavior_spec_hash != str(row["behavior_spec_hash"]):
            raise AgentRegistryIntegrityError("behavior spec hash mismatch")
        return manifest

    def resolve(self, agent_version_id: str) -> ResolvedAgentVersion:
        manifest = self.get(agent_version_id)
        paths = {
            ref.sha256: str(self._verify_artifact(ref))
            for ref in manifest.behavior_spec.artifact_refs()
        }
        return ResolvedAgentVersion(manifest=manifest, artifact_paths=paths)

    def list_versions(self, family_id: str | None = None) -> tuple[AgentVersionManifest, ...]:
        with self._connect() as connection:
            if family_id is None:
                rows = connection.execute(
                    "SELECT agent_version_id FROM agent_versions ORDER BY created_at, agent_version_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT agent_version_id FROM agent_versions WHERE family_id = ? "
                    "ORDER BY created_at, agent_version_id",
                    (family_id,),
                ).fetchall()
        return tuple(self.get(str(row["agent_version_id"])) for row in rows)

    def lifecycle(self, agent_version_id: str) -> AgentLifecycleRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_lifecycle WHERE agent_version_id = ?",
                (agent_version_id,),
            ).fetchone()
        if row is None:
            raise AgentVersionNotFoundError(agent_version_id)
        return AgentLifecycleRecord(
            agent_version_id=agent_version_id,
            state=str(row["state"]),
            display_name=row["display_name"],
            tags=tuple(json.loads(str(row["tags_json"]))),
            rating_milli=row["rating_milli"],
            games_played=int(row["games_played"]),
            updated_at=str(row["updated_at"]),
        )

    def update_lifecycle(
        self,
        agent_version_id: str,
        *,
        state: LifecycleState | None = None,
        display_name: str | None = None,
        tags: Iterable[str] | None = None,
        rating_milli: int | None = None,
        games_played: int | None = None,
    ) -> AgentLifecycleRecord:
        current = self.lifecycle(agent_version_id)
        next_state = state or current.state
        allowed = {
            "draft": {"draft", "candidate", "retired"},
            "candidate": {"candidate", "validated", "retired"},
            "validated": {"validated", "active", "retired"},
            "active": {"active", "champion", "historical", "retired"},
            "champion": {"champion", "historical", "retired"},
            "historical": {"historical", "active", "retired"},
            "retired": {"retired"},
        }
        if next_state not in allowed[current.state]:
            raise AgentRegistryError(
                f"invalid lifecycle transition: {current.state} -> {next_state}"
            )
        next_tags = current.tags if tags is None else tuple(sorted(set(tags)))
        next_games = current.games_played if games_played is None else games_played
        if next_games < 0:
            raise ValueError("games_played must be non-negative")
        with self._connect() as connection:
            connection.execute(
                "UPDATE agent_lifecycle SET state = ?, display_name = ?, tags_json = ?, "
                "rating_milli = ?, games_played = ?, updated_at = ? "
                "WHERE agent_version_id = ?",
                (
                    next_state,
                    current.display_name if display_name is None else display_name,
                    canonical_json(next_tags),
                    current.rating_milli if rating_milli is None else rating_milli,
                    next_games,
                    _now(),
                    agent_version_id,
                ),
            )
        return self.lifecycle(agent_version_id)

    def promote(self, agent_version_id: str, state: LifecycleState) -> AgentLifecycleRecord:
        return self.update_lifecycle(agent_version_id, state=state)

    def retire(self, agent_version_id: str) -> AgentLifecycleRecord:
        return self.update_lifecycle(agent_version_id, state="retired")

    def verify(self) -> None:
        for manifest in self.list_versions():
            resolved = self.resolve(manifest.agent_version_id)
            if resolved.manifest.manifest_hash != manifest.manifest_hash:
                raise AgentRegistryIntegrityError("registry resolution is unstable")
