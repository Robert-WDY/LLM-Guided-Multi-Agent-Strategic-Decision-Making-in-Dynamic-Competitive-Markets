"""Stable RNG, canonical JSON, and hash protocols for market replay."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Mapping


UINT64_MASK = (1 << 64) - 1
UNIFORM53_DENOMINATOR = 1 << 53


def _primitive(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _primitive(value.to_dict())
    if is_dataclass(value):
        return _primitive(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_primitive(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Canonical JSON does not support {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Canonical JSON for integer fixed-point protocol objects.

    RFC 8785 number edge cases do not apply because persisted protocol objects
    intentionally contain no floats. Sorting and compact UTF-8 encoding are
    therefore the JCS representation for the supported value domain.
    """

    return json.dumps(
        _primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_hash(value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return "sha256:" + digest


def state_hash(state_dict: Mapping[str, Any]) -> str:
    payload = dict(state_dict)
    payload.pop("state_hash", None)
    return sha256_hash(payload)


def seed_material(
    protocol_version: str,
    episode_seed: int,
    round_number: int,
    component_name: str,
    entity_id: str = "",
    draw_index: int = 0,
) -> bytes:
    protocol = protocol_version.encode("utf-8")
    component = component_name.encode("utf-8")
    entity = entity_id.encode("utf-8")
    if not 0 <= episode_seed <= UINT64_MASK:
        raise ValueError("episode_seed must fit uint64")
    if not 0 <= round_number < (1 << 32):
        raise ValueError("round_number must fit uint32")
    if not 0 <= draw_index < (1 << 32):
        raise ValueError("draw_index must fit uint32")
    for label, encoded in (
        ("protocol_version", protocol),
        ("component_name", component),
        ("entity_id", entity),
    ):
        if len(encoded) >= (1 << 16):
            raise ValueError(f"{label} is too long")
    return b"".join(
        (
            struct.pack(">H", len(protocol)),
            protocol,
            struct.pack(">Q", episode_seed),
            struct.pack(">I", round_number),
            struct.pack(">H", len(component)),
            component,
            struct.pack(">H", len(entity)),
            entity,
            struct.pack(">I", draw_index),
        )
    )


def derive_sub_seed(
    protocol_version: str,
    episode_seed: int,
    round_number: int,
    component_name: str,
    entity_id: str = "",
    draw_index: int = 0,
) -> int:
    material = seed_material(
        protocol_version,
        episode_seed,
        round_number,
        component_name,
        entity_id,
        draw_index,
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


class SplitMix64:
    """SplitMix64 with explicit unsigned overflow semantics."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & UINT64_MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & UINT64_MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
        return (value ^ (value >> 31)) & UINT64_MASK

    def uniform53_numerator(self) -> int:
        return self.next_u64() >> 11

    def uniform(self) -> float:
        return self.uniform53_numerator() / UNIFORM53_DENOMINATOR

    def normal_approx(self) -> float:
        return sum(self.uniform() for _ in range(12)) - 6.0


class ComponentRng:
    """One independently derived stream for one round/component/entity/draw."""

    __slots__ = ("sub_seed", "stream")

    def __init__(
        self,
        protocol_version: str,
        episode_seed: int,
        round_number: int,
        component_name: str,
        entity_id: str = "",
        draw_index: int = 0,
    ) -> None:
        self.sub_seed = derive_sub_seed(
            protocol_version,
            episode_seed,
            round_number,
            component_name,
            entity_id,
            draw_index,
        )
        self.stream = SplitMix64(self.sub_seed)

    def uniform(self) -> float:
        return self.stream.uniform()

    def normal_approx(self) -> float:
        return self.stream.normal_approx()

    def weighted_choice(self, weights_ppm: Mapping[str, int]) -> str:
        if not weights_ppm or sum(weights_ppm.values()) != 1_000_000:
            raise ValueError("weights_ppm must sum to 1000000")
        draw = (self.stream.uniform53_numerator() * 1_000_000) // UNIFORM53_DENOMINATOR
        cumulative = 0
        for key in sorted(weights_ppm):
            cumulative += weights_ppm[key]
            if draw < cumulative:
                return key
        return sorted(weights_ppm)[-1]
