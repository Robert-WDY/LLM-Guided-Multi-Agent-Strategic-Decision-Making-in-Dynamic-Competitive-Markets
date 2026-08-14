"""Loading and validation for the Engineering MVP v4 market configuration."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from game_theory_agent.market.exceptions import ConfigError


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _canonical_config_bytes(data: Mapping[str, Any]) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class MarketConfig:
    """Validated, deeply read-only v4 parameters loaded from YAML."""

    data: Mapping[str, Any]
    config_sha256: str

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> "MarketConfig":
        plain = _thaw(source)
        if not isinstance(plain, dict):
            raise ConfigError("Configuration root must be a mapping")
        config = cls(
            data=_freeze(plain),
            config_sha256="sha256:"
            + hashlib.sha256(_canonical_config_bytes(plain)).hexdigest(),
        )
        config._validate()
        return config

    @property
    def config_id(self) -> str:
        return self.text("config_id")

    @property
    def config_version(self) -> str:
        return self.text("config_version")

    @property
    def environment_version(self) -> str:
        return self.text("environment_version")

    @property
    def rounds(self) -> int:
        return self.integer("market", "rounds")

    @property
    def base_demand_orders(self) -> int:
        return self.integer("market", "base_demand_orders")

    @property
    def min_agents(self) -> int:
        return self.integer("market", "min_agents")

    @property
    def max_agents(self) -> int:
        return self.integer("market", "max_agents")

    @property
    def rng_protocol_version(self) -> str:
        return self.text("protocols", "rng")

    @property
    def hash_protocol_version(self) -> str:
        return self.text("protocols", "hash")

    def get(self, *path: str) -> Any:
        value: Any = self.data
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                raise ConfigError(f"Missing configuration field: {'.'.join(path)}")
            value = value[key]
        return value

    def mapping(self, *path: str) -> Mapping[str, Any]:
        value = self.get(*path)
        if not isinstance(value, Mapping):
            raise ConfigError(f"{'.'.join(path)} must be a mapping")
        return value

    def integer(self, *path: str) -> int:
        value = self.get(*path)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{'.'.join(path)} must be an integer")
        return value

    def text(self, *path: str) -> str:
        value = self.get(*path)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{'.'.join(path)} must be a non-empty string")
        return value

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self.data)

    def _validate(self) -> None:
        for field in ("config_id", "config_version", "environment_version"):
            self.text(field)

        if not 2 <= self.min_agents <= self.max_agents <= 8:
            raise ConfigError("market agent bounds must satisfy 2 <= min <= max <= 8")
        if self.rounds <= 0 or self.base_demand_orders <= 0:
            raise ConfigError("market rounds and base demand must be positive")

        episode_options = self.mapping("episode_options")
        round_options = episode_options.get("round_options")
        if (
            not isinstance(round_options, tuple)
            or not round_options
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in round_options
            )
        ):
            raise ConfigError("episode_options.round_options must contain positive integers")
        if episode_options.get("default_rounds") not in round_options:
            raise ConfigError("episode_options.default_rounds must be a round option")
        if episode_options.get("seed_min") != 0 or episode_options.get("seed_max") != (1 << 64) - 1:
            raise ConfigError("episode seed range must cover uint64")

        initial = self.mapping("company_initial")
        for field in (
            "cash_balance_cents",
            "base_capacity_orders",
            "base_unit_cost_cents",
            "brand_awareness_ppm",
            "service_quality_ppm",
            "reputation_ppm",
            "resilience_ppm",
        ):
            value = initial.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(f"company_initial.{field} must be non-negative")
        for field in (
            "brand_awareness_ppm",
            "service_quality_ppm",
            "reputation_ppm",
            "resilience_ppm",
        ):
            if int(initial[field]) > 1_000_000:
                raise ConfigError(f"company_initial.{field} must be in [0, 1000000]")

        bounds = self.mapping("action", "bounds")
        for name, raw in bounds.items():
            if not isinstance(raw, Mapping):
                raise ConfigError(f"action.bounds.{name} must be a mapping")
            low, high = raw.get("min"), raw.get("max")
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (low, high)):
                raise ConfigError(f"action.bounds.{name} min/max must be integers")
            if low < 0 or low > high:
                raise ConfigError(f"action.bounds.{name} range is invalid")

        operating = self.mapping("operating_costs")
        for field in (
            "fixed_overhead_cents",
            "fulfillment_cost_per_order_cents",
        ):
            value = operating.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(f"operating_costs.{field} must be non-negative")

        segments = self.mapping("consumer_choice", "segments")
        if len(segments) != 3:
            raise ConfigError("Engineering MVP requires exactly three segments")
        self._require_ppm_sum(
            {name: value.get("weight_ppm") for name, value in segments.items()},
            "consumer segment weights",
        )
        for name, raw in segments.items():
            if not isinstance(raw, Mapping) or not isinstance(
                raw.get("coefficients_ppm"), Mapping
            ):
                raise ConfigError(f"consumer segment {name} is incomplete")

        market_models = self.mapping("market_models")
        self._require_ppm_sum(
            market_models.get("selection_weights_ppm"),
            "market model selection weights",
        )
        profiles = self.mapping("market_models", "profiles")
        if set(market_models["selection_weights_ppm"]) != set(profiles):
            raise ConfigError("market model weights and profiles must use the same ids")
        for model_id, profile in profiles.items():
            if not isinstance(profile, Mapping):
                raise ConfigError(f"market model {model_id} must be a mapping")
            self._require_ppm_sum(
                profile.get("segment_weights_ppm"),
                f"market model {model_id} segment weights",
            )
            if set(profile["segment_weights_ppm"]) != set(segments):
                raise ConfigError(
                    f"market model {model_id} must define every consumer segment"
                )
            multipliers = profile.get("utility_multipliers_ppm")
            required_multipliers = {
                "price", "awareness", "service", "reputation", "prior_stockout"
            }
            if not isinstance(multipliers, Mapping) or set(multipliers) != required_multipliers:
                raise ConfigError(
                    f"market model {model_id} utility multipliers are incomplete"
                )
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in multipliers.values()
            ):
                raise ConfigError(
                    f"market model {model_id} utility multipliers must be non-negative integers"
                )
            for field in (
                "demand_bias_ppm",
                "price_anchor_cents",
                "price_band_cents",
            ):
                value = profile.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ConfigError(f"market model {model_id}.{field} must be positive")
            for field in ("label", "description"):
                value = profile.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ConfigError(f"market model {model_id}.{field} is required")

        events = self.mapping("events", "definitions")
        required_impacts = {
            "estimated_probability_ppm",
            "duration_weights_ppm",
            "demand_multiplier_ppm",
            "supply_cost_multiplier_ppm",
            "capacity_multiplier_ppm",
            "advertising_multiplier_ppm",
            "service_penalty_ppm",
            "reputation_penalty_ppm",
        }
        for event_name, event in events.items():
            if not isinstance(event, Mapping):
                raise ConfigError(f"event {event_name} must be a mapping")
            self._require_ppm_sum(
                event.get("severity_weights_ppm"), f"{event_name} severities"
            )
            severity = event.get("severity")
            if not isinstance(severity, Mapping):
                raise ConfigError(f"event {event_name} severity table is missing")
            for level, definition in severity.items():
                if not isinstance(definition, Mapping) or not required_impacts <= set(
                    definition
                ):
                    raise ConfigError(f"event {event_name}.{level} is incomplete")
                self._require_ppm_sum(
                    definition.get("duration_weights_ppm"),
                    f"{event_name}.{level} duration weights",
                )

        incidents = self.mapping("incidents")
        repair_mitigation = incidents.get("max_repair_mitigation_ppm")
        if (
            isinstance(repair_mitigation, bool)
            or not isinstance(repair_mitigation, int)
            or not 0 <= repair_mitigation <= 1_000_000
        ):
            raise ConfigError("incidents.max_repair_mitigation_ppm must be ppm")
        self._require_ppm_sum(
            incidents.get("type_weights_ppm"), "incident type weights"
        )
        self._require_ppm_sum(
            incidents.get("severity_weights_ppm"), "incident severity weights"
        )
        for incident_name, incident in self.mapping("incidents", "definitions").items():
            severity = (
                incident.get("severity") if isinstance(incident, Mapping) else None
            )
            if not isinstance(severity, Mapping):
                raise ConfigError(f"incident {incident_name} severity table is missing")
            for level, definition in severity.items():
                if not isinstance(definition, Mapping):
                    raise ConfigError(f"incident {incident_name}.{level} is invalid")
                for field in (
                    "duration_rounds",
                    "repair_required_cents",
                    "capacity_multiplier_ppm",
                    "advertising_multiplier_ppm",
                    "service_penalty_ppm",
                    "reputation_penalty_ppm",
                    "refund_rate_ppm",
                ):
                    if not isinstance(definition.get(field), int):
                        raise ConfigError(
                            f"incident {incident_name}.{level}.{field} is missing"
                        )

        for persona, weights in self.mapping(
            "persona_utilities", "weights_ppm"
        ).items():
            self._require_ppm_sum(weights, f"persona {persona} weights")

    @staticmethod
    def _require_ppm_sum(raw: Any, label: str) -> None:
        if not isinstance(raw, Mapping) or not raw:
            raise ConfigError(f"{label} must be a non-empty mapping")
        values = tuple(raw.values())
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
            raise ConfigError(f"{label} must contain non-negative integer ppm values")
        if sum(values) != 1_000_000:
            raise ConfigError(f"{label} must sum to 1000000")


def load_market_config(path: str | Path) -> MarketConfig:
    """Load a UTF-8 YAML config, validate it, and freeze an episode-safe snapshot."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Unable to load config {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError("Configuration root must be a mapping")
    return MarketConfig.from_mapping(raw)
