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

        regime = self.mapping("agent_context", "regime_thresholds")
        regime_fields = (
            "price_war_discount_ppm",
            "price_war_min_companies",
            "high_demand_ratio_ppm",
            "low_demand_ratio_ppm",
            "capacity_constrained_ppm",
            "capacity_slack_ppm",
            "supply_high_ppm",
            "supply_crisis_ppm",
            "supply_low_ppm",
            "hhi_moderate_ppm",
            "hhi_concentrated_ppm",
            "risk_warning_probability_ppm",
        )
        for field in regime_fields:
            value = regime.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(
                    f"agent_context.regime_thresholds.{field} must be non-negative"
                )
        if not (
            int(regime["low_demand_ratio_ppm"])
            < 1_000_000
            < int(regime["high_demand_ratio_ppm"])
        ):
            raise ConfigError("demand regime thresholds must bracket 1000000")
        if not (
            int(regime["capacity_slack_ppm"])
            < int(regime["capacity_constrained_ppm"])
            <= 1_000_000
        ):
            raise ConfigError("capacity regime thresholds are inconsistent")
        if not (
            int(regime["supply_low_ppm"])
            < 1_000_000
            < int(regime["supply_high_ppm"])
            < int(regime["supply_crisis_ppm"])
        ):
            raise ConfigError("supply regime thresholds are inconsistent")
        if not (
            int(regime["hhi_moderate_ppm"])
            < int(regime["hhi_concentrated_ppm"])
            <= 1_000_000
        ):
            raise ConfigError("HHI regime thresholds are inconsistent")

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

        policy = self.mapping("decision_policy")
        for field in (
            "future_overhead_reserve_rounds",
            "minimum_unit_contribution_cents",
            "recovery_loss_streak",
            "recovery_cash_drawdown_ppm",
            "liquidity_crisis_runway_milli_rounds",
            "recovery_spend_cap_ppm",
            "crisis_spend_cap_ppm",
        ):
            value = policy.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(f"decision_policy.{field} must be non-negative")
        for field in (
            "recovery_cash_drawdown_ppm",
            "recovery_spend_cap_ppm",
            "crisis_spend_cap_ppm",
        ):
            if int(policy[field]) > 1_000_000:
                raise ConfigError(f"decision_policy.{field} must be <= 1000000")
        if int(policy["crisis_spend_cap_ppm"]) > int(
            policy["recovery_spend_cap_ppm"]
        ):
            raise ConfigError("crisis spend cap must not exceed recovery spend cap")

        bounds = self.mapping("action", "bounds")
        for name, raw in bounds.items():
            if not isinstance(raw, Mapping):
                raise ConfigError(f"action.bounds.{name} must be a mapping")
            low, high = raw.get("min"), raw.get("max")
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (low, high)):
                raise ConfigError(f"action.bounds.{name} min/max must be integers")
            if low < 0 or low > high:
                raise ConfigError(f"action.bounds.{name} range is invalid")

        shared = self.mapping("shared_resilience")
        for field in (
            "initial_industry_resilience_ppm",
            "retention_ppm",
            "contribution_input_weight_ppm",
            "public_protection_weight_ppm",
        ):
            value = shared.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 1_000_000
            ):
                raise ConfigError(f"shared_resilience.{field} must be ppm")
        scale = shared.get("contribution_scale_cents")
        if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
            raise ConfigError(
                "shared_resilience.contribution_scale_cents must be positive"
            )

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

        max_event_supply = max(
            int(definition["supply_cost_multiplier_ppm"])
            for event in events.values()
            for definition in event["severity"].values()
        )
        max_refund = max(
            int(definition["refund_rate_ppm"])
            for incident in self.mapping("incidents", "definitions").values()
            for definition in incident["severity"].values()
        )
        worst_unit_cost = (
            int(initial["base_unit_cost_cents"])
            * self.integer("market", "supply_cost_max_ppm")
            * max_event_supply
            + 1_000_000_000_000 - 1
        ) // 1_000_000_000_000
        required_price_max = (
            (
                worst_unit_cost
                + self.integer(
                    "operating_costs", "fulfillment_cost_per_order_cents"
                )
                + int(policy["minimum_unit_contribution_cents"])
            )
            * 1_000_000
            + (1_000_000 - max_refund)
            - 1
        ) // (1_000_000 - max_refund)
        if int(bounds["price_cents"]["max"]) < required_price_max:
            raise ConfigError(
                "action.bounds.price_cents.max cannot cover worst-case safe price "
                f"{required_price_max}"
            )

        persona_cfg = self.mapping("persona_utilities")
        if persona_cfg.get("schema_version") not in {
            "persona-catalog-v1.0.0",
            "persona-catalog-v1.1.0",
        }:
            raise ConfigError(
                "persona_utilities.schema_version must be a supported persona catalog"
            )
        weights_by_persona = self.mapping("persona_utilities", "weights_ppm")
        traits_by_persona = self.mapping("persona_utilities", "traits_ppm")
        labels_by_persona = self.mapping("persona_utilities", "labels")
        objectives_by_persona = self.mapping("persona_utilities", "objectives")
        persona_ids = set(weights_by_persona)
        if not persona_ids or not (
            persona_ids
            == set(traits_by_persona)
            == set(labels_by_persona)
            == set(objectives_by_persona)
        ):
            raise ConfigError(
                "persona weights, traits, labels and objectives must use the same ids"
            )
        if persona_cfg.get("default_profile_id") not in persona_ids:
            raise ConfigError("persona_utilities.default_profile_id is unknown")
        utility_components = {
            "profit",
            "share",
            "growth",
            "stability",
            "cash",
            "reputation",
            "resilience",
            "social_welfare",
            "cooperation_reputation",
        }
        trait_components = {
            "time_discount",
            "risk_aversion",
            "reciprocity",
            "commitment_honesty",
            "opportunism",
        }
        capabilities = self.mapping("persona_utilities", "capabilities")
        for capability in ("social_welfare", "cooperation"):
            if not isinstance(capabilities.get(capability), bool):
                raise ConfigError(
                    f"persona_utilities.capabilities.{capability} must be boolean"
                )
        for persona, weights in weights_by_persona.items():
            if set(weights) != utility_components:
                raise ConfigError(
                    f"persona {persona} weights must define {sorted(utility_components)}"
                )
            self._require_ppm_sum(weights, f"persona {persona} weights")
            traits = traits_by_persona[persona]
            if not isinstance(traits, Mapping) or set(traits) != trait_components:
                raise ConfigError(
                    f"persona {persona} traits must define {sorted(trait_components)}"
                )
            for name, value in traits.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value <= 1_000_000
                ):
                    raise ConfigError(
                        f"persona {persona} trait {name} must be ppm"
                    )
            for field, capability in (
                ("social_welfare", "social_welfare"),
                ("cooperation_reputation", "cooperation"),
            ):
                if not capabilities[capability] and int(weights[field]) != 0:
                    raise ConfigError(
                        f"persona {persona} cannot weight disabled {capability}"
                    )
            for field, source in (
                ("label", labels_by_persona[persona]),
                ("objective", objectives_by_persona[persona]),
            ):
                if not isinstance(source, str) or not source.strip():
                    raise ConfigError(f"persona {persona} {field} must be text")

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
