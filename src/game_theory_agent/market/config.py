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


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge a versioned config overlay onto an immutable base."""

    merged = _thaw(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = _thaw(value)
    return merged


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
        supplier_strategy=self.data.get("supply_chain",{}).get("strategic_policy")
        if supplier_strategy is not None:
            expected={"contract_discount_ppm","minimum_contract_margin_ppm","max_inventory_orders","inventory_target_ppm",
                      "inventory_spoilage_ppm","fixed_overhead_cents","reserve_cash_cents","credit_limit_cents","interest_ppm",
                      "loan_term_rounds","reliability_budget_cents","reliability_cost_per_ppm","max_reliability_ppm","liquidation_recovery_ppm"}
            if not isinstance(supplier_strategy,Mapping) or set(supplier_strategy)!=expected:
                raise ConfigError("strategic supplier policy requires the complete versioned fields")
            for key,value in supplier_strategy.items():
                if type(value) is not int or value<0 or (key.endswith("_ppm") and value>1000000):
                    raise ConfigError("invalid strategic supplier parameter: "+key)
            if supplier_strategy["loan_term_rounds"]<1 or supplier_strategy["reliability_cost_per_ppm"]<1:
                raise ConfigError("supplier loan term and reliability unit cost must be positive")
        for field in ("config_id", "config_version", "environment_version"):
            self.text(field)

        if not 2 <= self.min_agents <= self.max_agents <= 10:
            raise ConfigError("market agent bounds must satisfy 2 <= min <= max <= 10")
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
            wtp = raw.get("wtp_distribution")
            if wtp is not None:
                if not isinstance(wtp, Mapping):
                    raise ConfigError(
                        f"consumer segment {name} wtp_distribution must be a mapping"
                    )
                weights = wtp.get("weights_ppm")
                offsets = wtp.get("offsets_cents")
                self._require_ppm_sum(
                    weights,
                    f"consumer segment {name} WTP cohort weights",
                )
                if not isinstance(offsets, Mapping) or set(offsets) != set(weights):
                    raise ConfigError(
                        f"consumer segment {name} WTP cohorts are inconsistent"
                    )
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in offsets.values()
                ):
                    raise ConfigError(
                        f"consumer segment {name} WTP offsets must be integer cents"
                    )

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
            "persona-catalog-v1.2.0",
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
        for field in (
            "profit_scale_cents",
            "share_growth_scale_ppm",
            "social_welfare_scale_cents",
        ):
            value = persona_cfg.get(field)
            if value is None and field == "social_welfare_scale_cents":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"persona_utilities.{field} must be positive")
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

        strategic = self.data.get("strategic_market")
        if strategic is not None:
            if not isinstance(strategic, Mapping):
                raise ConfigError("strategic_market must be a mapping")
            if strategic.get("enabled") is not True:
                raise ConfigError("strategic_market.enabled must be true when present")
            required_non_negative = (
                "distress_cash_threshold_cents",
                "immediate_exit_cash_threshold_cents",
                "distress_rounds_to_exit",
                "liquidation_recovery_ppm",
                "hhi_moderate_ppm",
                "hhi_concentrated_ppm",
                "dominance_share_ppm",
                "relative_price_weight_ppm",
                "absolute_price_weight_ppm",
                "regulatory_pressure_retention_ppm",
                "concentration_pressure_weight_ppm",
                "predatory_pressure_weight_ppm",
                "monopoly_markup_pressure_weight_ppm",
                "stockout_social_cost_per_order_cents",
            )
            for field in required_non_negative:
                value = strategic.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ConfigError(
                        f"strategic_market.{field} must be a non-negative integer"
                    )
            for field in (
                "liquidation_recovery_ppm",
                "hhi_moderate_ppm",
                "hhi_concentrated_ppm",
                "dominance_share_ppm",
                "relative_price_weight_ppm",
                "absolute_price_weight_ppm",
                "regulatory_pressure_retention_ppm",
                "concentration_pressure_weight_ppm",
                "predatory_pressure_weight_ppm",
                "monopoly_markup_pressure_weight_ppm",
            ):
                if int(strategic[field]) > 1_000_000:
                    raise ConfigError(f"strategic_market.{field} must be ppm")
            if not (
                int(strategic["immediate_exit_cash_threshold_cents"])
                <= int(strategic["distress_cash_threshold_cents"])
            ):
                raise ConfigError("immediate exit cash must not exceed distress cash")
            if int(strategic["distress_rounds_to_exit"]) <= 0:
                raise ConfigError("distress_rounds_to_exit must be positive")
            if not (
                int(strategic["hhi_moderate_ppm"])
                < int(strategic["hhi_concentrated_ppm"])
                < int(strategic["dominance_share_ppm"])
                <= 1_000_000
            ):
                raise ConfigError("strategic market concentration thresholds conflict")
            if (
                int(strategic["relative_price_weight_ppm"])
                + int(strategic["absolute_price_weight_ppm"])
                != 1_000_000
            ):
                raise ConfigError("strategic market price weights must sum to 1000000")
            project = strategic.get("threshold_project")
            if not isinstance(project, Mapping) or project.get("enabled") is not True:
                raise ConfigError("strategic_market.threshold_project must be enabled")
            for field in (
                "required_total_contribution_cents",
                "deadline_round",
                "failure_refund_rate_ppm",
                "public_protection_bonus_ppm",
                "supply_cost_reduction_ppm",
            ):
                value = project.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ConfigError(
                        f"strategic_market.threshold_project.{field} must be non-negative"
                    )
            if not isinstance(project.get("project_id"), str) or not str(
                project["project_id"]
            ).strip():
                raise ConfigError("threshold project id must be text")
            if int(project["required_total_contribution_cents"]) <= 0:
                raise ConfigError("threshold project contribution target must be positive")
            if int(project["deadline_round"]) <= 0:
                raise ConfigError("threshold project deadline must be positive")
            for field in (
                "failure_refund_rate_ppm",
                "public_protection_bonus_ppm",
                "supply_cost_reduction_ppm",
            ):
                if int(project[field]) > 1_000_000:
                    raise ConfigError(f"threshold project {field} must be ppm")
            mutual_aid = strategic.get("mutual_aid")
            if mutual_aid is not None:
                if (
                    not isinstance(mutual_aid, Mapping)
                    or mutual_aid.get("enabled") is not True
                ):
                    raise ConfigError(
                        "strategic_market.mutual_aid must be enabled"
                    )
                for field in (
                    "fee_per_order_cents",
                    "max_orders_per_company",
                ):
                    value = mutual_aid.get(field)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise ConfigError(
                            "strategic_market.mutual_aid."
                            f"{field} must be non-negative"
                        )
                if not isinstance(
                    mutual_aid.get("protocol_version"), str
                ) or not str(mutual_aid["protocol_version"]).strip():
                    raise ConfigError(
                        "mutual aid protocol version must be text"
                    )
                if int(mutual_aid["max_orders_per_company"]) <= 0:
                    raise ConfigError(
                        "mutual aid maximum orders must be positive"
                    )
                for field in (
                    "mutual_aid_capacity_offer_orders",
                    "mutual_aid_capacity_request_orders",
                ):
                    if field not in bounds:
                        raise ConfigError(f"action.bounds.{field} is required")
                    if int(bounds[field]["max"]) != int(
                        mutual_aid["max_orders_per_company"]
                    ):
                        raise ConfigError(
                            f"action.bounds.{field}.max must match mutual aid maximum"
                        )
            coordination = strategic.get("price_coordination")
            if coordination is not None:
                if (
                    not isinstance(coordination, Mapping)
                    or coordination.get("enabled") is not True
                ):
                    raise ConfigError(
                        "strategic_market.price_coordination must be enabled"
                    )
                for field in (
                    "adherence_tolerance_cents",
                    "initial_credibility_ppm",
                    "credibility_update_weight_ppm",
                    "base_detection_probability_ppm",
                    "markup_detection_weight_ppm",
                    "regulatory_pressure_detection_weight_ppm",
                    "coordination_pressure_weight_ppm",
                    "base_fine_cents",
                    "fine_revenue_share_ppm",
                ):
                    value = coordination.get(field)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise ConfigError(
                            "strategic_market.price_coordination."
                            f"{field} must be non-negative"
                        )
                for field in (
                    "initial_credibility_ppm",
                    "credibility_update_weight_ppm",
                    "base_detection_probability_ppm",
                    "markup_detection_weight_ppm",
                    "regulatory_pressure_detection_weight_ppm",
                    "coordination_pressure_weight_ppm",
                    "fine_revenue_share_ppm",
                ):
                    if int(coordination[field]) > 1_000_000:
                        raise ConfigError(
                            f"price coordination {field} must be ppm"
                        )
                if not isinstance(
                    coordination.get("protocol_version"), str
                ) or not str(coordination["protocol_version"]).strip():
                    raise ConfigError(
                        "price coordination protocol version must be text"
                    )

        supply_chain = self.data.get("supply_chain")
        if supply_chain is not None:
            if (
                not isinstance(supply_chain, Mapping)
                or supply_chain.get("enabled") is not True
            ):
                raise ConfigError("supply_chain must be an enabled mapping")
            for field in ("protocol_version", "default_supplier_id"):
                if not isinstance(supply_chain.get(field), str) or not str(
                    supply_chain[field]
                ).strip():
                    raise ConfigError(f"supply_chain.{field} must be text")
            for field in (
                "baseline_input_price_cents",
                "downstream_input_cost_share_ppm",
                "disruption_capacity_ppm",
            ):
                value = supply_chain.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ConfigError(
                        f"supply_chain.{field} must be a non-negative integer"
                    )
            if int(supply_chain["baseline_input_price_cents"]) <= 0:
                raise ConfigError("baseline input price must be positive")
            for field in (
                "downstream_input_cost_share_ppm",
                "disruption_capacity_ppm",
            ):
                if int(supply_chain[field]) > 1_000_000:
                    raise ConfigError(f"supply_chain.{field} must be ppm")
            suppliers = supply_chain.get("suppliers")
            if not isinstance(suppliers, Mapping) or len(suppliers) < 2:
                raise ConfigError("supply_chain requires at least two suppliers")
            if supply_chain["default_supplier_id"] not in suppliers:
                raise ConfigError("default supplier is unknown")
            for supplier_id, supplier in suppliers.items():
                if not isinstance(supplier_id, str) or not supplier_id.strip():
                    raise ConfigError("supplier ids must be non-empty text")
                if not isinstance(supplier, Mapping):
                    raise ConfigError(f"supplier {supplier_id} must be a mapping")
                if not isinstance(supplier.get("label"), str) or not str(
                    supplier["label"]
                ).strip():
                    raise ConfigError(f"supplier {supplier_id} label must be text")
                for field in (
                    "unit_price_cents",
                    "unit_cost_cents",
                    "base_capacity_orders",
                    "reliability_ppm",
                ):
                    value = supplier.get(field)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise ConfigError(
                            f"supplier {supplier_id} {field} must be non-negative"
                        )
                if int(supplier["unit_price_cents"]) < int(
                    supplier["unit_cost_cents"]
                ):
                    raise ConfigError(
                        f"supplier {supplier_id} price cannot be below unit cost"
                    )
                if int(supplier["base_capacity_orders"]) <= 0:
                    raise ConfigError(
                        f"supplier {supplier_id} capacity must be positive"
                    )
                if int(supplier["reliability_ppm"]) > 1_000_000:
                    raise ConfigError(
                        f"supplier {supplier_id} reliability must be ppm"
                    )
            accounting = supply_chain.get("transaction_accounting")
            if accounting is not None:
                if (not isinstance(accounting, Mapping) or accounting.get("policy_version") != "cash-material-v1.0.0"
                    or accounting.get("inventory") != "perishable"
                    or type(accounting.get("initial_supplier_cash_cents")) is not int
                    or accounting["initial_supplier_cash_cents"] < 0):
                    raise ConfigError("invalid cash material accounting policy")
            pricing = supply_chain.get("autonomous_pricing")
            if pricing is not None:
                if not isinstance(pricing, Mapping) or not isinstance(pricing.get("enabled"), bool):
                    raise ConfigError("autonomous_pricing.enabled must be boolean")
                if pricing.get("policy_version") != "bounded-utilization-pricing-v1.0.0":
                    raise ConfigError("unknown supplier pricing policy")
                for field, low, high in (
                    ("step_ppm", 1, 250000), ("min_markup_ppm", 0, 1000000),
                    ("max_base_price_ppm", 1000000, 3000000),
                    ("raise_threshold_ppm", 0, 1000000), ("lower_threshold_ppm", 0, 1000000),
                ):
                    value = pricing.get(field)
                    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                        raise ConfigError(f"invalid supplier pricing {field}")
                if pricing["lower_threshold_ppm"] >= pricing["raise_threshold_ppm"]:
                    raise ConfigError("supplier pricing thresholds must be ordered")
                for supplier in suppliers.values():
                    floor = (supplier["unit_cost_cents"] * (1000000 + pricing["min_markup_ppm"]) + 999999) // 1000000
                    if floor > supplier["unit_price_cents"]:
                        raise ConfigError("supplier initial quote is below pricing floor")
            supplier_share_bounds = bounds.get("primary_supplier_share_ppm")
            if not isinstance(supplier_share_bounds, Mapping) or (
                int(supplier_share_bounds.get("min", -1)) != 0
                or int(supplier_share_bounds.get("max", -1)) != 1_000_000
            ):
                raise ConfigError(
                    "action.bounds.primary_supplier_share_ppm must be [0, 1000000]"
                )

        autonomous = self.data.get("autonomous_market")
        if autonomous is not None:
            if (not isinstance(autonomous, Mapping) or autonomous.get("policy_version") != "four-actor-market-v1.0.0"
                or type(autonomous.get("company_procurement")) is not bool
                or not isinstance(supply_chain, Mapping) or not supply_chain.get("transaction_accounting")
                or len(supply_chain["suppliers"]) != 2):
                raise ConfigError("autonomous market requires cash accounting and two suppliers")
            for section, fields in {
                "supplier_investment": {"reserve_cash_cents":(0,10**12),"unit_capacity_cost_cents":(1,10**9),"max_addition_orders":(0,10000),"max_base_capacity_orders":(1,100000),"utilization_threshold_ppm":(1,1000000)},
                "government": {"initial_cash_cents":(0,10**12),"inspection_cost_per_case_cents":(1,10**9),"max_inspection_cases":(0,100),"concentration_threshold_ppm":(0,1000000),"detection_boost_ppm":(0,1000000),"stockout_threshold_ppm":(1,1000000),"support_cash_share_ppm":(0,1000000),"max_support_cents":(0,10**12)},
            }.items():
                policy = autonomous.get(section)
                if not isinstance(policy, Mapping) or type(policy.get("enabled")) is not bool:
                    raise ConfigError(f"invalid autonomous {section}")
                for field,(low,high) in fields.items():
                    if type(policy.get(field)) is not int or not low <= policy[field] <= high:
                        raise ConfigError(f"invalid autonomous {section}.{field}")
            if any("wtp_distribution" not in s for s in self.mapping("consumer_choice","segments").values()):
                raise ConfigError("autonomous consumers require explicit budget cohorts")
            government_strategy = autonomous["government"].get("strategic_policy")
            if government_strategy is not None:
                ranges = {"max_program_cents":(0,10**12), "program_cash_share_ppm":(0,1000000),
                          "rebate_share_ppm":(0,1000000), "fine_multiplier_ppm":(1000000,5000000)}
                if not isinstance(government_strategy, Mapping) or set(government_strategy) != set(ranges):
                    raise ConfigError("invalid government strategic policy fields")
                for key,(low,high) in ranges.items():
                    if type(government_strategy[key]) is not int or not low <= government_strategy[key] <= high:
                        raise ConfigError(f"invalid government strategic policy {key}")
            investment=autonomous["supplier_investment"]
            if investment.get("payback_guard_ppm") is not None:
                if supply_chain.get("track_supplier_demand") is not True:
                    raise ConfigError("investment payback requires supplier own order history")
                for field,low,high in (("payback_guard_ppm",1000000,10000000),("margin_realization_ppm",1,1000000)):
                    if type(investment.get(field)) is not int or not low<=investment[field]<=high:
                        raise ConfigError(f"invalid supplier investment {field}")
        procurement = self.data.get("rule_procurement")
        if procurement and isinstance(supply_chain, Mapping) and supply_chain.get("transaction_accounting"):
            raise ConfigError("legacy procurement projections require coefficient accounting")
        if procurement is not None:
            if (not isinstance(procurement, Mapping) or not isinstance(procurement.get("enabled"), bool)
                or procurement.get("mode") not in {"capacity", "gradual", "financial_guarded"}
                or procurement.get("policy_version") != ("financial-procurement-guard-v1.0.0" if procurement.get("mode")=="financial_guarded" else "public-capacity-procurement-v1.0.0")):
                raise ConfigError("invalid rule_procurement policy")
            if not isinstance(supply_chain, Mapping) or len(supply_chain["suppliers"]) != 2:
                raise ConfigError("rule_procurement requires exactly two suppliers")

        welfare = self.data.get("welfare_accounting")
        if welfare is not None:
            if (
                not isinstance(welfare, Mapping)
                or welfare.get("enabled") is not True
            ):
                raise ConfigError("welfare_accounting must be an enabled mapping")
            for field in ("protocol_version", "valuation_source"):
                if not isinstance(welfare.get(field), str) or not str(
                    welfare[field]
                ).strip():
                    raise ConfigError(f"welfare_accounting.{field} must be text")
            for field in (
                "government_enforcement_cost_per_case_cents",
                "stockout_externality_per_order_cents",
                "business_exit_externality_per_company_cents",
            ):
                value = welfare.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ConfigError(
                        f"welfare_accounting.{field} must be non-negative"
                    )
            if welfare.get("fines_are_social_transfers") is not True:
                raise ConfigError("welfare v2 requires fines to be transfers")
            if not isinstance(welfare.get("voluntary_no_purchase_is_externality"), bool):
                raise ConfigError(
                    "voluntary_no_purchase_is_externality must be boolean"
                )
            segments = self.mapping("consumer_choice", "segments")
            for segment_id, segment in segments.items():
                if not isinstance(segment, Mapping) or not isinstance(
                    segment.get("wtp_distribution"), Mapping
                ):
                    raise ConfigError(
                        "welfare v2 requires explicit WTP distribution for "
                        f"segment {segment_id}"
                    )

    @staticmethod
    def _require_ppm_sum(raw: Any, label: str) -> None:
        if not isinstance(raw, Mapping) or not raw:
            raise ConfigError(f"{label} must be a non-empty mapping")
        values = tuple(raw.values())
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
            raise ConfigError(f"{label} must contain non-negative integer ppm values")
        if sum(values) != 1_000_000:
            raise ConfigError(f"{label} must sum to 1000000")


def _load_config_mapping(path: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(item) for item in (*seen, resolved))
        raise ConfigError(f"Configuration inheritance cycle: {chain}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Unable to load config {resolved}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError("Configuration root must be a mapping")
    parent = raw.get("extends")
    overlay = {str(key): value for key, value in raw.items() if key != "extends"}
    if parent is None:
        return _thaw(overlay)
    if not isinstance(parent, str) or not parent.strip():
        raise ConfigError("extends must be a non-empty relative path")
    parent_path = (resolved.parent / parent).resolve()
    try:
        parent_path.relative_to(resolved.parent.resolve())
    except ValueError as exc:
        raise ConfigError("extends must stay inside the config directory") from exc
    base = _load_config_mapping(parent_path, (*seen, resolved))
    return _deep_merge(base, overlay)


def load_market_config(path: str | Path) -> MarketConfig:
    """Load a UTF-8 YAML config, validate it, and freeze an episode-safe snapshot."""

    config_path = Path(path)
    raw = _load_config_mapping(config_path, ())
    return MarketConfig.from_mapping(raw)
