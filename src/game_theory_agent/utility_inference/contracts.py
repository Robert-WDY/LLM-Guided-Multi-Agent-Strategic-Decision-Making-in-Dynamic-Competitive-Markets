"""Contracts for utility-weight inference from public opponent models."""

from __future__ import annotations

from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.market.protocols import sha256_hash


UtilityInferenceMode = Literal["off", "strategy_utility_v1"]
UTILITY_INFERENCE_SCHEMA_VERSION = "utility-inference-state-v1.0.0"
UTILITY_INFERENCE_MODEL_VERSION = "strategy-mixture-utility-v1.0.0"
UTILITY_INFERENCE_HASH_PROTOCOL_VERSION = "utility-inference-hash-v1.0.0"


class UtilityWeightEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mean_ppm: int = Field(ge=0, le=1_000_000)
    confidence_ppm: int = Field(ge=0, le=1_000_000)


class OpponentUtilityBelief(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_company_id: str
    profit: UtilityWeightEstimate
    market_share: UtilityWeightEstimate
    risk_avoidance: UtilityWeightEstimate
    cash_preservation: UtilityWeightEstimate
    growth: UtilityWeightEstimate
    social_welfare: UtilityWeightEstimate
    explanation_likelihood_ppm: int = Field(ge=0, le=1_000_000)
    source_opponent_model_hash: str

    @model_validator(mode="after")
    def validate_weight_total(self) -> "OpponentUtilityBelief":
        if sum(
            (
                self.profit.mean_ppm,
                self.market_share.mean_ppm,
                self.risk_avoidance.mean_ppm,
                self.cash_preservation.mean_ppm,
                self.growth.mean_ppm,
                self.social_welfare.mean_ppm,
            )
        ) != 1_000_000:
            raise ValueError("inferred utility means must sum to 1,000,000")
        return self


class UtilityInferenceState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    utility_inference_schema_version: Literal[
        "utility-inference-state-v1.0.0"
    ] = UTILITY_INFERENCE_SCHEMA_VERSION
    utility_inference_mode: Literal[
        "strategy_utility_v1"
    ] = "strategy_utility_v1"
    inference_model_version: Literal[
        "strategy-mixture-utility-v1.0.0"
    ] = UTILITY_INFERENCE_MODEL_VERSION
    episode_id: str
    observer_company_id: str
    prediction_target_round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    opponent_model_hash: str
    opponent_utilities: dict[str, OpponentUtilityBelief]
    uses_hidden_persona: Literal[False] = False
    uses_hidden_profit: Literal[False] = False
    uses_hidden_cash: Literal[False] = False
    inference_is_uncertain: Literal[True] = True

    @model_validator(mode="after")
    def validate_scope(self) -> "UtilityInferenceState":
        if self.observer_company_id in self.opponent_utilities:
            raise ValueError("utility inference cannot include the observer")
        for key, utility in self.opponent_utilities.items():
            if key != utility.opponent_company_id:
                raise ValueError("utility inference key mismatch")
            if utility.source_opponent_model_hash != self.opponent_model_hash:
                raise ValueError("utility inference source hash mismatch")
        return self


def compute_utility_inference_hash(
    state: UtilityInferenceState | Mapping[str, object],
) -> str:
    payload = (
        state.model_dump(mode="json")
        if isinstance(state, UtilityInferenceState)
        else dict(state)
    )
    return sha256_hash(
        {
            "hash_protocol_version": UTILITY_INFERENCE_HASH_PROTOCOL_VERSION,
            "schema_version": payload.get("utility_inference_schema_version"),
            "utility_inference_state": payload,
        }
    )
