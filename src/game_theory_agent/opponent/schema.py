"""Strict contracts for public-evidence opponent strategy models."""

from __future__ import annotations

from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.market.protocols import sha256_hash


OpponentModelMode = Literal["off", "public_strategy_v1"]
StrategyType = Literal["growth", "profit", "defensive", "cooperative"]
OPPONENT_MODEL_SCHEMA_VERSION = "opponent-model-state-v1.0.0"
OPPONENT_MODEL_UPDATER_VERSION = "public-strategy-rule-bayes-v1.0.0"
OPPONENT_MODEL_HASH_PROTOCOL_VERSION = "opponent-model-hash-v1.0.0"


class PublicStrategyEvidence(BaseModel):
    """One settled transition containing public facts only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_schema_version: Literal["public-strategy-evidence-v1.0.0"] = (
        "public-strategy-evidence-v1.0.0"
    )
    evidence_id: str
    episode_id: str
    settled_round: int = Field(ge=1)
    target_company_id: str
    previous_price_cents: int = Field(ge=0)
    settled_price_cents: int = Field(ge=0)
    price_direction: Literal["price_cut", "maintain", "price_raise"]
    market_share_delta_ppm: int = Field(ge=-1_000_000, le=1_000_000)
    public_sales_orders: int = Field(ge=0)
    reputation_delta_ppm: int = Field(ge=-1_000_000, le=1_000_000)
    public_shared_resilience_contribution_cents: int = Field(ge=0)
    source_fields: tuple[
        Literal[
            "price",
            "market_share",
            "sales_orders",
            "reputation",
            "public_shared_resilience_contribution",
        ],
        ...,
    ] = (
        "price",
        "market_share",
        "sales_orders",
        "reputation",
        "public_shared_resilience_contribution",
    )
    uses_hidden_state: Literal[False] = False


class OpponentBehaviorProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    price_aggressiveness_ppm: int = Field(ge=0, le=1_000_000)
    public_expansion_aggressiveness_ppm: int = Field(ge=0, le=1_000_000)
    risk_tolerance_ppm: int = Field(ge=0, le=1_000_000)
    cooperation_tendency_ppm: int = Field(ge=0, le=1_000_000)


class StrategyDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    growth_ppm: int = Field(ge=0, le=1_000_000)
    profit_ppm: int = Field(ge=0, le=1_000_000)
    defensive_ppm: int = Field(ge=0, le=1_000_000)
    cooperative_ppm: int = Field(ge=0, le=1_000_000)

    @model_validator(mode="after")
    def validate_total(self) -> "StrategyDistribution":
        if sum(
            (
                self.growth_ppm,
                self.profit_ppm,
                self.defensive_ppm,
                self.cooperative_ppm,
            )
        ) != 1_000_000:
            raise ValueError("strategy probabilities must sum to 1,000,000")
        return self

    @property
    def top_strategy(self) -> StrategyType:
        ranked = (
            (self.growth_ppm, 0, "growth"),
            (self.profit_ppm, 1, "profit"),
            (self.defensive_ppm, 2, "defensive"),
            (self.cooperative_ppm, 3, "cooperative"),
        )
        return max(ranked, key=lambda item: (item[0], -item[1]))[2]  # type: ignore[return-value]


class OpponentStrategyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_company_id: str
    evidence_count: int = Field(ge=0)
    latest_evidence_round: int | None = Field(default=None, ge=1)
    behavior_profile: OpponentBehaviorProfile
    strategy_distribution: StrategyDistribution
    confidence_ppm: int = Field(ge=0, le=1_000_000)
    public_evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence(self) -> "OpponentStrategyModel":
        if len(self.public_evidence_ids) != self.evidence_count:
            raise ValueError("opponent model evidence count mismatch")
        if len(set(self.public_evidence_ids)) != len(self.public_evidence_ids):
            raise ValueError("opponent model contains duplicate evidence")
        return self


class OpponentModelState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_model_schema_version: Literal[
        "opponent-model-state-v1.0.0"
    ] = OPPONENT_MODEL_SCHEMA_VERSION
    opponent_model_mode: Literal["public_strategy_v1"] = "public_strategy_v1"
    updater_version: Literal[
        "public-strategy-rule-bayes-v1.0.0"
    ] = OPPONENT_MODEL_UPDATER_VERSION
    episode_id: str
    observer_company_id: str
    prediction_target_round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    public_evidence_through_round: int = Field(ge=0)
    opponent_models: dict[str, OpponentStrategyModel]
    uses_hidden_cash: Literal[False] = False
    uses_hidden_cost: Literal[False] = False
    uses_hidden_persona: Literal[False] = False
    uses_hidden_prompt: Literal[False] = False

    @model_validator(mode="after")
    def validate_scope(self) -> "OpponentModelState":
        if self.observer_company_id in self.opponent_models:
            raise ValueError("opponent model cannot include the observer")
        for key, model in self.opponent_models.items():
            if key != model.opponent_company_id:
                raise ValueError("opponent model key mismatch")
        return self


def compute_opponent_model_hash(
    state: OpponentModelState | Mapping[str, object],
) -> str:
    payload = (
        state.model_dump(mode="json")
        if isinstance(state, OpponentModelState)
        else dict(state)
    )
    return sha256_hash(
        {
            "hash_protocol_version": OPPONENT_MODEL_HASH_PROTOCOL_VERSION,
            "schema_version": payload.get("opponent_model_schema_version"),
            "opponent_model_state": payload,
        }
    )
