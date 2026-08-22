"""Deterministic, company-scoped belief contracts for the Belief MVP."""

from __future__ import annotations

from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.market.protocols import sha256_hash


BeliefMode = Literal["off", "public_action_v1", "public_action_signal_v2"]
PriceDirection = Literal["price_cut", "maintain", "price_raise"]

BELIEF_SCHEMA_VERSION = "belief-state-v1.0.0"
BELIEF_UPDATER_VERSION = "dirichlet-public-price-v1.0.0"
BELIEF_HASH_PROTOCOL_VERSION = "belief-view-hash-v1.0.0"
SIGNAL_BELIEF_SCHEMA_VERSION = "belief-state-v2.0.0"
SIGNAL_BELIEF_UPDATER_VERSION = "dirichlet-public-price-signal-v2.0.0"


class CommunicationPriceSignal(BaseModel):
    """A visible, non-binding price claim; never a market fact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signal_schema_version: Literal["communication-price-signal-v1.0.0"] = (
        "communication-price-signal-v1.0.0"
    )
    message_id: str
    sender_company_id: str
    observer_company_id: str
    claimed_price_cents: int = Field(ge=0)
    direction_relative_to_public_price: PriceDirection
    historical_reliability_ppm: int = Field(ge=0, le=1_000_000)
    verified_fact: Literal[False] = False
    non_binding: Literal[True] = True


class PriceDirectionDistribution(BaseModel):
    """Integer predictive distribution; values always sum to one million."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    price_cut_ppm: int = Field(ge=0, le=1_000_000)
    maintain_ppm: int = Field(ge=0, le=1_000_000)
    price_raise_ppm: int = Field(ge=0, le=1_000_000)

    @model_validator(mode="after")
    def validate_total(self) -> "PriceDirectionDistribution":
        if (
            self.price_cut_ppm
            + self.maintain_ppm
            + self.price_raise_ppm
            != 1_000_000
        ):
            raise ValueError("price-direction probabilities must sum to 1,000,000")
        return self

    def probability_ppm(self, direction: PriceDirection) -> int:
        return {
            "price_cut": self.price_cut_ppm,
            "maintain": self.maintain_ppm,
            "price_raise": self.price_raise_ppm,
        }[direction]

    @property
    def top_direction(self) -> PriceDirection:
        ranked = (
            (self.price_cut_ppm, 0, "price_cut"),
            (self.maintain_ppm, 1, "maintain"),
            (self.price_raise_ppm, 2, "price_raise"),
        )
        return max(ranked, key=lambda item: (item[0], -item[1]))[2]  # type: ignore[return-value]


class PublicPriceEvidence(BaseModel):
    """One settled public price observation. No hidden state is permitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_schema_version: Literal["public-price-evidence-v1.0.0"] = (
        "public-price-evidence-v1.0.0"
    )
    evidence_id: str
    episode_id: str
    settled_round: int = Field(ge=1)
    target_company_id: str
    previous_public_price_cents: int = Field(ge=0)
    settled_public_price_cents: int = Field(ge=0)
    observed_direction: PriceDirection
    source: Literal["settled_public_action"] = "settled_public_action"


class OpponentPriceBelief(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_company_id: str
    prediction_target_round: int = Field(ge=1)
    evidence_count: int = Field(ge=0)
    latest_evidence_round: int | None = Field(default=None, ge=1)
    latest_observed_direction: PriceDirection | None = None
    prior_pseudocount_per_direction: int = Field(default=1, ge=1)
    observed_counts: dict[PriceDirection, int]
    next_price_direction: PriceDirectionDistribution
    signal_evidence_count: int = Field(default=0, ge=0)
    signal_direction_counts: dict[PriceDirection, int] = Field(
        default_factory=lambda: {
            "price_cut": 0,
            "maintain": 0,
            "price_raise": 0,
        }
    )
    signal_reliability_ppm: int | None = Field(
        default=None, ge=0, le=1_000_000
    )

    @model_validator(mode="after")
    def validate_counts(self) -> "OpponentPriceBelief":
        required = {"price_cut", "maintain", "price_raise"}
        if set(self.observed_counts) != required:
            raise ValueError("observed_counts must contain all price directions")
        if any(value < 0 for value in self.observed_counts.values()):
            raise ValueError("observed_counts cannot be negative")
        if sum(self.observed_counts.values()) != self.evidence_count:
            raise ValueError("evidence_count does not match observed_counts")
        if set(self.signal_direction_counts) != required:
            raise ValueError(
                "signal_direction_counts must contain all price directions"
            )
        if sum(self.signal_direction_counts.values()) != self.signal_evidence_count:
            raise ValueError(
                "signal_evidence_count does not match signal_direction_counts"
            )
        return self


class BeliefState(BaseModel):
    """Beliefs available before one company chooses its current-round action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    belief_schema_version: Literal[
        "belief-state-v1.0.0", "belief-state-v2.0.0"
    ] = BELIEF_SCHEMA_VERSION
    belief_mode: Literal[
        "public_action_v1", "public_action_signal_v2"
    ] = "public_action_v1"
    updater_version: Literal[
        "dirichlet-public-price-v1.0.0",
        "dirichlet-public-price-signal-v2.0.0",
    ] = (
        BELIEF_UPDATER_VERSION
    )
    episode_id: str
    observer_company_id: str
    prediction_target_round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    public_evidence_through_round: int = Field(ge=0)
    evidence_scope: Literal[
        "settled_public_prices_only",
        "settled_public_prices_and_visible_non_binding_claims",
    ] = (
        "settled_public_prices_only"
    )
    opponent_beliefs: dict[str, OpponentPriceBelief]
    visible_communication_signals: list[CommunicationPriceSignal] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_scope(self) -> "BeliefState":
        if self.belief_mode == "public_action_v1":
            if self.belief_schema_version != BELIEF_SCHEMA_VERSION:
                raise ValueError("public_action_v1 requires belief-state-v1.0.0")
            if self.updater_version != BELIEF_UPDATER_VERSION:
                raise ValueError("public_action_v1 updater mismatch")
            if self.visible_communication_signals:
                raise ValueError("public_action_v1 cannot contain communication signals")
        else:
            if self.belief_schema_version != SIGNAL_BELIEF_SCHEMA_VERSION:
                raise ValueError(
                    "public_action_signal_v2 requires belief-state-v2.0.0"
                )
            if self.updater_version != SIGNAL_BELIEF_UPDATER_VERSION:
                raise ValueError("public_action_signal_v2 updater mismatch")
        if self.observer_company_id in self.opponent_beliefs:
            raise ValueError("belief state cannot model the observer as an opponent")
        for company_id, belief in self.opponent_beliefs.items():
            if company_id != belief.opponent_company_id:
                raise ValueError("opponent belief key does not match company id")
            if belief.prediction_target_round != self.prediction_target_round:
                raise ValueError("opponent belief target round mismatch")
        for signal in self.visible_communication_signals:
            if signal.observer_company_id != self.observer_company_id:
                raise ValueError("communication signal observer mismatch")
            if signal.sender_company_id not in self.opponent_beliefs:
                raise ValueError("communication signal sender is not an opponent")
        return self


def compute_belief_hash(belief_state: BeliefState | Mapping[str, object]) -> str:
    payload = (
        belief_state.model_dump(mode="json")
        if isinstance(belief_state, BeliefState)
        else dict(belief_state)
    )
    return sha256_hash(
        {
            "hash_protocol_version": BELIEF_HASH_PROTOCOL_VERSION,
            "belief_schema_version": payload.get("belief_schema_version"),
            "belief_state": payload,
        }
    )
