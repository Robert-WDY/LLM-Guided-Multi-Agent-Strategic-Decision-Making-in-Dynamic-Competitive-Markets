"""Auditable contracts for the minimal Bayesian price-game advisor."""

from __future__ import annotations

from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.market.protocols import sha256_hash


AdvisorMode = Literal["off", "bayesian_price_v1", "bayesian_strategy_v2"]
ADVISOR_SCHEMA_VERSION = "bayesian-price-advice-v1.0.0"
ADVISOR_MODEL_VERSION = "independent-direction-payoff-proxy-v1.0.0"
ADVISOR_HASH_PROTOCOL_VERSION = "game-theory-advice-hash-v1.0.0"


class BayesianCandidateEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    price_cents: int = Field(ge=0)
    expected_margin_index_cents: int
    expected_competitive_pressure_ppm: int = Field(
        ge=-1_000_000, le=1_000_000
    )
    expected_payoff_proxy: int
    downside_payoff_proxy: int


class GameTheoryAdvice(BaseModel):
    """A deterministic advisory calculation; never an executable action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    advice_schema_version: Literal["bayesian-price-advice-v1.0.0"] = (
        ADVISOR_SCHEMA_VERSION
    )
    advisor_mode: Literal["bayesian_price_v1"] = "bayesian_price_v1"
    advisor_model_version: Literal[
        "independent-direction-payoff-proxy-v1.0.0"
    ] = ADVISOR_MODEL_VERSION
    episode_id: str
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    company_id: str
    belief_hash: str
    current_price_cents: int = Field(ge=0)
    unit_cost_cents: int = Field(ge=0)
    candidates: list[BayesianCandidateEvaluation] = Field(min_length=2)
    recommended_price_cents: int = Field(ge=0)
    recommendation_is_non_binding: Literal[True] = True
    uses_hidden_opponent_state: Literal[False] = False
    limitations: list[str] = Field(min_length=1)
    advice_hash: str

    @model_validator(mode="after")
    def validate_recommendation(self) -> "GameTheoryAdvice":
        ranked = max(
            self.candidates,
            key=lambda item: (
                item.expected_payoff_proxy,
                item.downside_payoff_proxy,
                -abs(item.price_cents - self.current_price_cents),
                -item.price_cents,
            ),
        )
        if self.recommended_price_cents != ranked.price_cents:
            raise ValueError("advisor recommendation is not the best candidate")
        expected = compute_advice_hash(self)
        if self.advice_hash != expected:
            raise ValueError("game theory advice hash mismatch")
        return self


def compute_advice_hash(advice: GameTheoryAdvice | Mapping[str, object]) -> str:
    payload = (
        advice.model_dump(mode="json")
        if isinstance(advice, GameTheoryAdvice)
        else dict(advice)
    )
    payload.pop("advice_hash", None)
    return sha256_hash(
        {
            "hash_protocol_version": ADVISOR_HASH_PROTOCOL_VERSION,
            "advice_schema_version": payload.get("advice_schema_version"),
            "advice": payload,
        }
    )


class PredictedOpponentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_company_id: str
    price_cut_ppm: int = Field(ge=0, le=1_000_000)
    maintain_ppm: int = Field(ge=0, le=1_000_000)
    price_raise_ppm: int = Field(ge=0, le=1_000_000)

    @model_validator(mode="after")
    def validate_total(self) -> "PredictedOpponentResponse":
        if self.price_cut_ppm + self.maintain_ppm + self.price_raise_ppm != 1_000_000:
            raise ValueError("predicted response probabilities must sum to 1,000,000")
        return self


class StrategicCandidateEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_label: Literal[
        "aggressive_price_cut", "price_cut", "maintain", "price_raise"
    ]
    price_cents: int = Field(ge=0)
    predicted_opponent_responses: list[PredictedOpponentResponse]
    expected_profit_proxy: int
    expected_market_share_ppm: int = Field(ge=0, le=1_000_000)
    strategic_risk_ppm: int = Field(ge=0, le=1_000_000)
    expected_utility_proxy: int
    worst_case_utility_proxy: int


class StrategicGameTheoryAdvice(BaseModel):
    """Approximate Bayesian best response over strategy and utility beliefs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    advice_schema_version: Literal[
        "bayesian-strategy-advice-v2.0.0"
    ] = "bayesian-strategy-advice-v2.0.0"
    advisor_mode: Literal["bayesian_strategy_v2"] = "bayesian_strategy_v2"
    advisor_model_version: Literal[
        "expected-strategic-response-v2.0.0"
    ] = "expected-strategic-response-v2.0.0"
    episode_id: str
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    company_id: str
    belief_hash: str
    opponent_model_hash: str
    utility_inference_hash: str
    current_price_cents: int = Field(ge=0)
    unit_cost_cents: int = Field(ge=0)
    candidate_actions: list[StrategicCandidateEvaluation] = Field(min_length=3)
    recommended_action: str
    recommended_price_cents: int = Field(ge=0)
    recommendation_reason: str
    recommendation_is_non_binding: Literal[True] = True
    approximate_best_response: Literal[True] = True
    claims_nash_equilibrium: Literal[False] = False
    uses_hidden_opponent_state: Literal[False] = False
    limitations: list[str] = Field(min_length=1)
    advice_hash: str

    @model_validator(mode="after")
    def validate_recommendation(self) -> "StrategicGameTheoryAdvice":
        best = max(
            self.candidate_actions,
            key=lambda item: (
                item.expected_utility_proxy,
                item.worst_case_utility_proxy,
                -item.strategic_risk_ppm,
                -abs(item.price_cents - self.current_price_cents),
            ),
        )
        if (
            self.recommended_action != best.action_label
            or self.recommended_price_cents != best.price_cents
        ):
            raise ValueError("strategic recommendation is not the best candidate")
        if self.advice_hash != compute_strategic_advice_hash(self):
            raise ValueError("strategic advice hash mismatch")
        return self


def compute_strategic_advice_hash(
    advice: StrategicGameTheoryAdvice | Mapping[str, object],
) -> str:
    payload = (
        advice.model_dump(mode="json")
        if isinstance(advice, StrategicGameTheoryAdvice)
        else dict(advice)
    )
    payload.pop("advice_hash", None)
    return sha256_hash(
        {
            "hash_protocol_version": "game-theory-advice-hash-v2.0.0",
            "advice_schema_version": payload.get("advice_schema_version"),
            "advice": payload,
        }
    )
