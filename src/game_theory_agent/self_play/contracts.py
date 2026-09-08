"""Immutable contracts for deterministic strategic-market self-play."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from game_theory_agent.market.protocols import sha256_hash


class StrategyVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_schema_version: Literal["market-strategy-v1.0.0"] = (
        "market-strategy-v1.0.0"
    )
    strategy_id: str = Field(pattern=r"^[a-z0-9_]+$")
    version: str
    label: str
    family: Literal[
        "baseline",
        "competition",
        "cooperation",
        "mutual_aid",
        "coordination",
        "adversarial",
    ]
    operationally_allowed: bool
    research_only_reason: str | None = None
    parameters: dict[str, Any]
    strategy_hash: str

    @classmethod
    def create(
        cls,
        *,
        strategy_id: str,
        version: str,
        label: str,
        family: str,
        operationally_allowed: bool,
        parameters: dict[str, Any],
        research_only_reason: str | None = None,
    ) -> "StrategyVersion":
        payload = {
            "strategy_schema_version": "market-strategy-v1.0.0",
            "strategy_id": strategy_id,
            "version": version,
            "label": label,
            "family": family,
            "operationally_allowed": operationally_allowed,
            "research_only_reason": research_only_reason,
            "parameters": parameters,
            "strategy_hash": "pending",
        }
        payload["strategy_hash"] = sha256_hash(
            {key: value for key, value in payload.items() if key != "strategy_hash"}
        )
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_hash_and_boundary(self) -> "StrategyVersion":
        expected = sha256_hash(
            self.model_dump(mode="json", exclude={"strategy_hash"})
        )
        if self.strategy_hash != expected:
            raise ValueError("strategy hash mismatch")
        if self.operationally_allowed and self.research_only_reason is not None:
            raise ValueError("operational strategy cannot have research-only reason")
        if not self.operationally_allowed and not self.research_only_reason:
            raise ValueError("research-only strategy must explain its boundary")
        return self


class FrozenPromotionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    promotion_schema_version: Literal[
        "self-play-promotion-candidate-v1.0.0"
    ] = "self-play-promotion-candidate-v1.0.0"
    config_sha256: str
    development_result_hash: str
    candidate_strategy_id: str
    candidate_strategy_hash: str
    selection_rule: str
    development_seeds: tuple[int, ...]
    excluded_holdout_seeds: tuple[int, ...]
    uses_holdout_for_selection: Literal[False] = False
    promotion_hash: str

    @model_validator(mode="after")
    def validate_hash(self) -> "FrozenPromotionCandidate":
        expected = sha256_hash(
            self.model_dump(mode="json", exclude={"promotion_hash"})
        )
        if self.promotion_hash != expected:
            raise ValueError("promotion candidate hash mismatch")
        if set(self.development_seeds).intersection(self.excluded_holdout_seeds):
            raise ValueError("development and holdout seeds overlap")
        return self
