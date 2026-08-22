"""Contracts for non-executing repeated-game strategy recommendations."""

from __future__ import annotations

from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from game_theory_agent.market.protocols import sha256_hash


RepeatedGameMode = Literal["off", "reciprocity_v1"]
RepeatedGameStance = Literal[
    "cooperate", "generous_cooperate", "cautious", "defect", "permanent_refusal"
]
REPEATED_GAME_SCHEMA_VERSION = "repeated-game-strategy-v1.0.0"
REPEATED_GAME_HASH_PROTOCOL_VERSION = "repeated-game-strategy-hash-v1.0.0"


class OpponentRepeatedGameStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_company_id: str
    trust_ppm: int = Field(ge=0, le=1_000_000)
    accepted_count: int = Field(ge=0)
    fulfilled_count: int = Field(ge=0)
    partial_betrayal_count: int = Field(ge=0)
    betrayal_count: int = Field(ge=0)
    tit_for_tat_stance: RepeatedGameStance
    grim_trigger_stance: RepeatedGameStance
    generous_tit_for_tat_stance: RepeatedGameStance
    recommended_stance: RepeatedGameStance
    contribution_multiplier_ppm: int = Field(ge=0, le=1_000_000)
    recommendation_is_non_binding: Literal[True] = True


class RepeatedGameStrategyState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repeated_game_schema_version: Literal[
        "repeated-game-strategy-v1.0.0"
    ] = REPEATED_GAME_SCHEMA_VERSION
    repeated_game_mode: Literal["reciprocity_v1"] = "reciprocity_v1"
    episode_id: str
    observer_company_id: str
    round: int = Field(ge=1)
    opponent_strategies: dict[str, OpponentRepeatedGameStrategy]
    uses_authoritative_cooperation_memory: Literal[True] = True
    changes_market_directly: Literal[False] = False


def compute_repeated_game_hash(
    state: RepeatedGameStrategyState | Mapping[str, object],
) -> str:
    payload = (
        state.model_dump(mode="json")
        if isinstance(state, RepeatedGameStrategyState)
        else dict(state)
    )
    return sha256_hash(
        {
            "hash_protocol_version": REPEATED_GAME_HASH_PROTOCOL_VERSION,
            "schema_version": payload.get("repeated_game_schema_version"),
            "repeated_game_strategy": payload,
        }
    )
