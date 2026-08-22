"""Opponent utility inference public API."""

from game_theory_agent.utility_inference.contracts import (
    UTILITY_INFERENCE_HASH_PROTOCOL_VERSION,
    UTILITY_INFERENCE_MODEL_VERSION,
    UTILITY_INFERENCE_SCHEMA_VERSION,
    OpponentUtilityBelief,
    UtilityInferenceMode,
    UtilityInferenceState,
    UtilityWeightEstimate,
    compute_utility_inference_hash,
)
from game_theory_agent.utility_inference.inference import (
    OpponentUtilityInferer,
    strategy_utility_template,
)
from game_theory_agent.utility_inference.replay import (
    UtilityInferenceReplayMismatchError,
    verify_utility_inference_replay,
)

__all__ = [
    "UTILITY_INFERENCE_HASH_PROTOCOL_VERSION",
    "UTILITY_INFERENCE_MODEL_VERSION",
    "UTILITY_INFERENCE_SCHEMA_VERSION",
    "OpponentUtilityBelief",
    "OpponentUtilityInferer",
    "UtilityInferenceMode",
    "UtilityInferenceReplayMismatchError",
    "UtilityInferenceState",
    "UtilityWeightEstimate",
    "compute_utility_inference_hash",
    "strategy_utility_template",
    "verify_utility_inference_replay",
]
