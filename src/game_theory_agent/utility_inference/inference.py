"""Rule-Bayesian utility inference from an auditable strategy mixture."""

from __future__ import annotations

from typing import Mapping

from game_theory_agent.opponent import (
    OpponentModelState,
    StrategyDistribution,
    compute_opponent_model_hash,
)
from game_theory_agent.utility_inference.contracts import (
    OpponentUtilityBelief,
    UtilityInferenceState,
    UtilityWeightEstimate,
    compute_utility_inference_hash,
)


_UTILITY_ORDER = (
    "profit",
    "market_share",
    "risk_avoidance",
    "cash_preservation",
    "growth",
    "social_welfare",
)

_STRATEGY_UTILITY: dict[str, tuple[int, ...]] = {
    "growth": (150_000, 300_000, 100_000, 50_000, 350_000, 50_000),
    "profit": (450_000, 100_000, 100_000, 200_000, 100_000, 50_000),
    "defensive": (250_000, 50_000, 350_000, 200_000, 100_000, 50_000),
    "cooperative": (150_000, 100_000, 150_000, 100_000, 100_000, 400_000),
}


def strategy_utility_template(strategy: str) -> dict[str, int]:
    values = _STRATEGY_UTILITY[strategy]
    return dict(zip(_UTILITY_ORDER, values))


def _weights(distribution: StrategyDistribution) -> dict[str, int]:
    probabilities = {
        "growth": distribution.growth_ppm,
        "profit": distribution.profit_ppm,
        "defensive": distribution.defensive_ppm,
        "cooperative": distribution.cooperative_ppm,
    }
    raw = {
        utility: sum(
            probabilities[strategy]
            * _STRATEGY_UTILITY[strategy][index]
            for strategy in probabilities
        )
        for index, utility in enumerate(_UTILITY_ORDER)
    }
    allocated = {key: value // 1_000_000 for key, value in raw.items()}
    remaining = 1_000_000 - sum(allocated.values())
    order = sorted(
        _UTILITY_ORDER,
        key=lambda key: (raw[key] % 1_000_000, -_UTILITY_ORDER.index(key)),
        reverse=True,
    )
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


class OpponentUtilityInferer:
    def infer(
        self, opponent_model: OpponentModelState | Mapping[str, object]
    ) -> tuple[UtilityInferenceState, str]:
        model = (
            opponent_model
            if isinstance(opponent_model, OpponentModelState)
            else OpponentModelState.model_validate(opponent_model)
        )
        model_hash = compute_opponent_model_hash(model)
        utilities: dict[str, OpponentUtilityBelief] = {}
        for company_id, profile in model.opponent_models.items():
            weights = _weights(profile.strategy_distribution)
            confidence = profile.confidence_ppm
            fields = {
                key: UtilityWeightEstimate(
                    mean_ppm=weights[key], confidence_ppm=confidence
                )
                for key in _UTILITY_ORDER
            }
            distribution = profile.strategy_distribution
            utilities[company_id] = OpponentUtilityBelief(
                opponent_company_id=company_id,
                **fields,
                explanation_likelihood_ppm=max(
                    distribution.growth_ppm,
                    distribution.profit_ppm,
                    distribution.defensive_ppm,
                    distribution.cooperative_ppm,
                ),
                source_opponent_model_hash=model_hash,
            )
        state = UtilityInferenceState(
            episode_id=model.episode_id,
            observer_company_id=model.observer_company_id,
            prediction_target_round=model.prediction_target_round,
            state_version=model.state_version,
            opponent_model_hash=model_hash,
            opponent_utilities=utilities,
        )
        return state, compute_utility_inference_hash(state)
