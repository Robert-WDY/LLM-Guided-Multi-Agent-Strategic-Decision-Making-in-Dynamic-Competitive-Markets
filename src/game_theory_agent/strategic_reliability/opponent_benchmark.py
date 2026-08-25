"""Known, mixed and unseen-opponent benchmarks for Stage 6.

The benchmark is synthetic and costs no model tokens.  It measures whether a
public-evidence strategy classifier is calibrated against latent utility
hypotheses, including a disjoint holdout pool sampled from a deterministic
Dirichlet(1) distribution.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.opponent import (
    OPPONENT_MODEL_CANDIDATE_UPDATER_VERSION,
    OpponentBehaviorProfile,
    OpponentStrategyModel,
    OpponentModelState,
    PublicStrategyEvidence,
    StrategyDistribution,
    build_strategy_model,
)


STRATEGIES = ("growth", "profit", "defensive", "cooperative")


def _draw(seed: int, case_id: str, round_number: int, component: str) -> float:
    digest = sha256_hash(
        {
            "protocol": "stage6-opponent-benchmark-rng-v1",
            "seed": seed,
            "case_id": case_id,
            "round": round_number,
            "component": component,
        }
    )
    value = int(digest.rsplit(":", 1)[-1][-16:], 16)
    return (value + 1) / ((1 << 64) + 1)


def _allocate_ppm(values: Sequence[float]) -> tuple[int, ...]:
    if not values or sum(values) <= 0:
        raise ValueError("probability weights must be positive")
    total = sum(values)
    exact = [value * 1_000_000 / total for value in values]
    allocated = [math.floor(value) for value in exact]
    remaining = 1_000_000 - sum(allocated)
    order = sorted(
        range(len(values)),
        key=lambda index: (exact[index] - allocated[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        allocated[index] += 1
    return tuple(allocated)


def deterministic_dirichlet_weights(
    seed: int, case_id: str
) -> StrategyDistribution:
    # Independent exponential samples normalize to Dirichlet(alpha=1).
    samples = [
        -math.log(max(1e-15, _draw(seed, case_id, 0, strategy)))
        for strategy in STRATEGIES
    ]
    weights = _allocate_ppm(samples)
    return StrategyDistribution(
        growth_ppm=weights[0],
        profit_ppm=weights[1],
        defensive_ppm=weights[2],
        cooperative_ppm=weights[3],
    )


def _distribution(values: Sequence[int]) -> StrategyDistribution:
    allocated = _allocate_ppm([float(value) for value in values])
    return StrategyDistribution(
        growth_ppm=allocated[0],
        profit_ppm=allocated[1],
        defensive_ppm=allocated[2],
        cooperative_ppm=allocated[3],
    )


def _select_strategy(
    distribution: StrategyDistribution,
    *,
    seed: int,
    case_id: str,
    round_number: int,
) -> str:
    draw = round(_draw(seed, case_id, round_number, "strategy") * 999_999)
    cumulative = 0
    for strategy in STRATEGIES:
        cumulative += int(getattr(distribution, f"{strategy}_ppm"))
        if draw < cumulative:
            return strategy
    return STRATEGIES[-1]


def _evidence_for_case(
    distribution: StrategyDistribution,
    *,
    seed: int,
    case_id: str,
    rounds: int,
) -> list[PublicStrategyEvidence]:
    patterns = {
        "growth": ("price_cut", 9_400, 22_000, -4_000, 0, 4_100),
        "profit": ("price_raise", 10_600, -3_000, 2_000, 0, 3_100),
        "defensive": ("maintain", 10_000, -5_000, 10_000, 0, 3_300),
        "cooperative": ("maintain", 10_000, 1_000, 8_000, 500_000, 3_400),
    }
    evidence: list[PublicStrategyEvidence] = []
    for round_number in range(1, rounds + 1):
        strategy = _select_strategy(
            distribution,
            seed=seed,
            case_id=case_id,
            round_number=round_number,
        )
        direction, price, share, reputation, contribution, sales = patterns[strategy]
        evidence.append(
            PublicStrategyEvidence(
                evidence_id=f"stage6:{case_id}:{round_number}",
                episode_id=f"stage6-{case_id}",
                settled_round=round_number,
                target_company_id="company_B",
                previous_price_cents=10_000,
                settled_price_cents=price,
                price_direction=direction,
                market_share_delta_ppm=share,
                public_sales_orders=sales,
                reputation_delta_ppm=reputation,
                public_shared_resilience_contribution_cents=contribution,
            )
        )
    return evidence


def build_calibrated_strategy_model_v2(
    company_id: str,
    evidence: list[PublicStrategyEvidence],
) -> OpponentStrategyModel:
    """Infer a utility hypothesis, not a hidden Persona identity.

    V2 uses public behavior likelihoods and recency weighting.  It remains a
    deterministic hypothesis over four strategic utility families.
    """

    if not evidence:
        return OpponentStrategyModel(
            opponent_company_id=company_id,
            evidence_count=0,
            latest_evidence_round=None,
            behavior_profile=OpponentBehaviorProfile(
                price_aggressiveness_ppm=500_000,
                public_expansion_aggressiveness_ppm=500_000,
                risk_tolerance_ppm=500_000,
                cooperation_tendency_ppm=500_000,
            ),
            strategy_distribution=_distribution((1, 1, 1, 1)),
            confidence_ppm=0,
            public_evidence_ids=[],
        )

    scores = {strategy: 2_000 for strategy in STRATEGIES}
    total_weight = 0
    cut_weight = 0
    expansion_weight = 0
    risky_weight = 0
    cooperation_weight = 0
    count = len(evidence)
    for index, item in enumerate(evidence):
        # New evidence carries up to twice the weight of the oldest item.
        weight = 1_000 + (index * 1_000 // max(1, count - 1))
        total_weight += weight
        if item.price_direction == "price_cut":
            cut_weight += weight
            scores["growth"] += 7 * weight
            scores["profit"] += weight
        elif item.price_direction == "price_raise":
            scores["profit"] += 7 * weight
            scores["defensive"] += weight
        else:
            scores["defensive"] += 5 * weight
            scores["cooperative"] += 3 * weight
            scores["profit"] += 2 * weight
        if item.market_share_delta_ppm > 0:
            expansion_weight += weight
            scores["growth"] += 4 * weight
        elif item.market_share_delta_ppm < 0:
            scores["defensive"] += 2 * weight
        if item.reputation_delta_ppm > 0:
            scores["defensive"] += 3 * weight
            scores["cooperative"] += 3 * weight
        elif item.reputation_delta_ppm < 0:
            risky_weight += weight
            scores["growth"] += 2 * weight
        if item.public_shared_resilience_contribution_cents > 0:
            cooperation_weight += weight
            scores["cooperative"] += 12 * weight
        else:
            scores["profit"] += weight

    values = [scores[strategy] for strategy in STRATEGIES]
    distribution = _distribution(values)

    def ratio(value: int) -> int:
        return (value + 1_000) * 1_000_000 // (total_weight + 2_000)

    return OpponentStrategyModel(
        opponent_company_id=company_id,
        evidence_count=count,
        latest_evidence_round=evidence[-1].settled_round,
        behavior_profile=OpponentBehaviorProfile(
            price_aggressiveness_ppm=ratio(cut_weight),
            public_expansion_aggressiveness_ppm=ratio(expansion_weight),
            risk_tolerance_ppm=ratio(risky_weight),
            cooperation_tendency_ppm=ratio(cooperation_weight),
        ),
        strategy_distribution=distribution,
        confidence_ppm=count * 1_000_000 // (count + 6),
        public_evidence_ids=[item.evidence_id for item in evidence],
    )


def build_calibrated_opponent_state_v2(
    base_state: OpponentModelState,
    evidence: Sequence[PublicStrategyEvidence],
) -> OpponentModelState:
    """Rebuild one company-scoped candidate state from public evidence only."""

    by_company = {
        company_id: [
            item for item in evidence if item.target_company_id == company_id
        ]
        for company_id in base_state.opponent_models
    }
    return OpponentModelState(
        updater_version=OPPONENT_MODEL_CANDIDATE_UPDATER_VERSION,
        episode_id=base_state.episode_id,
        observer_company_id=base_state.observer_company_id,
        prediction_target_round=base_state.prediction_target_round,
        state_version=base_state.state_version,
        public_evidence_through_round=base_state.public_evidence_through_round,
        opponent_models={
            company_id: build_calibrated_strategy_model_v2(
                company_id, by_company[company_id]
            )
            for company_id in base_state.opponent_models
        },
    )


def _score_rows(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("benchmark score requires rows")
    top_correct = 0
    brier = 0.0
    log_loss = 0.0
    for row in rows:
        truth = row["truth_distribution_ppm"]
        prediction = row[f"{key}_distribution_ppm"]
        if max(prediction, key=prediction.get) == max(truth, key=truth.get):
            top_correct += 1
        for strategy in STRATEGIES:
            actual = truth[strategy] / 1_000_000
            predicted = max(1e-12, prediction[strategy] / 1_000_000)
            brier += (predicted - actual) ** 2
            if actual > 0:
                log_loss -= actual * math.log(predicted)
    return {
        "case_count": len(rows),
        "top_strategy_accuracy_ppm": top_correct * 1_000_000 // len(rows),
        "mean_distribution_brier": brier / len(rows),
        "mean_cross_entropy": log_loss / len(rows),
    }


def run_opponent_benchmark(
    *,
    seed: int = 20260824,
    random_count: int = 100,
    evidence_rounds: int = 12,
) -> dict[str, Any]:
    if random_count < 10:
        raise ValueError("benchmark needs at least ten random opponents")
    known = {
        "known_growth": _distribution((80, 10, 5, 5)),
        "known_profit": _distribution((10, 80, 5, 5)),
        "known_defensive": _distribution((5, 10, 80, 5)),
        "known_cooperative": _distribution((5, 10, 15, 70)),
        "known_balanced": _distribution((25, 25, 25, 25)),
        "mixed_growth_profit": _distribution((60, 40, 0, 0)),
        "mixed_defensive_cooperative": _distribution((0, 10, 45, 45)),
    }
    random_cases = {
        f"unknown_{index:03d}": deterministic_dirichlet_weights(
            seed, f"unknown_{index:03d}"
        )
        for index in range(random_count)
    }
    development_cut = random_count * 7 // 10
    rows: list[dict[str, Any]] = []
    all_cases = {**known, **random_cases}
    for case_id, truth_distribution in all_cases.items():
        if case_id.startswith("known") or case_id.startswith("mixed"):
            split = "development"
        else:
            index = int(case_id.rsplit("_", 1)[-1])
            split = "development" if index < development_cut else "holdout"
        evidence = _evidence_for_case(
            truth_distribution,
            seed=seed,
            case_id=case_id,
            rounds=evidence_rounds,
        )
        v1 = build_strategy_model("company_B", evidence)
        v2 = build_calibrated_strategy_model_v2("company_B", evidence)
        rows.append(
            {
                "case_id": case_id,
                "split": split,
                "truth_distribution_ppm": {
                    strategy: int(
                        getattr(truth_distribution, f"{strategy}_ppm")
                    )
                    for strategy in STRATEGIES
                },
                "v1_distribution_ppm": {
                    strategy: int(
                        getattr(v1.strategy_distribution, f"{strategy}_ppm")
                    )
                    for strategy in STRATEGIES
                },
                "v2_distribution_ppm": {
                    strategy: int(
                        getattr(v2.strategy_distribution, f"{strategy}_ppm")
                    )
                    for strategy in STRATEGIES
                },
                "evidence_hash": sha256_hash(
                    [item.model_dump(mode="json") for item in evidence]
                ),
            }
        )
    development = [row for row in rows if row["split"] == "development"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    return {
        "benchmark_schema_version": "stage6-opponent-benchmark-v1.0.0",
        "benchmark_kind": "deterministic_synthetic_no_llm",
        "seed": seed,
        "evidence_rounds": evidence_rounds,
        "development_case_ids": [row["case_id"] for row in development],
        "holdout_case_ids": [row["case_id"] for row in holdout],
        "development_scores": {
            "v1": _score_rows(development, "v1"),
            "v2": _score_rows(development, "v2"),
        },
        "holdout_scores": {
            "v1": _score_rows(holdout, "v1"),
            "v2": _score_rows(holdout, "v2"),
        },
        "rows": rows,
        "limitations": [
            "synthetic evidence is not a real-model behavioral claim",
            "latent distributions are utility hypotheses, not hidden Persona labels",
            "holdout performance must precede integration into the online treatment",
        ],
    }
