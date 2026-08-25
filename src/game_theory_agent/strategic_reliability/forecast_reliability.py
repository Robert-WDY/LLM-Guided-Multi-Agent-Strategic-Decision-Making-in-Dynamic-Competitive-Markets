"""Versioned zero-token calibration contracts for public market forecasts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal, Mapping, Sequence

from pydantic import Field, model_validator

from game_theory_agent.market.protocols import sha256_hash

from .contracts import StrictModel


PPM = 1_000_000


class ForecastCalibrationSample(StrictModel):
    sample_schema_version: Literal[
        "forecast-calibration-sample-v1.0.0"
    ] = "forecast-calibration-sample-v1.0.0"
    sample_id: str
    split: Literal["development", "holdout"]
    episode_id: str
    seed: int = Field(ge=0)
    round: int = Field(ge=1)
    state_version: int = Field(ge=0)
    state_hash: str
    company_id: str
    persona_id: str
    event_regime: Literal["no_event", "warning", "active_event"]
    risk_level: Literal["low", "medium", "high"]
    rounds_remaining: int = Field(ge=1)
    rounds_remaining_bucket: Literal["1_2", "3_5", "6_plus"]
    opponent_price_structure: Literal[
        "opponents_lower", "stable", "opponents_higher"
    ]
    capacity_status: Literal["surplus", "near_capacity", "capacity_gap"]
    cash_status: Literal["normal", "tight", "recovery"]
    action_dimension: str
    candidate_id: str
    forecast_delta_ev_cents: int
    authoritative_delta_ev_cents: int
    sign_match: bool
    absolute_error_cents: int = Field(ge=0)
    forecast_rank: int = Field(ge=1)
    authoritative_rank: int = Field(ge=1)
    rank_error: int = Field(ge=0)
    individually_eligible: bool
    selected_in_uncalibrated_portfolio: bool
    forecast_state_hash: str
    public_decision_input_hash: str
    opponent_tape_hash: str
    uses_authoritative_state_for_label_only: Literal[True] = True
    allowed_in_agent_context: Literal[False] = False
    sample_hash: str

    @model_validator(mode="after")
    def validate_sample(self) -> "ForecastCalibrationSample":
        if self.sign_match != (
            _sign(self.forecast_delta_ev_cents)
            == _sign(self.authoritative_delta_ev_cents)
        ):
            raise ValueError("forecast calibration sign label mismatch")
        if self.absolute_error_cents != abs(
            self.forecast_delta_ev_cents - self.authoritative_delta_ev_cents
        ):
            raise ValueError("forecast calibration absolute error mismatch")
        if self.rank_error != abs(
            self.forecast_rank - self.authoritative_rank
        ):
            raise ValueError("forecast calibration rank error mismatch")
        if self.sample_hash != compute_forecast_sample_hash(self):
            raise ValueError("forecast calibration sample hash mismatch")
        return self


class ForecastStratumStats(StrictModel):
    stratum_key: str
    sample_count: int = Field(ge=1)
    direction_match_count: int = Field(ge=0)
    direction_accuracy_ppm: int = Field(ge=0, le=PPM)
    negative_actual_count: int = Field(ge=0)
    absolute_error_p50_cents: int = Field(ge=0)
    absolute_error_p90_cents: int = Field(ge=0)
    mean_absolute_error_cents: int = Field(ge=0)
    mean_rank_error_milli: int = Field(ge=0)
    reliable: bool


class ForecastReleasePolicy(StrictModel):
    policy_id: str
    minimum_stratum_samples: int = Field(ge=1)
    minimum_direction_accuracy_ppm: int = Field(ge=0, le=PPM)
    error_band_quantile: Literal["p50", "p90"]
    error_band_multiplier_ppm: int = Field(ge=0, le=2_000_000)
    require_positive_forecast: Literal[True] = True
    require_individual_eligibility: Literal[True] = True


class ForecastReliabilityProfile(StrictModel):
    profile_schema_version: Literal[
        "forecast-reliability-v1.0.0"
    ] = "forecast-reliability-v1.0.0"
    dataset_hash: str
    development_episode_ids: list[str] = Field(min_length=1)
    development_seeds: list[int] = Field(min_length=1)
    excluded_holdout_episode_ids: list[str] = Field(min_length=1)
    release_policy: ForecastReleasePolicy
    global_stats: ForecastStratumStats
    strata: dict[str, ForecastStratumStats]
    uses_development_samples_only: Literal[True] = True
    uses_current_holdout_for_tuning: Literal[False] = False
    profile_hash: str

    @model_validator(mode="after")
    def validate_profile(self) -> "ForecastReliabilityProfile":
        if set(self.development_episode_ids).intersection(
            self.excluded_holdout_episode_ids
        ):
            raise ValueError("forecast development/holdout episode leakage")
        if self.profile_hash != compute_forecast_profile_hash(self):
            raise ValueError("forecast reliability profile hash mismatch")
        return self


class ForecastReleaseDecision(StrictModel):
    decision_schema_version: Literal[
        "forecast-release-decision-v1.0.0"
    ] = "forecast-release-decision-v1.0.0"
    profile_hash: str
    sample_context_hash: str
    action_dimension: str
    matched_stratum_key: str
    matched_sample_count: int = Field(ge=1)
    historical_direction_accuracy_ppm: int = Field(ge=0, le=PPM)
    historical_error_band_cents: int = Field(ge=0)
    forecast_delta_ev_cents: int
    individually_eligible: bool
    released: bool
    reason_codes: list[str] = Field(min_length=1)
    uses_authoritative_current_label: Literal[False] = False
    decision_hash: str

    @model_validator(mode="after")
    def validate_decision(self) -> "ForecastReleaseDecision":
        if self.decision_hash != compute_forecast_release_decision_hash(self):
            raise ValueError("forecast release decision hash mismatch")
        return self


def _sign(value: int) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _percentile(values: Sequence[int], percentile: int) -> int:
    if not values:
        raise ValueError("percentile requires samples")
    ordered = sorted(values)
    index = max(0, (len(ordered) * percentile + 99) // 100 - 1)
    return ordered[min(index, len(ordered) - 1)]


def compute_forecast_sample_hash(
    sample: ForecastCalibrationSample | Mapping[str, Any],
) -> str:
    payload = (
        sample.model_dump(mode="json")
        if isinstance(sample, ForecastCalibrationSample)
        else dict(sample)
    )
    payload.pop("sample_hash", None)
    return sha256_hash(
        {
            "hash_protocol_version": "forecast-calibration-sample-hash-v1.0.0",
            "sample": payload,
        }
    )


def compute_forecast_dataset_hash(
    samples: Sequence[ForecastCalibrationSample | Mapping[str, Any]],
) -> str:
    rows = [
        item.model_dump(mode="json")
        if isinstance(item, ForecastCalibrationSample)
        else dict(item)
        for item in samples
    ]
    return sha256_hash(
        {
            "hash_protocol_version": "forecast-calibration-dataset-hash-v1.0.0",
            "sample_hashes": sorted(str(item["sample_hash"]) for item in rows),
        }
    )


def compute_forecast_profile_hash(
    profile: ForecastReliabilityProfile | Mapping[str, Any],
) -> str:
    payload = (
        profile.model_dump(mode="json")
        if isinstance(profile, ForecastReliabilityProfile)
        else dict(profile)
    )
    payload.pop("profile_hash", None)
    return sha256_hash(
        {
            "hash_protocol_version": "forecast-reliability-profile-hash-v1.0.0",
            "profile": payload,
        }
    )


def compute_forecast_release_decision_hash(
    decision: ForecastReleaseDecision | Mapping[str, Any],
) -> str:
    payload = (
        decision.model_dump(mode="json")
        if isinstance(decision, ForecastReleaseDecision)
        else dict(decision)
    )
    payload.pop("decision_hash", None)
    return sha256_hash(
        {
            "hash_protocol_version": "forecast-release-decision-hash-v1.0.0",
            "decision": payload,
        }
    )


def _stats(
    key: str,
    samples: Sequence[ForecastCalibrationSample],
    policy: ForecastReleasePolicy,
) -> ForecastStratumStats:
    matches = sum(item.sign_match for item in samples)
    errors = [item.absolute_error_cents for item in samples]
    rank_errors = [item.rank_error for item in samples]
    accuracy = matches * PPM // len(samples)
    return ForecastStratumStats(
        stratum_key=key,
        sample_count=len(samples),
        direction_match_count=matches,
        direction_accuracy_ppm=accuracy,
        negative_actual_count=sum(
            item.authoritative_delta_ev_cents < 0 for item in samples
        ),
        absolute_error_p50_cents=_percentile(errors, 50),
        absolute_error_p90_cents=_percentile(errors, 90),
        mean_absolute_error_cents=sum(errors) // len(errors),
        mean_rank_error_milli=sum(rank_errors) * 1_000 // len(rank_errors),
        reliable=(
            len(samples) >= policy.minimum_stratum_samples
            and accuracy >= policy.minimum_direction_accuracy_ppm
        ),
    )


def sample_stratum_keys(
    sample: ForecastCalibrationSample | Mapping[str, Any],
) -> tuple[str, ...]:
    raw = (
        sample.model_dump(mode="json")
        if isinstance(sample, ForecastCalibrationSample)
        else dict(sample)
    )
    action = str(raw["action_dimension"])
    return (
        f"event_action:{raw['event_regime']}:{action}",
        f"risk_action:{raw['risk_level']}:{action}",
        f"remaining_action:{raw['rounds_remaining_bucket']}:{action}",
        f"opponent_price_action:{raw['opponent_price_structure']}:{action}",
        f"capacity_action:{raw['capacity_status']}:{action}",
        f"cash_action:{raw['cash_status']}:{action}",
        f"action:{action}",
        f"event:{raw['event_regime']}",
    )


def build_forecast_reliability_profile(
    *,
    samples: Sequence[ForecastCalibrationSample],
    dataset_hash: str,
    holdout_episode_ids: Sequence[str],
    policy: ForecastReleasePolicy,
) -> ForecastReliabilityProfile:
    development = [item for item in samples if item.split == "development"]
    if not development:
        raise ValueError("forecast profile requires development samples")
    if any(item.split != "development" for item in development):
        raise ValueError("forecast profile cannot use holdout samples")
    grouped: dict[str, list[ForecastCalibrationSample]] = defaultdict(list)
    for sample in development:
        for key in sample_stratum_keys(sample):
            grouped[key].append(sample)
    strata = {
        key: _stats(key, rows, policy) for key, rows in sorted(grouped.items())
    }
    payload: dict[str, Any] = {
        "profile_schema_version": "forecast-reliability-v1.0.0",
        "dataset_hash": dataset_hash,
        "development_episode_ids": sorted(
            {item.episode_id for item in development}
        ),
        "development_seeds": sorted({item.seed for item in development}),
        "excluded_holdout_episode_ids": sorted(set(holdout_episode_ids)),
        "release_policy": policy.model_dump(mode="json"),
        "global_stats": _stats("global", development, policy).model_dump(
            mode="json"
        ),
        "strata": {
            key: value.model_dump(mode="json") for key, value in strata.items()
        },
        "uses_development_samples_only": True,
        "uses_current_holdout_for_tuning": False,
        "profile_hash": "pending",
    }
    payload["profile_hash"] = compute_forecast_profile_hash(payload)
    return ForecastReliabilityProfile.model_validate(payload)


def release_forecast_candidate(
    *,
    profile: ForecastReliabilityProfile,
    context: ForecastCalibrationSample | Mapping[str, Any],
    action_dimension: str,
    forecast_delta_ev_cents: int,
    individually_eligible: bool,
) -> ForecastReleaseDecision:
    raw = (
        context.model_dump(mode="json")
        if isinstance(context, ForecastCalibrationSample)
        else dict(context)
    )
    raw["action_dimension"] = action_dimension
    keys = sample_stratum_keys(raw)
    matched = next(
        (
            profile.strata[key]
            for key in keys
            if key in profile.strata and profile.strata[key].reliable
        ),
        profile.global_stats,
    )
    policy = profile.release_policy
    error = (
        matched.absolute_error_p90_cents
        if policy.error_band_quantile == "p90"
        else matched.absolute_error_p50_cents
    )
    error_band = error * policy.error_band_multiplier_ppm // PPM
    reasons: list[str] = []
    if not matched.reliable:
        reasons.append("historical_stratum_unreliable")
    if not individually_eligible:
        reasons.append("individual_marginal_floors_failed")
    if forecast_delta_ev_cents <= 0:
        reasons.append("forecast_gain_not_positive")
    if forecast_delta_ev_cents <= error_band:
        reasons.append("forecast_gain_within_historical_error_band")
    released = not reasons
    if released:
        reasons.append("released_by_historical_forecast_profile")
    context_hash = sha256_hash(
        {
            "protocol": "forecast-release-context-v1.0.0",
            "event_regime": raw["event_regime"],
            "risk_level": raw["risk_level"],
            "rounds_remaining_bucket": raw["rounds_remaining_bucket"],
            "opponent_price_structure": raw["opponent_price_structure"],
            "capacity_status": raw["capacity_status"],
            "cash_status": raw["cash_status"],
            "action_dimension": action_dimension,
            "forecast_delta_ev_cents": forecast_delta_ev_cents,
            "individually_eligible": individually_eligible,
        }
    )
    payload: dict[str, Any] = {
        "decision_schema_version": "forecast-release-decision-v1.0.0",
        "profile_hash": profile.profile_hash,
        "sample_context_hash": context_hash,
        "action_dimension": action_dimension,
        "matched_stratum_key": matched.stratum_key,
        "matched_sample_count": matched.sample_count,
        "historical_direction_accuracy_ppm": matched.direction_accuracy_ppm,
        "historical_error_band_cents": error_band,
        "forecast_delta_ev_cents": forecast_delta_ev_cents,
        "individually_eligible": individually_eligible,
        "released": released,
        "reason_codes": reasons,
        "uses_authoritative_current_label": False,
        "decision_hash": "pending",
    }
    payload["decision_hash"] = compute_forecast_release_decision_hash(payload)
    return ForecastReleaseDecision.model_validate(payload)
