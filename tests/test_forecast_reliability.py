from game_theory_agent.strategic_reliability.forecast_reliability import (
    ForecastCalibrationSample,
    ForecastReleasePolicy,
    build_forecast_reliability_profile,
    compute_forecast_dataset_hash,
    compute_forecast_sample_hash,
    release_forecast_candidate,
)


def _sample(index: int, *, split: str = "development") -> ForecastCalibrationSample:
    payload = {
        "sample_schema_version": "forecast-calibration-sample-v1.0.0",
        "sample_id": f"episode-dev:r{index}:service",
        "split": split,
        "episode_id": "episode-dev" if split == "development" else "episode-holdout",
        "seed": 1 if split == "development" else 2,
        "round": index,
        "state_version": index - 1,
        "state_hash": f"sha256:state-{index}",
        "company_id": "company_A",
        "persona_id": "balanced_v1",
        "event_regime": "no_event",
        "risk_level": "low",
        "rounds_remaining": 6,
        "rounds_remaining_bucket": "6_plus",
        "opponent_price_structure": "stable",
        "capacity_status": "surplus",
        "cash_status": "normal",
        "action_dimension": "service",
        "candidate_id": "increase_service",
        "forecast_delta_ev_cents": 500_000,
        "authoritative_delta_ev_cents": 450_000,
        "sign_match": True,
        "absolute_error_cents": 50_000,
        "forecast_rank": 1,
        "authoritative_rank": 1,
        "rank_error": 0,
        "individually_eligible": True,
        "selected_in_uncalibrated_portfolio": True,
        "forecast_state_hash": f"sha256:forecast-{index}",
        "public_decision_input_hash": f"sha256:public-{index}",
        "opponent_tape_hash": f"sha256:tape-{index}",
        "uses_authoritative_state_for_label_only": True,
        "allowed_in_agent_context": False,
        "sample_hash": "pending",
    }
    payload["sample_hash"] = compute_forecast_sample_hash(payload)
    return ForecastCalibrationSample.model_validate(payload)


def test_forecast_profile_is_development_only_hashed_and_replayable():
    development = [_sample(index) for index in range(1, 9)]
    holdout = _sample(1, split="holdout")
    policy = ForecastReleasePolicy(
        policy_id="test",
        minimum_stratum_samples=8,
        minimum_direction_accuracy_ppm=800_000,
        error_band_quantile="p90",
        error_band_multiplier_ppm=1_000_000,
    )
    profile = build_forecast_reliability_profile(
        samples=[*development, holdout],
        dataset_hash=compute_forecast_dataset_hash(development),
        holdout_episode_ids=[holdout.episode_id],
        policy=policy,
    )
    assert profile.development_episode_ids == ["episode-dev"]
    assert profile.excluded_holdout_episode_ids == ["episode-holdout"]
    rebuilt = build_forecast_reliability_profile(
        samples=development,
        dataset_hash=compute_forecast_dataset_hash(development),
        holdout_episode_ids=[holdout.episode_id],
        policy=policy,
    )
    assert rebuilt.profile_hash == profile.profile_hash


def test_forecast_release_uses_historical_error_not_current_label():
    development = [_sample(index) for index in range(1, 9)]
    policy = ForecastReleasePolicy(
        policy_id="test",
        minimum_stratum_samples=8,
        minimum_direction_accuracy_ppm=800_000,
        error_band_quantile="p90",
        error_band_multiplier_ppm=1_000_000,
    )
    profile = build_forecast_reliability_profile(
        samples=development,
        dataset_hash=compute_forecast_dataset_hash(development),
        holdout_episode_ids=["episode-holdout"],
        policy=policy,
    )
    context = _sample(1, split="holdout")
    released = release_forecast_candidate(
        profile=profile,
        context=context,
        action_dimension="service",
        forecast_delta_ev_cents=500_000,
        individually_eligible=True,
    )
    abstained = release_forecast_candidate(
        profile=profile,
        context=context,
        action_dimension="service",
        forecast_delta_ev_cents=40_000,
        individually_eligible=True,
    )
    assert released.released is True
    assert released.uses_authoritative_current_label is False
    assert abstained.released is False
    assert "forecast_gain_within_historical_error_band" in abstained.reason_codes
