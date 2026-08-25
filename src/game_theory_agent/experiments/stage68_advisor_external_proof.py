"""Pre-registered, zero-token external audit of the v6 strategic advisor.

The audit never fits or edits the frozen forecast profile.  It applies the
released v0.7 profile and current v6 planner to historical real-LLM episodes
that were not used by either the development or first holdout split.  Public
belief and opponent state are rebuilt only from prior settled public actions.
Authoritative market state is used solely for frozen-tape outcome labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.belief import BeliefLedger
from game_theory_agent.market import MarketState, load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.opponent import OpponentModelLedger
from game_theory_agent.strategic_reliability import PublicMarketRolloutAdvisor
from game_theory_agent.strategic_reliability.forecast_reliability import (
    PPM,
    ForecastCalibrationSample,
    ForecastReliabilityProfile,
    compute_forecast_dataset_hash,
    compute_forecast_sample_hash,
    release_forecast_candidate,
)

from .midterm_forecast_calibration import (
    _direction_metrics,
    _rounds_bucket,
    _special_cells,
    _strata,
)
from .stage66_failure_forensics import (
    CONFIG_PATH,
    FOCAL_COMPANY,
    _events,
    economic_action,
    simulate_frozen_tape,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE = (
    PROJECT_ROOT
    / "midterm-release-v0.7"
    / "selected-runs"
    / "forecast-calibration"
    / "profile.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "stage6.8-advisor-external-proof"
DEFAULT_PREREGISTRATION = (
    PROJECT_ROOT
    / "experiment-specs"
    / "stage6.8-advisor-external-proof"
    / "PREREGISTRATION.json"
)
RELEASE_TAG = "strategic-research-mvp-v0.7.0"
RELEASE_COMMIT = "0747032bbd7b468c03010d874a820060153f933c"
HORIZON_ROUNDS = 3
SCENARIO_COUNT = 5
MAX_REAL_LLM_CALLS = 6

EXTERNAL_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "source_id": "belief-real-701",
        "seed": 701,
        "persona_id": "balanced_v1",
        "directory": "runs/belief-real-paired-20260821/seed-701/belief_on",
    },
    {
        "source_id": "belief-real-702",
        "seed": 702,
        "persona_id": "balanced_v1",
        "directory": "runs/belief-real-paired-20260821/seed-702/belief_on",
    },
    {
        "source_id": "belief-real-703",
        "seed": 703,
        "persona_id": "balanced_v1",
        "directory": "runs/belief-real-paired-20260821/seed-703/belief_on",
    },
    {
        "source_id": "privileged-public-aggressive-6101",
        "seed": 6101,
        "persona_id": "aggressive_v1_extreme",
        "directory": (
            "runs/privileged-information-persona-doubao-3seed-20260821/"
            "aggressive_v1_extreme/seed-6101/public_control"
        ),
    },
    {
        "source_id": "privileged-public-balanced-6101",
        "seed": 6101,
        "persona_id": "balanced_v1",
        "directory": (
            "runs/privileged-information-persona-doubao-3seed-20260821/"
            "balanced_v1/seed-6101/public_control"
        ),
    },
    {
        "source_id": "privileged-public-conservative-6101",
        "seed": 6101,
        "persona_id": "conservative_v1_extreme",
        "directory": (
            "runs/privileged-information-persona-doubao-3seed-20260821/"
            "conservative_v1_extreme/seed-6101/public_control"
        ),
    },
)

PREDICTION_THRESHOLDS = {
    "all_candidate_direction_accuracy_ppm": 800_000,
    "released_direction_accuracy_ppm": 900_000,
    "released_coverage_ppm": 300_000,
    "released_negative_actual_count": 0,
}
OUTCOME_THRESHOLDS = {
    "minimum_nonnegative_episode_count": 5,
    "minimum_mean_ev_delta_cents": 1,
    "minimum_worst_episode_mean_ev_delta_cents": 0,
    "minimum_worst_window_ev_delta_cents": -1_000_000,
    "minimum_mean_regret_reduction_cents": 1,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _sign(value: int) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def exact_two_sided_sign_test_ppm(values: Sequence[int]) -> int | None:
    """Return an exact two-sided sign-test p-value in integer ppm."""

    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    count = positive + negative
    if count == 0:
        return None
    tail = min(positive, negative)
    numerator = 2 * sum(math.comb(count, index) for index in range(tail + 1))
    denominator = 2**count
    return min(PPM, (numerator * PPM + denominator // 2) // denominator)


def _source_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in EXTERNAL_SOURCES:
        directory = PROJECT_ROOT / str(source["directory"])
        summary_path = directory / "summary.json"
        events_path = directory / "round-events.jsonl"
        if not summary_path.is_file() or not events_path.is_file():
            raise FileNotFoundError(f"missing external audit source: {directory}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not bool(summary.get("passed")):
            raise ValueError(f"external source did not pass its own checks: {directory}")
        rows.append(
            {
                **source,
                "evidence_type": "REAL_RECORDED_LLM_EPISODE",
                "episode_id": str(summary["episode_id"]),
                "round_count": int(summary["rounds"]),
                "provider": str(summary.get("provider") or "doubao"),
                "model": summary.get("model"),
                "summary_sha256": _sha256_file(summary_path),
                "round_events_sha256": _sha256_file(events_path),
            }
        )
    return rows


def _with_hash(payload: Mapping[str, Any], *, protocol: str) -> dict[str, Any]:
    result = dict(payload)
    result["preregistration_hash"] = sha256_hash(
        {"protocol": protocol, "payload": result}
    )
    return result


def build_preregistration(profile_path: Path = DEFAULT_PROFILE) -> dict[str, Any]:
    profile = ForecastReliabilityProfile.model_validate_json(
        profile_path.read_text(encoding="utf-8")
    )
    sources = _source_manifest()
    episode_ids = {str(item["episode_id"]) for item in sources}
    if episode_ids.intersection(profile.development_episode_ids):
        raise ValueError("external audit episodes overlap profile development")
    if episode_ids.intersection(profile.excluded_holdout_episode_ids):
        raise ValueError("external audit episodes overlap first holdout")
    payload: dict[str, Any] = {
        "preregistration_schema_version": "stage6.8-preregistration-v1.0.0",
        "release_tag": RELEASE_TAG,
        "release_commit": RELEASE_COMMIT,
        "advisor_mode": "pareto_reliable_v6",
        "profile_path": str(profile_path.resolve()),
        "frozen_profile_hash": profile.profile_hash,
        "profile_is_not_refit": True,
        "sources": sources,
        "source_episode_count": len(sources),
        "independent_seed_count": len({int(item["seed"]) for item in sources}),
        "source_round_window_count": sum(int(item["round_count"]) for item in sources),
        "focal_company_id": FOCAL_COMPANY,
        "horizon_rounds": HORIZON_ROUNDS,
        "scenario_count": SCENARIO_COUNT,
        "primary_outcome": "frozen_tape_final_enterprise_value_cents",
        "secondary_outcomes": [
            "counterfactual_regret_cents",
            "cumulative_profit_cents",
            "forecast_direction_accuracy",
            "released_candidate_negative_actual_count",
        ],
        "prediction_thresholds": PREDICTION_THRESHOLDS,
        "outcome_thresholds": OUTCOME_THRESHOLDS,
        "real_llm_gate": {
            "requires_all_engineering_prediction_and_outcome_checks": True,
            "maximum_calls_if_open": MAX_REAL_LLM_CALLS,
            "paired_design_if_open": "3 new seeds x control/treatment",
            "calls_if_closed": 0,
        },
        "statistical_boundary": (
            "four independent seeds cannot establish a general statistical "
            "conclusion; report exact seed-cluster sign test and directional evidence"
        ),
        "forbidden_after_audit": [
            "refit the frozen profile on these labels",
            "change thresholds after observing outcomes",
            "call the model when any hard gate fails",
            "describe overlapping round windows as independent samples",
        ],
    }
    return _with_hash(payload, protocol="stage6.8-preregistration-hash-v1.0.0")


def write_preregistration(path: Path, profile_path: Path) -> dict[str, Any]:
    preregistration = build_preregistration(profile_path)
    _write_json(path, preregistration)
    return preregistration


def _mean(values: Sequence[int]) -> int:
    return sum(values) // max(1, len(values))


def _aggregate_windows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [int(item["advisor_minus_historical_ev_cents"]) for item in rows]
    regret = [int(item["regret_reduction_cents"]) for item in rows]
    profit = [int(item["advisor_minus_historical_profit_cents"]) for item in rows]
    return {
        "window_count": len(rows),
        "positive_zero_negative_windows": {
            "positive": sum(value > 0 for value in deltas),
            "zero": sum(value == 0 for value in deltas),
            "negative": sum(value < 0 for value in deltas),
        },
        "mean_ev_delta_cents": _mean(deltas),
        "worst_ev_delta_cents": min(deltas),
        "best_ev_delta_cents": max(deltas),
        "mean_regret_reduction_cents": _mean(regret),
        "mean_profit_delta_cents": _mean(profit),
    }


def _group_aggregates(
    rows: Sequence[Mapping[str, Any]], key: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return [
        {key: value, **_aggregate_windows(grouped[value])}
        for value in sorted(grouped)
    ]


def _release_metrics(
    pairs: Sequence[tuple[ForecastCalibrationSample, Mapping[str, Any]]]
) -> dict[str, Any]:
    eligible = [
        item
        for item, _ in pairs
        if item.individually_eligible and item.forecast_delta_ev_cents > 0
    ]
    released = [(item, decision) for item, decision in pairs if decision["released"]]
    matches = sum(item.sign_match for item, _ in released)
    return {
        "sample_count": len(pairs),
        "eligible_candidate_count": len(eligible),
        "released_candidate_count": len(released),
        "coverage_ppm": len(released) * PPM // max(1, len(eligible)),
        "released_direction_match_count": matches,
        "released_direction_accuracy_ppm": matches * PPM // max(1, len(released)),
        "released_negative_actual_count": sum(
            item.authoritative_delta_ev_cents < 0 for item, _ in released
        ),
        "released_negative_sample_ids": [
            item.sample_id
            for item, _ in released
            if item.authoritative_delta_ev_cents < 0
        ],
    }


def _load_and_validate_preregistration(
    path: Path, profile_path: Path
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "PREREGISTRATION.json is required; run with --plan-only first"
        )
    recorded = json.loads(path.read_text(encoding="utf-8"))
    expected = build_preregistration(profile_path)
    if recorded != expected:
        raise ValueError("preregistration no longer matches frozen source/profile inputs")
    return recorded


def _build_audit(
    preregistration: Mapping[str, Any],
    profile: ForecastReliabilityProfile,
) -> tuple[
    list[ForecastCalibrationSample],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    advisor = PublicMarketRolloutAdvisor(config)
    samples: list[ForecastCalibrationSample] = []
    release_rows: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    deterministic_advice = 0
    deterministic_market = 0

    for source in preregistration["sources"]:
        events = _events(PROJECT_ROOT / str(source["directory"]))
        first_state = MarketState.from_dict(events[0].state_before)
        belief_ledger = BeliefLedger(
            episode_id=first_state.episode_id,
            company_ids=first_state.company_ids,
        )
        opponent_ledger = OpponentModelLedger(
            episode_id=first_state.episode_id,
            company_ids=first_state.company_ids,
        )
        profile_for_persona = personas.get(str(source["persona_id"]))

        for event_index, event in enumerate(events):
            initial_state = MarketState.from_dict(event.state_before)
            trace = next(
                item for item in event.traces if item.company_id == FOCAL_COMPANY
            )
            belief, _ = belief_ledger.company_view(
                observer_company_id=FOCAL_COMPANY,
                round_number=initial_state.round,
                state_version=initial_state.state_version,
            )
            opponent_model, _ = opponent_ledger.company_view(
                observer_company_id=FOCAL_COMPANY,
                round_number=initial_state.round,
                state_version=initial_state.state_version,
            )
            horizon = min(HORIZON_ROUNDS, len(events) - event_index)
            advice = advisor.advise(
                observation=trace.observation,
                company_id=FOCAL_COMPANY,
                persona_profile=profile_for_persona,
                belief_state=belief,
                opponent_model=opponent_model,
                horizon_rounds=horizon,
                scenario_count=SCENARIO_COUNT,
                advisor_mode="pareto_reliable_v6",
            )
            rebuilt = advisor.advise(
                observation=trace.observation,
                company_id=FOCAL_COMPANY,
                persona_profile=profile_for_persona,
                belief_state=belief,
                opponent_model=opponent_model,
                horizon_rounds=horizon,
                scenario_count=SCENARIO_COUNT,
                advisor_mode="pareto_reliable_v6",
            )
            advice_matches = advice.model_dump(mode="json") == rebuilt.model_dump(
                mode="json"
            )
            deterministic_advice += int(advice_matches)
            plan = advice.investment_marginal_plan
            if plan is None:
                raise ValueError("v6 external audit requires a marginal plan")
            tape = events[event_index : event_index + horizon]
            historical_action = economic_action(event.joint_action[FOCAL_COMPANY])
            historical = simulate_frozen_tape(
                initial_state=initial_state,
                initial_action=historical_action,
                recorded_events=tape,
            )
            advised = simulate_frozen_tape(
                initial_state=initial_state,
                initial_action=advice.recommended_action,
                recorded_events=tape,
            )
            advised_repeat = simulate_frozen_tape(
                initial_state=initial_state,
                initial_action=advice.recommended_action,
                recorded_events=tape,
            )
            market_matches = advised == advised_repeat
            deterministic_market += int(market_matches)

            candidate_values: dict[str, int] = {}
            for evaluation in advice.candidate_actions:
                candidate_values[evaluation.candidate.candidate_id] = int(
                    simulate_frozen_tape(
                        initial_state=initial_state,
                        initial_action=evaluation.candidate.action.model_dump(
                            mode="json"
                        ),
                        recorded_events=tape,
                    )["final_enterprise_value_cents"]
                )
            historical_ev = int(historical["final_enterprise_value_cents"])
            advised_ev = int(advised["final_enterprise_value_cents"])
            authoritative_best = max(
                historical_ev, advised_ev, *candidate_values.values()
            )
            historical_profit = int(
                historical["rounds"][-1]["cumulative_profit_cents"]
            )
            advised_profit = int(advised["rounds"][-1]["cumulative_profit_cents"])
            window_id = f"{event.episode_id}:r{event.settled_round}"
            windows.append(
                {
                    "window_id": window_id,
                    "source_id": source["source_id"],
                    "episode_id": event.episode_id,
                    "seed": int(source["seed"]),
                    "persona_id": source["persona_id"],
                    "round": int(event.settled_round),
                    "horizon_rounds": horizon,
                    "state_hash": event.state_before_hash,
                    "historical_real_llm_action": historical_action,
                    "v6_recommended_candidate_id": advice.recommended_candidate_id,
                    "v6_recommended_action": economic_action(
                        advice.recommended_action
                    ),
                    "historical_enterprise_value_cents": historical_ev,
                    "advisor_enterprise_value_cents": advised_ev,
                    "advisor_minus_historical_ev_cents": advised_ev - historical_ev,
                    "historical_cumulative_profit_cents": historical_profit,
                    "advisor_cumulative_profit_cents": advised_profit,
                    "advisor_minus_historical_profit_cents": (
                        advised_profit - historical_profit
                    ),
                    "authoritative_best_observed_candidate_ev_cents": authoritative_best,
                    "historical_regret_cents": authoritative_best - historical_ev,
                    "advisor_regret_cents": authoritative_best - advised_ev,
                    "regret_reduction_cents": advised_ev - historical_ev,
                    "advice_hash": advice.advice_hash,
                    "deterministic_advice_rebuild": advice_matches,
                    "deterministic_market_replay": market_matches,
                    "uses_authoritative_hidden_state_in_advice": (
                        advice.uses_authoritative_hidden_market_state
                    ),
                }
            )

            baseline = simulate_frozen_tape(
                initial_state=initial_state,
                initial_action=plan["baseline_action"],
                recorded_events=tape,
            )
            baseline_ev = int(baseline["final_enterprise_value_cents"])
            probes: list[dict[str, Any]] = []
            for assessment in plan["assessments"]:
                settled = simulate_frozen_tape(
                    initial_state=initial_state,
                    initial_action=assessment["probe_action"],
                    recorded_events=tape,
                )
                probes.append(
                    {
                        "assessment": assessment,
                        "forecast": int(
                            assessment["marginal_expected_enterprise_value_cents"]
                        ),
                        "actual": int(settled["final_enterprise_value_cents"])
                        - baseline_ev,
                    }
                )
            forecast_order = {
                id(row): rank
                for rank, row in enumerate(
                    sorted(
                        probes,
                        key=lambda item: (
                            -item["forecast"],
                            item["assessment"]["dimension"],
                        ),
                    ),
                    start=1,
                )
            }
            actual_order = {
                id(row): rank
                for rank, row in enumerate(
                    sorted(
                        probes,
                        key=lambda item: (
                            -item["actual"],
                            item["assessment"]["dimension"],
                        ),
                    ),
                    start=1,
                )
            }
            context = _strata(trace.observation)
            rounds_remaining = int(trace.observation["rounds_remaining"])
            tape_hash = sha256_hash(
                {
                    "protocol": "forecast-authoritative-tape-v1.0.0",
                    "state_before_hash": event.state_before_hash,
                    "event_ids": [item.event_id for item in tape],
                    "joint_action_hashes": [item.joint_action_hash for item in tape],
                }
            )
            for row in probes:
                assessment = row["assessment"]
                payload: dict[str, Any] = {
                    "sample_schema_version": "forecast-calibration-sample-v1.0.0",
                    "sample_id": f"{window_id}:{assessment['dimension']}",
                    "split": "holdout",
                    "episode_id": event.episode_id,
                    "seed": int(source["seed"]),
                    "round": int(event.settled_round),
                    "state_version": int(trace.observation["state_version"]),
                    "state_hash": event.state_before_hash,
                    "company_id": FOCAL_COMPANY,
                    "persona_id": source["persona_id"],
                    **context,
                    "rounds_remaining": rounds_remaining,
                    "rounds_remaining_bucket": _rounds_bucket(rounds_remaining),
                    "action_dimension": str(assessment["dimension"]),
                    "candidate_id": str(assessment["probe_candidate_id"]),
                    "forecast_delta_ev_cents": row["forecast"],
                    "authoritative_delta_ev_cents": row["actual"],
                    "sign_match": _sign(row["forecast"]) == _sign(row["actual"]),
                    "absolute_error_cents": abs(row["forecast"] - row["actual"]),
                    "forecast_rank": forecast_order[id(row)],
                    "authoritative_rank": actual_order[id(row)],
                    "rank_error": abs(
                        forecast_order[id(row)] - actual_order[id(row)]
                    ),
                    "individually_eligible": bool(
                        assessment["individually_eligible"]
                    ),
                    "selected_in_uncalibrated_portfolio": bool(
                        assessment["selected_in_portfolio"]
                    ),
                    "forecast_state_hash": advice.forecast_state_hash,
                    "public_decision_input_hash": advice.public_decision_input_hash,
                    "opponent_tape_hash": tape_hash,
                    "uses_authoritative_state_for_label_only": True,
                    "allowed_in_agent_context": False,
                    "sample_hash": "pending",
                }
                payload["sample_hash"] = compute_forecast_sample_hash(payload)
                sample = ForecastCalibrationSample.model_validate(payload)
                decision = release_forecast_candidate(
                    profile=profile,
                    context=sample,
                    action_dimension=sample.action_dimension,
                    forecast_delta_ev_cents=sample.forecast_delta_ev_cents,
                    individually_eligible=sample.individually_eligible,
                )
                rebuilt_decision = release_forecast_candidate(
                    profile=profile,
                    context=sample,
                    action_dimension=sample.action_dimension,
                    forecast_delta_ev_cents=sample.forecast_delta_ev_cents,
                    individually_eligible=sample.individually_eligible,
                )
                samples.append(sample)
                release_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        **decision.model_dump(mode="json"),
                        "decision_replay_match": decision == rebuilt_decision,
                    }
                )

            belief_ledger.update_after_settlement(
                initial_state, event.joint_action
            )
            opponent_ledger.update_after_settlement(
                initial_state,
                MarketState.from_dict(event.state_after),
                event.joint_action,
            )

    checks = {
        "deterministic_advice_rebuild_count": deterministic_advice,
        "deterministic_market_replay_count": deterministic_market,
        "window_count": len(windows),
        "release_decision_replay_count": sum(
            bool(item["decision_replay_match"]) for item in release_rows
        ),
        "hidden_authoritative_state_used_by_advice_count": sum(
            bool(item["uses_authoritative_hidden_state_in_advice"])
            for item in windows
        ),
    }
    return samples, release_rows, windows, checks


def run(
    profile_path: Path, output: Path, preregistration_path: Path
) -> dict[str, Any]:
    preregistration = _load_and_validate_preregistration(
        preregistration_path, profile_path
    )
    profile = ForecastReliabilityProfile.model_validate_json(
        profile_path.read_text(encoding="utf-8")
    )
    samples, release_rows, windows, raw_checks = _build_audit(
        preregistration, profile
    )
    pairs = list(zip(samples, release_rows, strict=True))
    direction = _direction_metrics(samples)
    released = _release_metrics(pairs)
    overall = _aggregate_windows(windows)
    by_episode = _group_aggregates(windows, "episode_id")
    by_persona = _group_aggregates(windows, "persona_id")
    by_seed = _group_aggregates(windows, "seed")
    episode_means = [int(item["mean_ev_delta_cents"]) for item in by_episode]
    seed_means = [int(item["mean_ev_delta_cents"]) for item in by_seed]

    engineering_checks = {
        "preregistration_hash_matches": True,
        "profile_hash_unchanged": profile.profile_hash
        == preregistration["frozen_profile_hash"],
        "external_episode_overlap_with_profile_zero": not (
            {item["episode_id"] for item in preregistration["sources"]}
            & (
                set(profile.development_episode_ids)
                | set(profile.excluded_holdout_episode_ids)
            )
        ),
        "all_windows_rebuilt_deterministically": (
            raw_checks["deterministic_advice_rebuild_count"] == len(windows)
            and raw_checks["deterministic_market_replay_count"] == len(windows)
        ),
        "all_release_decisions_rebuilt_deterministically": (
            raw_checks["release_decision_replay_count"] == len(release_rows)
        ),
        "hidden_authoritative_state_used_by_advice_count": raw_checks[
            "hidden_authoritative_state_used_by_advice_count"
        ],
        "new_real_model_calls": 0,
        "new_prompt_tokens": 0,
        "new_completion_tokens": 0,
        "new_total_tokens": 0,
        "new_estimated_cost_cny": 0,
    }
    prediction_checks = {
        "all_candidate_direction_accuracy_at_least_80_percent": direction[
            "direction_accuracy_ppm"
        ]
        >= PREDICTION_THRESHOLDS["all_candidate_direction_accuracy_ppm"],
        "released_direction_accuracy_at_least_90_percent": released[
            "released_direction_accuracy_ppm"
        ]
        >= PREDICTION_THRESHOLDS["released_direction_accuracy_ppm"]
        and released["released_candidate_count"] > 0,
        "released_coverage_at_least_30_percent": released["coverage_ppm"]
        >= PREDICTION_THRESHOLDS["released_coverage_ppm"],
        "released_negative_actual_count_zero": released[
            "released_negative_actual_count"
        ]
        == PREDICTION_THRESHOLDS["released_negative_actual_count"],
    }
    outcome_checks = {
        "advisor_mean_ev_delta_positive": overall["mean_ev_delta_cents"]
        >= OUTCOME_THRESHOLDS["minimum_mean_ev_delta_cents"],
        "at_least_five_of_six_episode_means_nonnegative": sum(
            value >= 0 for value in episode_means
        )
        >= OUTCOME_THRESHOLDS["minimum_nonnegative_episode_count"],
        "worst_episode_mean_ev_delta_nonnegative": min(episode_means)
        >= OUTCOME_THRESHOLDS["minimum_worst_episode_mean_ev_delta_cents"],
        "worst_window_ev_delta_within_tail_limit": overall[
            "worst_ev_delta_cents"
        ]
        >= OUTCOME_THRESHOLDS["minimum_worst_window_ev_delta_cents"],
        "mean_regret_reduction_positive": overall["mean_regret_reduction_cents"]
        >= OUTCOME_THRESHOLDS["minimum_mean_regret_reduction_cents"],
    }
    engineering_passed = all(
        value is True or value == 0 for value in engineering_checks.values()
    )
    prediction_passed = all(prediction_checks.values())
    outcome_passed = all(outcome_checks.values())
    paid_gate = engineering_passed and prediction_passed and outcome_passed
    summary: dict[str, Any] = {
        "experiment_schema_version": "stage6.8-advisor-external-proof-v1.0.0",
        "evidence_level": "zero-token-external-historical-real-llm-frozen-tape",
        "preregistration_hash": preregistration["preregistration_hash"],
        "frozen_profile_hash": profile.profile_hash,
        "dataset_hash": compute_forecast_dataset_hash(samples),
        "episode_count": len(by_episode),
        "independent_seed_count": len(by_seed),
        "window_count": len(windows),
        "forecast_sample_count": len(samples),
        "direction_metrics": direction,
        "release_metrics": released,
        "special_cells": _special_cells(samples),
        "overall_outcomes": overall,
        "by_episode": by_episode,
        "by_persona": by_persona,
        "by_seed": by_seed,
        "seed_cluster_exact_sign_test_two_sided_p_ppm": (
            exact_two_sided_sign_test_ppm(seed_means)
        ),
        "engineering_checks": engineering_checks,
        "prediction_checks": prediction_checks,
        "outcome_checks": outcome_checks,
        "engineering_passed": engineering_passed,
        "prediction_gate_passed": prediction_passed,
        "outcome_gate_passed": outcome_passed,
        "real_llm_gate": (
            "open_max_6_calls" if paid_gate else "closed_no_real_llm_calls"
        ),
        "new_real_model_calls": 0,
        "new_total_tokens": 0,
        "new_estimated_cost_cny": 0,
        "statistical_conclusion_supported": False,
        "conclusion_limits": [
            "only four independent seeds are present across six episodes",
            "round windows overlap within an episode and are not independent samples",
            "the market is synthetic and not empirically calibrated",
            "v6 advice is auto-executed here rather than adopted by a new model call",
            "historical episodes are external to v6 fitting but were created in earlier project stages",
        ],
        "report_hash": "pending",
    }
    summary["report_hash"] = sha256_hash(
        {key: value for key, value in summary.items() if key != "report_hash"}
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "dataset.jsonl").open("w", encoding="utf-8") as handle:
        for item in samples:
            handle.write(item.model_dump_json())
            handle.write("\n")
    with (output / "release-decisions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for item in release_rows:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    with (output / "decision-windows.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for item in windows:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=DEFAULT_PREREGISTRATION,
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    result = (
        write_preregistration(args.preregistration, args.profile)
        if args.plan_only
        else run(args.profile, args.output, args.preregistration)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.plan_only:
        return 0
    return 0 if result["real_llm_gate"] == "open_max_6_calls" else 2


if __name__ == "__main__":
    raise SystemExit(main())
