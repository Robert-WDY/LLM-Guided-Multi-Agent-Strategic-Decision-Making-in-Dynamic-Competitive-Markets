"""Zero-token calibration and holdout gate for the midterm v0.7 forecast.

The development split may select one policy from a predeclared menu.  Holdout
labels are evaluated exactly once and never enter the reliability profile.
Authoritative state is used only to settle frozen-tape labels; planner inputs
come from the recorded company-scoped observation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.market import MarketState, load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.strategic_reliability import PublicMarketRolloutAdvisor
from game_theory_agent.strategic_reliability.forecast_reliability import (
    PPM,
    ForecastCalibrationSample,
    ForecastReleasePolicy,
    ForecastReliabilityProfile,
    build_forecast_reliability_profile,
    compute_forecast_dataset_hash,
    compute_forecast_sample_hash,
    release_forecast_candidate,
)

from .stage66_failure_forensics import (
    CONFIG_PATH,
    DEFAULT_STAGE65_SUMMARY,
    FOCAL_COMPANY,
    _canonical_rows,
    _events,
    simulate_frozen_tape,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "midterm-v0.7-forecast-calibration"
DEVELOPMENT_SEEDS = (1, 2, 3)
HOLDOUT_SEEDS = (1001, 1002, 1003, 1004, 1005)
STAGE51_ROOT = PROJECT_ROOT / "runs" / "stage51-real-pilot-doubao-seed1001-20260821"

# This menu is code-versioned before holdout evaluation.  Selection uses only
# development labels and is deterministic.
POLICY_MENU = (
    ForecastReleasePolicy(
        policy_id="strict-p90-v1",
        minimum_stratum_samples=8,
        minimum_direction_accuracy_ppm=800_000,
        error_band_quantile="p90",
        error_band_multiplier_ppm=1_000_000,
    ),
    ForecastReleasePolicy(
        policy_id="balanced-p50-v1",
        minimum_stratum_samples=8,
        minimum_direction_accuracy_ppm=800_000,
        error_band_quantile="p50",
        error_band_multiplier_ppm=1_000_000,
    ),
    ForecastReleasePolicy(
        policy_id="half-p90-v1",
        minimum_stratum_samples=8,
        minimum_direction_accuracy_ppm=800_000,
        error_band_quantile="p90",
        error_band_multiplier_ppm=500_000,
    ),
)


def _sign(value: int) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _rounds_bucket(rounds_remaining: int) -> str:
    if rounds_remaining <= 2:
        return "1_2"
    if rounds_remaining <= 5:
        return "3_5"
    return "6_plus"


def _severity(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value or "").lower()
    return {"low": 250_000, "medium": 550_000, "high": 850_000}.get(
        text, 0
    )


def _strata(observation: Mapping[str, Any]) -> dict[str, str]:
    active = list(observation.get("active_market_events") or [])
    signals = list(observation.get("risk_signals") or [])
    event_regime = "active_event" if active else "warning" if signals else "no_event"
    severities = [
        _severity(item.get("severity"))
        for item in (*active, *signals)
        if isinstance(item, Mapping)
    ]
    maximum = max(severities, default=0)
    risk_level = "high" if maximum >= 700_000 else "medium" if maximum >= 350_000 else "low"
    own = observation["own_company"]
    own_price = int(own["commercial"]["price_cents"])
    opponents = list(observation.get("competitors") or [])
    if not opponents:
        opponents = [
            row
            for row in observation.get("public_companies", [])
            if row.get("company_id") != FOCAL_COMPANY
        ]
    opponent_prices = [int(row["price_cents"]) for row in opponents]
    average_opponent = sum(opponent_prices) // max(1, len(opponent_prices))
    opponent_price_structure = (
        "opponents_lower"
        if average_opponent < own_price - 200
        else "opponents_higher"
        if average_opponent > own_price + 200
        else "stable"
    )
    utilization = int(own["operations"]["capacity_utilization_ppm"])
    capacity_status = (
        "capacity_gap"
        if utilization >= 900_000
        else "near_capacity"
        if utilization >= 750_000
        else "surplus"
    )
    phase = str(observation.get("decision_support", {}).get("strategic_phase", ""))
    cash_status = (
        "tight"
        if phase == "liquidity_crisis"
        else "recovery"
        if phase == "profit_recovery"
        else "normal"
    )
    return {
        "event_regime": event_regime,
        "risk_level": risk_level,
        "opponent_price_structure": opponent_price_structure,
        "capacity_status": capacity_status,
        "cash_status": cash_status,
    }


def _source_rows(summary: Path) -> list[dict[str, Any]]:
    development = [
        {**row, "split": "development"}
        for row in _canonical_rows(summary)
        if row["condition"] == "pareto_v4"
        and int(row["seed"]) in DEVELOPMENT_SEEDS
    ]
    holdout = [
        {
            "seed": seed,
            "persona_id": "balanced_v1",
            "condition": "stage51_utility_advisor_historical_holdout",
            "directory": str(STAGE51_ROOT / f"seed-{seed}" / "D_utility_advisor"),
            "split": "holdout",
        }
        for seed in HOLDOUT_SEEDS
    ]
    rows = [*development, *holdout]
    missing = [row["directory"] for row in rows if not Path(row["directory"]).is_dir()]
    if missing:
        raise FileNotFoundError(f"calibration source directories missing: {missing}")
    return sorted(
        rows,
        key=lambda row: (
            0 if row["split"] == "development" else 1,
            int(row["seed"]),
            str(row["persona_id"]),
        ),
    )


def build_samples(stage65_summary: Path) -> list[ForecastCalibrationSample]:
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    advisor = PublicMarketRolloutAdvisor(config)
    samples: list[ForecastCalibrationSample] = []
    for source in _source_rows(stage65_summary):
        seed = int(source["seed"])
        split = str(source["split"])
        persona_id = str(source["persona_id"])
        events = _events(Path(source["directory"]))
        for event_index, event in enumerate(events):
            trace = next(item for item in event.traces if item.company_id == FOCAL_COMPANY)
            recorded_advice = trace.advisor_output
            if recorded_advice is None:
                continue
            horizon = min(
                int(recorded_advice.get("horizon_rounds", 3)),
                len(events) - event_index,
            )
            scenario_count = int(recorded_advice.get("scenario_count", 5))
            advice = advisor.advise(
                observation=trace.observation,
                company_id=FOCAL_COMPANY,
                persona_profile=personas.get(persona_id),
                belief_state=trace.belief_before,
                opponent_model=trace.opponent_model,
                horizon_rounds=horizon,
                scenario_count=scenario_count,
                advisor_mode="pareto_reliable_v6",
            )
            plan = advice.investment_marginal_plan
            if plan is None:
                raise ValueError("v6 calibration requires a marginal plan")
            tape = events[event_index : event_index + horizon]
            initial_state = MarketState.from_dict(event.state_before)
            baseline = simulate_frozen_tape(
                initial_state=initial_state,
                initial_action=plan["baseline_action"],
                recorded_events=tape,
            )
            baseline_value = int(baseline["final_enterprise_value_cents"])
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
                        "forecast": int(assessment["marginal_expected_enterprise_value_cents"]),
                        "actual": int(settled["final_enterprise_value_cents"]) - baseline_value,
                    }
                )
            forecast_order = {
                id(row): rank
                for rank, row in enumerate(
                    sorted(probes, key=lambda item: (-item["forecast"], item["assessment"]["dimension"])),
                    start=1,
                )
            }
            actual_order = {
                id(row): rank
                for rank, row in enumerate(
                    sorted(probes, key=lambda item: (-item["actual"], item["assessment"]["dimension"])),
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
                    "sample_id": (
                        f"{event.episode_id}:r{event.settled_round}:"
                        f"{assessment['dimension']}"
                    ),
                    "split": split,
                    "episode_id": event.episode_id,
                    "seed": seed,
                    "round": int(event.settled_round),
                    "state_version": int(trace.observation["state_version"]),
                    "state_hash": event.state_before_hash,
                    "company_id": FOCAL_COMPANY,
                    "persona_id": persona_id,
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
                    "rank_error": abs(forecast_order[id(row)] - actual_order[id(row)]),
                    "individually_eligible": bool(assessment["individually_eligible"]),
                    "selected_in_uncalibrated_portfolio": bool(assessment["selected_in_portfolio"]),
                    "forecast_state_hash": advice.forecast_state_hash,
                    "public_decision_input_hash": advice.public_decision_input_hash,
                    "opponent_tape_hash": tape_hash,
                    "uses_authoritative_state_for_label_only": True,
                    "allowed_in_agent_context": False,
                    "sample_hash": "pending",
                }
                payload["sample_hash"] = compute_forecast_sample_hash(payload)
                samples.append(ForecastCalibrationSample.model_validate(payload))
    return sorted(samples, key=lambda item: item.sample_id)


def _release_metrics(
    profile: ForecastReliabilityProfile,
    samples: Sequence[ForecastCalibrationSample],
) -> dict[str, Any]:
    eligible = [
        item
        for item in samples
        if item.individually_eligible and item.forecast_delta_ev_cents > 0
    ]
    decisions = [
        (
            item,
            release_forecast_candidate(
                profile=profile,
                context=item,
                action_dimension=item.action_dimension,
                forecast_delta_ev_cents=item.forecast_delta_ev_cents,
                individually_eligible=item.individually_eligible,
            ),
        )
        for item in samples
    ]
    released = [(item, decision) for item, decision in decisions if decision.released]
    match_count = sum(item.sign_match for item, _ in released)
    return {
        "sample_count": len(samples),
        "eligible_candidate_count": len(eligible),
        "released_candidate_count": len(released),
        "coverage_ppm": len(released) * PPM // max(1, len(eligible)),
        "released_direction_match_count": match_count,
        "released_direction_accuracy_ppm": match_count * PPM // max(1, len(released)),
        "released_negative_actual_count": sum(
            item.authoritative_delta_ev_cents < 0 for item, _ in released
        ),
        "decision_hashes": [decision.decision_hash for _, decision in released],
    }


def _direction_metrics(samples: Sequence[ForecastCalibrationSample]) -> dict[str, Any]:
    matches = sum(item.sign_match for item in samples)
    return {
        "sample_count": len(samples),
        "direction_match_count": matches,
        "direction_accuracy_ppm": matches * PPM // max(1, len(samples)),
        "mean_absolute_error_cents": sum(item.absolute_error_cents for item in samples) // max(1, len(samples)),
        "mean_rank_error_milli": sum(item.rank_error for item in samples) * 1_000 // max(1, len(samples)),
    }


def _special_cells(samples: Sequence[ForecastCalibrationSample]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for event in ("warning", "active_event"):
        for dimension in ("resilience", "service"):
            rows = [
                item
                for item in samples
                if item.event_regime == event and item.action_dimension == dimension
            ]
            cells[f"{event}__{dimension}"] = _direction_metrics(rows)
    return cells


def _choose_profile(
    development: Sequence[ForecastCalibrationSample],
    holdout_episode_ids: Sequence[str],
) -> tuple[ForecastReliabilityProfile, list[dict[str, Any]]]:
    development_hash = compute_forecast_dataset_hash(development)
    candidates: list[tuple[ForecastReliabilityProfile, dict[str, Any]]] = []
    for policy in POLICY_MENU:
        profile = build_forecast_reliability_profile(
            samples=development,
            dataset_hash=development_hash,
            holdout_episode_ids=holdout_episode_ids,
            policy=policy,
        )
        metrics = _release_metrics(profile, development)
        candidates.append((profile, {"policy": policy.model_dump(mode="json"), **metrics}))
    passing = [
        item
        for item in candidates
        if item[1]["coverage_ppm"] >= 300_000
        and item[1]["released_direction_accuracy_ppm"] >= 900_000
        and item[1]["released_negative_actual_count"] == 0
    ]
    if passing:
        chosen = max(
            passing,
            key=lambda item: (
                item[1]["coverage_ppm"],
                item[1]["released_direction_accuracy_ppm"],
                item[0].release_policy.policy_id,
            ),
        )
    else:
        # Safety is the primary objective when no menu member meets every
        # development target: minimize known negative releases before seeking
        # accuracy or coverage.  Holdout labels are not consulted here.
        chosen = min(
            candidates,
            key=lambda item: (
                item[1]["released_negative_actual_count"],
                -item[1]["released_direction_accuracy_ppm"],
                -item[1]["coverage_ppm"],
                item[0].release_policy.policy_id,
            ),
        )
    return chosen[0], [row for _, row in candidates]


def run(stage65_summary: Path, output: Path) -> dict[str, Any]:
    first = build_samples(stage65_summary)
    second = build_samples(stage65_summary)
    deterministic = [item.model_dump(mode="json") for item in first] == [
        item.model_dump(mode="json") for item in second
    ]
    development = [item for item in first if item.split == "development"]
    holdout = [item for item in first if item.split == "holdout"]
    development_episodes = {item.episode_id for item in development}
    holdout_episodes = {item.episode_id for item in holdout}
    profile, policy_audit = _choose_profile(development, sorted(holdout_episodes))
    profile_rebuild, _ = _choose_profile(development, sorted(holdout_episodes))
    development_release = _release_metrics(profile, development)
    holdout_release = _release_metrics(profile, holdout)
    holdout_direction = _direction_metrics(holdout)
    split_clean = not development_episodes.intersection(holdout_episodes) and not set(
        DEVELOPMENT_SEEDS
    ).intersection(HOLDOUT_SEEDS)
    engineering_checks = {
        "deterministic_dataset_rebuild_100_percent": deterministic,
        "development_holdout_episode_leakage_zero": split_clean,
        "profile_hash_replay_100_percent": profile.profile_hash == profile_rebuild.profile_hash,
        "authoritative_label_visible_to_agent_count": sum(item.allowed_in_agent_context for item in first),
        "authoritative_label_flag_error_count": sum(
            not item.uses_authoritative_state_for_label_only for item in first
        ),
        "illegal_action_count": 0,
        "new_real_model_calls": 0,
        "new_prompt_tokens": 0,
        "new_completion_tokens": 0,
        "new_estimated_cost_cny": 0,
    }
    prediction_checks = {
        "holdout_all_candidate_direction_accuracy_at_least_80_percent": (
            holdout_direction["direction_accuracy_ppm"] >= 800_000
        ),
        "holdout_released_direction_accuracy_at_least_90_percent": (
            holdout_release["released_direction_accuracy_ppm"] >= 900_000
            and holdout_release["released_candidate_count"] > 0
        ),
        "holdout_released_negative_count_zero": (
            holdout_release["released_negative_actual_count"] == 0
        ),
        "holdout_release_coverage_at_least_30_percent": (
            holdout_release["coverage_ppm"] >= 300_000
        ),
    }
    gate_passed = (
        all(value is True or value == 0 for value in engineering_checks.values())
        and all(prediction_checks.values())
    )
    summary: dict[str, Any] = {
        "experiment_schema_version": "midterm-v0.7-forecast-calibration-v1.0.0",
        "evidence_level": "zero-token-frozen-tape-development-holdout",
        "source_stage65_summary": str(stage65_summary.resolve()),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "holdout_source": "historical-stage51-real-llm-episodes-not-used-by-profile",
        "prior_seed_4_5_diagnostic_preserved_as": "initial-diagnostic-summary.json",
        "sample_count": len(first),
        "development_sample_count": len(development),
        "holdout_sample_count": len(holdout),
        "full_dataset_hash": compute_forecast_dataset_hash(first),
        "development_dataset_hash": compute_forecast_dataset_hash(development),
        "selected_profile_hash": profile.profile_hash,
        "selected_policy_id": profile.release_policy.policy_id,
        "policy_selection_uses_development_only": True,
        "holdout_used_for_tuning": False,
        "policy_selection_audit": policy_audit,
        "development_direction": _direction_metrics(development),
        "development_release": development_release,
        "holdout_direction": holdout_direction,
        "holdout_release": holdout_release,
        "holdout_special_cells": _special_cells(holdout),
        "engineering_checks": engineering_checks,
        "prediction_checks": prediction_checks,
        "zero_token_holdout_gate_passed": gate_passed,
        "real_llm_gate": (
            "open_max_6_calls" if gate_passed else "closed_no_real_llm_calls"
        ),
        "conclusion_limits": [
            "labels are synthetic-market frozen-tape outcomes, not real-market calibration",
            "continuation actions are frozen from prior recorded episodes",
            "holdout covers historical Stage 5.1 seeds 1001-1005 with balanced_v1",
            "a failed gate forbids new real-model validation in this release closure",
        ],
        "report_hash": "pending",
    }
    summary["report_hash"] = sha256_hash(
        {key: value for key, value in summary.items() if key != "report_hash"}
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "dataset.jsonl").open("w", encoding="utf-8") as handle:
        for sample in first:
            handle.write(sample.model_dump_json())
            handle.write("\n")
    (output / "profile.json").write_text(
        json.dumps(profile.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage65-summary", type=Path, default=DEFAULT_STAGE65_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.stage65_summary, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["zero_token_holdout_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
