"""Develop and hold out a zero-token v8 execution gate over Pareto v7 advice."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.belief import BeliefLedger
from game_theory_agent.experiments.stage610_v7_zero_token_holdout import (
    _build_rule_episode,
    _simulate_closed_loop,
)
from game_theory_agent.experiments.stage66_failure_forensics import economic_action
from game_theory_agent.information import compute_observation_hash
from game_theory_agent.market import load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.opponent import OpponentModelLedger
from game_theory_agent.strategic_reliability import PublicMarketRolloutAdvisor


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v4.yaml"
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "stage8-advisor-v8-zero-token"
DEVELOPMENT_SEEDS = tuple(range(260_902_001, 260_902_006))
HOLDOUT_SEEDS = tuple(range(260_903_001, 260_903_011))
PERSONA_IDS = (
    "balanced_v1",
    "aggressive_v1_extreme",
    "risk_guarded_v1",
)
FOCAL_COMPANY = "company_A"
ROUNDS = 20
HORIZON = 3
SCENARIOS = 5
PPM = 1_000_000


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def build_candidate_rows(seeds: Sequence[int]) -> list[dict[str, Any]]:
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    advisor = PublicMarketRolloutAdvisor(config)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        _initial, transitions, replay_ok = _build_rule_episode(
            config, int(seed), rounds=ROUNDS
        )
        if not replay_ok:
            raise RuntimeError("rule episode replay failed")
        first = transitions[0].state_before
        beliefs = BeliefLedger(
            episode_id=first.episode_id, company_ids=first.company_ids
        )
        opponents = OpponentModelLedger(
            episode_id=first.episode_id, company_ids=first.company_ids
        )
        for index, transition in enumerate(transitions):
            state = transition.state_before
            belief, belief_hash = beliefs.company_view(
                observer_company_id=FOCAL_COMPANY,
                round_number=state.round,
                state_version=state.state_version,
            )
            opponent, opponent_hash = opponents.company_view(
                observer_company_id=FOCAL_COMPANY,
                round_number=state.round,
                state_version=state.state_version,
            )
            observation = ObservationBuilder().build(
                state,
                FOCAL_COMPANY,
                "public",
                belief_state=belief.model_dump(mode="json"),
                belief_hash=belief_hash,
                belief_schema_version=belief.belief_schema_version,
            )
            observation["opponent_model_hash"] = opponent_hash
            observation["opponent_model_state"] = opponent.model_dump(mode="json")
            observation["observation_hash"] = compute_observation_hash(observation)
            baseline_action = dict(transition.joint_action)[FOCAL_COMPANY]
            horizon = min(HORIZON, len(transitions) - index)
            baseline = _simulate_closed_loop(
                config=config,
                initial_state=state,
                first_focal_action=baseline_action,
                horizon_rounds=horizon,
            )
            for persona_id in PERSONA_IDS:
                advice = advisor.advise(
                    observation=observation,
                    company_id=FOCAL_COMPANY,
                    persona_profile=personas.get(persona_id),
                    belief_state=belief,
                    opponent_model=opponent,
                    horizon_rounds=horizon,
                    scenario_count=SCENARIOS,
                    advisor_mode="pareto_reliable_v7",
                )
                gate = dict(advice.reliability_gate or {})
                planner_id = str(gate["planner_candidate_id"])
                planner = next(
                    item
                    for item in advice.candidate_actions
                    if item.candidate.candidate_id == planner_id
                )
                maintain = next(
                    item
                    for item in advice.candidate_actions
                    if item.candidate.candidate_id == "maintain"
                )
                planner_action = planner.candidate.action.model_dump(mode="json")
                outcome = _simulate_closed_loop(
                    config=config,
                    initial_state=state,
                    first_focal_action=planner_action,
                    horizon_rounds=horizon,
                )
                uncertainty = int(gate["forecast_uncertainty_cents"])
                gap = int(gate["top_two_value_gap_cents"])
                row = {
                    "seed": int(seed),
                    "round": state.round,
                    "persona_id": persona_id,
                    "horizon_rounds": horizon,
                    "state_hash": state.state_hash,
                    "observation_hash": observation["observation_hash"],
                    "advice_hash": advice.advice_hash,
                    "v7_execution_disposition": advice.execution_disposition,
                    "v7_reason_codes": list(gate["abstain_reason_codes"]),
                    "planner_candidate_id": planner_id,
                    "planner_action": planner_action,
                    "baseline_action": economic_action(baseline_action),
                    "action_changed": (
                        economic_action(planner_action)
                        != economic_action(baseline_action)
                    ),
                    "opponent_confidence_ppm": int(
                        gate["opponent_model_confidence_ppm"]
                    ),
                    "top_gap_ratio_ppm": gap * PPM // max(1, uncertainty),
                    "forecast_uncertainty_cents": uncertainty,
                    "predicted_gain_over_maintain_cents": (
                        planner.certainty_equivalent_value_cents
                        - maintain.certainty_equivalent_value_cents
                    ),
                    "passes_fallback_value_floor": bool(
                        gate["selected_passes_fallback_value_floor"]
                    ),
                    "passes_fallback_worst_floor": bool(
                        gate["selected_passes_fallback_worst_floor"]
                    ),
                    "passes_fallback_persona_floor": bool(
                        gate["selected_passes_fallback_persona_floor"]
                    ),
                    "baseline_enterprise_value_cents": int(
                        baseline["enterprise_value_cents"]
                    ),
                    "planner_enterprise_value_cents": int(
                        outcome["enterprise_value_cents"]
                    ),
                    "planner_minus_baseline_ev_cents": int(
                        outcome["enterprise_value_cents"]
                    )
                    - int(baseline["enterprise_value_cents"]),
                    "uses_hidden_state": bool(
                        advice.uses_authoritative_hidden_market_state
                        or advice.uses_hidden_opponent_state
                    ),
                }
                rows.append(row)
            beliefs.update_after_settlement(state, dict(transition.joint_action))
            opponents.update_after_settlement(
                state, transition.state_after, dict(transition.joint_action)
            )
    return rows


def _policy_applies(row: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    if int(row["horizon_rounds"]) < int(policy["minimum_horizon_rounds"]):
        return False
    if not bool(row["action_changed"]):
        return False
    if int(row["opponent_confidence_ppm"]) < int(
        policy["minimum_opponent_confidence_ppm"]
    ):
        return False
    if int(row["top_gap_ratio_ppm"]) < int(policy["minimum_top_gap_ratio_ppm"]):
        return False
    if int(row["predicted_gain_over_maintain_cents"]) < int(
        policy["minimum_predicted_gain_cents"]
    ):
        return False
    return all(
        bool(row[field])
        for field in (
            "passes_fallback_value_floor",
            "passes_fallback_worst_floor",
            "passes_fallback_persona_floor",
        )
    )


def evaluate_policy(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    released = [row for row in rows if _policy_applies(row, policy)]
    deltas = [int(row["planner_minus_baseline_ev_cents"]) for row in released]
    clusters: dict[tuple[int, str], list[int]] = defaultdict(list)
    for row in rows:
        value = (
            int(row["planner_minus_baseline_ev_cents"])
            if _policy_applies(row, policy)
            else 0
        )
        clusters[(int(row["seed"]), str(row["persona_id"]))].append(value)
    return {
        "window_count": len(rows),
        "released_count": len(released),
        "coverage_ppm": len(released) * PPM // max(1, len(rows)),
        "positive_zero_negative_released": {
            "positive": sum(value > 0 for value in deltas),
            "zero": sum(value == 0 for value in deltas),
            "negative": sum(value < 0 for value in deltas),
        },
        "mean_released_ev_delta_cents": (
            sum(deltas) // len(deltas) if deltas else 0
        ),
        "worst_released_ev_delta_cents": min(deltas) if deltas else 0,
        "mean_all_window_ev_delta_cents": (
            sum(deltas) // max(1, len(rows))
        ),
        "worst_seed_persona_mean_ev_delta_cents": min(
            sum(values) // len(values) for values in clusters.values()
        ),
        "hidden_state_leak_count": sum(bool(row["uses_hidden_state"]) for row in rows),
    }


def select_development_policy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for confidence in range(500_000, 850_001, 50_000):
        for gap_ratio in (0, 100_000, 250_000, 500_000, 750_000):
            for predicted_gain in (0, 100_000, 250_000, 500_000, 1_000_000):
                policy = {
                    "policy_schema_version": "pareto-v8-execution-gate-v1.0.0",
                    "minimum_horizon_rounds": 3,
                    "minimum_opponent_confidence_ppm": confidence,
                    "minimum_top_gap_ratio_ppm": gap_ratio,
                    "minimum_predicted_gain_cents": predicted_gain,
                    "requires_all_fallback_floors": True,
                    "requires_action_change": True,
                }
                metrics = evaluate_policy(rows, policy)
                if (
                    metrics["positive_zero_negative_released"]["negative"] == 0
                    and metrics["worst_seed_persona_mean_ev_delta_cents"] >= 0
                ):
                    candidates.append({"policy": policy, "metrics": metrics})
    if not candidates:
        raise RuntimeError("no development policy removes negative released windows")
    selected = max(
        candidates,
        key=lambda item: (
            int(item["metrics"]["coverage_ppm"]),
            int(item["metrics"]["mean_all_window_ev_delta_cents"]),
            -int(item["policy"]["minimum_opponent_confidence_ppm"]),
        ),
    )
    policy = dict(selected["policy"])
    policy["development_seeds"] = list(DEVELOPMENT_SEEDS)
    policy["selection_rule"] = (
        "maximize coverage among grid policies with zero negative released windows "
        "and nonnegative worst seed-persona mean; break ties by mean all-window EV"
    )
    policy["policy_hash"] = sha256_hash(
        {key: value for key, value in policy.items() if key != "policy_hash"}
    )
    return {"policy": policy, "development_metrics": selected["metrics"]}


def run_development(output: Path) -> dict[str, Any]:
    rows = build_candidate_rows(DEVELOPMENT_SEEDS)
    selected = select_development_policy(rows)
    summary = {
        "schema_version": "stage8-v8-development-v1.0.0",
        "evidence_type": "post_hoc_development_zero_token",
        "seeds": list(DEVELOPMENT_SEEDS),
        "rounds": ROUNDS,
        **selected,
        "minimum_holdout_coverage_ppm": 100_000,
        "new_real_model_calls": 0,
        "report_hash": "pending",
    }
    summary["report_hash"] = sha256_hash(
        {key: value for key, value in summary.items() if key != "report_hash"}
    )
    _write_jsonl(output / "candidate-windows.jsonl", rows)
    _write_json(output / "summary.json", summary)
    _write_json(output / "FROZEN_V8_POLICY.json", summary["policy"])
    return summary


def _load_frozen_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    recorded = policy.pop("policy_hash")
    calculated = sha256_hash(policy)
    policy["policy_hash"] = recorded
    if calculated != recorded:
        raise ValueError("frozen v8 policy hash mismatch")
    return policy


def run_holdout(policy_path: Path, output: Path) -> dict[str, Any]:
    policy = _load_frozen_policy(policy_path)
    if set(policy["development_seeds"]) & set(HOLDOUT_SEEDS):
        raise ValueError("holdout seeds overlap development seeds")
    rows = build_candidate_rows(HOLDOUT_SEEDS)
    metrics = evaluate_policy(rows, policy)
    checks = {
        "coverage": metrics["coverage_ppm"] >= 100_000,
        "no_negative_released": (
            metrics["positive_zero_negative_released"]["negative"] == 0
        ),
        "worst_released_nonnegative": metrics["worst_released_ev_delta_cents"] >= 0,
        "mean_all_windows_nonnegative": metrics["mean_all_window_ev_delta_cents"] >= 0,
        "worst_cluster_nonnegative": (
            metrics["worst_seed_persona_mean_ev_delta_cents"] >= 0
        ),
        "no_hidden_state_leak": metrics["hidden_state_leak_count"] == 0,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": "stage8-v8-holdout-v1.0.0",
        "evidence_type": "pre_frozen_synthetic_rule_holdout",
        "policy": policy,
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "rounds": ROUNDS,
        "metrics": metrics,
        "checks": checks,
        "holdout_passed": passed,
        "real_llm_gate_open": passed,
        "new_real_model_calls": 0,
        "report_hash": "pending",
    }
    summary["report_hash"] = sha256_hash(
        {key: value for key, value in summary.items() if key != "report_hash"}
    )
    _write_jsonl(output / "candidate-windows.jsonl", rows)
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "holdout"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    if args.phase == "development":
        result = run_development(args.output)
    else:
        if args.policy is None:
            raise ValueError("--policy is required for holdout")
        result = run_holdout(args.policy, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
