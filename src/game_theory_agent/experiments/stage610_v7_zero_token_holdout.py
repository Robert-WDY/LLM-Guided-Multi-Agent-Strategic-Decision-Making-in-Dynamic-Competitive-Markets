"""Pre-registered zero-token holdout for fail-closed Pareto Advisor v7."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.belief import BeliefLedger
from game_theory_agent.gameplay import build_rule_action, build_terminal_rankings
from game_theory_agent.information import compute_observation_hash
from game_theory_agent.market import CompanyAction, MarketEnv, MarketState, load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.market.replay import MarketTransition, replay
from game_theory_agent.opponent import OpponentModelLedger
from game_theory_agent.strategic_reliability import PublicMarketRolloutAdvisor

from .stage66_failure_forensics import _to_company_action, economic_action
from .stage68_advisor_external_proof import _write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v4.yaml"
DEFAULT_SPEC = (
    PROJECT_ROOT
    / "experiment-specs"
    / "stage6.10-v7-zero-token-holdout"
    / "PREREGISTRATION.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "stage6.10-v7-zero-token-holdout"
REPAIR_COMMIT = "77eb4d565e42fed644b73f05acfc2122633aad52"
SEEDS = tuple(range(260_825_001, 260_825_011))
PERSONA_IDS = (
    "balanced_v1",
    "aggressive_v1_extreme",
    "risk_guarded_v1",
)
COMPANIES = ("company_A", "company_B", "company_C", "company_D")
FOCAL_COMPANY = "company_A"
ROUNDS = 10
HORIZON_ROUNDS = 3
SCENARIO_COUNT = 5
PPM = 1_000_000


def _with_hash(payload: Mapping[str, Any], protocol: str) -> dict[str, Any]:
    result = dict(payload)
    result["preregistration_hash"] = sha256_hash(
        {
            "hash_protocol_version": protocol,
            "preregistration": result,
        }
    )
    return result


def build_preregistration() -> dict[str, Any]:
    return _with_hash(
        {
            "preregistration_schema_version": (
                "stage6.10-v7-zero-token-holdout-v1.0.0"
            ),
            "repair_commit": REPAIR_COMMIT,
            "advisor_mode": "pareto_reliable_v7",
            "evidence_type": "SYNTHETIC_RULE_HOLDOUT",
            "seeds": list(SEEDS),
            "seed_count": len(SEEDS),
            "seed_identifiers_absent_before_preregistration": True,
            "persona_ids": list(PERSONA_IDS),
            "market_model": "balanced",
            "company_ids": list(COMPANIES),
            "focal_company_id": FOCAL_COMPANY,
            "rounds_per_seed": ROUNDS,
            "horizon_rounds": HORIZON_ROUNDS,
            "scenario_count": SCENARIO_COUNT,
            "baseline_policy": "build_rule_action-current-checkout",
            "treatment_policy": (
                "execute v7 recommendation only when disposition=recommend; "
                "otherwise retain the focal rule action"
            ),
            "continuation_policy": (
                "after the first focal action, all companies use the same "
                "deterministic build_rule_action policy for up to three rounds"
            ),
            "primary_outcome": "final_composite_enterprise_value_delta_cents",
            "secondary_outcomes": [
                "counterfactual_regret_reduction_cents",
                "cumulative_profit_delta_cents",
                "recommendation_coverage_ppm",
                "abstention_executable_action_count",
                "hidden_state_leak_count",
            ],
            "engineering_thresholds": {
                "deterministic_advice_rebuild_rate_ppm": PPM,
                "deterministic_market_replay_rate_ppm": PPM,
                "deterministic_observation_hash_rate_ppm": PPM,
                "safe_set_mismatch_count": 0,
                "abstention_executable_action_count": 0,
                "abstention_nonzero_delta_count": 0,
                "illegal_action_count": 0,
                "hidden_state_leak_count": 0,
            },
            "outcome_thresholds": {
                "minimum_recommendation_coverage_ppm": 100_000,
                "released_recommendation_negative_window_count": 0,
                "minimum_mean_ev_delta_cents": 0,
                "minimum_mean_regret_reduction_cents": 0,
                "minimum_worst_seed_persona_mean_ev_delta_cents": 0,
            },
            "real_llm_gate": {
                "requires_all_engineering_and_outcome_checks": True,
                "maximum_calls_if_open": 6,
                "calls_during_this_holdout": 0,
                "paired_design_if_open": "3 new seeds x baseline/treatment",
                "does_not_auto_start_paid_calls": True,
            },
            "research_boundary": (
                "This is a deterministic synthetic Rule holdout. Passing can "
                "open a small paid Smoke gate but cannot establish real-LLM or "
                "real-market effectiveness. Overlapping round windows are not "
                "independent; aggregate conclusions use seed-persona clusters."
            ),
            "forbidden_after_results": [
                "replace or delete a seed after observing its result",
                "change thresholds after observing outcomes",
                "tune Market v4, Persona weights or Advisor v7 on these labels",
                "describe round windows as independent samples",
                "start paid calls when any hard gate fails",
            ],
        },
        "stage6.10-preregistration-hash-v1.0.0",
    )


def write_preregistration(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    payload = build_preregistration()
    _write_json(path, payload)
    return payload


def _validate_preregistration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("run --plan-only and commit the preregistration first")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    expected = build_preregistration()
    if recorded != expected:
        raise ValueError("Stage 6.10 preregistration differs from the frozen plan")
    return recorded


def _terminal_value(state: MarketState, config: Any) -> int:
    return next(
        int(item["value_cents"])
        for item in build_terminal_rankings(state, config)["composite"]
        if item["company_id"] == FOCAL_COMPANY
    )


def _build_rule_episode(
    config: Any,
    seed: int,
    *,
    rounds: int = ROUNDS,
) -> tuple[MarketState, tuple[MarketTransition, ...], bool]:
    env = MarketEnv(config)
    initial = env.reset(
        COMPANIES,
        episode_id=f"stage610-rule-{seed}",
        episode_seed=seed,
        market_model="balanced",
        max_rounds=rounds,
    )
    state = initial
    transitions: list[MarketTransition] = []
    while not state.terminal:
        joint = {
            company_id: build_rule_action(config, state, company_id)
            for company_id in state.company_ids
        }
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", joint
        )
        transitions.append(MarketTransition.create(state, joint, result))
        state = result.state_after
    replayed = replay(
        MarketEnv(config),
        initial,
        [dict(item.joint_action) for item in transitions],
    )
    replay_ok = [item.state_hash for item in replayed[1:]] == [
        item.state_after.state_hash for item in transitions
    ]
    return initial, tuple(transitions), replay_ok


def _simulate_closed_loop(
    *,
    config: Any,
    initial_state: MarketState,
    first_focal_action: Mapping[str, Any] | CompanyAction,
    horizon_rounds: int,
) -> dict[str, int]:
    env = MarketEnv(config)
    env.load_state(initial_state)
    for offset in range(horizon_rounds):
        state = env.get_state()
        joint = {
            company_id: build_rule_action(config, state, company_id)
            for company_id in state.company_ids
        }
        if offset == 0:
            joint[FOCAL_COMPANY] = _to_company_action(
                state,
                FOCAL_COMPANY,
                economic_action(first_focal_action),
                (
                    f"stage610:first:{state.episode_seed}:{state.round}:"
                    f"{sha256_hash(economic_action(first_focal_action))}"
                ),
            )
        for company_id, action in joint.items():
            validation = env.validate_action(action, company_id)
            if not validation.valid:
                raise ValueError(
                    f"illegal closed-loop action for {company_id}: "
                    f"{validation.errors}"
                )
        env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", joint
        )
    final = env.get_state()
    focal = final.company(FOCAL_COMPANY)
    return {
        "enterprise_value_cents": _terminal_value(final, config),
        "cumulative_profit_cents": focal.financial.cumulative_profit_cents,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [int(item["treatment_minus_baseline_ev_cents"]) for item in rows]
    regrets = [int(item["regret_reduction_cents"]) for item in rows]
    profits = [int(item["treatment_minus_baseline_profit_cents"]) for item in rows]
    return {
        "window_count": len(rows),
        "positive_zero_negative_windows": {
            "positive": sum(value > 0 for value in deltas),
            "zero": sum(value == 0 for value in deltas),
            "negative": sum(value < 0 for value in deltas),
        },
        "mean_ev_delta_cents": sum(deltas) // max(1, len(deltas)),
        "mean_regret_reduction_cents": sum(regrets) // max(1, len(regrets)),
        "mean_profit_delta_cents": sum(profits) // max(1, len(profits)),
        "worst_ev_delta_cents": min(deltas),
        "best_ev_delta_cents": max(deltas),
    }


def _grouped(rows: Sequence[Mapping[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    return [
        {**dict(zip(keys, group_key, strict=True)), **_aggregate(groups[group_key])}
        for group_key in sorted(groups)
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")


def run(
    *,
    preregistration_path: Path = DEFAULT_SPEC,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    preregistration = _validate_preregistration(preregistration_path)
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    advisor = PublicMarketRolloutAdvisor(config)
    rows: list[dict[str, Any]] = []
    deterministic_tape_replay_count = 0
    illegal_action_count = 0

    for seed in preregistration["seeds"]:
        _initial, transitions, replay_ok = _build_rule_episode(config, int(seed))
        deterministic_tape_replay_count += int(replay_ok)
        first_state = transitions[0].state_before
        belief_ledger = BeliefLedger(
            episode_id=first_state.episode_id,
            company_ids=first_state.company_ids,
        )
        opponent_ledger = OpponentModelLedger(
            episode_id=first_state.episode_id,
            company_ids=first_state.company_ids,
        )

        for event_index, transition in enumerate(transitions):
            state = transition.state_before
            belief, belief_hash = belief_ledger.company_view(
                observer_company_id=FOCAL_COMPANY,
                round_number=state.round,
                state_version=state.state_version,
            )
            opponent, opponent_hash = opponent_ledger.company_view(
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
            observation_hash = compute_observation_hash(observation)
            observation_rebuild_hash = compute_observation_hash(dict(observation))
            baseline_action = dict(transition.joint_action)[FOCAL_COMPANY]
            horizon = min(HORIZON_ROUNDS, len(transitions) - event_index)

            for persona_id in preregistration["persona_ids"]:
                profile = personas.get(str(persona_id))
                advice = advisor.advise(
                    observation=observation,
                    company_id=FOCAL_COMPANY,
                    persona_profile=profile,
                    belief_state=belief,
                    opponent_model=opponent,
                    horizon_rounds=horizon,
                    scenario_count=SCENARIO_COUNT,
                    advisor_mode="pareto_reliable_v7",
                )
                rebuilt = advisor.advise(
                    observation=observation,
                    company_id=FOCAL_COMPANY,
                    persona_profile=profile,
                    belief_state=belief,
                    opponent_model=opponent,
                    horizon_rounds=horizon,
                    scenario_count=SCENARIO_COUNT,
                    advisor_mode="pareto_reliable_v7",
                )
                gate = advice.reliability_gate or {}
                disposition = str(advice.execution_disposition)
                treatment_action = (
                    economic_action(baseline_action)
                    if disposition == "defer_to_agent"
                    else economic_action(advice.recommended_action or {})
                )
                probe_env = MarketEnv(config)
                probe_env.load_state(state)
                probe_action = _to_company_action(
                    state,
                    FOCAL_COMPANY,
                    treatment_action,
                    f"stage610:validate:{seed}:{state.round}:{persona_id}",
                )
                if not probe_env.validate_action(probe_action, FOCAL_COMPANY).valid:
                    illegal_action_count += 1

                baseline = _simulate_closed_loop(
                    config=config,
                    initial_state=state,
                    first_focal_action=baseline_action,
                    horizon_rounds=horizon,
                )
                treatment = _simulate_closed_loop(
                    config=config,
                    initial_state=state,
                    first_focal_action=treatment_action,
                    horizon_rounds=horizon,
                )
                candidate_values = {
                    item.candidate.candidate_id: _simulate_closed_loop(
                        config=config,
                        initial_state=state,
                        first_focal_action=item.candidate.action,
                        horizon_rounds=horizon,
                    )["enterprise_value_cents"]
                    for item in advice.candidate_actions
                }
                baseline_ev = int(baseline["enterprise_value_cents"])
                treatment_ev = int(treatment["enterprise_value_cents"])
                best_ev = max(baseline_ev, *candidate_values.values())
                eligible = sorted(
                    item["candidate_id"]
                    for item in advice.pareto_decision["candidate_assessments"]
                    if item["eligible"]
                )
                safe_ids = sorted(gate.get("safe_candidate_ids", []))
                rows.append(
                    {
                        "window_id": f"{state.episode_id}:r{state.round}:{persona_id}",
                        "seed": int(seed),
                        "round": state.round,
                        "persona_id": persona_id,
                        "state_hash": state.state_hash,
                        "observation_hash": observation_hash,
                        "observation_hash_rebuild_matches": (
                            observation_hash == observation_rebuild_hash
                        ),
                        "advice_hash": advice.advice_hash,
                        "advice_rebuild_matches": advice == rebuilt,
                        "execution_disposition": disposition,
                        "recommended_candidate_id": advice.recommended_candidate_id,
                        "recommended_action": advice.recommended_action,
                        "planner_candidate_id": gate.get("planner_candidate_id"),
                        "diagnostic_fallback_candidate_id": gate.get(
                            "diagnostic_fallback_candidate_id"
                        ),
                        "effective_candidate_id": gate.get("effective_candidate_id"),
                        "fallback_is_proven_safe": gate.get(
                            "fallback_is_proven_safe"
                        ),
                        "safe_set_matches_true_eligible": safe_ids == eligible,
                        "uses_authoritative_hidden_market_state": (
                            advice.uses_authoritative_hidden_market_state
                        ),
                        "uses_hidden_opponent_state": advice.uses_hidden_opponent_state,
                        "baseline_action": economic_action(baseline_action),
                        "treatment_action": treatment_action,
                        "baseline_enterprise_value_cents": baseline_ev,
                        "treatment_enterprise_value_cents": treatment_ev,
                        "treatment_minus_baseline_ev_cents": (
                            treatment_ev - baseline_ev
                        ),
                        "baseline_cumulative_profit_cents": baseline[
                            "cumulative_profit_cents"
                        ],
                        "treatment_cumulative_profit_cents": treatment[
                            "cumulative_profit_cents"
                        ],
                        "treatment_minus_baseline_profit_cents": (
                            int(treatment["cumulative_profit_cents"])
                            - int(baseline["cumulative_profit_cents"])
                        ),
                        "authoritative_best_candidate_ev_cents": best_ev,
                        "baseline_regret_cents": best_ev - baseline_ev,
                        "treatment_regret_cents": best_ev - treatment_ev,
                        "regret_reduction_cents": treatment_ev - baseline_ev,
                    }
                )

            belief_ledger.update_after_settlement(
                state, dict(transition.joint_action)
            )
            opponent_ledger.update_after_settlement(
                state, transition.state_after, dict(transition.joint_action)
            )

    clusters = _grouped(rows, ("seed", "persona_id"))
    dispositions = Counter(str(item["execution_disposition"]) for item in rows)
    recommendations = [
        item for item in rows if item["execution_disposition"] == "recommend"
    ]
    abstentions = [
        item for item in rows if item["execution_disposition"] == "defer_to_agent"
    ]
    aggregate = _aggregate(rows)
    metrics = {
        "window_count": len(rows),
        "seed_persona_cluster_count": len(clusters),
        "disposition_counts": dict(sorted(dispositions.items())),
        "recommendation_coverage_ppm": len(recommendations) * PPM // len(rows),
        "released_recommendation_negative_window_count": sum(
            int(item["treatment_minus_baseline_ev_cents"]) < 0
            for item in recommendations
        ),
        "abstention_executable_action_count": sum(
            item["recommended_action"] is not None
            or item["effective_candidate_id"] is not None
            for item in abstentions
        ),
        "abstention_nonzero_delta_count": sum(
            int(item["treatment_minus_baseline_ev_cents"]) != 0
            for item in abstentions
        ),
        "safe_set_mismatch_count": sum(
            not item["safe_set_matches_true_eligible"] for item in rows
        ),
        "hidden_state_leak_count": sum(
            item["uses_authoritative_hidden_market_state"]
            or item["uses_hidden_opponent_state"]
            for item in rows
        ),
        "illegal_action_count": illegal_action_count,
        "deterministic_advice_rebuild_rate_ppm": sum(
            item["advice_rebuild_matches"] for item in rows
        )
        * PPM
        // len(rows),
        "deterministic_observation_hash_rate_ppm": sum(
            item["observation_hash_rebuild_matches"] for item in rows
        )
        * PPM
        // len(rows),
        "deterministic_market_replay_rate_ppm": (
            deterministic_tape_replay_count * PPM // len(SEEDS)
        ),
        **aggregate,
        "worst_seed_persona_mean_ev_delta_cents": min(
            int(item["mean_ev_delta_cents"]) for item in clusters
        ),
    }
    engineering_thresholds = preregistration["engineering_thresholds"]
    outcome_thresholds = preregistration["outcome_thresholds"]
    checks = {
        "engineering": {
            "deterministic_advice_rebuild": metrics[
                "deterministic_advice_rebuild_rate_ppm"
            ]
            >= engineering_thresholds["deterministic_advice_rebuild_rate_ppm"],
            "deterministic_market_replay": metrics[
                "deterministic_market_replay_rate_ppm"
            ]
            >= engineering_thresholds["deterministic_market_replay_rate_ppm"],
            "deterministic_observation_hash": metrics[
                "deterministic_observation_hash_rate_ppm"
            ]
            >= engineering_thresholds["deterministic_observation_hash_rate_ppm"],
            "safe_set_exact": metrics["safe_set_mismatch_count"]
            == engineering_thresholds["safe_set_mismatch_count"],
            "abstention_has_no_action": metrics["abstention_executable_action_count"]
            == engineering_thresholds["abstention_executable_action_count"],
            "abstention_retains_agent_outcome": metrics[
                "abstention_nonzero_delta_count"
            ]
            == engineering_thresholds["abstention_nonzero_delta_count"],
            "all_actions_legal": metrics["illegal_action_count"]
            == engineering_thresholds["illegal_action_count"],
            "no_hidden_state_leak": metrics["hidden_state_leak_count"]
            == engineering_thresholds["hidden_state_leak_count"],
        },
        "outcome": {
            "coverage": metrics["recommendation_coverage_ppm"]
            >= outcome_thresholds["minimum_recommendation_coverage_ppm"],
            "no_negative_released_recommendation": metrics[
                "released_recommendation_negative_window_count"
            ]
            == outcome_thresholds[
                "released_recommendation_negative_window_count"
            ],
            "mean_ev_nonnegative": metrics["mean_ev_delta_cents"]
            >= outcome_thresholds["minimum_mean_ev_delta_cents"],
            "mean_regret_nonnegative": metrics["mean_regret_reduction_cents"]
            >= outcome_thresholds["minimum_mean_regret_reduction_cents"],
            "worst_cluster_nonnegative": metrics[
                "worst_seed_persona_mean_ev_delta_cents"
            ]
            >= outcome_thresholds[
                "minimum_worst_seed_persona_mean_ev_delta_cents"
            ],
        },
    }
    engineering_passed = all(checks["engineering"].values())
    outcome_passed = all(checks["outcome"].values())
    summary: dict[str, Any] = {
        "result_schema_version": "stage6.10-v7-zero-token-result-v1.0.0",
        "preregistration_hash": preregistration["preregistration_hash"],
        "evidence_type": preregistration["evidence_type"],
        "metrics": metrics,
        "checks": checks,
        "engineering_passed": engineering_passed,
        "outcome_passed": outcome_passed,
        "real_llm_smoke_gate_open": engineering_passed and outcome_passed,
        "by_seed_persona": clusters,
        "by_persona": _grouped(rows, ("persona_id",)),
        "by_seed": _grouped(rows, ("seed",)),
        "research_boundary": preregistration["research_boundary"],
        "new_real_model_calls": 0,
        "new_prompt_tokens": 0,
        "new_completion_tokens": 0,
        "new_estimated_cost_cny": 0,
        "report_hash": "pending",
    }
    summary["report_hash"] = sha256_hash(
        {key: value for key, value in summary.items() if key != "report_hash"}
    )
    _write_jsonl(output / "decision-windows.jsonl", rows)
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        result = write_preregistration(args.preregistration)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    result = run(preregistration_path=args.preregistration, output=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["real_llm_smoke_gate_open"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
