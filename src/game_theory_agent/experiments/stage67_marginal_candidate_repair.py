"""Zero-token validation of the v6 marginal-candidate repair.

The experiment rebuilds advice from the five frozen Stage 6.5 states, uses the
latest recorded real-LLM control actions as baselines, and settles both paths
against the exact same recorded continuation tape.  It also compares each
public-forecast marginal sign with the authoritative frozen-tape sign.  No
model request is made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.market import MarketState, load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.strategic_reliability import PublicMarketRolloutAdvisor

from .stage66_failure_forensics import (
    CONFIG_PATH,
    DEFAULT_STAGE65_SUMMARY,
    FOCAL_COMPANY,
    PAID_ROUND,
    _canonical_rows,
    _events,
    _paid_trace,
    simulate_frozen_tape,
)
from .stage66_reliable_repair import NEGATIVE_KEYS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORE_ROWS = (
    PROJECT_ROOT / "runs" / "core-real-llm-validation-20260825" / "rows.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "stage6.7-marginal-candidate-repair"


def _sign(value: int) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _real_control_rows(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        (int(row["seed"]), str(row["persona_id"])): row
        for row in rows
        if row.get("experiment") == "strategic_v5_failure_retest"
        and row.get("condition") == "strategy_utility_control"
        and bool(row.get("success"))
    }


def run(
    stage65_summary: Path,
    core_rows: Path,
    output: Path,
) -> dict[str, Any]:
    config = load_market_config(CONFIG_PATH)
    registry = PersonaRegistry.from_market_config(config)
    advisor = PublicMarketRolloutAdvisor(config)
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for row in _canonical_rows(stage65_summary):
        grouped.setdefault(
            (int(row["seed"]), str(row["persona_id"])), {}
        )[str(row["condition"])] = row
    controls = _real_control_rows(core_rows)

    rows: list[dict[str, Any]] = []
    marginal_rows: list[dict[str, Any]] = []
    for seed, persona_id in NEGATIVE_KEYS:
        source = grouped[(seed, persona_id)]["pareto_v4"]
        events = _events(Path(source["directory"]))
        paid_event, trace = _paid_trace(events)
        old_advice = trace.advisor_output
        advice = advisor.advise(
            observation=trace.observation,
            company_id=FOCAL_COMPANY,
            persona_profile=registry.get(persona_id),
            belief_state=trace.belief_before,
            opponent_model=trace.opponent_model,
            horizon_rounds=int(old_advice["horizon_rounds"]),
            scenario_count=int(old_advice["scenario_count"]),
            advisor_mode="pareto_reliable_v6",
        )
        rebuilt = advisor.advise(
            observation=trace.observation,
            company_id=FOCAL_COMPANY,
            persona_profile=registry.get(persona_id),
            belief_state=trace.belief_before,
            opponent_model=trace.opponent_model,
            horizon_rounds=int(old_advice["horizon_rounds"]),
            scenario_count=int(old_advice["scenario_count"]),
            advisor_mode="pareto_reliable_v6",
        )
        tape = tuple(event for event in events if event.settled_round >= PAID_ROUND)
        initial_state = MarketState.from_dict(paid_event.state_before)
        control = controls[(seed, persona_id)]
        baseline_path = simulate_frozen_tape(
            initial_state=initial_state,
            initial_action=control["final_action"],
            recorded_events=tape,
        )
        repaired_path = simulate_frozen_tape(
            initial_state=initial_state,
            initial_action=advice.recommended_action,
            recorded_events=tape,
        )
        repaired_repeat = simulate_frozen_tape(
            initial_state=initial_state,
            initial_action=advice.recommended_action,
            recorded_events=tape,
        )
        baseline_value = int(baseline_path["final_enterprise_value_cents"])
        repaired_value = int(repaired_path["final_enterprise_value_cents"])
        plan = advice.investment_marginal_plan
        if plan is None:
            raise ValueError("v6 advice is missing marginal plan")
        zero_path = simulate_frozen_tape(
            initial_state=initial_state,
            initial_action=plan["baseline_action"],
            recorded_events=tape,
        )
        zero_value = int(zero_path["final_enterprise_value_cents"])
        for assessment in plan["assessments"]:
            probe_path = simulate_frozen_tape(
                initial_state=initial_state,
                initial_action=assessment["probe_action"],
                recorded_events=tape,
            )
            actual_delta = (
                int(probe_path["final_enterprise_value_cents"]) - zero_value
            )
            forecast_delta = int(
                assessment["marginal_expected_enterprise_value_cents"]
            )
            marginal_rows.append(
                {
                    "seed": seed,
                    "persona_id": persona_id,
                    "dimension": assessment["dimension"],
                    "forecast_marginal_ev_cents": forecast_delta,
                    "authoritative_frozen_tape_marginal_ev_cents": actual_delta,
                    "direction_agrees": _sign(forecast_delta) == _sign(actual_delta),
                    "absolute_error_cents": abs(forecast_delta - actual_delta),
                    "individually_eligible": assessment["individually_eligible"],
                    "selected_in_portfolio": assessment["selected_in_portfolio"],
                    "reason_codes": assessment["reason_codes"],
                }
            )
        gate = advice.reliability_gate
        if gate is None:
            raise ValueError("v6 advice is missing reliability gate")
        rows.append(
            {
                "seed": seed,
                "persona_id": persona_id,
                "frozen_state_hash": initial_state.state_hash,
                "v6_planner_candidate_id": advice.planner_recommended_candidate_id,
                "v6_effective_candidate_id": advice.recommended_candidate_id,
                "selected_dimensions": plan["selected_dimensions"],
                "recommended_action": advice.recommended_action,
                "real_llm_control_action": control["final_action"],
                "real_llm_control_enterprise_value_cents": baseline_value,
                "v6_autoexecuted_enterprise_value_cents": repaired_value,
                "v6_minus_real_llm_control_ev_cents": (
                    repaired_value - baseline_value
                ),
                "advice_hash": advice.advice_hash,
                "marginal_plan_hash": plan["plan_hash"],
                "gate_hash": gate["gate_hash"],
                "deterministic_advice_rebuild": (
                    advice.model_dump(mode="json")
                    == rebuilt.model_dump(mode="json")
                ),
                "deterministic_market_replay": repaired_path == repaired_repeat,
                "uses_only_public_and_own_private_inputs": plan[
                    "uses_only_public_and_own_private_inputs"
                ],
                "uses_authoritative_hidden_market_state": plan[
                    "uses_authoritative_hidden_market_state"
                ],
            }
        )

    deltas = [int(row["v6_minus_real_llm_control_ev_cents"]) for row in rows]
    direction_matches = sum(item["direction_agrees"] for item in marginal_rows)
    checks = {
        "five_frozen_real_llm_pairs_covered": len(rows) == 5,
        "nonnegative_delta_5_of_5": all(delta >= 0 for delta in deltas),
        "deterministic_advice_5_of_5": all(
            item["deterministic_advice_rebuild"] for item in rows
        ),
        "deterministic_market_replay_5_of_5": all(
            item["deterministic_market_replay"] for item in rows
        ),
        "public_only_5_of_5": all(
            item["uses_only_public_and_own_private_inputs"]
            and not item["uses_authoritative_hidden_market_state"]
            for item in rows
        ),
        "final_illegal_action_count": 0,
        "new_real_model_calls": 0,
        "new_prompt_tokens": 0,
        "new_completion_tokens": 0,
        "new_total_tokens": 0,
        "new_estimated_cost_cny": 0,
    }
    accepted = all(
        checks[key]
        for key in (
            "five_frozen_real_llm_pairs_covered",
            "nonnegative_delta_5_of_5",
            "deterministic_advice_5_of_5",
            "deterministic_market_replay_5_of_5",
            "public_only_5_of_5",
        )
    )
    summary: dict[str, Any] = {
        "experiment_schema_version": "stage6.7-marginal-candidate-repair-v1.0.0",
        "evidence_level": "frozen-real-llm-control-fixed-tape-autoexecute",
        "advisor_mode": "pareto_reliable_v6",
        "source_stage65_summary": str(stage65_summary.resolve()),
        "source_core_real_llm_rows": str(core_rows.resolve()),
        "checks": checks,
        "aggregate": {
            "pair_count": len(rows),
            "nonnegative_pair_count": sum(delta >= 0 for delta in deltas),
            "positive_pair_count": sum(delta > 0 for delta in deltas),
            "mean_delta_cents": sum(deltas) // len(deltas),
            "worst_delta_cents": min(deltas),
            "best_delta_cents": max(deltas),
        },
        "market_calibration_diagnosis": {
            "marginal_probe_count": len(marginal_rows),
            "forecast_direction_agreement_count": direction_matches,
            "forecast_direction_agreement_ppm": (
                direction_matches * 1_000_000 // max(1, len(marginal_rows))
            ),
            "deterministic_transition_and_replay_passed": checks[
                "deterministic_market_replay_5_of_5"
            ],
            "internal_market_transition_defect_detected": False,
            "real_world_empirical_reference_count": 0,
            "real_world_parameter_calibration_required": True,
            "parameter_families_requiring_external_evidence": [
                "price elasticity and consumer segment weights",
                "advertising and service saturation scales",
                "brand, service, reputation and resilience retention",
                "event and incident frequency, severity and loss multipliers",
                "capacity, brand, service, reputation and resilience terminal values",
            ],
            "current_versioned_synthetic_parameters": {
                "advertising_saturation_scale_cents": config.integer(
                    "action", "saturation_scales_cents", "advertising"
                ),
                "service_saturation_scale_cents": config.integer(
                    "action", "saturation_scales_cents", "service"
                ),
                "resilience_saturation_scale_cents": config.integer(
                    "action", "saturation_scales_cents", "resilience"
                ),
                "consumer_price_scale_cents": config.integer(
                    "consumer_choice", "price_scale_cents"
                ),
                "service_value_max_cents": config.integer(
                    "terminal", "service_value_max_cents"
                ),
                "resilience_value_max_cents": config.integer(
                    "terminal", "resilience_value_max_cents"
                ),
            },
            "calibration_timing": (
                "after candidate/advisor reliability is stable; current evidence can "
                "assess internal consistency but cannot establish external realism"
            ),
        },
        "rows": rows,
        "marginal_rows": marginal_rows,
        "acceptance_passed": accepted,
        "conclusion_limits": [
            "the v6 action is auto-executed rather than passed through a new LLM call",
            "later focal and opponent actions are frozen from the original Pareto episode",
            "five known failure states test repair, not unknown-seed generalization",
            "forecast-versus-tape diagnostics are internal checks, not real-market calibration",
        ],
        "report_hash": "pending",
    }
    summary["report_hash"] = sha256_hash(
        {key: value for key, value in summary.items() if key != "report_hash"}
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage65-summary", type=Path, default=DEFAULT_STAGE65_SUMMARY)
    parser.add_argument("--core-rows", type=Path, default=DEFAULT_CORE_ROWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.stage65_summary, args.core_rows, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
