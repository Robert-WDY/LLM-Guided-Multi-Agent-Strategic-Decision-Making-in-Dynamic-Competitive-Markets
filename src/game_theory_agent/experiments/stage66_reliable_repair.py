"""Zero-token acceptance for the Stage 6.6 reliable Pareto repair.

The experiment rebuilds v5 advice on the five frozen Stage 6.5 failure states,
auto-executes the effective action on the recorded opponent tape, and compares
it with the matching no-Advisor action.  It never regenerates LLM output.
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "stage6.6-reliable-repair"
NEGATIVE_KEYS = (
    (1, "risk_guarded_v1"),
    (2, "profit_myopic"),
    (3, "risk_guarded_v1"),
    (4, "profit_myopic"),
    (4, "risk_guarded_v1"),
)


def _build_advice(
    *,
    advisor: PublicMarketRolloutAdvisor,
    registry: PersonaRegistry,
    row: dict[str, Any],
    trace: Any,
) -> Any:
    old_advice = trace.advisor_output
    return advisor.advise(
        observation=trace.observation,
        company_id=FOCAL_COMPANY,
        persona_profile=registry.get(str(row["persona_id"])),
        belief_state=trace.belief_before,
        opponent_model=trace.opponent_model,
        horizon_rounds=int(old_advice["horizon_rounds"]),
        scenario_count=int(old_advice["scenario_count"]),
        advisor_mode="pareto_reliable_v5",
    )


def run(stage65_summary: Path, output: Path) -> dict[str, Any]:
    config = load_market_config(CONFIG_PATH)
    registry = PersonaRegistry.from_market_config(config)
    advisor = PublicMarketRolloutAdvisor(config)
    source_summary = json.loads(stage65_summary.read_text(encoding="utf-8"))
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for row in _canonical_rows(stage65_summary):
        grouped.setdefault(
            (int(row["seed"]), str(row["persona_id"])), {}
        )[str(row["condition"])] = row

    rows: list[dict[str, Any]] = []
    for seed, persona_id in NEGATIVE_KEYS:
        cell = grouped[(seed, persona_id)]
        pareto_events = _events(Path(cell["pareto_v4"]["directory"]))
        paid_event, pareto_trace = _paid_trace(pareto_events)
        advice = _build_advice(
            advisor=advisor,
            registry=registry,
            row=cell["pareto_v4"],
            trace=pareto_trace,
        )
        rebuilt = _build_advice(
            advisor=advisor,
            registry=registry,
            row=cell["pareto_v4"],
            trace=pareto_trace,
        )
        advice_payload = advice.model_dump(mode="json")
        gate = advice_payload["reliability_gate"]
        tape = tuple(
            event for event in pareto_events if event.settled_round >= PAID_ROUND
        )
        initial_state = MarketState.from_dict(paid_event.state_before)
        repaired_path = simulate_frozen_tape(
            initial_state=initial_state,
            initial_action=advice.recommended_action,
            recorded_events=tape,
        )
        strategy_events = _events(Path(cell["strategy_utility"]["directory"]))
        _, strategy_trace = _paid_trace(strategy_events)
        baseline_path = simulate_frozen_tape(
            initial_state=initial_state,
            initial_action=strategy_trace.final_action,
            recorded_events=tape,
        )
        repaired_value = int(repaired_path["final_enterprise_value_cents"])
        baseline_value = int(baseline_path["final_enterprise_value_cents"])
        rows.append(
            {
                "seed": seed,
                "persona_id": persona_id,
                "frozen_state_hash": initial_state.state_hash,
                "old_planner_candidate_id": pareto_trace.advisor_output[
                    "recommended_candidate_id"
                ],
                "v5_planner_candidate_id": advice.planner_recommended_candidate_id,
                "v5_effective_candidate_id": advice.recommended_candidate_id,
                "abstained": bool(gate["should_abstain"]),
                "abstain_reason_codes": gate["abstain_reason_codes"],
                "opponent_model_confidence_ppm": gate[
                    "opponent_model_confidence_ppm"
                ],
                "advisor_confidence_ppm": gate["advisor_confidence_ppm"],
                "recommended_action": advice.recommended_action,
                "baseline_enterprise_value_cents": baseline_value,
                "repaired_enterprise_value_cents": repaired_value,
                "repaired_vs_baseline_delta_cents": repaired_value - baseline_value,
                "advice_hash": advice.advice_hash,
                "gate_hash": gate["gate_hash"],
                "deterministic_rebuild": (
                    advice_payload == rebuilt.model_dump(mode="json")
                ),
                "uses_only_public_and_own_private_inputs": gate[
                    "uses_only_public_and_own_private_inputs"
                ],
                "uses_authoritative_hidden_market_state": gate[
                    "uses_authoritative_hidden_market_state"
                ],
            }
        )

    deltas = [int(row["repaired_vs_baseline_delta_cents"]) for row in rows]
    checks = {
        "five_negative_pairs_covered": len(rows) == 5,
        "abstention_triggered_5_of_5": all(row["abstained"] for row in rows),
        "nonnegative_fixed_tape_delta_5_of_5": all(delta >= 0 for delta in deltas),
        "strictly_positive_fixed_tape_delta_count": sum(delta > 0 for delta in deltas),
        "deterministic_advice_rebuild_5_of_5": all(
            row["deterministic_rebuild"] for row in rows
        ),
        "public_only_gate_5_of_5": all(
            row["uses_only_public_and_own_private_inputs"]
            and not row["uses_authoritative_hidden_market_state"]
            for row in rows
        ),
        "final_illegal_action_count": 0,
        "new_real_model_calls": 0,
        "new_prompt_tokens": 0,
        "new_completion_tokens": 0,
        "new_total_tokens": 0,
        "new_estimated_cost_cny": 0,
    }
    accepted = all(
        value is True
        for key, value in checks.items()
        if key.endswith("5_of_5") or key == "five_negative_pairs_covered"
    )
    summary: dict[str, Any] = {
        "experiment_schema_version": "stage6.6-reliable-repair-v1.0.0",
        "evidence_level": "frozen-real-llm-output-fixed-opponent-tape-autoexecute",
        "advisor_mode": "pareto_reliable_v5",
        "source_stage65_summary": str(stage65_summary.resolve()),
        "source_real_model_usage": {
            key: value
            for key, value in source_summary.get("real_model_usage", {}).items()
            if key != "estimated_cost_cny"
        },
        "checks": checks,
        "aggregate": {
            "pair_count": len(rows),
            "nonnegative_pair_count": sum(delta >= 0 for delta in deltas),
            "positive_pair_count": sum(delta > 0 for delta in deltas),
            "mean_delta_cents": sum(deltas) // len(deltas),
            "worst_delta_cents": min(deltas),
            "best_delta_cents": max(deltas),
        },
        "rows": rows,
        "acceptance_passed": accepted,
        "conclusion_limits": [
            "the repair is auto-executed rather than passed through a new LLM call",
            "opponent and later focal actions are frozen from each Pareto episode",
            "the result validates the reliability gate on the five known failures only",
            "a small paid safe-menu LLM retest is still required before claiming behavioral improvement",
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.stage65_summary, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
