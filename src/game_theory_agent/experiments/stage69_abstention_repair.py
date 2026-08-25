"""Zero-token regression for v7 fail-closed Advisor abstention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.market import load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.strategic_reliability import PublicMarketRolloutAdvisor

from .stage66_failure_forensics import (
    CONFIG_PATH,
    FOCAL_COMPANY,
    _events,
    economic_action,
    simulate_frozen_tape,
)
from .stage68_advisor_external_proof import (
    DEFAULT_OUTPUT as DEFAULT_AUDIT_OUTPUT,
    DEFAULT_PREREGISTRATION,
    HORIZON_ROUNDS,
    PROJECT_ROOT,
    SCENARIO_COUNT,
    _write_json,
)
from .stage68_failure_forensics import _context, _jsonl


DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "stage6.9-advisor-abstention-regression"


def run(
    audit_output: Path = DEFAULT_AUDIT_OUTPUT,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    preregistration = json.loads(
        preregistration_path.read_text(encoding="utf-8")
    )
    audit_summary = json.loads(
        (audit_output / "summary.json").read_text(encoding="utf-8")
    )
    negative = [
        item
        for item in _jsonl(audit_output / "decision-windows.jsonl")
        if int(item["advisor_minus_historical_ev_cents"]) < 0
    ]
    sources = {
        item["episode_id"]: item for item in preregistration["sources"]
    }
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    advisor = PublicMarketRolloutAdvisor(config)
    rows: list[dict[str, Any]] = []

    for window in sorted(negative, key=lambda item: item["window_id"]):
        source = sources[window["episode_id"]]
        events = _events(PROJECT_ROOT / source["directory"])
        event_index = next(
            index
            for index, item in enumerate(events)
            if int(item.settled_round) == int(window["round"])
        )
        trace, belief, opponent_model, initial_state = _context(
            events, event_index
        )
        horizon = min(HORIZON_ROUNDS, len(events) - event_index)
        v6 = advisor.advise(
            observation=trace.observation,
            company_id=FOCAL_COMPANY,
            persona_profile=personas.get(source["persona_id"]),
            belief_state=belief,
            opponent_model=opponent_model,
            horizon_rounds=horizon,
            scenario_count=SCENARIO_COUNT,
            advisor_mode="pareto_reliable_v6",
        )
        if v6.advice_hash != window["advice_hash"]:
            raise ValueError(
                f"v6 replay mismatch: {window['window_id']} "
                f"expected={window['advice_hash']} actual={v6.advice_hash}"
            )
        v7 = advisor.advise(
            observation=trace.observation,
            company_id=FOCAL_COMPANY,
            persona_profile=personas.get(source["persona_id"]),
            belief_state=belief,
            opponent_model=opponent_model,
            horizon_rounds=horizon,
            scenario_count=SCENARIO_COUNT,
            advisor_mode="pareto_reliable_v7",
        )
        gate = v7.reliability_gate or {}
        if not (
            gate.get("should_abstain")
            and gate.get("execution_disposition") == "defer_to_agent"
            and gate.get("effective_candidate_id") is None
            and v7.recommended_candidate_id is None
            and v7.recommended_action is None
        ):
            raise ValueError(f"v7 failed closed: {window['window_id']}")

        tape = events[event_index : event_index + horizon]
        retained_action = economic_action(window["historical_real_llm_action"])
        retained = simulate_frozen_tape(
            initial_state=initial_state,
            initial_action=retained_action,
            recorded_events=tape,
        )
        retained_ev = int(retained["final_enterprise_value_cents"])
        expected_historical_ev = int(
            window["historical_enterprise_value_cents"]
        )
        if retained_ev != expected_historical_ev:
            raise ValueError(
                f"historical action replay mismatch: {window['window_id']}"
            )
        rows.append(
            {
                "window_id": window["window_id"],
                "episode_id": window["episode_id"],
                "round": window["round"],
                "persona_id": window["persona_id"],
                "v6_effective_candidate_id": v6.recommended_candidate_id,
                "v6_advisor_minus_historical_ev_cents": int(
                    window["advisor_minus_historical_ev_cents"]
                ),
                "v7_execution_disposition": v7.execution_disposition,
                "v7_recommended_candidate_id": v7.recommended_candidate_id,
                "v7_recommended_action": v7.recommended_action,
                "retained_agent_action": retained_action,
                "retained_agent_ev_cents": retained_ev,
                "v7_abstention_delta_vs_retained_agent_cents": 0,
                "prevented_fallback_loss_cents": -int(
                    window["advisor_minus_historical_ev_cents"]
                ),
                "v7_advice_hash": v7.advice_hash,
                "v7_gate_hash": gate["gate_hash"],
            }
        )

    summary: dict[str, Any] = {
        "regression_schema_version": "stage6.9-abstention-regression-v1.0.0",
        "evidence_level": "post-hoc-zero-token-engineering-regression",
        "source_stage68_report_hash": audit_summary["report_hash"],
        "negative_window_count": len(rows),
        "v7_defer_count": sum(
            item["v7_execution_disposition"] == "defer_to_agent"
            for item in rows
        ),
        "v7_executable_recommendation_count": sum(
            item["v7_recommended_action"] is not None for item in rows
        ),
        "v7_abstention_negative_delta_count": sum(
            int(item["v7_abstention_delta_vs_retained_agent_cents"]) < 0
            for item in rows
        ),
        "prevented_fallback_loss_cents": sum(
            int(item["prevented_fallback_loss_cents"]) for item in rows
        ),
        "rows": rows,
        "engineering_passed": (
            len(rows) == 5
            and all(
                item["v7_execution_disposition"] == "defer_to_agent"
                and item["v7_recommended_action"] is None
                and item["v7_abstention_delta_vs_retained_agent_cents"] == 0
                for item in rows
            )
        ),
        "research_boundary": (
            "The five windows were already inspected and are diagnostic regression "
            "cases only. Zero delta means v7 no longer overrides the recorded Agent "
            "action when abstaining; it is not new holdout evidence that v7 improves "
            "future real-LLM outcomes."
        ),
        "new_real_model_calls": 0,
        "new_total_tokens": 0,
        "new_estimated_cost_cny": 0,
        "report_hash": "pending",
    }
    summary["report_hash"] = sha256_hash(
        {key: value for key, value in summary.items() if key != "report_hash"}
    )
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.audit_output, args.preregistration, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["engineering_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
