"""Zero-token forensic decomposition of Stage 6.8 negative outcome windows."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.belief import BeliefLedger
from game_theory_agent.market import MarketState, load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.opponent import OpponentModelLedger
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


DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "stage6.8-advisor-failure-forensics"
ACTION_FIELDS = (
    "price_cents",
    "advertising_budget_cents",
    "service_budget_cents",
    "capacity_investment_cents",
    "resilience_budget_cents",
    "shared_resilience_contribution_cents",
    "incident_response_mode",
    "repair_budget_cents",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _context(
    events: tuple[Any, ...], event_index: int
) -> tuple[Any, Any, Any, MarketState]:
    initial = MarketState.from_dict(events[0].state_before)
    belief_ledger = BeliefLedger(
        episode_id=initial.episode_id,
        company_ids=initial.company_ids,
    )
    opponent_ledger = OpponentModelLedger(
        episode_id=initial.episode_id,
        company_ids=initial.company_ids,
    )
    for prior in events[:event_index]:
        before = MarketState.from_dict(prior.state_before)
        after = MarketState.from_dict(prior.state_after)
        belief_ledger.update_after_settlement(before, prior.joint_action)
        opponent_ledger.update_after_settlement(
            before, after, prior.joint_action
        )
    event = events[event_index]
    state = MarketState.from_dict(event.state_before)
    belief, _ = belief_ledger.company_view(
        observer_company_id=FOCAL_COMPANY,
        round_number=state.round,
        state_version=state.state_version,
    )
    opponent_model, _ = opponent_ledger.company_view(
        observer_company_id=FOCAL_COMPANY,
        round_number=state.round,
        state_version=state.state_version,
    )
    trace = next(
        item for item in event.traces if item.company_id == FOCAL_COMPANY
    )
    return trace, belief, opponent_model, state


def _shapley_decomposition(
    *,
    historical_action: Mapping[str, Any],
    advisor_action: Mapping[str, Any],
    initial_state: MarketState,
    tape: tuple[Any, ...],
) -> tuple[list[dict[str, Any]], int]:
    historical = economic_action(historical_action)
    advised = economic_action(advisor_action)
    changed = [field for field in ACTION_FIELDS if historical[field] != advised[field]]
    count = len(changed)
    cache: dict[tuple[str, ...], int] = {}

    def value(fields: tuple[str, ...]) -> int:
        key = tuple(sorted(fields))
        if key not in cache:
            action = dict(historical)
            for field in key:
                action[field] = advised[field]
            cache[key] = int(
                simulate_frozen_tape(
                    initial_state=initial_state,
                    initial_action=action,
                    recorded_events=tape,
                )["final_enterprise_value_cents"]
            )
        return cache[key]

    contributions: list[dict[str, Any]] = []
    for field in changed:
        others = [item for item in changed if item != field]
        contribution = Fraction(0, 1)
        for size in range(len(others) + 1):
            weight = Fraction(
                math.factorial(size)
                * math.factorial(count - size - 1),
                math.factorial(count),
            )
            for subset in combinations(others, size):
                contribution += weight * (
                    value((*subset, field)) - value(tuple(subset))
                )
        contributions.append(
            {
                "dimension": field,
                "historical_value": historical[field],
                "advisor_value": advised[field],
                "shapley_ev_contribution_numerator": contribution.numerator,
                "shapley_ev_contribution_denominator": contribution.denominator,
                "shapley_ev_contribution_cents_rounded": round(float(contribution)),
            }
        )
    total_delta = value(tuple(changed)) - value(())
    return contributions, total_delta


def run(
    audit_output: Path,
    preregistration_path: Path,
    output: Path,
) -> dict[str, Any]:
    preregistration = json.loads(
        preregistration_path.read_text(encoding="utf-8")
    )
    audit_summary = json.loads(
        (audit_output / "summary.json").read_text(encoding="utf-8")
    )
    windows = _jsonl(audit_output / "decision-windows.jsonl")
    samples = {
        item["sample_id"]: item
        for item in _jsonl(audit_output / "dataset.jsonl")
    }
    release_decisions = {
        item["sample_id"]: item
        for item in _jsonl(audit_output / "release-decisions.jsonl")
    }
    negative = [
        item
        for item in windows
        if int(item["advisor_minus_historical_ev_cents"]) < 0
    ]
    sources = {
        item["episode_id"]: item for item in preregistration["sources"]
    }
    config = load_market_config(CONFIG_PATH)
    personas = PersonaRegistry.from_market_config(config)
    advisor = PublicMarketRolloutAdvisor(config)
    rows: list[dict[str, Any]] = []

    for window in sorted(
        negative, key=lambda item: int(item["advisor_minus_historical_ev_cents"])
    ):
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
        advice = advisor.advise(
            observation=trace.observation,
            company_id=FOCAL_COMPANY,
            persona_profile=personas.get(source["persona_id"]),
            belief_state=belief,
            opponent_model=opponent_model,
            horizon_rounds=horizon,
            scenario_count=SCENARIO_COUNT,
            advisor_mode="pareto_reliable_v6",
        )
        if advice.advice_hash != window["advice_hash"]:
            raise ValueError(f"advice replay mismatch: {window['window_id']}")
        tape = events[event_index : event_index + horizon]
        candidate_rows: list[dict[str, Any]] = []
        for evaluation in advice.candidate_actions:
            settled = simulate_frozen_tape(
                initial_state=initial_state,
                initial_action=evaluation.candidate.action.model_dump(mode="json"),
                recorded_events=tape,
            )
            candidate_rows.append(
                {
                    "candidate_id": evaluation.candidate.candidate_id,
                    "action": economic_action(evaluation.candidate.action),
                    "public_expected_ev_delta_cents": (
                        evaluation.expected_enterprise_value_delta_cents
                    ),
                    "public_certainty_equivalent_value_cents": (
                        evaluation.certainty_equivalent_value_cents
                    ),
                    "authoritative_frozen_tape_ev_cents": int(
                        settled["final_enterprise_value_cents"]
                    ),
                    "selected": evaluation.candidate.candidate_id
                    == advice.recommended_candidate_id,
                }
            )
        predicted = sorted(
            candidate_rows,
            key=lambda item: (
                -int(item["public_certainty_equivalent_value_cents"]),
                str(item["candidate_id"]),
            ),
        )
        actual = sorted(
            candidate_rows,
            key=lambda item: (
                -int(item["authoritative_frozen_tape_ev_cents"]),
                str(item["candidate_id"]),
            ),
        )
        predicted_rank = {
            item["candidate_id"]: index
            for index, item in enumerate(predicted, start=1)
        }
        actual_rank = {
            item["candidate_id"]: index
            for index, item in enumerate(actual, start=1)
        }
        for item in candidate_rows:
            item["public_ce_rank"] = predicted_rank[item["candidate_id"]]
            item["authoritative_ev_rank"] = actual_rank[item["candidate_id"]]

        shapley, reconstructed_delta = _shapley_decomposition(
            historical_action=window["historical_real_llm_action"],
            advisor_action=window["v6_recommended_action"],
            initial_state=initial_state,
            tape=tape,
        )
        prefix = f"{window['episode_id']}:r{window['round']}:"
        marginal_rows = []
        for sample_id, sample in samples.items():
            if not sample_id.startswith(prefix):
                continue
            decision = release_decisions[sample_id]
            marginal_rows.append(
                {
                    "dimension": sample["action_dimension"],
                    "forecast_delta_ev_cents": sample["forecast_delta_ev_cents"],
                    "authoritative_delta_ev_cents": sample[
                        "authoritative_delta_ev_cents"
                    ],
                    "released": decision["released"],
                    "release_reason_codes": decision["reason_codes"],
                }
            )
        historical_action = economic_action(
            window["historical_real_llm_action"]
        )
        historical_in_candidate_set = any(
            item["action"] == historical_action for item in candidate_rows
        )
        selected_id = advice.recommended_candidate_id
        rows.append(
            {
                "window_id": window["window_id"],
                "seed": window["seed"],
                "persona_id": window["persona_id"],
                "advisor_minus_historical_ev_cents": window[
                    "advisor_minus_historical_ev_cents"
                ],
                "selected_candidate_id": selected_id,
                "planner_recommended_candidate_id": (
                    advice.planner_recommended_candidate_id
                ),
                "recommendation_reason": advice.recommendation_reason,
                "selection_situation": advice.selection_situation,
                "safe_candidate_ids": advice.safe_candidate_ids,
                "reliability_gate": advice.reliability_gate,
                "excluded_candidates": advice.excluded_candidates,
                "selected_public_ce_rank": predicted_rank[selected_id],
                "selected_authoritative_ev_rank": actual_rank[selected_id],
                "candidate_count": len(candidate_rows),
                "historical_action_in_candidate_set": historical_in_candidate_set,
                "historical_action": historical_action,
                "advisor_action": economic_action(window["v6_recommended_action"]),
                "shapley_decomposition": shapley,
                "shapley_reconstructed_total_delta_cents": reconstructed_delta,
                "marginal_forecast_release_rows": sorted(
                    marginal_rows, key=lambda item: item["dimension"]
                ),
                "candidate_rank_rows": sorted(
                    candidate_rows,
                    key=lambda item: item["authoritative_ev_rank"],
                ),
            }
        )

    dimension_totals: dict[str, int] = defaultdict(int)
    dimension_occurrences: Counter[str] = Counter()
    for row in rows:
        for item in row["shapley_decomposition"]:
            dimension = str(item["dimension"])
            dimension_totals[dimension] += int(
                item["shapley_ev_contribution_cents_rounded"]
            )
            dimension_occurrences[dimension] += 1
    summary: dict[str, Any] = {
        "forensic_schema_version": "stage6.8-failure-forensics-v1.0.0",
        "evidence_level": "post-gate-zero-token-diagnostic",
        "source_audit_report_hash": audit_summary["report_hash"],
        "negative_window_count": len(rows),
        "selected_candidate_counts": dict(
            sorted(Counter(row["selected_candidate_id"] for row in rows).items())
        ),
        "historical_action_absent_from_candidate_set_count": sum(
            not row["historical_action_in_candidate_set"] for row in rows
        ),
        "selected_candidate_authoritative_rank_worse_than_one_count": sum(
            int(row["selected_authoritative_ev_rank"]) > 1 for row in rows
        ),
        "shapley_dimension_totals_cents": dict(sorted(dimension_totals.items())),
        "shapley_dimension_occurrences": dict(sorted(dimension_occurrences.items())),
        "rows": rows,
        "diagnosis_boundary": (
            "post-hoc forensic explanation only; it does not reopen the paid gate "
            "and must not be used to alter the pre-registered Stage 6.8 result"
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
    parser.add_argument(
        "--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT
    )
    parser.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.audit_output, args.preregistration, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
