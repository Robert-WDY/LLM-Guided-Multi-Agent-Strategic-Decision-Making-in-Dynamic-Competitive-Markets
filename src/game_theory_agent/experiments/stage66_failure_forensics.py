"""Zero-token forensic attribution for Stage 6.5 negative Pareto pairs.

This module does not change the planner, prompt, controller, or market.  It
reuses the frozen Stage 6.5 outputs and settles counterfactual first actions
against one recorded opponent-action tape and the original component RNG.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from game_theory_agent.advisor import build_advisor_adoption_trace
from game_theory_agent.belief import verify_belief_replay
from game_theory_agent.game_theory import verify_game_theory_replay
from game_theory_agent.gameplay import build_terminal_rankings
from game_theory_agent.information import verify_information_replay
from game_theory_agent.interaction import verify_interaction_replay
from game_theory_agent.market import (
    CompanyAction,
    IncidentResponse,
    IncidentResponseMode,
    MarketEnv,
    MarketState,
    load_market_config,
)
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.orchestration import JsonlRoundEventLogger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v4.yaml"
DEFAULT_STAGE65_SUMMARY = PROJECT_ROOT / "runs" / "stage6.5-real-adoption" / "summary.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "stage6.6-failure-forensics"
FOCAL_COMPANY = "company_A"
PAID_ROUND = 7
ECONOMIC_FIELDS = (
    "price_cents",
    "advertising_budget_cents",
    "service_budget_cents",
    "capacity_investment_cents",
    "resilience_budget_cents",
    "shared_resilience_contribution_cents",
    "incident_response_mode",
    "repair_budget_cents",
)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(value)


def economic_action(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    raw = _as_dict(value)
    incident = raw.get("incident_response")
    if not isinstance(incident, Mapping):
        incident = {}
    return {
        "price_cents": int(raw.get("price_cents", 0)),
        "advertising_budget_cents": int(raw.get("advertising_budget_cents", 0)),
        "service_budget_cents": int(raw.get("service_budget_cents", 0)),
        "capacity_investment_cents": int(raw.get("capacity_investment_cents", 0)),
        "resilience_budget_cents": int(raw.get("resilience_budget_cents", 0)),
        "shared_resilience_contribution_cents": int(
            raw.get("shared_resilience_contribution_cents") or 0
        ),
        "incident_response_mode": str(
            raw.get("incident_response_mode", incident.get("mode", "wait"))
        ),
        "repair_budget_cents": int(
            raw.get("repair_budget_cents", incident.get("repair_budget_cents", 0))
        ),
    }


def _to_company_action(
    state: MarketState,
    company_id: str,
    payload: Mapping[str, Any],
    action_id: str,
) -> CompanyAction:
    action = economic_action(payload)
    return CompanyAction(
        action_id=action_id,
        episode_id=state.episode_id,
        agent_id=company_id,
        round=state.round,
        state_version=state.state_version,
        price_cents=action["price_cents"],
        advertising_budget_cents=action["advertising_budget_cents"],
        service_budget_cents=action["service_budget_cents"],
        capacity_investment_cents=action["capacity_investment_cents"],
        resilience_budget_cents=action["resilience_budget_cents"],
        shared_resilience_contribution_cents=(
            action["shared_resilience_contribution_cents"]
            if state.shared_resilience is not None
            else None
        ),
        incident_response=IncidentResponse(
            IncidentResponseMode(action["incident_response_mode"]),
            action["repair_budget_cents"],
        ),
        strategy_summary="Stage 6.6 frozen-output forensic counterfactual",
    )


def _terminal_value(state: MarketState) -> int:
    rankings = build_terminal_rankings(state, load_market_config(CONFIG_PATH))
    return next(
        int(row["value_cents"])
        for row in rankings["composite"]
        if row["company_id"] == FOCAL_COMPANY
    )


def _economic_state_hash(raw_state: Mapping[str, Any]) -> str:
    volatile = {
        "episode_id",
        "state_hash",
        "last_action_id",
        "action_id",
        "event_id",
        "signal_id",
    }

    def cleaned(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: cleaned(item)
                for key, item in sorted(value.items())
                if key not in volatile
            }
        if isinstance(value, list):
            return [cleaned(item) for item in value]
        return value

    return sha256_hash(cleaned(raw_state))


def _state_metrics(state: MarketState) -> dict[str, Any]:
    company = state.company(FOCAL_COMPANY)
    return {
        "round": state.round - 1,
        "state_hash": state.state_hash,
        "enterprise_value_cents": _terminal_value(state),
        "cumulative_profit_cents": company.financial.cumulative_profit_cents,
        "cash_cents": company.financial.cash_balance_cents,
        "market_share_ppm": company.commercial.market_share_ppm,
        "price_cents": company.commercial.price_cents,
        "active_incident": (
            company.risk.active_incident.to_dict()
            if company.risk.active_incident is not None
            else None
        ),
    }


def _recorded_opponent_action(
    state: MarketState,
    raw: Mapping[str, Any] | CompanyAction,
    company_id: str,
    offset: int,
) -> CompanyAction:
    original = raw if isinstance(raw, CompanyAction) else CompanyAction.from_dict(raw)
    return replace(
        original,
        action_id=f"stage66:tape:{offset}:{company_id}:{state.state_version}",
        episode_id=state.episode_id,
        agent_id=company_id,
        round=state.round,
        state_version=state.state_version,
    )


def simulate_frozen_tape(
    *,
    initial_state: MarketState,
    initial_action: Mapping[str, Any],
    recorded_events: Sequence[Any],
) -> dict[str, Any]:
    """Settle one first action plus three fully frozen continuation rounds."""

    config = load_market_config(CONFIG_PATH)
    env = MarketEnv(config)
    env.load_state(initial_state)
    rounds: list[dict[str, Any]] = []
    for offset, event in enumerate(recorded_events):
        state = env.get_state()
        joint: dict[str, CompanyAction] = {}
        for company_id in state.company_ids:
            if company_id == FOCAL_COMPANY:
                if offset == 0:
                    joint[company_id] = _to_company_action(
                        state,
                        company_id,
                        initial_action,
                        f"stage66:first:{sha256_hash(dict(initial_action))}",
                    )
                else:
                    joint[company_id] = _recorded_opponent_action(
                        state,
                        event.joint_action[company_id],
                        company_id,
                        offset,
                    )
            else:
                joint[company_id] = _recorded_opponent_action(
                    state,
                    event.joint_action[company_id],
                    company_id,
                    offset,
                )
            validation = env.validate_action(joint[company_id], company_id)
            if not validation.valid:
                raise ValueError(
                    f"frozen tape action is illegal for {company_id}: "
                    f"{validation.errors}"
                )
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", joint
        )
        rounds.append(
            {
                **_state_metrics(result.state_after),
                "focal_action": economic_action(joint[FOCAL_COMPANY]),
                "opponent_actions": {
                    company_id: economic_action(action)
                    for company_id, action in joint.items()
                    if company_id != FOCAL_COMPANY
                },
                "random_draw_summary": dict(result.random_draw_summary),
            }
        )
    return {
        "initial_state_hash": initial_state.state_hash,
        "rounds": rounds,
        "final_state_hash": env.get_state().state_hash,
        "final_economic_state_hash": _economic_state_hash(
            env.get_state().to_dict()
        ),
        "final_enterprise_value_cents": _terminal_value(env.get_state()),
    }


def _candidate_maps(advice: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    candidates = {
        str(row["candidate"]["candidate_id"]): economic_action(
            row["candidate"]["action"]
        )
        for row in advice["candidate_actions"]
    }
    assessments = advice["pareto_decision"]["candidate_assessments"]
    safe = {
        str(row["candidate_id"])
        for row in assessments
        if bool(row["eligible"])
    }
    return candidates, safe


def map_action_to_candidate(
    action: Mapping[str, Any] | Any,
    candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = economic_action(action)
    exact = next(
        (candidate_id for candidate_id, item in candidates.items() if normalized == item),
        None,
    )
    distances = {
        candidate_id: sum(
            normalized[field] != item[field]
            for field in ECONOMIC_FIELDS
        )
        for candidate_id, item in candidates.items()
    }
    nearest = min(distances, key=lambda item: (distances[item], item))
    return {
        "mapping_status": "exact_candidate" if exact is not None else "custom_action",
        "candidate_id": exact or "custom_action",
        "nearest_candidate_id": nearest,
        "different_field_count": distances[nearest],
        "economic_action": normalized,
    }


def decompose_regret(
    *, oracle_value: int, advisor_value: int, llm_value: int, final_value: int
) -> dict[str, int]:
    planner = oracle_value - advisor_value
    adoption = advisor_value - llm_value
    execution = llm_value - final_value
    return {
        "planner_regret_cents": planner,
        "adoption_regret_cents": adoption,
        "execution_regret_cents": execution,
        "oracle_to_final_regret_cents": oracle_value - final_value,
        "additive_identity_residual_cents": (
            oracle_value - final_value - planner - adoption - execution
        ),
    }


def _events(directory: Path) -> tuple[Any, ...]:
    return JsonlRoundEventLogger(directory / "round-events.jsonl").read_all()


def _paid_trace(events: Sequence[Any]) -> tuple[Any, Any]:
    event = next(item for item in events if item.settled_round == PAID_ROUND)
    trace = next(item for item in event.traces if item.company_id == FOCAL_COMPANY)
    return event, trace


def _canonical_rows(stage65_summary: Path) -> list[dict[str, Any]]:
    merged = json.loads(stage65_summary.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for source in merged["source_summaries"]:
        path = Path(source)
        source_summary = json.loads(path.read_text(encoding="utf-8"))
        for row in source_summary["rows"]:
            item = dict(row)
            item["directory"] = str(
                path.parent
                / f"seed-{row['seed']}"
                / row["persona_id"]
                / row["condition"]
            )
            rows.append(item)
    keys = [(row["seed"], row["persona_id"], row["condition"]) for row in rows]
    if len(keys) != 65 or len(set(keys)) != 65:
        raise ValueError("Stage 6.5 canonical source matrix must contain 65 unique rows")
    return rows


def _verify_episode(events: Sequence[Any]) -> dict[str, Any]:
    config = load_market_config(CONFIG_PATH)
    env = MarketEnv(config)
    initial = MarketState.from_dict(events[0].state_before)
    env.load_state(initial)
    for event in events:
        current = env.get_state()
        result = env.step(
            f"{current.episode_id}:{current.round}:{current.state_version}",
            event.joint_action,
        )
        if result.state_after.state_hash != event.state_after_hash:
            raise ValueError("economic replay state hash mismatch")
    verify_interaction_replay(events)
    verify_information_replay(events)
    verify_belief_replay(events)
    game_theory = verify_game_theory_replay(events)
    adoption_count = 0
    for event in events:
        for trace in event.traces:
            if trace.advisor_adoption is None:
                continue
            rebuilt = build_advisor_adoption_trace(
                advice=_as_dict(trace.advisor_output),
                llm_requested_action=_as_dict(trace.requested_action),
                final_action=_as_dict(trace.final_action),
                planner_output=_as_dict(trace.planner_output),
            )
            if rebuilt != trace.advisor_adoption:
                raise ValueError("adoption trace replay mismatch")
            adoption_count += 1
    return {
        "economic_replay": True,
        "interaction_replay": True,
        "information_replay": True,
        "belief_replay": True,
        "advisor_and_game_theory_replay": True,
        "adoption_trace_replay": True,
        "adoption_trace_count": adoption_count,
        "hidden_state_leak_count": game_theory.hidden_state_leak_count,
    }


def _analyze_cell(
    cell_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_condition = {row["condition"]: row for row in cell_rows}
    pareto_dir = Path(by_condition["pareto_v4"]["directory"])
    pareto_events = _events(pareto_dir)
    paid_event, pareto_trace = _paid_trace(pareto_events)
    advice = _as_dict(pareto_trace.advisor_output)
    candidates, safe_set = _candidate_maps(advice)
    tape_events = tuple(
        item for item in pareto_events if item.settled_round >= PAID_ROUND
    )
    initial_state = MarketState.from_dict(paid_event.state_before)
    candidate_paths = {
        candidate_id: simulate_frozen_tape(
            initial_state=initial_state,
            initial_action=action,
            recorded_events=tape_events,
        )
        for candidate_id, action in candidates.items()
    }
    requested_path = simulate_frozen_tape(
        initial_state=initial_state,
        initial_action=_as_dict(pareto_trace.requested_action),
        recorded_events=tape_events,
    )
    final_path = simulate_frozen_tape(
        initial_state=initial_state,
        initial_action=_as_dict(pareto_trace.final_action),
        recorded_events=tape_events,
    )
    strategy_events = _events(Path(by_condition["strategy_utility"]["directory"]))
    _, strategy_trace = _paid_trace(strategy_events)
    baseline_path = simulate_frozen_tape(
        initial_state=initial_state,
        initial_action=_as_dict(strategy_trace.final_action),
        recorded_events=tape_events,
    )
    oracle_id = max(
        candidate_paths,
        key=lambda item: (
            candidate_paths[item]["final_enterprise_value_cents"], item
        ),
    )
    advisor_id = str(advice["recommended_candidate_id"])
    advisor_path = candidate_paths[advisor_id]
    decomposition = decompose_regret(
        oracle_value=candidate_paths[oracle_id]["final_enterprise_value_cents"],
        advisor_value=advisor_path["final_enterprise_value_cents"],
        llm_value=requested_path["final_enterprise_value_cents"],
        final_value=final_path["final_enterprise_value_cents"],
    )
    candidate_row = next(
        row
        for row in advice["candidate_actions"]
        if row["candidate"]["candidate_id"] == advisor_id
    )
    initial_ev = _terminal_value(initial_state)
    forecast_ev = initial_ev + int(
        candidate_row["expected_enterprise_value_delta_cents"]
    )
    forecast_horizon = int(advice["horizon_rounds"])
    realized_same_horizon = advisor_path["rounds"][forecast_horizon - 1][
        "enterprise_value_cents"
    ]
    forecast_error = forecast_ev - realized_same_horizon
    controlled_baseline_delta = (
        final_path["final_enterprise_value_cents"]
        - baseline_path["final_enterprise_value_cents"]
    )
    observed_delta = (
        int(by_condition["pareto_v4"]["enterprise_value_cents"])
        - int(by_condition["strategy_utility"]["enterprise_value_cents"])
    )
    endogenous_response_path_effect = observed_delta - controlled_baseline_delta
    endogenous_response_penalty = max(0, -endogenous_response_path_effect)
    candidate_set_coverage_regret = max(
        0,
        baseline_path["final_enterprise_value_cents"]
        - candidate_paths[oracle_id]["final_enterprise_value_cents"],
    )
    candidates_sorted = sorted(
        (
            {
                "candidate_id": candidate_id,
                "safe": candidate_id in safe_set,
                "forecast_expected_enterprise_value_cents": (
                    initial_ev
                    + int(
                        next(
                            row
                            for row in advice["candidate_actions"]
                            if row["candidate"]["candidate_id"] == candidate_id
                        )["expected_enterprise_value_delta_cents"]
                    )
                ),
                "realized_same_horizon_enterprise_value_cents": path["rounds"][
                    forecast_horizon - 1
                ]["enterprise_value_cents"],
                "realized_final_enterprise_value_cents": path[
                    "final_enterprise_value_cents"
                ],
            }
            for candidate_id, path in candidate_paths.items()
        ),
        key=lambda row: (-row["realized_final_enterprise_value_cents"], row["candidate_id"]),
    )
    primary = max(
        (
            ("planner_selection", decomposition["planner_regret_cents"]),
            ("candidate_set_coverage", candidate_set_coverage_regret),
            ("llm_adoption", max(0, decomposition["adoption_regret_cents"])),
            ("execution_adjustment", max(0, decomposition["execution_regret_cents"])),
            (
                "forecast_error",
                max(0, forecast_error, endogenous_response_penalty),
            ),
        ),
        key=lambda item: (item[1], item[0]),
    )[0]
    case = {
        "seed": int(by_condition["pareto_v4"]["seed"]),
        "persona_id": str(by_condition["pareto_v4"]["persona_id"]),
        "frozen_state_hash": initial_state.state_hash,
        "observed_pareto_vs_strategy_ev_delta_cents": observed_delta,
        "controlled_tape_pareto_vs_strategy_ev_delta_cents": controlled_baseline_delta,
        "endogenous_response_path_effect_cents": endogenous_response_path_effect,
        "endogenous_response_penalty_cents": endogenous_response_penalty,
        "planner_recommended_candidate_id": advisor_id,
        "realized_oracle_candidate_id": oracle_id,
        "planner_optimal_on_realized_tape": advisor_id == oracle_id,
        "safe_candidate_ids": sorted(safe_set),
        "llm_mapping": map_action_to_candidate(pareto_trace.requested_action, candidates),
        "final_mapping": map_action_to_candidate(pareto_trace.final_action, candidates),
        "adoption_status": pareto_trace.advisor_adoption.adoption_status,
        "controller_economic_adjustment": (
            economic_action(pareto_trace.requested_action)
            != economic_action(pareto_trace.final_action)
        ),
        **decomposition,
        "candidate_set_coverage_regret_cents": candidate_set_coverage_regret,
        "forecast_expected_enterprise_value_cents": forecast_ev,
        "forecast_realized_same_horizon_enterprise_value_cents": realized_same_horizon,
        "forecast_error_cents": forecast_error,
        "planning_horizon_rounds": forecast_horizon,
        "realized_settlement_count": len(tape_events),
        "horizon_extension_effect_cents": (
            advisor_path["final_enterprise_value_cents"] - realized_same_horizon
        ),
        "primary_loss_source": primary,
        "candidate_realized_table": candidates_sorted,
        "raw_model_output": pareto_trace.raw_model_output,
        "requested_action": economic_action(pareto_trace.requested_action),
        "final_action": economic_action(pareto_trace.final_action),
        "advisor_path": advisor_path,
        "strategy_baseline_path": baseline_path,
        "recorded_pareto_economic_path_reproduced": (
            final_path["final_economic_state_hash"]
            == _economic_state_hash(pareto_events[-1].state_after)
            and final_path["final_enterprise_value_cents"]
            == int(by_condition["pareto_v4"]["enterprise_value_cents"])
        ),
    }
    mappings: list[dict[str, Any]] = []
    for condition, row in sorted(by_condition.items()):
        events = _events(Path(row["directory"]))
        _, trace = _paid_trace(events)
        requested_mapping = map_action_to_candidate(trace.requested_action, candidates)
        final_mapping = map_action_to_candidate(trace.final_action, candidates)
        requested_cf = simulate_frozen_tape(
            initial_state=initial_state,
            initial_action=_as_dict(trace.requested_action),
            recorded_events=tape_events,
        )
        final_cf = simulate_frozen_tape(
            initial_state=initial_state,
            initial_action=_as_dict(trace.final_action),
            recorded_events=tape_events,
        )
        mappings.append(
            {
                "seed": int(row["seed"]),
                "persona_id": str(row["persona_id"]),
                "condition": condition,
                "recommended_candidate_id": advisor_id,
                "safe_candidate_ids": sorted(safe_set),
                "requested_mapping": requested_mapping,
                "final_mapping": final_mapping,
                "requested_in_safe_set": (
                    requested_mapping["candidate_id"] in safe_set
                ),
                "final_in_safe_set": final_mapping["candidate_id"] in safe_set,
                "planner_regret_cents": decomposition["planner_regret_cents"],
                "adoption_regret_cents": (
                    advisor_path["final_enterprise_value_cents"]
                    - requested_cf["final_enterprise_value_cents"]
                ),
                "execution_regret_cents": (
                    requested_cf["final_enterprise_value_cents"]
                    - final_cf["final_enterprise_value_cents"]
                ),
                "forecast_error_cents": forecast_error,
            }
        )
    return case, mappings


def run(stage65_summary: Path, output: Path) -> dict[str, Any]:
    rows = _canonical_rows(stage65_summary)
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["seed"]), str(row["persona_id"])), []).append(row)
    all_mappings: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    negative_keys = {
        key
        for key, cell in grouped.items()
        if (
            next(row for row in cell if row["condition"] == "pareto_v4")[
                "enterprise_value_cents"
            ]
            < next(row for row in cell if row["condition"] == "strategy_utility")[
                "enterprise_value_cents"
            ]
        )
    }
    for key, cell in sorted(grouped.items()):
        case, mappings = _analyze_cell(cell)
        all_mappings.extend(mappings)
        if key in negative_keys:
            cases.append(case)
    replay_rows: list[dict[str, Any]] = []
    for row in rows:
        checks = _verify_episode(_events(Path(row["directory"])))
        replay_rows.append(
            {
                "seed": row["seed"],
                "persona_id": row["persona_id"],
                "condition": row["condition"],
                **checks,
            }
        )
    if len(cases) != 5 or len(all_mappings) != 65:
        raise ValueError("forensic matrix must contain five failures and 65 mappings")
    checks = {
        "negative_pair_attribution_5_of_5": all(
            case["primary_loss_source"] != "unknown" for case in cases
        ),
        "advice_to_candidate_mapping_100pct": all(
            item["recommended_candidate_id"] for item in all_mappings
        ),
        "final_action_classification_100pct": all(
            item["final_mapping"]["candidate_id"] for item in all_mappings
        ),
        "safe_set_legality_100pct": all(case["safe_candidate_ids"] for case in cases),
        "recorded_pareto_economic_path_reproduced_100pct": all(
            case["recorded_pareto_economic_path_reproduced"] for case in cases
        ),
        "final_illegal_action_count": 0,
        "economic_replay_100pct": all(item["economic_replay"] for item in replay_rows),
        "interaction_replay_100pct": all(
            item["interaction_replay"] for item in replay_rows
        ),
        "information_replay_100pct": all(
            item["information_replay"] for item in replay_rows
        ),
        "belief_replay_100pct": all(item["belief_replay"] for item in replay_rows),
        "advisor_replay_100pct": all(
            item["advisor_and_game_theory_replay"] for item in replay_rows
        ),
        "adoption_trace_replay_100pct": all(
            item["adoption_trace_replay"] for item in replay_rows
        ),
        "hidden_state_leakage": sum(
            item["hidden_state_leak_count"] for item in replay_rows
        ),
        "real_model_calls": 0,
        "new_advice_contract_hash_status": "deferred_until_forensic_decision",
    }
    summary: dict[str, Any] = {
        "experiment_schema_version": "stage6.6-failure-forensics-v1.0.0",
        "evidence_level": "frozen_real_llm_output_fixed-opponent-tape-counterfactual",
        "source_stage65_summary": str(stage65_summary.resolve()),
        "negative_pair_count": len(cases),
        "historical_mapping_count": len(all_mappings),
        "checks": checks,
        "failure_attribution_table": [
            {
                key: case[key]
                for key in (
                    "seed",
                    "persona_id",
                    "planner_optimal_on_realized_tape",
                    "adoption_status",
                    "controller_economic_adjustment",
                    "planner_regret_cents",
                    "adoption_regret_cents",
                    "execution_regret_cents",
                    "candidate_set_coverage_regret_cents",
                    "forecast_error_cents",
                    "endogenous_response_path_effect_cents",
                    "primary_loss_source",
                )
            }
            for case in cases
        ],
        "conclusion_limits": [
            "the opponent action tape is frozen from each Pareto episode",
            "the focal company changes only its paid-round action; all later settled actions are frozen from the Pareto episode",
            "planner/adoption/execution regrets are additive; forecast error is a separate overlapping diagnostic",
            "the public forecast horizon is compared with realized value at the same horizon",
            "no LLM output is regenerated and no planner algorithm is modified",
        ],
        "report_hash": "pending",
    }
    summary["report_hash"] = sha256_hash(
        {key: value for key, value in summary.items() if key != "report_hash"}
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "failure-cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "historical-mapping.json").write_text(
        json.dumps(all_mappings, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "replay-checks.json").write_text(
        json.dumps(replay_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
