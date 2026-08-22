"""Plan, execute, and aggregate paired real-model communication Smoke runs.

This experiment validates the non-binding communication pipeline. It does not
claim that cooperation, commitment, or causal market improvement exists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace
from typing import Any, Iterable

from game_theory_agent.experiments.four_agent_acceptance import (
    run as run_episode,
)
from game_theory_agent.orchestration import JsonlRoundEventLogger


COMMUNICATION_CONDITIONS = ("off", "public_only", "public_private")
ACTION_FIELDS = (
    "price_cents",
    "advertising_budget_cents",
    "service_budget_cents",
    "capacity_investment_cents",
    "resilience_budget_cents",
)
MARKET_METRIC_PATHS = {
    "market_total_profit_cents": ("market_total_profit_cents",),
    "cumulative_unserved_demand_orders": (
        "cumulative_unserved_demand_orders",
    ),
    "cumulative_outside_option_orders": (
        "cumulative_outside_option_orders",
    ),
    "mean_price_dispersion_cents": (
        "strategy_diversity",
        "mean_price_dispersion_cents",
    ),
    "mean_normalized_full_action_distance": (
        "strategy_diversity",
        "mean_normalized_full_action_distance",
    ),
    "mean_demand_capacity_absolute_gap_orders": (
        "resource_allocation",
        "mean_demand_capacity_absolute_gap_orders",
    ),
    "minimum_resilience_ppm": (
        "resource_allocation",
        "minimum_resilience_ppm",
    ),
}


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("at least one Seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    return seeds


def build_matrix(seeds: Iterable[int]) -> list[dict[str, Any]]:
    """Return a blocked three-condition matrix with Seed as the unit."""

    return [
        {
            "seed": int(seed),
            "condition": condition,
            "primary_experiment_unit": "paired_seed",
            "communication_is_non_binding": True,
        }
        for seed in seeds
        for condition in COMMUNICATION_CONDITIONS
    ]


def parse_artifacts(values: Iterable[str]) -> dict[tuple[int, str], Path]:
    """Parse SEED:CONDITION=DIR references to already completed runs."""

    artifacts: dict[tuple[int, str], Path] = {}
    for value in values:
        selector, separator, raw_path = value.partition("=")
        if not separator or ":" not in selector or not raw_path:
            raise ValueError(
                "artifact must use SEED:CONDITION=DIR syntax"
            )
        raw_seed, condition = selector.split(":", 1)
        seed = int(raw_seed)
        if condition not in COMMUNICATION_CONDITIONS:
            raise ValueError(f"unsupported artifact condition: {condition}")
        key = (seed, condition)
        if key in artifacts:
            raise ValueError(f"duplicate artifact selector: {selector}")
        artifacts[key] = Path(raw_path).resolve()
    return artifacts


def _validate_episode_summary(
    summary: dict[str, Any],
    *,
    seed: int,
    condition: str,
    args: argparse.Namespace,
) -> None:
    expected = {
        "seed": seed,
        "rounds": args.rounds,
        "provider": args.provider,
        "model": args.model,
        "persona": args.persona,
        "communication_mode": condition,
        "llm_count": 2,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": summary.get(key)}
        for key, expected_value in expected.items()
        if summary.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            "episode artifact does not match the Smoke plan: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _nested_number(value: dict[str, Any], path: tuple[str, ...]) -> float:
    current: Any = value
    for key in path:
        current = current[key]
    return float(current)


def _event_interaction_evidence(events: Iterable[Any]) -> dict[str, Any]:
    """Audit message references and the observable response/action chain."""

    message_count = 0
    channel_counts: dict[str, int] = {}
    speech_act_counts: dict[str, int] = {}
    disposition_counts: dict[str, int] = {}
    generation_status_counts: dict[str, int] = {}
    hallucinated_reference_count = 0
    invisible_reference_count = 0
    response_count = 0
    response_chain_count = 0
    chain_rows: list[dict[str, Any]] = []

    for event in events:
        phase = event.communication_phase
        all_messages = list(phase.closure.all_messages) if phase else []
        messages_by_id = {message.message_id: message for message in all_messages}
        message_count += len(all_messages)
        for message in all_messages:
            channel_counts[message.channel] = (
                channel_counts.get(message.channel, 0) + 1
            )
            speech_act_counts[message.speech_act] = (
                speech_act_counts.get(message.speech_act, 0) + 1
            )
        if phase is not None:
            for generation in phase.generation_traces:
                status = generation.generation_status
                generation_status_counts[status] = (
                    generation_status_counts.get(status, 0) + 1
                )

        for trace in event.traces:
            visible_ids = {
                message.message_id
                for message in (
                    trace.communication_view.visible_messages
                    if trace.communication_view is not None
                    else []
                )
            }
            for response in trace.message_responses or []:
                message_id = (
                    response.get("message_id")
                    if isinstance(response, dict)
                    else response.message_id
                )
                disposition = (
                    response.get("disposition")
                    if isinstance(response, dict)
                    else response.disposition
                )
                response_count += 1
                disposition_counts[str(disposition)] = (
                    disposition_counts.get(str(disposition), 0) + 1
                )
                known = message_id in messages_by_id
                visible = message_id in visible_ids
                hallucinated_reference_count += int(not known)
                invisible_reference_count += int(known and not visible)
                requested_action_present = trace.requested_action is not None
                final_action_present = bool(trace.final_action)
                complete = bool(
                    known
                    and visible
                    and requested_action_present
                    and final_action_present
                )
                response_chain_count += int(complete)
                chain_rows.append(
                    {
                        "round": event.settled_round,
                        "company_id": trace.company_id,
                        "message_id": message_id,
                        "sender_company_id": (
                            messages_by_id[message_id].sender_company_id
                            if known
                            else None
                        ),
                        "channel": (
                            messages_by_id[message_id].channel if known else None
                        ),
                        "disposition": disposition,
                        "message_exists": known,
                        "message_visible": visible,
                        "requested_action_present": requested_action_present,
                        "final_action_present": final_action_present,
                        "complete_observable_chain": complete,
                    }
                )

    return {
        "message_count": message_count,
        "channel_counts": dict(sorted(channel_counts.items())),
        "speech_act_counts": dict(sorted(speech_act_counts.items())),
        "generation_status_counts": dict(
            sorted(generation_status_counts.items())
        ),
        "response_count": response_count,
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "hallucinated_message_reference_count": hallucinated_reference_count,
        "invisible_message_reference_count": invisible_reference_count,
        "complete_response_chain_count": response_chain_count,
        "all_response_chains_complete": (
            response_chain_count == response_count
        ),
        "response_chains": chain_rows,
    }


def _paired_seed_evidence(
    summaries: dict[str, dict[str, Any]],
    events: dict[str, list[Any]],
) -> dict[str, Any]:
    """Compare each communication condition with the same-Seed off run."""

    off_summary = summaries["off"]
    off_events = events["off"]
    rows: list[dict[str, Any]] = []
    for condition in ("public_only", "public_private"):
        treatment_summary = summaries[condition]
        treatment_events = events[condition]
        same_initial_state = bool(
            off_events
            and treatment_events
            and off_events[0].state_before.get("round")
            == treatment_events[0].state_before.get("round")
            and off_events[0].state_before.get("state_version")
            == treatment_events[0].state_before.get("state_version")
            and off_events[0].state_before.get("market")
            == treatment_events[0].state_before.get("market")
            and off_events[0].state_before.get("companies")
            == treatment_events[0].state_before.get("companies")
        )
        market_deltas = {
            name: _nested_number(
                treatment_summary["research_metrics"], path
            )
            - _nested_number(off_summary["research_metrics"], path)
            for name, path in MARKET_METRIC_PATHS.items()
        }

        first_round_action_deltas: dict[str, dict[str, float]] = {}
        first_round_changed_companies: list[str] = []
        if off_events and treatment_events:
            for company_id in treatment_summary["llm_companies"]:
                baseline = off_events[0].joint_action[company_id]
                treatment = treatment_events[0].joint_action[company_id]
                deltas = {
                    field: float(treatment[field]) - float(baseline[field])
                    for field in ACTION_FIELDS
                }
                first_round_action_deltas[company_id] = deltas
                if any(delta != 0 for delta in deltas.values()):
                    first_round_changed_companies.append(company_id)

        treatment_evidence = _event_interaction_evidence(treatment_events)
        responding_companies = {
            row["company_id"]
            for row in treatment_evidence["response_chains"]
        }
        first_round_responding_companies = {
            row["company_id"]
            for row in treatment_evidence["response_chains"]
            if row["round"] == 1
        }
        rows.append(
            {
                "condition": condition,
                "same_initial_market_and_company_state": same_initial_state,
                "market_metric_deltas_vs_off": market_deltas,
                "first_round_action_deltas_vs_off": (
                    first_round_action_deltas
                ),
                "first_round_changed_llm_companies": (
                    first_round_changed_companies
                ),
                "responding_companies": sorted(responding_companies),
                "first_round_responding_companies": sorted(
                    first_round_responding_companies
                ),
                "first_round_response_and_action_delta_companies": sorted(
                    first_round_responding_companies
                    & set(first_round_changed_companies)
                ),
                "interpretation": (
                    "paired observational evidence; separate real-model calls "
                    "are not a deterministic causal intervention"
                ),
            }
        )
    return {"conditions": rows}


def aggregate_results(
    *,
    seeds: Iterable[int],
    summaries: dict[tuple[int, str], dict[str, Any]],
    events: dict[tuple[int, str], list[Any]],
) -> dict[str, Any]:
    """Build engineering gates and paired research summaries."""

    seed_values = tuple(seeds)
    expected_keys = {
        (seed, condition)
        for seed in seed_values
        for condition in COMMUNICATION_CONDITIONS
    }
    complete_matrix = set(summaries) == expected_keys and set(events) == expected_keys
    episode_evidence = {
        f"seed_{seed}/{condition}": _event_interaction_evidence(
            events[(seed, condition)]
        )
        for seed, condition in sorted(events)
    }
    paired = {
        str(seed): _paired_seed_evidence(
            {
                condition: summaries[(seed, condition)]
                for condition in COMMUNICATION_CONDITIONS
            },
            {
                condition: events[(seed, condition)]
                for condition in COMMUNICATION_CONDITIONS
            },
        )
        for seed in seed_values
        if all((seed, condition) in summaries for condition in COMMUNICATION_CONDITIONS)
        and all((seed, condition) in events for condition in COMMUNICATION_CONDITIONS)
    }

    paired_metric_summary: dict[str, dict[str, Any]] = {}
    for condition in ("public_only", "public_private"):
        by_metric: dict[str, list[float]] = {
            name: [] for name in MARKET_METRIC_PATHS
        }
        for seed_evidence in paired.values():
            row = next(
                item
                for item in seed_evidence["conditions"]
                if item["condition"] == condition
            )
            for name, delta in row["market_metric_deltas_vs_off"].items():
                by_metric[name].append(float(delta))
        paired_metric_summary[condition] = {
            name: {
                "paired_seed_count": len(values),
                "mean_delta_vs_off": round(mean(values), 6) if values else None,
                "median_delta_vs_off": round(median(values), 6) if values else None,
                "positive_seed_count": sum(value > 0 for value in values),
                "zero_seed_count": sum(value == 0 for value in values),
                "negative_seed_count": sum(value < 0 for value in values),
            }
            for name, values in by_metric.items()
        }

    protocol_passed = complete_matrix and all(
        bool(summary.get("protocol_passed"))
        and bool(summary.get("passed"))
        and bool(
            summary.get("protocol_checks", {}).get("replay_match_100pct")
        )
        and bool(
            summary.get("protocol_checks", {}).get(
                "interaction_replay_match_100pct"
            )
        )
        for summary in summaries.values()
    )
    no_hallucinated_references = all(
        item["hallucinated_message_reference_count"] == 0
        for item in episode_evidence.values()
    )
    no_invisible_references = all(
        item["invisible_message_reference_count"] == 0
        for item in episode_evidence.values()
    )
    communication_failure_count = sum(
        count
        for item in episode_evidence.values()
        for status, count in item["generation_status_counts"].items()
        if status in {"fallback", "invalid"}
    )
    llm_generation_attempts = sum(
        count
        for item in episode_evidence.values()
        for status, count in item["generation_status_counts"].items()
        if status not in {"not_applicable", "disabled"}
    )
    minimum_design_complete = len(seed_values) >= 5 and complete_matrix
    smoke_passed = bool(
        minimum_design_complete
        and protocol_passed
        and no_hallucinated_references
        and no_invisible_references
        and communication_failure_count == 0
    )
    return {
        "aggregate_schema_version": "real-communication-smoke-v1.0.0",
        "stage_claim": "non_binding_cheap_talk_communication_only",
        "cooperation_mechanism_implemented": False,
        "primary_experiment_unit": "paired_seed",
        "decisions_are_nested_independent_samples": False,
        "seeds": list(seed_values),
        "conditions": list(COMMUNICATION_CONDITIONS),
        "expected_episode_count": len(expected_keys),
        "executed_episode_count": len(summaries),
        "minimum_five_seed_design_complete": minimum_design_complete,
        "engineering_checks": {
            "complete_three_condition_matrix": complete_matrix,
            "all_episode_protocol_and_dual_replay_passed": protocol_passed,
            "hallucinated_message_reference_zero": (
                no_hallucinated_references
            ),
            "invisible_message_reference_zero": no_invisible_references,
            "communication_failure_count": communication_failure_count,
            "llm_generation_attempt_count": llm_generation_attempts,
        },
        "smoke_passed": smoke_passed,
        "episode_evidence": episode_evidence,
        "paired_seed_evidence": paired,
        "paired_market_metric_summary": paired_metric_summary,
        "causal_claim": "not_established_by_independent_real_model_calls",
    }


def _episode_args(
    args: argparse.Namespace,
    *,
    seed: int,
    condition: str,
    output: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        episode_id=None,
        seed=seed,
        rounds=args.rounds,
        market_model=args.market_model,
        provider=args.provider,
        model=args.model,
        persona=args.persona,
        personas=args.persona,
        condition=None,
        llm_count=2,
        rotation_index=0,
        decision_support_version="economic_v2",
        persona_semantics_version="economic_v2",
        diagnostic_mode="off",
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        communication_mode=condition,
        communication_timeout=args.communication_timeout,
        mock_communication_scenario="silence",
        output=output,
    )


async def run(args: argparse.Namespace) -> int:
    seeds = parse_seeds(args.seeds)
    matrix = build_matrix(seeds)
    artifact_sources = parse_artifacts(args.artifact)
    matrix_keys = {
        (int(row["seed"]), str(row["condition"])) for row in matrix
    }
    unexpected_artifacts = set(artifact_sources) - matrix_keys
    if unexpected_artifacts:
        raise ValueError(
            "artifact selectors are outside the requested matrix: "
            f"{sorted(unexpected_artifacts)}"
        )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "smoke-plan.json"
    result_path = output / "smoke-results.json"
    plan = {
        "plan_schema_version": "real-communication-smoke-plan-v1.0.0",
        "stage_claim": "non_binding_cheap_talk_communication_only",
        "provider": args.provider,
        "model": args.model,
        "rounds": args.rounds,
        "llm_count": 2,
        "rule_count": 2,
        "persona": args.persona,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed_count": len(seeds),
        "episode_count": len(matrix),
        "primary_experiment_unit": "paired_seed",
        "conditions": list(COMMUNICATION_CONDITIONS),
        "artifact_sources": {
            f"seed_{seed}/{condition}": str(path)
            for (seed, condition), path in sorted(artifact_sources.items())
        },
        "rows": matrix,
    }
    if plan_path.exists() and plan_path.read_text(encoding="utf-8").strip():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise FileExistsError(
                f"existing smoke plan differs: {plan_path}"
            )
    else:
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    summaries: dict[tuple[int, str], dict[str, Any]] = {}
    events: dict[tuple[int, str], list[Any]] = {}
    summary_paths: dict[tuple[int, str], Path] = {}
    for index, row in enumerate(matrix, start=1):
        seed = int(row["seed"])
        condition = str(row["condition"])
        key = (seed, condition)
        episode_output = artifact_sources.get(
            key, output / f"seed-{seed}" / condition
        )
        summary_path = episode_output / "summary.json"
        event_path = episode_output / "round-events.jsonl"
        if key in artifact_sources:
            if not summary_path.exists() or not event_path.exists():
                raise FileNotFoundError(
                    f"referenced artifact is incomplete: {episode_output}"
                )
            print(
                f"[{index}/{len(matrix)}] import seed={seed} "
                f"condition={condition}",
                flush=True,
            )
        elif summary_path.exists() or event_path.exists():
            if not args.resume:
                raise FileExistsError(
                    "existing episode artifact requires --resume: "
                    f"{episode_output}"
                )
            if not summary_path.exists() or not event_path.exists():
                raise FileNotFoundError(
                    f"incomplete episode artifact: {episode_output}"
                )
            print(
                f"[{index}/{len(matrix)}] reuse seed={seed} "
                f"condition={condition}",
                flush=True,
            )
        else:
            print(
                f"[{index}/{len(matrix)}] run seed={seed} "
                f"condition={condition}",
                flush=True,
            )
            exit_code = await run_episode(
                _episode_args(
                    args,
                    seed=seed,
                    condition=condition,
                    output=episode_output,
                )
            )
            if exit_code != 0 and args.stop_on_failure:
                return exit_code
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _validate_episode_summary(
            summary,
            seed=seed,
            condition=condition,
            args=args,
        )
        summaries[key] = summary
        events[key] = list(
            JsonlRoundEventLogger(event_path).read_all()
        )
        summary_paths[key] = summary_path.resolve()

    aggregate = aggregate_results(
        seeds=seeds,
        summaries=summaries,
        events=events,
    )
    aggregate["plan_path"] = str(plan_path.resolve())
    aggregate["episode_summaries"] = {
        f"seed_{seed}/{condition}": str(path)
        for (seed, condition), path in sorted(summary_paths.items())
    }
    result_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0 if aggregate["smoke_passed"] else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="810,811,812,813,814")
    parser.add_argument("--rounds", type=int, choices=(5, 10), default=5)
    parser.add_argument("--provider", choices=("doubao", "deepseek"), default="doubao")
    parser.add_argument("--model", required=True)
    parser.add_argument("--persona", default="balanced_v1")
    parser.add_argument("--market-model", default="balanced")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--communication-timeout", type=float, default=60.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="SEED:CONDITION=DIR",
        help="reuse a validated completed single-run artifact",
    )
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    return asyncio.run(run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
