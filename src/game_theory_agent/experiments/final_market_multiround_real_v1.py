"""Cost-bounded 1-LLM + 3-rule, ten-round final-market Advisor comparison."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Mapping

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v6_final.yaml"
_PROJECT_ENV = dotenv_values(PROJECT_ROOT / ".env")
os.environ["MARKET_CONFIG_PATH"] = str(CONFIG_PATH)
for _key, _value in _PROJECT_ENV.items():
    if _value is not None:
        os.environ.setdefault(str(_key), str(_value))

from game_theory_agent.experiments.four_agent_acceptance import run as run_episode
from game_theory_agent.gameplay import build_terminal_rankings
from game_theory_agent.game_theory import verify_game_theory_replay
from game_theory_agent.market import MarketEnv, MarketState, load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.market.replay import EpisodeManifest
from game_theory_agent.orchestration import JsonlRoundEventLogger
from game_theory_agent.strategic_reliability import RealModelBudget, RealModelCostGuard


DEFAULT_SPEC = (
    PROJECT_ROOT
    / "experiment-specs"
    / "final-market-multiround-real-v1"
    / "PREREGISTRATION.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "final-market-multiround-real-v1"
SEEDS = (95301, 95302, 95303)
CONDITIONS = ("advisor_off", "advisor_v9")
ROUNDS = 10
PERSONA = "balanced_v1"
RESERVED_PROMPT_TOKENS = 32_000
RESERVED_COMPLETION_TOKENS = 4_000
INPUT_PRICE_MICROUNITS = 3
OUTPUT_PRICE_MICROUNITS = 9


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalized_hash(payload: Mapping[str, Any], field: str) -> str:
    normalized = dict(payload)
    normalized.pop(field, None)
    return sha256_hash(normalized)


def prepare(spec_path: Path) -> dict[str, Any]:
    if spec_path.exists():
        raise RuntimeError("preregistration already exists; refusing to overwrite it")
    config = load_market_config(CONFIG_PATH)
    cells: list[dict[str, Any]] = []
    for index, seed in enumerate(SEEDS):
        order = CONDITIONS if index % 2 == 0 else tuple(reversed(CONDITIONS))
        cells.append(
            {
                "seed": seed,
                "episode_id": f"final-market-multiround-real-{seed}",
                "condition_order": list(order),
            }
        )
    calls = len(SEEDS) * len(CONDITIONS) * ROUNDS
    spec: dict[str, Any] = {
        "experiment_schema_version": "final-market-multiround-real-prereg-v1.0.0",
        "evidence_level": "DIRECTIONAL_REAL_LLM_MULTIROUND_PAIRED_EVIDENCE",
        "provider": "deepseek",
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "temperature_ppm": 0,
        "top_p_ppm": 100_000,
        "config_path": str(CONFIG_PATH),
        "config_sha256": config.config_sha256,
        "composition": "1 LLM + 3 Rule",
        "persona": PERSONA,
        "seeds": list(SEEDS),
        "rounds": ROUNDS,
        "conditions": {
            "advisor_off": {
                "belief_mode": "public_action_v1",
                "opponent_model_mode": "public_strategy_v1",
                "utility_inference_mode": "strategy_utility_v1",
                "advisor_mode": "off",
            },
            "advisor_v9": {
                "belief_mode": "public_action_v1",
                "opponent_model_mode": "public_strategy_v1",
                "utility_inference_mode": "strategy_utility_v1",
                "advisor_mode": "strategic_market_v9",
            },
        },
        "frozen_controls": [
            "seed",
            "episode_id",
            "market_model",
            "information_mode",
            "communication_mode",
            "cooperation_mode",
            "persona",
            "model",
            "rule_opponents",
        ],
        "maximum_logical_provider_calls": calls,
        "reserved_prompt_tokens_per_call": RESERVED_PROMPT_TOKENS,
        "reserved_completion_tokens_per_call": RESERVED_COMPLETION_TOKENS,
        "maximum_prompt_tokens": calls * RESERVED_PROMPT_TOKENS,
        "maximum_completion_tokens": calls * RESERVED_COMPLETION_TOKENS,
        "maximum_estimated_cost_microunits": calls
        * (
            RESERVED_PROMPT_TOKENS * INPUT_PRICE_MICROUNITS
            + RESERVED_COMPLETION_TOKENS * OUTPUT_PRICE_MICROUNITS
        ),
        "price_snapshot": {
            "input_price_microunits_per_token": INPUT_PRICE_MICROUNITS,
            "output_price_microunits_per_token": OUTPUT_PRICE_MICROUNITS,
            "currency": "CNY",
            "status": "conservative_project_estimate_not_provider_invoice",
        },
        "primary_metrics": [
            "terminal_enterprise_value",
            "cumulative_profit",
            "survival",
            "worst_paired_enterprise_value_delta",
            "released_advice_adoption",
        ],
        "cells": cells,
    }
    spec["preregistration_hash"] = _normalized_hash(spec, "preregistration_hash")
    _write_json(spec_path, spec)
    return spec


def _episode_args(
    spec: Mapping[str, Any],
    *,
    seed: int,
    condition: str,
    output: Path,
    guard: RealModelCostGuard,
) -> SimpleNamespace:
    treatment = spec["conditions"][condition]
    return SimpleNamespace(
        episode_id=f"final-market-multiround-real-{seed}",
        seed=seed,
        rounds=ROUNDS,
        market_model="balanced",
        information_mode="public",
        privileged_observer_company_id=None,
        provider="deepseek",
        belief_mode=treatment["belief_mode"],
        opponent_model_mode=treatment["opponent_model_mode"],
        utility_inference_mode=treatment["utility_inference_mode"],
        advisor_mode=treatment["advisor_mode"],
        repeated_game_mode="off",
        cooperation_mode="combined_v1",
        honor_game_theory_advice=False,
        model=str(spec["model"]),
        persona=PERSONA,
        condition=None,
        personas=None,
        llm_count=1,
        rotation_index=0,
        decision_support_version="economic_v2",
        persona_semantics_version="economic_v2",
        diagnostic_mode="off",
        temperature=int(spec["temperature_ppm"]) / 1_000_000,
        top_p=int(spec["top_p_ppm"]) / 1_000_000,
        timeout=90.0,
        communication_mode="off",
        communication_timeout=30.0,
        mock_communication_scenario="silence",
        output=output,
        quiet=True,
        paid_decision_rounds=tuple(range(1, ROUNDS + 1)),
        real_model_cost_guard=guard,
        agent_configs={"company_A": {"persona": {"persona_id": PERSONA}}},
    )


def _summarize_episode(output: Path, seed: int, condition: str) -> dict[str, Any]:
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    events = list(JsonlRoundEventLogger(output / "round-events.jsonl").read_all())
    final_state = MarketState.from_dict(events[-1].state_after)
    config = load_market_config(CONFIG_PATH)
    initial_state = MarketState.from_dict(events[0].state_before)
    replay_env = MarketEnv(config)
    replay_env.load_state(initial_state)
    manifest = EpisodeManifest.create(
        replay_env,
        initial_state,
        experiment_id="final-market-multiround-real-v1-recovery",
        information_mode="public",
        communication_mode="off",
        cooperation_mode="combined_v1",
        belief_mode="public_action_v1",
        opponent_model_mode="public_strategy_v1",
        utility_inference_mode="strategy_utility_v1",
        advisor_mode=("strategic_market_v9" if condition == "advisor_v9" else "off"),
    )
    game_theory_replay = verify_game_theory_replay(events, manifest)
    protocol_checks = dict(summary["protocol_checks"])
    protocol_checks["game_theory_replay_match_100pct"] = (
        game_theory_replay.hidden_state_leak_count == 0
    )
    passed = all(bool(value) for value in protocol_checks.values())
    focal = final_state.company("company_A")
    enterprise_value = next(
        int(item["value_cents"])
        for item in build_terminal_rankings(final_state, config)["composite"]
        if item["company_id"] == "company_A"
    )
    traces = [
        trace
        for event in events
        for trace in event.traces
        if trace.company_id == "company_A"
    ]
    released = [
        trace
        for trace in traces
        if isinstance(trace.advisor_output, dict)
        and trace.advisor_output.get("execution_disposition") == "recommend"
    ]
    return {
        "seed": seed,
        "condition": condition,
        "passed": passed,
        "rounds": len(events),
        "final_state_hash": final_state.state_hash,
        "enterprise_value_cents": enterprise_value,
        "cumulative_profit_cents": focal.financial.cumulative_profit_cents,
        "cash_balance_cents": focal.financial.cash_balance_cents,
        "final_market_share_ppm": focal.commercial.market_share_ppm,
        "operating_status": (
            final_state.strategic_market.lifecycle("company_A").status.value
            if final_state.strategic_market is not None
            else "operating"
        ),
        "input_tokens": sum(int(trace.input_tokens or 0) for trace in traces),
        "output_tokens": sum(int(trace.output_tokens or 0) for trace in traces),
        "submitted_decisions": sum(trace.decision_status == "submitted" for trace in traces),
        "released_advice_count": len(released),
        "released_advice_accepted_count": sum(
            trace.advisor_adoption is not None and trace.advisor_adoption.accepted
            for trace in released
        ),
        "research_metrics": summary["research_metrics"],
        "round_event_sha256": summary["reproducibility"]["round_event_log_sha256"],
        "game_theory_replay_hash": game_theory_replay.replay_hash,
    }


async def run(spec_path: Path, output: Path) -> dict[str, Any]:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec["preregistration_hash"] != _normalized_hash(spec, "preregistration_hash"):
        raise RuntimeError("preregistration hash mismatch")
    if load_market_config(CONFIG_PATH).config_sha256 != spec["config_sha256"]:
        raise RuntimeError("market config changed after preregistration")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "plan.json", spec)
    rows_path = output / "rows.json"
    rows: list[dict[str, Any]] = (
        json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else []
    )
    refreshed_rows: list[dict[str, Any]] = []
    for existing in rows:
        if int(existing.get("rounds", 0)) == ROUNDS:
            episode_output = (
                output
                / f"seed-{int(existing['seed'])}"
                / str(existing["condition"])
            )
            refreshed = _summarize_episode(
                episode_output,
                int(existing["seed"]),
                str(existing["condition"]),
            )
            refreshed["runner_exit_code"] = existing.get("runner_exit_code", 0)
            refreshed["provider_calls_attempted"] = int(
                existing.get("provider_calls_attempted", ROUNDS)
            )
            if existing.get("recovered_after_postprocessing_failure"):
                refreshed["recovered_after_postprocessing_failure"] = True
            refreshed_rows.append(refreshed)
        else:
            refreshed_rows.append(existing)
    rows = refreshed_rows
    if rows:
        _write_json(rows_path, rows)
    completed = {(int(row["seed"]), str(row["condition"])) for row in rows}
    completed_calls = sum(
        int(row.get("provider_calls_attempted", row["rounds"])) for row in rows
    )
    remaining_calls = int(spec["maximum_logical_provider_calls"]) - completed_calls
    price = spec["price_snapshot"]
    guard = RealModelCostGuard(
        RealModelBudget(
            max_calls=max(1, remaining_calls),
            max_prompt_tokens=max(1, remaining_calls) * RESERVED_PROMPT_TOKENS,
            max_completion_tokens=max(1, remaining_calls) * RESERVED_COMPLETION_TOKENS,
            max_estimated_cost_microunits=max(1, remaining_calls)
            * (
                RESERVED_PROMPT_TOKENS * int(price["input_price_microunits_per_token"])
                + RESERVED_COMPLETION_TOKENS
                * int(price["output_price_microunits_per_token"])
            ),
        ),
        explicitly_authorized=True,
    )
    total_episodes = len(spec["cells"]) * len(CONDITIONS)
    for cell in spec["cells"]:
        seed = int(cell["seed"])
        for condition in cell["condition_order"]:
            key = (seed, str(condition))
            if key in completed:
                continue
            episode_output = output / f"seed-{seed}" / str(condition)
            if (
                (episode_output / "summary.json").exists()
                and (episode_output / "round-events.jsonl").exists()
            ):
                recovered = _summarize_episode(
                    episode_output, seed, str(condition)
                )
                if int(recovered["rounds"]) != ROUNDS:
                    raise RuntimeError(
                        f"incomplete prior episode cannot be recovered: {key}"
                    )
                recovered["runner_exit_code"] = 0
                recovered["provider_calls_attempted"] = int(
                    recovered["rounds"]
                )
                recovered["recovered_after_postprocessing_failure"] = True
                rows.append(recovered)
                completed.add(key)
                _write_json(rows_path, rows)
                print(
                    f"[{len(rows)}/{total_episodes}] recovered seed={seed} "
                    f"condition={condition} without provider calls",
                    flush=True,
                )
                continue
            result_code = await run_episode(
                _episode_args(
                    spec,
                    seed=seed,
                    condition=str(condition),
                    output=episode_output,
                    guard=guard,
                )
            )
            row = _summarize_episode(episode_output, seed, str(condition))
            row["runner_exit_code"] = result_code
            row["provider_calls_attempted"] = int(row["rounds"])
            rows.append(row)
            completed.add(key)
            _write_json(rows_path, rows)
            print(
                f"[{len(rows)}/{total_episodes}] seed={seed} condition={condition} "
                f"passed={row['passed']} tokens={row['input_tokens'] + row['output_tokens']}",
                flush=True,
            )
    pairs: list[dict[str, Any]] = []
    incomplete_pairs: list[dict[str, Any]] = []
    for seed in spec["seeds"]:
        baseline = next(
            row for row in rows if row["seed"] == seed and row["condition"] == "advisor_off"
        )
        treatment = next(
            row for row in rows if row["seed"] == seed and row["condition"] == "advisor_v9"
        )
        if not baseline.get("passed") or not treatment.get("passed"):
            incomplete_pairs.append(
                {
                    "seed": seed,
                    "baseline_status": baseline.get("status", "failed"),
                    "treatment_status": treatment.get("status", "failed"),
                }
            )
            continue
        pairs.append(
            {
                "seed": seed,
                "enterprise_value_delta_cents": int(treatment["enterprise_value_cents"])
                - int(baseline["enterprise_value_cents"]),
                "cumulative_profit_delta_cents": int(treatment["cumulative_profit_cents"])
                - int(baseline["cumulative_profit_cents"]),
                "market_share_delta_ppm": int(treatment["final_market_share_ppm"])
                - int(baseline["final_market_share_ppm"]),
                "released_advice_count": int(treatment["released_advice_count"]),
                "released_advice_accepted_count": int(
                    treatment["released_advice_accepted_count"]
                ),
            }
        )
    ev_deltas = [int(pair["enterprise_value_delta_cents"]) for pair in pairs]
    released_count = sum(int(pair["released_advice_count"]) for pair in pairs)
    accepted_count = sum(
        int(pair["released_advice_accepted_count"]) for pair in pairs
    )
    usage = {
        "calls": sum(
            int(row.get("provider_calls_attempted", row["rounds"])) for row in rows
        ),
        "prompt_tokens": sum(int(row["input_tokens"]) for row in rows),
        "completion_tokens": sum(int(row["output_tokens"]) for row in rows),
        "estimated_cost_microunits": sum(
            int(row["input_tokens"])
            * int(price["input_price_microunits_per_token"])
            + int(row["output_tokens"])
            * int(price["output_price_microunits_per_token"])
            for row in rows
        ),
        "latest_process_guard_actual": asdict(guard.actual),
    }
    summary: dict[str, Any] = {
        "result_schema_version": "final-market-multiround-real-result-v1.0.0",
        "evidence_level": spec["evidence_level"],
        "preregistration_hash": spec["preregistration_hash"],
        "episode_count": len(rows),
        "pair_count": len(pairs),
        "mean_enterprise_value_delta_cents": (
            round(mean(ev_deltas)) if ev_deltas else None
        ),
        "worst_enterprise_value_delta_cents": min(ev_deltas) if ev_deltas else None,
        "positive_zero_negative_pairs": {
            "positive": sum(value > 0 for value in ev_deltas),
            "zero": sum(value == 0 for value in ev_deltas),
            "negative": sum(value < 0 for value in ev_deltas),
        },
        "released_advice_count": released_count,
        "released_advice_adoption_rate_ppm": (
            round(accepted_count * 1_000_000 / released_count)
            if released_count
            else None
        ),
        "usage": usage,
        "gates": {
            "all_episodes_recorded": len(rows) == len(spec["cells"]) * len(CONDITIONS),
            "all_episodes_passed": all(bool(row["passed"]) for row in rows),
            "all_llm_decisions_submitted": all(
                int(row["submitted_decisions"]) == ROUNDS for row in rows
            ),
            "mean_enterprise_value_non_decreasing": bool(ev_deltas)
            and mean(ev_deltas) >= 0,
            "worst_enterprise_value_non_decreasing": bool(ev_deltas)
            and min(ev_deltas) >= 0,
        },
        "pairs": pairs,
        "incomplete_pairs": incomplete_pairs,
    }
    summary["result_hash"] = _normalized_hash(summary, "result_hash")
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--authorize-real-model", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.spec.resolve()), ensure_ascii=False, indent=2))
        return
    if not args.authorize_real_model:
        raise RuntimeError("--authorize-real-model is required")
    print(
        json.dumps(
            asyncio.run(run(args.spec.resolve(), args.output.resolve())),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
