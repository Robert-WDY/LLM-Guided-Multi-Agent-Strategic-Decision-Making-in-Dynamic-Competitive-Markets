"""Stage 6.5 token-efficient real-LLM strategic adoption experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

from game_theory_agent.experiments.four_agent_acceptance import run as run_episode
from game_theory_agent.gameplay import build_terminal_rankings
from game_theory_agent.market import MarketState, load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.orchestration import JsonlRoundEventLogger
from game_theory_agent.strategic_reliability import (
    RealModelBudget,
    RealModelCostGuard,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v4.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "stage6.5-real-adoption"
COMPANIES = ("company_A", "company_B", "company_C", "company_D")
PERSONAS = (
    "aggressive_v1_extreme",
    "profit_myopic",
    "risk_guarded_v1",
)
CONDITIONS: dict[str, dict[str, str]] = {
    "persona_only": {
        "belief_mode": "off",
        "opponent_model_mode": "off",
        "utility_inference_mode": "off",
        "advisor_mode": "off",
    },
    "action_belief": {
        "belief_mode": "public_action_v1",
        "opponent_model_mode": "off",
        "utility_inference_mode": "off",
        "advisor_mode": "off",
    },
    "strategy_utility": {
        "belief_mode": "public_action_v1",
        "opponent_model_mode": "public_strategy_v1",
        "utility_inference_mode": "strategy_utility_v1",
        "advisor_mode": "off",
    },
    "bayesian_v2": {
        "belief_mode": "public_action_v1",
        "opponent_model_mode": "public_strategy_v1",
        "utility_inference_mode": "strategy_utility_v1",
        "advisor_mode": "bayesian_strategy_v2",
    },
    "pareto_v4": {
        "belief_mode": "public_action_v1",
        "opponent_model_mode": "public_strategy_v1",
        "utility_inference_mode": "strategy_utility_v1",
        "advisor_mode": "pareto_rollout_v4",
    },
}
ECONOMIC_FIELDS = (
    "price_cents",
    "advertising_budget_cents",
    "service_budget_cents",
    "capacity_investment_cents",
    "resilience_budget_cents",
    "shared_resilience_contribution_cents",
    "incident_response",
)


def parse_csv(value: str) -> tuple[str, ...]:
    rows = tuple(item.strip() for item in value.split(",") if item.strip())
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("values must be a non-empty unique CSV")
    return rows


def parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in parse_csv(value))


def build_plan(
    seeds: tuple[int, ...], personas: tuple[str, ...]
) -> list[dict[str, Any]]:
    names = tuple(CONDITIONS)
    plan: list[dict[str, Any]] = []
    for cell_index, (seed, persona) in enumerate(
        (seed, persona) for seed in seeds for persona in personas
    ):
        shift = cell_index % len(names)
        order = names[shift:] + names[:shift]
        for call_order, condition in enumerate(order, start=1):
            plan.append(
                {
                    "seed": seed,
                    "persona_id": persona,
                    "condition": condition,
                    "condition_call_order": call_order,
                    **CONDITIONS[condition],
                }
            )
    return plan


def _episode_args(
    args: argparse.Namespace,
    item: dict[str, Any],
    output: Path,
    guard: RealModelCostGuard,
) -> SimpleNamespace:
    return SimpleNamespace(
        # Conditions intentionally reuse one episode id.  Rule opponents derive
        # seeded variation from this id, so changing it would break pairing.
        episode_id=f"stage65-{item['seed']}-{item['persona_id']}",
        seed=item["seed"],
        rounds=args.rounds,
        market_model="balanced",
        information_mode="public",
        provider=args.provider,
        belief_mode=item["belief_mode"],
        opponent_model_mode=item["opponent_model_mode"],
        utility_inference_mode=item["utility_inference_mode"],
        advisor_mode=item["advisor_mode"],
        repeated_game_mode="off",
        cooperation_mode="off",
        honor_game_theory_advice=False,
        model=args.model,
        persona=item["persona_id"],
        condition=None,
        personas=None,
        llm_count=1,
        rotation_index=0,
        decision_support_version="economic_v2",
        persona_semantics_version="economic_v2",
        diagnostic_mode="off",
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.timeout,
        communication_mode="off",
        communication_timeout=30.0,
        mock_communication_scenario="silence",
        quiet=True,
        output=output,
        paid_decision_rounds=(args.paid_round,),
        real_model_cost_guard=guard,
        agent_configs={
            "company_A": {
                "persona": {"persona_id": item["persona_id"]}
            }
        },
    )


def _economic_action(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {field: action.get(field) for field in ECONOMIC_FIELDS}


def _economic_state_hash(raw_state: dict[str, Any]) -> str:
    volatile = {
        "episode_id",
        "state_hash",
        "last_action_id",
        "action_id",
        "event_id",
        "signal_id",
    }

    def cleaned(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: cleaned(item)
                for key, item in sorted(value.items())
                if key not in volatile
            }
        if isinstance(value, list):
            return [cleaned(item) for item in value]
        return value

    return sha256_hash(cleaned(raw_state))


def summarize_episode(
    directory: Path, item: dict[str, Any], paid_round: int
) -> dict[str, Any]:
    events = list(
        JsonlRoundEventLogger(directory / "round-events.jsonl").read_all()
    )
    event = next(row for row in events if row.settled_round == paid_round)
    trace = next(row for row in event.traces if row.company_id == "company_A")
    terminal = MarketState.from_dict(events[-1].state_after)
    company = terminal.company("company_A")
    enterprise_value = next(
        int(row["value_cents"])
        for row in build_terminal_rankings(
            terminal, load_market_config(CONFIG_PATH)
        )["composite"]
        if row["company_id"] == "company_A"
    )
    adoption = (
        trace.advisor_adoption.model_dump(mode="json")
        if trace.advisor_adoption is not None
        else None
    )
    return {
        **item,
        "paid_round": paid_round,
        "decision_status": trace.decision_status,
        "model_name": trace.model_name,
        "input_tokens": int(trace.input_tokens or 0),
        "output_tokens": int(trace.output_tokens or 0),
        "retry_count": trace.retry_count,
        "pre_decision_economic_state_hash": _economic_state_hash(
            event.state_before
        ),
        "requested_action": _economic_action(trace.requested_action),
        "final_action": _economic_action(trace.final_action),
        "planner_output": trace.planner_output,
        "advisor_adoption": adoption,
        "enterprise_value_cents": enterprise_value,
        "cumulative_profit_cents": (
            company.financial.cumulative_profit_cents
        ),
        "final_market_share_ppm": company.commercial.market_share_ppm,
        "final_cash_cents": company.financial.cash_balance_cents,
        "final_state_hash": terminal.state_hash,
    }


def _paired(rows: list[dict[str, Any]], treatment: str, control: str) -> dict[str, Any]:
    controls = {
        (row["seed"], row["persona_id"]): row
        for row in rows
        if row["condition"] == control
    }
    treated = [row for row in rows if row["condition"] == treatment]
    if len(treated) != len(controls):
        raise ValueError("Stage 6.5 comparison is not completely paired")
    ev_deltas: list[int] = []
    profit_deltas: list[int] = []
    regret_deltas: list[int] = []
    action_changes = 0
    for row in treated:
        baseline = controls[(row["seed"], row["persona_id"])]
        ev_deltas.append(
            row["enterprise_value_cents"] - baseline["enterprise_value_cents"]
        )
        profit_deltas.append(
            row["cumulative_profit_cents"] - baseline["cumulative_profit_cents"]
        )
        regret_deltas.append(
            row["observed_action_regret_cents"]
            - baseline["observed_action_regret_cents"]
        )
        action_changes += row["requested_action"] != baseline["requested_action"]
    return {
        "treatment": treatment,
        "control": control,
        "pair_count": len(treated),
        "action_change_rate_ppm": action_changes * 1_000_000 // len(treated),
        "mean_enterprise_value_delta_cents": round(mean(ev_deltas)),
        "worst_enterprise_value_delta_cents": min(ev_deltas),
        "positive_zero_negative_ev_pairs": {
            "positive": sum(value > 0 for value in ev_deltas),
            "zero": sum(value == 0 for value in ev_deltas),
            "negative": sum(value < 0 for value in ev_deltas),
        },
        "mean_cumulative_profit_delta_cents": round(mean(profit_deltas)),
        "mean_observed_action_regret_delta_cents": round(mean(regret_deltas)),
    }


def aggregate(
    rows: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    personas: tuple[str, ...],
    guard: RealModelCostGuard,
    rounds: int,
    paid_round: int,
) -> dict[str, Any]:
    cell_keys = {
        (row["seed"], row["persona_id"])
        for row in rows
    }
    best_value_by_cell = {
        (seed, persona): max(
            row["enterprise_value_cents"]
            for row in rows
            if row["seed"] == seed and row["persona_id"] == persona
        )
        for seed, persona in cell_keys
    }
    for row in rows:
        row["observed_action_regret_cents"] = (
            best_value_by_cell[(row["seed"], row["persona_id"])]
            - row["enterprise_value_cents"]
        )
    cells: dict[tuple[int, str], set[str]] = {}
    for row in rows:
        cells.setdefault((row["seed"], row["persona_id"]), set()).add(
            row["pre_decision_economic_state_hash"]
        )
    adoption_rows = [
        row["advisor_adoption"]
        for row in rows
        if row["condition"] in {"bayesian_v2", "pareto_v4"}
        and row["advisor_adoption"] is not None
    ]
    adoption_by_mode = {
        condition: {
            "trace_count": len(items),
            "accepted_count": sum(item["accepted"] for item in items),
            "adoption_rate_ppm": (
                sum(item["accepted"] for item in items) * 1_000_000 // len(items)
                if items
                else 0
            ),
            "mean_target_alignment_ppm": (
                round(mean(item["target_alignment_ppm"] for item in items))
                if items
                else 0
            ),
            "status_counts": {
                status: sum(item["adoption_status"] == status for item in items)
                for status in (
                    "exact_action",
                    "accepted_target",
                    "partial",
                    "rejected",
                )
            },
        }
        for condition in ("bayesian_v2", "pareto_v4")
        for items in [[
            row["advisor_adoption"]
            for row in rows
            if row["condition"] == condition
            and row["advisor_adoption"] is not None
        ]]
    }
    actual = guard.actual
    actual_cost_cny = actual.estimated_cost_microunits / 1_000_000
    comparisons = {
        "action_belief_vs_persona": _paired(
            rows, "action_belief", "persona_only"
        ),
        "strategy_utility_vs_action_belief": _paired(
            rows, "strategy_utility", "action_belief"
        ),
        "bayesian_v2_vs_strategy_utility": _paired(
            rows, "bayesian_v2", "strategy_utility"
        ),
        "pareto_v4_vs_strategy_utility": _paired(
            rows, "pareto_v4", "strategy_utility"
        ),
        "pareto_v4_vs_persona": _paired(rows, "pareto_v4", "persona_only"),
    }
    engineering_checks = {
        "matrix_complete": len(rows)
        == len(cell_keys) * len(CONDITIONS),
        "exactly_one_paid_call_per_episode": actual.calls == len(rows),
        "all_paid_decisions_submitted": all(
            row["decision_status"] == "submitted" for row in rows
        ),
        "provider_usage_complete": all(
            row["input_tokens"] > 0 and row["output_tokens"] > 0 for row in rows
        ),
        "matched_pre_decision_states": all(len(items) == 1 for items in cells.values()),
        "advisor_traces_complete": len(adoption_rows)
        == len(cell_keys) * 2,
        "budget_not_exceeded": True,
    }
    directional_gate = {
        "pareto_changes_actions": comparisons[
            "pareto_v4_vs_strategy_utility"
        ]["action_change_rate_ppm"]
        >= 200_000,
        "pareto_is_adopted_sometimes": adoption_by_mode["pareto_v4"][
            "accepted_count"
        ]
        > 0,
        "pareto_mean_ev_not_lower": comparisons[
            "pareto_v4_vs_strategy_utility"
        ]["mean_enterprise_value_delta_cents"]
        >= 0,
        "pareto_mean_observed_regret_not_higher": comparisons[
            "pareto_v4_vs_strategy_utility"
        ]["mean_observed_action_regret_delta_cents"]
        <= 0,
        "pareto_worst_value_not_lower": comparisons[
            "pareto_v4_vs_strategy_utility"
        ]["worst_enterprise_value_delta_cents"]
        >= 0,
    }
    seed_coverage = {
        persona: sorted(
            seed for seed, item_persona in cell_keys
            if item_persona == persona
        )
        for persona in personas
    }
    research_completion_checks = {
        "three_personas_covered": set(PERSONAS).issubset(personas),
        "at_least_ten_seeds_per_persona": bool(seed_coverage)
        and all(len(items) >= 10 for items in seed_coverage.values()),
        "known_mixed_holdout_opponents_covered": False,
        "directional_gate_passed": all(directional_gate.values()),
    }
    return {
        "experiment_schema_version": "stage6.5-real-adoption-v1.0.0",
        "evidence_level": "paired_real_llm_single_paid_decision_per_episode",
        "seeds": list(seeds),
        "personas": list(personas),
        "seed_coverage_by_persona": seed_coverage,
        "conditions": list(CONDITIONS),
        "rounds": rounds,
        "paid_decision_round": paid_round,
        "real_model_usage": {
            **asdict(actual),
            "estimated_cost_cny": round(actual_cost_cny, 6),
            "pricing_note": (
                "conservative guard accounting: input 1 and output 4 CNY "
                "microunits per token; actual provider invoice may differ"
            ),
        },
        "adoption_by_mode": adoption_by_mode,
        "comparisons": comparisons,
        "engineering_checks": engineering_checks,
        "engineering_passed": all(engineering_checks.values()),
        "directional_gate": directional_gate,
        "directional_gate_passed": all(directional_gate.values()),
        "research_completion_checks": research_completion_checks,
        "stage65_complete": all(research_completion_checks.values()),
        "real_model_expansion_stopped_by_gate": not all(
            directional_gate.values()
        ),
        "rows": rows,
        "conclusion_limits": [
            "only one paid LLM decision is sampled in each ten-round episode",
            "long-term Regret is bounded to the five observed LLM actions and three post-decision rounds",
            "three deterministic Rule opponents are used; Known/Mixed/Holdout latent opponent generalization is not tested",
            "adaptive Seed coverage is directional paired evidence, not statistical significance",
            "provider stochastic output is recorded rather than regenerated during replay",
        ],
    }


def merge_existing_summaries(
    sources: tuple[Path, ...], output: Path
) -> dict[str, Any]:
    loaded = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sources
    ]
    rows = [row for summary in loaded for row in summary["rows"]]
    keys = [(row["seed"], row["persona_id"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("merged Stage 6.5 summaries contain duplicate cells")
    budget = RealModelBudget(
        max_calls=len(rows),
        max_prompt_tokens=len(rows) * 32_000,
        max_completion_tokens=len(rows) * 4_000,
        max_estimated_cost_microunits=len(rows) * 48_000,
    )
    guard = RealModelCostGuard(budget, explicitly_authorized=True)
    for row in rows:
        guard.reserve(
            prompt_tokens=32_000,
            completion_tokens=4_000,
            estimated_cost_microunits=48_000,
        )
        guard.record_actual(
            prompt_tokens=int(row["input_tokens"]),
            completion_tokens=int(row["output_tokens"]),
            estimated_cost_microunits=(
                int(row["input_tokens"]) + 4 * int(row["output_tokens"])
            ),
        )
    seeds = tuple(sorted({int(row["seed"]) for row in rows}))
    personas = tuple(sorted({str(row["persona_id"]) for row in rows}))
    summary = aggregate(
        rows,
        seeds=seeds,
        personas=personas,
        guard=guard,
        rounds=int(loaded[0]["rounds"]),
        paid_round=int(loaded[0]["paid_decision_round"]),
    )
    summary["adaptive_sampling"] = True
    summary["source_summaries"] = [str(path.resolve()) for path in sources]
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


async def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    seeds = parse_seeds(args.seeds)
    personas = parse_csv(args.personas)
    plan = build_plan(seeds, personas)
    budget = RealModelBudget(
        max_calls=len(plan),
        max_prompt_tokens=len(plan) * 32_000,
        max_completion_tokens=len(plan) * 4_000,
        max_estimated_cost_microunits=len(plan) * 48_000,
    )
    guard = RealModelCostGuard(
        budget, explicitly_authorized=args.authorize_real_model
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    partial_path = output / "partial-summary.json"
    for index, item in enumerate(plan, start=1):
        directory = (
            output
            / f"seed-{item['seed']}"
            / item["persona_id"]
            / item["condition"]
        )
        row_path = directory / "stage65-row.json"
        if row_path.exists() and args.resume:
            row = json.loads(row_path.read_text(encoding="utf-8"))
            # Reconstruct both the conservative reservation and actual usage so a
            # resumed run remains subject to exactly the same global cost limits.
            guard.reserve(
                prompt_tokens=32_000,
                completion_tokens=4_000,
                estimated_cost_microunits=48_000,
            )
            guard.record_actual(
                prompt_tokens=int(row["input_tokens"]),
                completion_tokens=int(row["output_tokens"]),
                estimated_cost_microunits=(
                    int(row["input_tokens"]) + 4 * int(row["output_tokens"])
                ),
            )
            rows.append(row)
            continue
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"existing result requires a fresh output directory: {directory}"
            )
        exit_code = await run_episode(
            _episode_args(args, item, directory, guard)
        )
        if exit_code != 0:
            raise RuntimeError(f"Stage 6.5 episode failed: {item}")
        row = summarize_episode(directory, item, args.paid_round)
        row_path.write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append(row)
        partial_path.write_text(
            json.dumps(
                {
                    "completed": index,
                    "planned": len(plan),
                    "actual_usage": asdict(guard.actual),
                    "latest": {
                        "seed": item["seed"],
                        "persona": item["persona_id"],
                        "condition": item["condition"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if not args.quiet:
            print(
                f"[{index}/{len(plan)}] seed={item['seed']} "
                f"persona={item['persona_id']} condition={item['condition']} "
                f"tokens={row['input_tokens'] + row['output_tokens']}"
            )
    summary = aggregate(
        rows,
        seeds=seeds,
        personas=personas,
        guard=guard,
        rounds=args.rounds,
        paid_round=args.paid_round,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(i) for i in range(1, 11)))
    parser.add_argument("--personas", default=",".join(PERSONAS))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--paid-round", type=int, default=7)
    parser.add_argument(
        "--provider",
        choices=("mock", "doubao", "deepseek"),
        default="doubao",
    )
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--authorize-real-model", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--merge-summaries", nargs="+", type=Path)
    args = parser.parse_args()
    if not 1 <= args.paid_round <= args.rounds:
        parser.error("--paid-round must be inside the episode")
    summary = (
        merge_existing_summaries(
            tuple(args.merge_summaries), Path(args.output).resolve()
        )
        if args.merge_summaries
        else asyncio.run(run(args))
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve() / "summary.json"),
                "engineering_passed": summary["engineering_passed"],
                "directional_gate_passed": summary["directional_gate_passed"],
                "real_model_usage": summary["real_model_usage"],
                "adoption_by_mode": summary["adoption_by_mode"],
                "comparisons": summary["comparisons"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
