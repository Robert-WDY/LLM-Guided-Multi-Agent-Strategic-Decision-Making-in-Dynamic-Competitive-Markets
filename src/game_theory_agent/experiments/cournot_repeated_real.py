"""Repeated Cournot real-LLM closed-loop evaluation with public-history advice."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from game_theory_agent.benchmark_games import (
    CournotAction,
    CournotEnvironment,
    CournotRoundEvent,
    replay_cournot_events,
)
from game_theory_agent.benchmark_games.repeated import (
    forecast_opponent_belief,
    rule_opponent_quantity,
)
from game_theory_agent.experiments.canonical_validity_real import (
    PRICE_SNAPSHOT,
    PROJECT_ROOT,
    _cost,
    _valid_integer,
    _write_json,
)
from game_theory_agent.experiments.real_json_client import RealJSONClient
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.strategic_reliability import RealModelBudget, RealModelCostGuard


DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "cournot-repeated-real-v1"
SEEDS = (910_201, 910_202, 910_203)
ROUNDS = 10
CONDITIONS = ("persona_only", "exact_online_advisor")
OPPONENTS = ("fixed_nash", "adaptive_best_response", "deepseek_llm")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _decision_prompt(
    *,
    role: str,
    round_number: int,
    focal_history: list[int],
    opponent_history: list[int],
    belief_ppm: dict[int, int],
    advice: dict[str, Any] | None,
) -> str:
    payload: dict[str, Any] = {
        "role": role,
        "round": round_number,
        "total_rounds": ROUNDS,
        "simultaneous_actions": True,
        "current_opponent_action_visible": False,
        "quantity_bounds": [0, 30],
        "price_formula": "max(30 - own_quantity - opponent_quantity, 0)",
        "profit_formula": "(price - 6) * own_quantity",
        "public_history": {
            "own_quantities": focal_history[-5:],
            "opponent_quantities": opponent_history[-5:],
        },
        "opponent_quantity_belief_ppm": {
            str(quantity): probability for quantity, probability in belief_ppm.items()
        },
        "objective": "最大化10轮累计私人利润；不得假设转账、通信或约束协议。",
    }
    if advice is not None:
        payload["advisor"] = {
            "recommended_quantity": advice["recommended_quantity"],
            "expected_profit_microunits": advice["expected_profit_microunits"],
            "method": "基于公开历史信念穷举0到30的全部整数候选",
            "non_binding": True,
        }
    return "\n".join(
        (
            "你在两公司重复 Cournot 数量竞争中选择本轮产量。",
            "两家公司同时行动，你不能看到对手本轮选择。",
            "只输出 JSON：{\"quantity\":0到30的整数,\"reason\":\"不超过60字的可审计理由\"}。",
            "不要输出隐藏推理过程。",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )
    )


async def _audited_generate(
    *,
    client: RealJSONClient,
    prompt: str,
    call_id: str,
    guard: RealModelCostGuard,
    audit_path: Path,
) -> Any:
    provider = client.provider
    guard.reserve(
        prompt_tokens=1_500,
        completion_tokens=300,
        estimated_cost_microunits=_cost(provider, 1_500, 300),
    )
    generation = await client.generate(prompt)
    # Persist the provider boundary before parsing or market settlement.
    _append_jsonl(
        audit_path,
        {
            "call_id": call_id,
            "provider": generation.provider,
            "model": generation.model,
            "prompt_hash": sha256_hash(prompt),
            "prompt": prompt,
            "raw_response": generation.raw_response,
            "parsed_output": generation.parsed,
            "parse_error": generation.parse_error,
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
            "latency_ms": generation.latency_ms,
            "request_started_at": generation.request_started_at,
            "response_received_at": generation.response_received_at,
            "request_id": generation.request_id,
            "response_model": generation.response_model,
        },
    )
    guard.record_actual(
        prompt_tokens=generation.input_tokens,
        completion_tokens=generation.output_tokens,
        estimated_cost_microunits=_cost(
            provider, generation.input_tokens, generation.output_tokens
        ),
    )
    return generation


async def _audited_quantity(
    *,
    client: RealJSONClient,
    prompt: str,
    call_id: str,
    guard: RealModelCostGuard,
    audit_path: Path,
) -> tuple[Any, int, bool]:
    generation = await _audited_generate(
        client=client,
        prompt=prompt,
        call_id=call_id,
        guard=guard,
        audit_path=audit_path,
    )
    quantity = _valid_integer((generation.parsed or {}).get("quantity"), 0, 30)
    if quantity is not None:
        return generation, quantity, False
    repair_prompt = "\n".join(
        (
            "上一次输出不是完整合法 JSON。根据原问题和上次文本，修复为最终答案。",
            "只能输出 {\"quantity\":0到30的整数,\"reason\":\"不超过30字\"}。",
            "quantity 必须反映上次文本最后确认的选择，不要使用开头被后文纠正的草稿数字。",
            "[原问题]",
            prompt,
            "[上次无效输出]",
            generation.raw_response,
        )
    )
    repaired = await _audited_generate(
        client=client,
        prompt=repair_prompt,
        call_id=f"{call_id}|schema_repair",
        guard=guard,
        audit_path=audit_path,
    )
    quantity = _valid_integer((repaired.parsed or {}).get("quantity"), 0, 30)
    if quantity is None:
        raise RuntimeError(f"schema repair failed: {call_id}")
    return repaired, quantity, True


async def _run_episode(
    *,
    seed: int,
    condition: str,
    opponent_type: str,
    focal_client: RealJSONClient,
    opponent_client: RealJSONClient,
    guard: RealModelCostGuard,
    provider_audit_path: Path,
    call_id_attempt_suffix: str = "",
) -> tuple[list[dict[str, Any]], list[CournotRoundEvent]]:
    env = CournotEnvironment()
    focal_history: list[int] = []
    opponent_history: list[int] = []
    rows: list[dict[str, Any]] = []
    events: list[CournotRoundEvent] = []
    previous_hash: str | None = None
    for round_number in range(1, ROUNDS + 1):
        belief = forecast_opponent_belief(
            opponent_type,
            focal_history=focal_history,
            opponent_history=opponent_history,
        )
        advice = env.advise(company_id="A", belief_ppm=belief).model_dump(mode="json")
        focal_prompt = _decision_prompt(
            role="company_A",
            round_number=round_number,
            focal_history=focal_history,
            opponent_history=opponent_history,
            belief_ppm=belief,
            advice=advice if condition == "exact_online_advisor" else None,
        )
        prefix = (
            f"seed{seed}|{condition}|{opponent_type}|round{round_number}"
            f"{call_id_attempt_suffix}"
        )
        if opponent_type == "deepseek_llm":
            reverse_belief = forecast_opponent_belief(
                "deepseek_llm",
                focal_history=opponent_history,
                opponent_history=focal_history,
            )
            opponent_prompt = _decision_prompt(
                role="company_B",
                round_number=round_number,
                focal_history=opponent_history,
                opponent_history=focal_history,
                belief_ppm=reverse_belief,
                advice=None,
            )
            prompts = {
                "focal": (focal_client, focal_prompt, f"{prefix}|focal"),
                "opponent": (
                    opponent_client,
                    opponent_prompt,
                    f"{prefix}|opponent",
                ),
            }
            generations: dict[str, Any] = {}
            order = ("focal", "opponent") if (seed + round_number) % 2 else ("opponent", "focal")
            for name in order:
                client, prompt, call_id = prompts[name]
                generations[name] = await _audited_quantity(
                    client=client,
                    prompt=prompt,
                    call_id=call_id,
                    guard=guard,
                    audit_path=provider_audit_path,
                )
            focal_generation, focal_quantity, focal_repaired = generations["focal"]
            opponent_generation, opponent_quantity, opponent_repaired = generations["opponent"]
        else:
            focal_generation, focal_quantity, focal_repaired = await _audited_quantity(
                client=focal_client,
                prompt=focal_prompt,
                call_id=f"{prefix}|focal",
                guard=guard,
                audit_path=provider_audit_path,
            )
            opponent_repaired = False
            opponent_generation = None
            opponent_quantity = rule_opponent_quantity(
                opponent_type,
                focal_history=focal_history,
            )
        actions = (
            CournotAction(company_id="A", quantity=focal_quantity),
            CournotAction(company_id="B", quantity=opponent_quantity),
        )
        outcome = env.settle(actions)
        event = CournotRoundEvent.create(
            episode_id=f"cournot-repeated-{seed}-{condition}-{opponent_type}",
            round_number=round_number,
            config=env.config,
            actions=actions,
            outcome=outcome,
            previous_event_hash=previous_hash,
        )
        previous_hash = event.event_hash
        events.append(event)
        rows.append(
            {
                "seed": seed,
                "condition": condition,
                "opponent_type": opponent_type,
                "round": round_number,
                "belief_ppm": belief,
                "advice": advice,
                "focal_quantity": focal_quantity,
                "opponent_quantity": opponent_quantity,
                "price": outcome.price,
                "focal_profit": outcome.profits["A"],
                "opponent_profit": outcome.profits["B"],
                "exact_ex_post_regret": env.regret(focal_quantity, opponent_quantity),
                "consumer_surplus_twice": outcome.consumer_surplus_twice,
                "total_welfare_twice": outcome.total_welfare_twice,
                "nash_distance": abs(focal_quantity - 8) + abs(opponent_quantity - 8),
                "collusion_distance": abs(focal_quantity - 6) + abs(opponent_quantity - 6),
                "advisor_adopted": (
                    focal_quantity == advice["recommended_quantity"]
                    if condition == "exact_online_advisor"
                    else None
                ),
                "focal_schema_repaired": focal_repaired,
                "opponent_schema_repaired": opponent_repaired,
                "focal_reason": (focal_generation.parsed or {}).get("reason"),
                "opponent_reason": (
                    (opponent_generation.parsed or {}).get("reason")
                    if opponent_generation is not None
                    else None
                ),
                "event_hash": event.event_hash,
            }
        )
        focal_history.append(focal_quantity)
        opponent_history.append(opponent_quantity)
    if not replay_cournot_events(events):
        raise RuntimeError("Cournot episode replay failed")
    return rows, events


def _episode_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cumulative_focal_profit": sum(row["focal_profit"] for row in rows),
        "cumulative_opponent_profit": sum(row["opponent_profit"] for row in rows),
        "cumulative_ex_post_regret": sum(row["exact_ex_post_regret"] for row in rows),
        "mean_price": mean(row["price"] for row in rows),
        "mean_nash_distance": mean(row["nash_distance"] for row in rows),
        "mean_collusion_distance": mean(row["collusion_distance"] for row in rows),
        "cumulative_consumer_surplus_twice": sum(row["consumer_surplus_twice"] for row in rows),
        "cumulative_total_welfare_twice": sum(row["total_welfare_twice"] for row in rows),
        "advisor_adoption_rate": (
            mean(bool(row["advisor_adopted"]) for row in rows)
            if rows[0]["condition"] == "exact_online_advisor"
            else None
        ),
    }


async def run(output: Path) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    fresh_manifest = {
        "schema_version": "cournot-repeated-real-manifest-v1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "seeds": list(SEEDS),
        "rounds": ROUNDS,
        "conditions": list(CONDITIONS),
        "opponents": list(OPPONENTS),
        "focal_provider": "doubao",
        "llm_opponent_provider": "deepseek",
        "provider_seed_supported": False,
        "planned_calls": 240,
        "maximum_provider_calls_including_failed_attempts": 250,
        "price_snapshot": PRICE_SNAPSHOT,
        "evidence_level": "small_closed_loop_directional_real_model_evidence",
        "information_boundary": "both LLMs decide from prior public history; neither sees the simultaneous current action",
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != fresh_manifest["schema_version"]:
            raise RuntimeError("cannot resume an incompatible manifest")
    else:
        manifest = fresh_manifest
        _write_json(manifest_path, manifest)
    guard = RealModelCostGuard(
        RealModelBudget(
            max_calls=250,
            max_prompt_tokens=375_000,
            max_completion_tokens=75_000,
            max_estimated_cost_microunits=1_000_000,
        ),
        explicitly_authorized=True,
    )
    focal_client = RealJSONClient("doubao", temperature=0.0)
    opponent_client = RealJSONClient("deepseek", temperature=0.0)
    provider_audit_path = output / "provider_calls.jsonl"
    provider_rows = (
        [json.loads(line) for line in provider_audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if provider_audit_path.exists()
        else []
    )
    for row in provider_rows:
        guard.reserve(
            prompt_tokens=1_500,
            completion_tokens=300,
            estimated_cost_microunits=_cost(row["provider"], 1_500, 300),
        )
        guard.record_actual(
            prompt_tokens=int(row["input_tokens"]),
            completion_tokens=int(row["output_tokens"]),
            estimated_cost_microunits=_cost(
                row["provider"], int(row["input_tokens"]), int(row["output_tokens"])
            ),
        )
    rounds_path = output / "rounds.jsonl"
    episodes_path = output / "episodes.jsonl"
    round_rows = (
        [json.loads(line) for line in rounds_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rounds_path.exists()
        else []
    )
    episode_rows = (
        [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if episodes_path.exists()
        else []
    )
    completed_keys = {
        (int(row["seed"]), str(row["condition"]), str(row["opponent_type"]))
        for row in episode_rows
    }
    completed_episodes = len(episode_rows)
    for seed in SEEDS:
        opponent_order = OPPONENTS if seed % 2 else tuple(reversed(OPPONENTS))
        for opponent_index, opponent_type in enumerate(opponent_order):
            condition_order = CONDITIONS if (seed + opponent_index) % 2 else tuple(reversed(CONDITIONS))
            for condition in condition_order:
                episode_key = (seed, condition, opponent_type)
                if episode_key in completed_keys:
                    continue
                episode_prefix = f"seed{seed}|{condition}|{opponent_type}|"
                has_incomplete_attempt = any(
                    str(row["call_id"]).startswith(episode_prefix)
                    for row in provider_rows
                )
                attempt_suffix = (
                    f"|resume_after_call_{len(provider_rows)}"
                    if has_incomplete_attempt
                    else ""
                )
                if has_incomplete_attempt:
                    _write_json(
                        output / "incident.json",
                        {
                            "incident_schema_version": "cournot-repeated-incident-v1.0.0",
                            "episode_key": list(episode_key),
                            "cause": "invalid model JSON was preserved before settlement",
                            "recovery": "restart only the incomplete episode with unique call IDs; never overwrite or silently coerce the invalid response",
                            "provider_calls_before_resume": len(provider_rows),
                        },
                    )
                rows, events = await _run_episode(
                    seed=seed,
                    condition=condition,
                    opponent_type=opponent_type,
                    focal_client=focal_client,
                    opponent_client=opponent_client,
                    guard=guard,
                    provider_audit_path=provider_audit_path,
                    call_id_attempt_suffix=attempt_suffix,
                )
                for row in rows:
                    _append_jsonl(output / "rounds.jsonl", row)
                round_rows.extend(rows)
                episode = {
                    "seed": seed,
                    "condition": condition,
                    "opponent_type": opponent_type,
                    "rounds": len(rows),
                    "replay_passed": replay_cournot_events(events),
                    **_episode_summary(rows),
                }
                episode_rows.append(episode)
                _append_jsonl(output / "episodes.jsonl", episode)
                completed_episodes += 1
                print(
                    f"[{completed_episodes}/18] seed={seed} opponent={opponent_type} condition={condition} profit={episode['cumulative_focal_profit']} regret={episode['cumulative_ex_post_regret']} calls={guard.actual.calls}",
                    flush=True,
                )
    pair_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for opponent_type in OPPONENTS:
            baseline = next(row for row in episode_rows if row["seed"] == seed and row["opponent_type"] == opponent_type and row["condition"] == "persona_only")
            treatment = next(row for row in episode_rows if row["seed"] == seed and row["opponent_type"] == opponent_type and row["condition"] == "exact_online_advisor")
            pair_rows.append(
                {
                    "seed": seed,
                    "opponent_type": opponent_type,
                    "profit_delta": treatment["cumulative_focal_profit"] - baseline["cumulative_focal_profit"],
                    "regret_reduction": baseline["cumulative_ex_post_regret"] - treatment["cumulative_ex_post_regret"],
                    "consumer_surplus_twice_delta": treatment["cumulative_consumer_surplus_twice"] - baseline["cumulative_consumer_surplus_twice"],
                    "total_welfare_twice_delta": treatment["cumulative_total_welfare_twice"] - baseline["cumulative_total_welfare_twice"],
                    "mean_price_delta": treatment["mean_price"] - baseline["mean_price"],
                    "mean_nash_distance_delta": treatment["mean_nash_distance"] - baseline["mean_nash_distance"],
                    "mean_collusion_distance_delta": treatment["mean_collusion_distance"] - baseline["mean_collusion_distance"],
                    "treatment_advisor_adoption_rate": treatment["advisor_adoption_rate"],
                }
            )
    by_opponent: dict[str, Any] = {}
    for opponent_type in OPPONENTS:
        selected = [row for row in pair_rows if row["opponent_type"] == opponent_type]
        by_opponent[opponent_type] = {
            "pairs": len(selected),
            "mean_profit_delta": mean(row["profit_delta"] for row in selected),
            "mean_regret_reduction": mean(row["regret_reduction"] for row in selected),
            "positive_zero_negative_profit": {
                "positive": sum(row["profit_delta"] > 0 for row in selected),
                "zero": sum(row["profit_delta"] == 0 for row in selected),
                "negative": sum(row["profit_delta"] < 0 for row in selected),
            },
            "mean_consumer_surplus_twice_delta": mean(row["consumer_surplus_twice_delta"] for row in selected),
            "mean_total_welfare_twice_delta": mean(row["total_welfare_twice_delta"] for row in selected),
            "mean_price_delta": mean(row["mean_price_delta"] for row in selected),
            "mean_nash_distance_delta": mean(row["mean_nash_distance_delta"] for row in selected),
            "mean_collusion_distance_delta": mean(row["mean_collusion_distance_delta"] for row in selected),
            "mean_advisor_adoption_rate": mean(row["treatment_advisor_adoption_rate"] for row in selected),
        }
    summary = {
        "schema_version": "cournot-repeated-real-summary-v1.0.0",
        "evidence_level": manifest["evidence_level"],
        "episodes": len(episode_rows),
        "rounds": len(round_rows),
        "provider_calls": guard.actual.calls,
        "planned_provider_calls": manifest["planned_calls"],
        "additional_calls_from_interrupted_attempts_and_schema_repairs": guard.actual.calls - manifest["planned_calls"],
        "all_replays_passed": all(row["replay_passed"] for row in episode_rows),
        "invalid_provider_outputs": [
            {
                "call_id": row["call_id"],
                "parse_error": row["parse_error"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
            }
            for row in provider_rows
            if row.get("parse_error")
        ],
        "pairs": pair_rows,
        "by_opponent": by_opponent,
        "overall": {
            "mean_profit_delta": mean(row["profit_delta"] for row in pair_rows),
            "mean_regret_reduction": mean(row["regret_reduction"] for row in pair_rows),
            "positive_zero_negative_profit": {
                "positive": sum(row["profit_delta"] > 0 for row in pair_rows),
                "zero": sum(row["profit_delta"] == 0 for row in pair_rows),
                "negative": sum(row["profit_delta"] < 0 for row in pair_rows),
            },
        },
        "usage": asdict(guard.actual),
        "estimated_cost_cny_conservative": round(guard.actual.estimated_cost_microunits / 1_000_000, 6),
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--authorize-real-model", action="store_true")
    args = parser.parse_args()
    if not args.authorize_real_model:
        raise RuntimeError("--authorize-real-model is required")
    print(json.dumps(asyncio.run(run(args.output.resolve())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
