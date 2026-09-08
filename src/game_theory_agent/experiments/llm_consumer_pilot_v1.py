"""Small, cost-bounded LLM-as-synthetic-consumer preference pilot."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from dotenv import load_dotenv

from game_theory_agent.market.protocols import sha256_hash


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = (
    PROJECT_ROOT
    / "experiment-specs"
    / "llm-consumer-pilot-v1"
    / "PREREGISTRATION.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "llm-consumer-pilot-v1"
SEGMENTS = {
    "price_sensitive": "你的预算较紧，价格通常是第一考虑，但明显差的服务也会让你放弃。",
    "brand_sensitive": "你重视品牌信誉与品质确定性，愿为可靠品牌支付合理溢价。",
    "service_sensitive": "你重视服务体验和履约稳定，愿为明显更好的服务支付合理溢价。",
}
TASKS = {
    "normal_tradeoff": [
        {"id": "A", "price_cents": 9200, "brand": 42, "service": 48, "reliability": 82},
        {"id": "B", "price_cents": 10500, "brand": 86, "service": 62, "reliability": 90},
        {"id": "C", "price_cents": 10300, "brand": 58, "service": 88, "reliability": 94},
    ],
    "high_price_outside": [
        {"id": "A", "price_cents": 14000, "brand": 48, "service": 55, "reliability": 84},
        {"id": "B", "price_cents": 15500, "brand": 88, "service": 68, "reliability": 91},
        {"id": "C", "price_cents": 17000, "brand": 65, "service": 91, "reliability": 95},
    ],
    "moderate_competition": [
        {"id": "A", "price_cents": 10000, "brand": 58, "service": 60, "reliability": 88},
        {"id": "B", "price_cents": 11200, "brand": 90, "service": 66, "reliability": 93},
        {"id": "C", "price_cents": 10800, "brand": 64, "service": 89, "reliability": 96},
    ],
}
REPEATS = 2
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


def _extract_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object without importing the model-client package graph."""

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    return value


def prepare(spec_path: Path) -> dict[str, Any]:
    if spec_path.exists():
        raise RuntimeError("preregistration already exists; refusing to overwrite it")
    cells = [
        {"segment": segment, "task": task, "repeat": repeat}
        for segment in SEGMENTS
        for task in TASKS
        for repeat in range(1, REPEATS + 1)
    ]
    calls = len(cells)
    spec: dict[str, Any] = {
        "experiment_schema_version": "llm-consumer-pilot-prereg-v1.0.0",
        "evidence_level": "EXPLORATORY_SYNTHETIC_LLM_RESPONDENT_EVIDENCE",
        "provider": "deepseek",
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "temperature_ppm": 400_000,
        "top_p_ppm": 800_000,
        "segments": SEGMENTS,
        "tasks": TASKS,
        "repeats": REPEATS,
        "cells": cells,
        "maximum_provider_calls": calls,
        "maximum_prompt_tokens": calls * 2_000,
        "maximum_completion_tokens": calls * 800,
        "maximum_estimated_cost_microunits": calls
        * (2_000 * INPUT_PRICE_MICROUNITS + 800 * OUTPUT_PRICE_MICROUNITS),
        "claim_boundary": (
            "LLM answers are synthetic preference probes, not people, survey data, "
            "revealed preference, or external market calibration"
        ),
        "parameter_use_rule": (
            "use only direction and broad ranges; retain explicit synthetic status "
            "and deterministic economic-shape gates"
        ),
    }
    spec["preregistration_hash"] = _normalized_hash(spec, "preregistration_hash")
    _write_json(spec_path, spec)
    return spec


def _prompt(segment: str, task: str, repeat: int) -> str:
    return (
        "这是一个匿名消费选择实验。你不是企业，也不要考虑企业利润。\n"
        f"消费者特征：{SEGMENTS[segment]}\n"
        "商品功能相同，价格单位为分；品牌、服务和履约可靠性为0到100分。\n"
        f"选项：{json.dumps(TASKS[task], ensure_ascii=False)}\n"
        "可以选择A、B、C，也可以选择OUTSIDE表示本轮不购买。\n"
        "只输出JSON，字段必须为：choice；max_willingness_to_pay_cents（0到20000整数）；"
        "importance（price、brand、service、reliability四个0到100整数且总和100）；"
        "reason（一句中文）。不要假设折扣、借款或题外信息。\n"
        f"独立重复编号：{repeat}。"
    )


def _validate(payload: Mapping[str, Any]) -> dict[str, Any]:
    choice = str(payload.get("choice", "")).upper()
    if choice not in {"A", "B", "C", "OUTSIDE"}:
        raise ValueError("choice must be A/B/C/OUTSIDE")
    wtp = payload.get("max_willingness_to_pay_cents")
    if isinstance(wtp, bool) or not isinstance(wtp, int) or not 0 <= wtp <= 20_000:
        raise ValueError("invalid max_willingness_to_pay_cents")
    importance = payload.get("importance")
    keys = {"price", "brand", "service", "reliability"}
    if not isinstance(importance, Mapping) or set(importance) != keys:
        raise ValueError("invalid importance object")
    values = {key: importance[key] for key in keys}
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100
        for value in values.values()
    ) or sum(values.values()) != 100:
        raise ValueError("importance must be integer percentages summing to 100")
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise ValueError("reason is required")
    return {
        "choice": choice,
        "max_willingness_to_pay_cents": wtp,
        "importance": {key: int(importance[key]) for key in sorted(keys)},
        "reason": reason[:500],
    }


async def run(spec_path: Path, output: Path) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec["preregistration_hash"] != _normalized_hash(spec, "preregistration_hash"):
        raise RuntimeError("preregistration hash mismatch")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout=45.0,
        max_retries=0,
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "plan.json", spec)
    rows_path = output / "rows.json"
    rows: list[dict[str, Any]] = (
        json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else []
    )
    completed = {
        (row["segment"], row["task"], int(row["repeat"])) for row in rows
    }
    for cell in spec["cells"]:
        key = (cell["segment"], cell["task"], int(cell["repeat"]))
        if key in completed:
            continue
        started = time.perf_counter()
        request_started_at = datetime.now(UTC).isoformat()
        row: dict[str, Any] = dict(cell)
        try:
            response = await client.chat.completions.create(
                model=str(spec["model"]),
                messages=[
                    {
                        "role": "system",
                        "content": "你是消费选择实验中的合成受访者。只输出合法JSON。",
                    },
                    {"role": "user", "content": _prompt(*key)},
                ],
                response_format={"type": "json_object"},
                max_tokens=800,
                stream=False,
                temperature=int(spec["temperature_ppm"]) / 1_000_000,
                top_p=int(spec["top_p_ppm"]) / 1_000_000,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = response.choices[0].message.content
            if not isinstance(raw, str):
                raise ValueError("provider returned empty response")
            parsed = _validate(_extract_json_object(raw))
            usage = response.usage
            row.update(
                {
                    "success": True,
                    **parsed,
                    "raw_response": raw,
                    "input_tokens": int(usage.prompt_tokens or 0),
                    "output_tokens": int(usage.completion_tokens or 0),
                    "request_id": getattr(response, "id", None),
                    "response_model": getattr(response, "model", None),
                    "request_started_at": request_started_at,
                }
            )
        except Exception as exc:
            row.update(
                {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            )
        row["latency_ms"] = round((time.perf_counter() - started) * 1000)
        rows.append(row)
        completed.add(key)
        _write_json(rows_path, rows)
        print(
            f"[{len(rows)}/{len(spec['cells'])}] {key} success={row['success']} "
            f"tokens={row['input_tokens'] + row['output_tokens']}",
            flush=True,
        )
    successful = [row for row in rows if row.get("success")]
    by_segment: dict[str, Any] = {}
    for segment in SEGMENTS:
        selected = [row for row in successful if row["segment"] == segment]
        by_segment[segment] = {
            "response_count": len(selected),
            "choice_counts": {
                choice: sum(row["choice"] == choice for row in selected)
                for choice in ("A", "B", "C", "OUTSIDE")
            },
            "mean_max_willingness_to_pay_cents": (
                round(mean(row["max_willingness_to_pay_cents"] for row in selected))
                if selected
                else None
            ),
            "mean_importance_pct": {
                field: (
                    round(mean(row["importance"][field] for row in selected))
                    if selected
                    else None
                )
                for field in ("price", "brand", "service", "reliability")
            },
        }
    high_price = [
        row for row in successful if row["task"] == "high_price_outside"
    ]
    usage = {
        "calls": len(rows),
        "successful_calls": len(successful),
        "prompt_tokens": sum(int(row["input_tokens"]) for row in rows),
        "completion_tokens": sum(int(row["output_tokens"]) for row in rows),
    }
    usage["estimated_cost_microunits"] = (
        usage["prompt_tokens"] * INPUT_PRICE_MICROUNITS
        + usage["completion_tokens"] * OUTPUT_PRICE_MICROUNITS
    )
    summary: dict[str, Any] = {
        "result_schema_version": "llm-consumer-pilot-result-v1.0.0",
        "evidence_level": spec["evidence_level"],
        "preregistration_hash": spec["preregistration_hash"],
        "claim_boundary": spec["claim_boundary"],
        "by_segment": by_segment,
        "high_price_outside_choice_rate_ppm": (
            round(
                sum(row["choice"] == "OUTSIDE" for row in high_price)
                * 1_000_000
                / len(high_price)
            )
            if high_price
            else None
        ),
        "usage": usage,
        "gates": {
            "all_calls_recorded": len(rows) == len(spec["cells"]),
            "all_calls_successful": len(successful) == len(spec["cells"]),
            "all_segments_covered": all(
                by_segment[item]["response_count"] == len(TASKS) * REPEATS
                for item in SEGMENTS
            ),
        },
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
