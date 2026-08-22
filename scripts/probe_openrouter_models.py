"""逐个探测 WebUI 候选模型，只输出不含凭据或模型原文的审计结果。"""

from __future__ import annotations

import json
from pathlib import Path

from game_theory_agent.agents.single.provider import (
    ALLOWED_FREE_MODELS,
    OpenRouterProvider,
    ProviderError,
    load_openrouter_api_key,
)
from game_theory_agent.agents.single.models import (
    DecisionContext,
    EpisodeMemoryView,
    RoundFeedback,
    SnapshotKey,
    StrategyReflection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRET_PATH = PROJECT_ROOT / "secrets" / "open_router-api_key.env"
WEBUI_MODELS = (
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
)


def probe_context() -> DecisionContext:
    return DecisionContext(
        snapshot_key=SnapshotKey(
            episode_id="model-probe",
            company_id="company_A",
            round=2,
            state_version=1,
            state_hash="probe-state",
        ),
        observation={
            "episode_id": "model-probe",
            "company_id": "company_A",
            "round": 2,
            "own_company": {"financial": {"cash_balance_cents": 10_000_000}},
            "public_companies": [{"company_id": "company_B", "rank": 2}],
        },
        action_contract={
            "constraints": {"bounds": {"price_cents": {"min": 8_000, "max": 12_000}}}
        },
        memory=EpisodeMemoryView(
            recent_feedback=[
                RoundFeedback(
                    settled_round=1,
                    own_action={"price_cents": 9_800},
                    own_result={"round_profit_cents": 500_000, "market_share_ppm": 230_000},
                    market={"realized_demand_orders": 12_000},
                )
            ],
            previous_selected_candidate_id="balanced",
            previous_expected_outcome="份额小幅提高",
        ),
        reflection=StrategyReflection(
            source="deterministic",
            lesson_codes=["profit_positive"],
            adjustments=["保持现金约束内的有效投入。"],
            evidence_paths=["memory.recent_feedback[-1].own_result.round_profit_cents"],
            summary="保持现金约束内的有效投入。",
        ),
    )


def main() -> int:
    unknown = set(WEBUI_MODELS) - set(ALLOWED_FREE_MODELS)
    if unknown:
        raise SystemExit(f"模型不在项目白名单：{sorted(unknown)}")
    provider = OpenRouterProvider(
        api_key=load_openrouter_api_key(SECRET_PATH),
        timeout_seconds=45,
    )
    results: list[dict[str, object]] = []
    for model_id in WEBUI_MODELS:
        try:
            result = provider.generate_decision(
                model_id=model_id,
                context=probe_context(),
            )
        except ProviderError as exc:
            results.append(
                {
                    "model": model_id,
                    "ok": False,
                    "error_code": exc.code,
                    "latency_ms": exc.latency_ms,
                    "total_tokens": exc.usage.total_tokens,
                    "finish_reason": exc.finish_reason,
                }
            )
            continue
        results.append(
            {
                "model": model_id,
                "ok": True,
                "error_code": None,
                "latency_ms": result.latency_ms,
                "total_tokens": result.usage.total_tokens,
                "finish_reason": result.finish_reason,
            }
        )
    print(json.dumps(results, ensure_ascii=False))
    return 0 if all(bool(item["ok"]) for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
