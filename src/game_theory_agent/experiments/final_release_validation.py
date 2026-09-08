"""Assemble the final strategic-market release gate from frozen evidence.

This command does not call a model and does not rerun the expensive tournament.
It verifies the hashes and declared gates of the already generated artifacts,
then writes a compact, reviewable release manifest under experiment-results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from game_theory_agent.market.protocols import sha256_hash


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiment-results" / "stage7-final-release-v1"
)

ArtifactHashMode = Literal["result", "legacy_pending_result", "manifest"]

ARTIFACTS: tuple[tuple[str, str, ArtifactHashMode], ...] = (
    ("market_lifecycle", "runs/strategic-market-v6-stage71/summary.json", "result"),
    ("threshold_project", "runs/strategic-market-v6-stage72/summary.json", "result"),
    ("mutual_aid", "runs/strategic-market-v6-stage73/summary.json", "result"),
    ("price_coordination", "runs/strategic-market-v6-stage74/summary.json", "result"),
    ("combined_market", "runs/strategic-market-v6-stage75/summary.json", "result"),
    ("advisor_v9", "runs/final-market-advisor-v9/summary.json", "result"),
    (
        "real_llm_smoke",
        "runs/final-market-v9-real-llm-smoke/summary.json",
        "legacy_pending_result",
    ),
    (
        "self_play_holdout",
        "runs/final-market-self-play-v1/holdout-summary.json",
        "result",
    ),
    (
        "agent_iteration_holdout",
        "runs/final-market-agent-iteration-v1/holdout-summary.json",
        "result",
    ),
    (
        "training_export",
        "runs/final-market-training-data-v1/manifest.json",
        "manifest",
    ),
)


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read release artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"release artifact must be a JSON object: {path}")
    return payload


def verify_artifact_hash(
    payload: dict[str, Any], mode: ArtifactHashMode
) -> tuple[bool, str, str]:
    """Return match, stored hash and recomputed hash.

    The first real-LLM v9 smoke predates the canonical exclude-field rule and
    hashed a literal ``result_hash='pending'``. Its raw calls cannot be rerun
    merely to rewrite metadata, so the release gate preserves and names that
    historical protocol instead of silently changing the evidence.
    """

    field = "manifest_hash" if mode == "manifest" else "result_hash"
    stored = str(payload.get(field, ""))
    canonical = {key: value for key, value in payload.items() if key != field}
    if mode == "legacy_pending_result":
        canonical[field] = "pending"
    actual = sha256_hash(canonical)
    return stored == actual, stored, actual


def _all_declared_gates(payload: dict[str, Any]) -> bool:
    gates = payload.get("gates", {})
    return isinstance(gates, dict) and bool(gates) and all(
        bool(value) for value in gates.values()
    )


def _artifact_row(
    name: str,
    relative_path: str,
    mode: ArtifactHashMode,
    payload: dict[str, Any],
) -> dict[str, Any]:
    matches, stored, actual = verify_artifact_hash(payload, mode)
    return {
        "artifact_id": name,
        "source_path": relative_path,
        "schema_version": payload.get(
            "result_schema_version", payload.get("dataset_schema_version")
        ),
        "hash_mode": mode,
        "stored_hash": stored,
        "recomputed_hash": actual,
        "hash_matches": matches,
    }


def build_release(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    payloads: dict[str, dict[str, Any]] = {}
    evidence: list[dict[str, Any]] = []
    for name, relative_path, mode in ARTIFACTS:
        payload = _read(project_root / relative_path)
        payloads[name] = payload
        evidence.append(_artifact_row(name, relative_path, mode, payload))

    mechanism_ids = (
        "market_lifecycle",
        "threshold_project",
        "mutual_aid",
        "price_coordination",
        "combined_market",
    )
    advisor = payloads["advisor_v9"]
    real_llm = payloads["real_llm_smoke"]
    self_play = payloads["self_play_holdout"]
    iteration = payloads["agent_iteration_holdout"]
    training = payloads["training_export"]
    self_play_gates = self_play.get("gates", {})
    phenomena = self_play.get("phenomena", {})
    iteration_metrics = iteration.get("metrics", {}).get(
        "contextual_vs_balanced", {}
    )

    gates = {
        "all_artifact_hashes_match_declared_protocol": all(
            row["hash_matches"] for row in evidence
        ),
        "all_five_market_mechanism_suites_pass": all(
            bool(payloads[item].get("all_gates_passed"))
            for item in mechanism_ids
        ),
        "advisor_v9_zero_token_holdout_passes": bool(
            advisor.get("all_gates_passed")
        ),
        "real_llm_integration_is_directionally_positive": (
            int(real_llm.get("paired_complete", 0)) >= 2
            and int(real_llm.get("successful_calls", 0)) >= 5
            and int(real_llm.get("worst_enterprise_value_delta_cents", -1))
            >= 0
            and int(
                real_llm.get(
                    "research_only_price_coordination_action_count", 1
                )
            )
            == 0
        ),
        "self_play_mechanisms_and_replay_pass": all(
            bool(self_play_gates.get(key))
            for key in (
                "replay_all",
                "demand_closure_all",
                "cash_non_negative_all",
                "competition_bankruptcy_and_price_war_observed",
                "cooperation_and_free_riding_observed",
                "mutual_aid_observed",
                "coordination_betrayal_and_enforcement_observed",
            )
        )
        and int(phenomena.get("monopoly_episode_count", 0)) > 0
        and int(phenomena.get("threshold_success_episode_count", 0)) > 0
        and int(phenomena.get("threshold_failure_episode_count", 0)) > 0,
        "unsafe_pure_self_play_candidate_is_rejected": (
            self_play.get("promotion_accepted") is False
            and self_play_gates.get("candidate_tail_loss_bounded") is False
        ),
        "contextual_policy_passes_disjoint_holdout": (
            iteration.get("promotion_accepted") is True
            and _all_declared_gates(iteration)
            and int(iteration_metrics.get("negative_pair_count", 1)) == 0
            and int(
                iteration_metrics.get(
                    "worst_enterprise_value_delta_cents", -1
                )
            )
            >= 0
        ),
        "fine_tuning_is_correctly_blocked": (
            training.get("fine_tuning_release_allowed") is False
            and training.get("contains_holdout_rows") is False
            and int(training.get("example_count", 0)) == 108
        ),
    }
    release_allowed = all(gates.values())
    usage = real_llm.get("usage", {})
    return {
        "release_schema_version": "final-strategic-market-release-v1.0.0",
        "release_name": "最终战略市场研究版 v1",
        "config_id": "market-v6-final",
        "config_sha256": payloads["combined_market"]["config_sha256"],
        "engineering_release_allowed": release_allowed,
        "research_claim_level": (
            "合成市场的确定性与多随机种子证据，外加方向性真实模型证据"
        ),
        "evidence": evidence,
        "gates": gates,
        "headline_metrics": {
            "combined_market_episode_count": len(
                payloads["combined_market"].get("episode_rows", [])
            ),
            "self_play_holdout_episode_count": int(
                phenomena.get("unique_episode_count", 0)
            ),
            "agent_iteration_holdout_pair_count": int(
                iteration_metrics.get("pair_count", 0)
            ),
            "agent_iteration_mean_ev_gain_cents": int(
                iteration_metrics.get(
                    "mean_enterprise_value_delta_cents", 0
                )
            ),
            "agent_iteration_worst_ev_gain_cents": int(
                iteration_metrics.get(
                    "worst_enterprise_value_delta_cents", 0
                )
            ),
            "real_llm_attempted_calls": int(
                real_llm.get("attempted_provider_calls", 0)
            ),
            "real_llm_known_prompt_tokens": int(
                usage.get("prompt_tokens", 0)
            ),
            "real_llm_known_completion_tokens": int(
                usage.get("completion_tokens", 0)
            ),
            "real_llm_estimated_cost_microunits": int(
                usage.get("estimated_cost_microunits", 0)
            ),
        },
        "boundaries": {
            "real_llm_statistical_generalization": False,
            "real_market_calibration": False,
            "small_model_fine_tuning_allowed": False,
            "price_coordination_is_executable_advice": False,
            "self_play_is_full_psro": False,
        },
    }


def _result_markdown(summary: dict[str, Any]) -> str:
    status = "通过" if summary["engineering_release_allowed"] else "未通过"
    metrics = summary["headline_metrics"]
    failed = [key for key, value in summary["gates"].items() if not value]
    failed_text = "无" if not failed else "、".join(failed)
    estimated_cost_cny = (
        metrics["real_llm_estimated_cost_microunits"] / 1_000_000
    )
    return f"""# 最终战略市场研究版 v1：发布门禁

结论：**工程发布门禁{status}**。失败门禁：{failed_text}。

## 已验证范围

- 最终组合市场：{metrics['combined_market_episode_count']} 个 20 轮 Episode，覆盖破产、市场集中、价格战、门槛公共项目、应急互助、价格协调背叛与监管。
- 自博弈留出集：{metrics['self_play_holdout_episode_count']} 个 20 轮 Episode；纯高价防守因尾部损失被拒绝晋级。
- 策略迭代留出集：{metrics['agent_iteration_holdout_pair_count']} 个配对；情境防守相对均衡平均企业价值增加 {metrics['agent_iteration_mean_ev_gain_cents']} 分，最差差值 {metrics['agent_iteration_worst_ev_gain_cents']} 分。
- 真实模型集成：尝试 {metrics['real_llm_attempted_calls']} 次调用；已知 Token 为 {metrics['real_llm_known_prompt_tokens']} 输入 + {metrics['real_llm_known_completion_tokens']} 输出，估算费用 ¥{estimated_cost_cny:.6f}。

## 不能越界的结论

当前版本可以作为经过合成市场、多随机种子、回放和方向性真实模型验证的研究平台；不能声称已经完成真实市场校准、跨模型统计泛化、完整 PSRO，或已经具备用这些数据微调小模型的证据条件。价格协调只保留为危害与监管研究动作，不会由 v9 建议器发布为可执行建议。

发布清单哈希：`{summary['release_hash']}`。
"""


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    summary = build_release()
    summary["release_hash"] = sha256_hash(summary)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output / "RESULT.md").write_text(
        _result_markdown(summary), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.output.resolve())
    print(
        json.dumps(
            {
                "engineering_release_allowed": summary[
                    "engineering_release_allowed"
                ],
                "failed_gates": [
                    key
                    for key, value in summary["gates"].items()
                    if not value
                ],
                "release_hash": summary["release_hash"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
