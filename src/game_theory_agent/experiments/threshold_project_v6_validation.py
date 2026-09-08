"""Provision-point / Stag Hunt validation for Strategic Market v6."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from game_theory_agent.market import CompanyAction, MarketEnv, load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition, verify_replay


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_v6_strategic.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "strategic-market-v6-stage72"
SEEDS = tuple(range(72001, 72021))
CONTRIBUTION_CENTS = 2_000_000
CONDITIONS = {
    "all_cooperate": {
        "company_A": CONTRIBUTION_CENTS,
        "company_B": CONTRIBUTION_CENTS,
        "company_C": CONTRIBUTION_CENTS,
        "company_D": CONTRIBUTION_CENTS,
    },
    "pivotal_defection": {
        "company_A": 0,
        "company_B": CONTRIBUTION_CENTS,
        "company_C": CONTRIBUTION_CENTS,
        "company_D": CONTRIBUTION_CENTS,
    },
    "lone_cooperation": {
        "company_A": CONTRIBUTION_CENTS,
        "company_B": 0,
        "company_C": 0,
        "company_D": 0,
    },
    "all_defect": {
        "company_A": 0,
        "company_B": 0,
        "company_C": 0,
        "company_D": 0,
    },
}


def _mean_int(values) -> int:
    items = tuple(int(value) for value in values)
    return sum(items) // len(items)


def _run_episode(config, seed: int, condition: str) -> dict[str, object]:
    env = MarketEnv(config)
    state = env.reset(
        # Episode id is deliberately identical across paired conditions because
        # risk-signal ids enter component RNG entity ids.
        episode_id=f"stage72-{seed}",
        episode_seed=seed,
        max_rounds=5,
        cooperation_modes=("threshold_project_v1",),
    )
    initial = state
    manifest = EpisodeManifest.create(
        env, initial, experiment_id="stage72-threshold-project"
    )
    transitions = []
    project_status_path = []
    for index in range(5):
        actions = {
            company_id: CompanyAction(
                action_id=(
                    f"fixed:{state.episode_id}:{state.round}:{company_id}"
                ),
                episode_id=state.episode_id,
                agent_id=company_id,
                round=state.round,
                state_version=state.state_version,
                price_cents=10_000,
                threshold_project_contribution_cents=0,
                strategy_summary="frozen economic action tape",
            )
            for company_id in state.company_ids
        }
        if index == 0:
            actions = {
                company_id: replace(
                    action,
                    threshold_project_contribution_cents=CONDITIONS[
                        condition
                    ][company_id],
                )
                for company_id, action in actions.items()
            }
        result = env.step(
            f"{state.episode_id}:{state.round}:{state.state_version}", actions
        )
        transitions.append(MarketTransition.create(state, actions, result))
        state = result.state_after
        project_status_path.append(
            state.strategic_market.threshold_project.status.value
        )
    replayed = verify_replay(MarketEnv(config), manifest, tuple(transitions))[-1]
    project = state.strategic_market.threshold_project
    values = dict(state.terminal_enterprise_values_cents)
    return {
        "seed": seed,
        "condition": condition,
        "project_status": project.status.value,
        "project_status_path": project_status_path,
        "project_total_cents": project.accumulated_total_contribution_cents,
        "refund_by_company_cents": dict(
            project.last_refund_by_company_cents
        ),
        "focal_enterprise_value_cents": values["company_A"],
        "market_enterprise_value_cents": sum(values.values()),
        "cumulative_social_welfare_proxy_cents": (
            state.strategic_market.cumulative_social_welfare_proxy_cents
        ),
        "final_supply_cost_index_ppm": state.market.actual_supply_cost_index_ppm,
        "final_regulatory_pressure_ppm": (
            state.strategic_market.regulatory_pressure_ppm
        ),
        "replay_passed": replayed.state_hash == state.state_hash,
    }


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    config = load_market_config(CONFIG_PATH)
    rows = [
        _run_episode(config, seed, condition)
        for seed in SEEDS
        for condition in CONDITIONS
    ]
    indexed = {
        (int(row["seed"]), str(row["condition"])): row for row in rows
    }
    paired = []
    for seed in SEEDS:
        all_cooperate = indexed[(seed, "all_cooperate")]
        pivotal = indexed[(seed, "pivotal_defection")]
        lone = indexed[(seed, "lone_cooperation")]
        defect = indexed[(seed, "all_defect")]
        paired.append(
            {
                "seed": seed,
                "pivotal_cooperation_private_gain_cents": (
                    int(all_cooperate["focal_enterprise_value_cents"])
                    - int(pivotal["focal_enterprise_value_cents"])
                ),
                "lone_cooperation_private_loss_cents": (
                    int(defect["focal_enterprise_value_cents"])
                    - int(lone["focal_enterprise_value_cents"])
                ),
                "successful_project_market_value_gain_cents": (
                    int(all_cooperate["market_enterprise_value_cents"])
                    - int(pivotal["market_enterprise_value_cents"])
                ),
                "successful_project_social_welfare_gain_cents": (
                    int(all_cooperate["cumulative_social_welfare_proxy_cents"])
                    - int(pivotal["cumulative_social_welfare_proxy_cents"])
                ),
            }
        )
    gates = {
        "all_cooperate_succeeds_20_of_20": all(
            indexed[(seed, "all_cooperate")]["project_status"] == "succeeded"
            for seed in SEEDS
        ),
        "one_pivotal_defection_fails_20_of_20": all(
            indexed[(seed, "pivotal_defection")]["project_status"] == "failed"
            for seed in SEEDS
        ),
        "lone_cooperation_fails_20_of_20": all(
            indexed[(seed, "lone_cooperation")]["project_status"] == "failed"
            for seed in SEEDS
        ),
        "pivotal_cooperation_is_privately_rational_20_of_20": all(
            row["pivotal_cooperation_private_gain_cents"] > 0 for row in paired
        ),
        "lone_cooperation_is_privately_costly_20_of_20": all(
            row["lone_cooperation_private_loss_cents"] > 0 for row in paired
        ),
        "successful_project_improves_market_value_20_of_20": all(
            row["successful_project_market_value_gain_cents"] > 0
            for row in paired
        ),
        "successful_project_improves_social_welfare_20_of_20": all(
            row["successful_project_social_welfare_gain_cents"] > 0
            for row in paired
        ),
        "replay_80_of_80": all(row["replay_passed"] for row in rows),
    }
    summary: dict[str, object] = {
        "result_schema_version": "strategic-market-v6-stage72-result-v1.0.0",
        "config_id": config.config_id,
        "config_sha256": config.config_sha256,
        "seeds": list(SEEDS),
        "conditions": CONDITIONS,
        "provider_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "gates": gates,
        "aggregate": {
            "mean_pivotal_cooperation_private_gain_cents": _mean_int(
                row["pivotal_cooperation_private_gain_cents"] for row in paired
            ),
            "mean_lone_cooperation_private_loss_cents": _mean_int(
                row["lone_cooperation_private_loss_cents"] for row in paired
            ),
            "mean_successful_project_market_value_gain_cents": _mean_int(
                row["successful_project_market_value_gain_cents"] for row in paired
            ),
            "mean_successful_project_social_welfare_gain_cents": _mean_int(
                row["successful_project_social_welfare_gain_cents"]
                for row in paired
            ),
        },
        "paired_rows": paired,
        "episode_rows": rows,
    }
    summary["all_gates_passed"] = all(gates.values())
    summary["result_hash"] = sha256_hash(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.output)
    print(
        json.dumps(
            {
                "all_gates_passed": summary["all_gates_passed"],
                "aggregate": summary["aggregate"],
                "failed_gates": [
                    key for key, value in summary["gates"].items() if not value
                ],
                "result_hash": summary["result_hash"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
