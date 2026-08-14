"""Repeatable multi-seed evaluations for UI investment anchors."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import MarketConfig, MarketEnv


def evaluate_presets(
    config: MarketConfig,
    *,
    seed_start: int = 0,
    seed_count: int = 200,
) -> dict[str, Any]:
    """Play each fixed investment anchor against the same adaptive opponents."""

    if not 1 <= seed_count <= 500:
        raise ValueError("seed_count must be in [1, 500]")
    presets = config.mapping("action", "presets")
    results: dict[str, Any] = {}
    company_ids = ["company_A", "company_B", "company_C", "company_D"]
    for preset_name, preset in presets.items():
        ranks: Counter[int] = Counter()
        terminal_values: list[int] = []
        adjustment_counts = 0
        for seed in range(seed_start, seed_start + seed_count):
            episode_id = f"calibration-{preset_name}-{seed}"
            env = MarketEnv(config)
            state = env.reset(
                company_ids, episode_id=episode_id, episode_seed=seed
            )
            while not state.terminal:
                actions = {}
                for company_id in state.company_ids:
                    raw = (
                        dict(preset)
                        if company_id == "company_A"
                        else build_rule_action(config, state, company_id).to_dict()
                    )
                    decision = resolve_action_request(
                        config,
                        state,
                        company_id,
                        raw,
                        source=(
                            f"calibration-preset:{preset_name}"
                            if company_id == "company_A"
                            else "calibration-rule-opponent"
                        ),
                        action_id=f"{episode_id}:{state.round}:{company_id}",
                    )
                    actions[company_id] = decision.action
                    if company_id == "company_A":
                        adjustment_counts += len(decision.adjustments)
                state = env.step(
                    f"{episode_id}:{state.round}:{state.state_version}", actions
                ).state_after
            values = dict(state.terminal_enterprise_values_cents)
            ranking = sorted(state.company_ids, key=values.get, reverse=True)
            ranks[ranking.index("company_A") + 1] += 1
            terminal_values.append(values["company_A"])
        first_ratio = ranks[1] / seed_count
        last_ratio = ranks[len(company_ids)] / seed_count
        results[preset_name] = {
            "rank_counts": {
                str(rank): ranks[rank] for rank in range(1, len(company_ids) + 1)
            },
            "first_place_ratio": first_ratio,
            "last_place_ratio": last_ratio,
            "extreme_ratio_max": max(first_ratio, last_ratio),
            "passes_extreme_threshold": max(first_ratio, last_ratio) <= 0.70,
            "average_terminal_value_cents": round(mean(terminal_values)),
            "guardrail_adjustment_count": adjustment_counts,
        }
    return {
        "method": "fixed investment anchor vs three adaptive rule opponents",
        "seed_start": seed_start,
        "seed_count": seed_count,
        "threshold": 0.70,
        "presets": results,
        "passed": all(
            item["passes_extreme_threshold"] for item in results.values()
        ),
    }
