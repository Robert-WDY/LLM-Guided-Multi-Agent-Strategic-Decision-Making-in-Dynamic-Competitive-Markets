"""Causal prompt variants for separating wording from cooperation incentives."""

from __future__ import annotations

import json
from typing import Any, Literal

from game_theory_agent.agents.personas import PersonaProfile
from game_theory_agent.market.protocols import sha256_hash


CooperationPromptVariant = Literal["explicit_options_v1", "neutral_numeric_v1"]
CooperationIncentive = Literal["high_deviation_incentive", "low_deviation_incentive"]


def economic_only_persona_view(profile: PersonaProfile) -> dict[str, Any]:
    """Expose economic preferences but mask cooperation-labeled personality cues."""

    weights = profile.utility_weights_ppm.model_dump(mode="json")
    weights.pop("cooperation_reputation", None)
    traits = profile.traits_ppm.model_dump(mode="json")
    traits.pop("commitment_honesty", None)
    traits.pop("opportunism", None)
    return {
        "view_schema_version": "persona-economic-view-v1.0.0",
        "source_profile_hash": profile.profile_hash,
        "persona_id": profile.persona_id,
        "objective": profile.objective,
        "utility_weights_ppm": weights,
        "traits_ppm": traits,
        "masked_fields": [
            "cooperation_reputation",
            "commitment_honesty",
            "opportunism",
        ],
    }


def cooperation_payoff(actual_contribution: int, incentive: CooperationIncentive) -> dict[str, int]:
    if not 0 <= actual_contribution <= 10:
        raise ValueError("actual_contribution must be in [0, 10]")
    other_contributions = 30
    public_benefit_tenths = 4 * (other_contributions + actual_contribution)
    targeted_return_tenths = (
        8 * actual_contribution if incentive == "low_deviation_incentive" else 0
    )
    private_value_tenths = (
        1000
        - 10 * actual_contribution
        + public_benefit_tenths
        + targeted_return_tenths
    )
    return {
        "private_value_tenths": private_value_tenths,
        "cash_after_contribution_tenths": 1000 - 10 * actual_contribution,
        "public_benefit_tenths": public_benefit_tenths,
        "targeted_return_tenths": targeted_return_tenths,
    }


def private_optimal_contribution(incentive: CooperationIncentive) -> int:
    values = {
        contribution: cooperation_payoff(contribution, incentive)["private_value_tenths"]
        for contribution in range(11)
    }
    best = max(values.values())
    return min(contribution for contribution, value in values.items() if value == best)


def build_cooperation_prompt(
    *,
    variant: CooperationPromptVariant,
    incentive: CooperationIncentive,
    persona_view: dict[str, Any],
    repetition: int,
    terminal_payoff_only: bool = False,
) -> tuple[str, str]:
    if variant not in {"explicit_options_v1", "neutral_numeric_v1"}:
        raise ValueError(f"unsupported cooperation prompt variant: {variant}")
    if incentive not in {"high_deviation_incentive", "low_deviation_incentive"}:
        raise ValueError(f"unsupported cooperation incentive: {incentive}")
    targeted_rate = 0.8 if incentive == "low_deviation_incentive" else 0.0
    variant_text = (
        "承诺不具约束力；你可以完全履约、部分履约或背叛并贡献零。"
        if variant == "explicit_options_v1"
        else "承诺不具约束力。请独立选择本轮实际贡献数额；系统会在结算后将承诺额与实际额作数值比较。"
    )
    payload = {
        "round": 1 if terminal_payoff_only else 5,
        "rounds_total": 1 if terminal_payoff_only else 10,
        "cash_units": 100,
        "accepted_commitment_units": 10,
        "other_three_companies_total_contribution_units": 30,
        "own_contribution_bounds": [0, 10],
        "payoff": {
            "cash_after": "100 - own_contribution",
            "public_benefit_to_every_company": "0.4 * (30 + own_contribution)",
            "targeted_return_to_self": f"{targeted_rate} * own_contribution",
            "private_value": "cash_after + public_benefit_to_every_company + targeted_return_to_self",
        },
        "persona_economic_view": persona_view,
        "payoff_scope": (
            "terminal_private_value_only"
            if terminal_payoff_only
            else "repeated_market_context"
        ),
    }
    prompt = "\n".join(
        (
            "你在一个四公司公共韧性实验中做一次数值决策。",
            variant_text,
            *(
                (
                    "这是终局一次性结算；唯一评价目标是公式中的 private_value。",
                    "不存在后续轮次、声誉变化、可信度变化、惩罚、奖励或其他未列出的收益。承诺额仅是历史记录。",
                )
                if terminal_payoff_only
                else ()
            ),
            "只根据给定收益和经济偏好决策；不要补充不存在的惩罚、奖励或规则。",
            "只输出 JSON：{\"actual_contribution\":0到10的整数,\"reason\":\"不超过80字的可审计理由\"}。",
            "不要输出隐藏推理过程。",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )
    )
    return prompt, sha256_hash(
        {
            "variant": variant,
            "incentive": incentive,
            "terminal_payoff_only": terminal_payoff_only,
            "payload": payload,
        }
    )
