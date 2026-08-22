"""Tit-for-Tat family strategies over authoritative cooperation memory."""

from __future__ import annotations

from typing import Any, Mapping

from game_theory_agent.repeated_game.contracts import (
    OpponentRepeatedGameStrategy,
    RepeatedGameStrategyState,
    compute_repeated_game_hash,
)


class RepeatedGameStrategist:
    def build(
        self,
        *,
        episode_id: str,
        observer_company_id: str,
        round_number: int,
        cooperation_view: Mapping[str, Any],
    ) -> tuple[RepeatedGameStrategyState, str]:
        memory = cooperation_view.get("cooperation_memory", {})
        if not isinstance(memory, Mapping):
            raise ValueError("cooperation memory must be an object")
        strategies: dict[str, OpponentRepeatedGameStrategy] = {}
        for opponent, raw in sorted(memory.items()):
            record = dict(raw)
            trust = int(record.get("credibility_ppm", 500_000))
            fulfilled = int(record.get("fulfilled_by_opponent", 0))
            partial = int(record.get("partial_betrayals_by_opponent", 0))
            betrayed = int(record.get("betrayed_by_opponent", 0))
            accepted = int(record.get("accepted_by_opponent", 0))
            if betrayed > 0:
                tft = "defect"
            elif partial > 0:
                tft = "cautious"
            elif fulfilled > 0:
                tft = "cooperate"
            else:
                tft = "cautious"
            grim = "permanent_refusal" if betrayed + partial > 0 else "cooperate"
            generous = (
                "generous_cooperate"
                if trust >= 400_000
                else "cautious"
            )
            recommended = (
                "permanent_refusal"
                if betrayed >= 2
                else (
                    "defect"
                    if betrayed == 1 and trust < 350_000
                    else (
                        "cautious"
                        if partial + betrayed > 0
                        else (
                            "cooperate"
                            if fulfilled > 0 or trust >= 600_000
                            else "generous_cooperate"
                        )
                    )
                )
            )
            multiplier = {
                "cooperate": 1_000_000,
                "generous_cooperate": 750_000,
                "cautious": 400_000,
                "defect": 0,
                "permanent_refusal": 0,
            }[recommended]
            strategies[str(opponent)] = OpponentRepeatedGameStrategy(
                opponent_company_id=str(opponent),
                trust_ppm=trust,
                accepted_count=accepted,
                fulfilled_count=fulfilled,
                partial_betrayal_count=partial,
                betrayal_count=betrayed,
                tit_for_tat_stance=tft,
                grim_trigger_stance=grim,
                generous_tit_for_tat_stance=generous,
                recommended_stance=recommended,
                contribution_multiplier_ppm=multiplier,
            )
        state = RepeatedGameStrategyState(
            episode_id=episode_id,
            observer_company_id=observer_company_id,
            round=round_number,
            opponent_strategies=strategies,
        )
        return state, compute_repeated_game_hash(state)
