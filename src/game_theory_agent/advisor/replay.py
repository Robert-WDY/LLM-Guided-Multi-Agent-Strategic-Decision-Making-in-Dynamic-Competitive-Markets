"""Replay for Bayesian advisor payloads embedded in observations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from game_theory_agent.advisor.advisor import (
    BayesianGameAdvisor,
    BayesianStrategyAdvisor,
)
from game_theory_agent.advisor.contracts import (
    GameTheoryAdvice,
    StrategicGameTheoryAdvice,
)


class AdvisorReplayMismatchError(RuntimeError):
    pass


def verify_advisor_replay(
    events: Sequence[Any],
    manifest: Any | None = None,
) -> tuple[GameTheoryAdvice | StrategicGameTheoryAdvice, ...]:
    expected_mode = getattr(manifest, "advisor_mode", None)
    verified: list[GameTheoryAdvice | StrategicGameTheoryAdvice] = []
    for event in events:
        snapshots = []
        phase = getattr(event, "communication_phase", None)
        if phase is not None:
            snapshots.extend(
                trace.information_snapshot
                for trace in phase.generation_traces
                if trace.information_snapshot is not None
            )
        snapshots.extend(
            trace.information_snapshot
            for trace in event.traces
            if trace.information_snapshot is not None
        )
        for snapshot in snapshots:
            observation = snapshot.observation
            raw = observation.get("game_theory_advice")
            if raw is None:
                if expected_mode not in {None, "off"}:
                    raise AdvisorReplayMismatchError(
                        "enabled treatment is missing advisor output"
                    )
                continue
            if expected_mode == "off":
                raise AdvisorReplayMismatchError(
                    "off treatment contains advisor output"
                )
            belief = observation.get("belief_state")
            if not isinstance(belief, dict):
                raise AdvisorReplayMismatchError(
                    "advisor payload exists without belief state"
                )
            if raw.get("advisor_mode") == "bayesian_strategy_v2":
                opponent_model = observation.get("opponent_model_state")
                utility = observation.get("utility_inference_state")
                if not isinstance(opponent_model, dict) or not isinstance(
                    utility, dict
                ):
                    raise AdvisorReplayMismatchError(
                        "v2 advisor is missing strategic inputs"
                    )
                expected = BayesianStrategyAdvisor().advise(
                    belief_state=belief,
                    opponent_model=opponent_model,
                    utility_inference=utility,
                    own_company=observation["own_company"],
                    action_constraints=observation["action_constraints"],
                )
                recorded = StrategicGameTheoryAdvice.model_validate(raw)
            else:
                expected = BayesianGameAdvisor().advise(
                    belief_state=belief,
                    own_company=observation["own_company"],
                    action_constraints=observation["action_constraints"],
                )
                recorded = GameTheoryAdvice.model_validate(raw)
            if recorded != expected:
                raise AdvisorReplayMismatchError(
                    f"advisor payload differs for {snapshot.company_id}"
                )
            verified.append(recorded)
    return tuple(verified)
