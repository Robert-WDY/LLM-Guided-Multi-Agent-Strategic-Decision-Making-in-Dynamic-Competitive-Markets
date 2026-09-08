"""Replay for Bayesian advisor payloads embedded in observations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from game_theory_agent.advisor.advisor import (
    BayesianGameAdvisor,
    BayesianStrategyAdvisor,
)
from game_theory_agent.advisor.context_view import build_agent_advice_view
from game_theory_agent.advisor.contracts import (
    GameTheoryAdvice,
    StrategicGameTheoryAdvice,
)
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.market import load_market_config
from game_theory_agent.strategic_reliability import (
    PublicMarketRolloutAdvisor,
    PublicStrategicAdvice,
)


class AdvisorReplayMismatchError(RuntimeError):
    pass


def verify_advisor_replay(
    events: Sequence[Any],
    manifest: Any | None = None,
) -> tuple[
    GameTheoryAdvice | StrategicGameTheoryAdvice | PublicStrategicAdvice, ...
]:
    expected_mode = getattr(manifest, "advisor_mode", None)
    verified: list[
        GameTheoryAdvice | StrategicGameTheoryAdvice | PublicStrategicAdvice
    ] = []
    for event in events:
        snapshots: list[tuple[Any, str | None]] = []
        phase = getattr(event, "communication_phase", None)
        if phase is not None:
            snapshots.extend(
                (trace.information_snapshot, getattr(trace, "persona", None))
                for trace in phase.generation_traces
                if trace.information_snapshot is not None
            )
        snapshots.extend(
            (trace.information_snapshot, getattr(trace, "persona", None))
            for trace in event.traces
            if trace.information_snapshot is not None
        )
        for snapshot, trace_persona_id in snapshots:
            observation = snapshot.observation
            raw = observation.get("game_theory_advice")
            if raw is None:
                if expected_mode == "strategic_market_v9":
                    belief = observation.get("belief_state")
                    opponent_model = observation.get("opponent_model_state")
                    if not isinstance(belief, dict) or not isinstance(
                        opponent_model, dict
                    ):
                        raise AdvisorReplayMismatchError(
                            "v9 abstention is missing public strategic inputs"
                        )
                    if not trace_persona_id:
                        raise AdvisorReplayMismatchError(
                            "v9 abstention is missing persona binding"
                        )
                    config = load_market_config(
                        Path(__file__).resolve().parents[3]
                        / "configs"
                        / "market_v6_final.yaml"
                    )
                    registry = PersonaRegistry.from_market_config(config)
                    recomputed = PublicMarketRolloutAdvisor(config).advise(
                        observation=observation,
                        company_id=snapshot.company_id,
                        persona_profile=registry.get(str(trace_persona_id)),
                        belief_state=belief,
                        opponent_model=opponent_model,
                        horizon_rounds=min(
                            3, int(observation.get("rounds_remaining", 1))
                        ),
                        scenario_count=5,
                        advisor_mode="strategic_market_v9",
                    ).model_dump(mode="json")
                    if build_agent_advice_view(recomputed) is not None:
                        raise AdvisorReplayMismatchError(
                            "released v9 recommendation is missing from observation"
                        )
                    continue
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
            if raw.get("advisor_mode") in {
                "public_rollout_v3",
                "pareto_rollout_v4",
                "pareto_reliable_v5",
                "pareto_reliable_v6",
                "pareto_reliable_v7",
                "strategic_market_v9",
            }:
                opponent_model = observation.get("opponent_model_state")
                if not isinstance(opponent_model, dict):
                    raise AdvisorReplayMismatchError(
                        "public rollout advisor is missing the public opponent model"
                    )
                config_name = (
                    "market_v6_final.yaml"
                    if raw.get("advisor_mode") == "strategic_market_v9"
                    else "market_v4.yaml"
                )
                config = load_market_config(
                    Path(__file__).resolve().parents[3] / "configs" / config_name
                )
                registry = PersonaRegistry.from_market_config(config)
                recorded = PublicStrategicAdvice.model_validate(raw)
                expected = PublicMarketRolloutAdvisor(config).advise(
                    observation=observation,
                    company_id=snapshot.company_id,
                    persona_profile=registry.get(recorded.persona_id),
                    belief_state=belief,
                    opponent_model=opponent_model,
                    horizon_rounds=recorded.horizon_rounds,
                    scenario_count=recorded.scenario_count,
                    advisor_mode=recorded.advisor_mode,
                )
            elif raw.get("advisor_mode") == "bayesian_strategy_v2":
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
