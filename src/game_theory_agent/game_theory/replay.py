"""Composite replay for opponent model -> utility -> advisor -> strategy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from game_theory_agent.advisor import verify_advisor_replay
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.opponent import verify_opponent_model_replay
from game_theory_agent.repeated_game import verify_repeated_game_replay
from game_theory_agent.utility_inference import verify_utility_inference_replay


class GameTheoryReplayMismatchError(RuntimeError):
    pass


class GameTheoryReplayReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_schema_version: str = "game-theory-replay-report-v1.0.0"
    event_count: int = Field(ge=0)
    opponent_model_view_count: int = Field(ge=0)
    utility_inference_view_count: int = Field(ge=0)
    advisor_view_count: int = Field(ge=0)
    repeated_game_view_count: int = Field(ge=0)
    trace_binding_count: int = Field(ge=0)
    hidden_state_leak_count: int = Field(ge=0)
    replay_hash: str


def verify_game_theory_replay(
    events: Sequence[Any], manifest: Any | None = None
) -> GameTheoryReplayReport:
    opponent_views = verify_opponent_model_replay(events, manifest)
    utility_views = verify_utility_inference_replay(events, manifest)
    advisor_views = verify_advisor_replay(events, manifest)
    repeated_views = verify_repeated_game_replay(events, manifest)
    trace_bindings = 0
    hidden_leaks = 0
    forbidden_keys = {
        "opponent_private_financials",
        "opponent_hidden_cost",
        "opponent_persona",
        "opponent_prompt",
    }
    for event in events:
        for trace in event.traces:
            snapshot = trace.information_snapshot
            if snapshot is None:
                continue
            observation = snapshot.observation
            strategic_payloads = {
                "opponent_model": observation.get("opponent_model_state"),
                "utility_inference": observation.get(
                    "utility_inference_state"
                ),
                "advisor_output": observation.get("game_theory_advice"),
                "repeated_game_strategy": observation.get(
                    "repeated_game_strategy"
                ),
            }
            for payload in strategic_payloads.values():
                if isinstance(payload, dict):
                    hidden_leaks += len(forbidden_keys & set(payload))
            bindings = (
                ("opponent_model", "opponent_model"),
                ("utility_inference", "utility_inference"),
                ("advisor_output", "advisor_output"),
                ("repeated_game_strategy", "repeated_game_strategy"),
            )
            for trace_field, payload_key in bindings:
                recorded = getattr(trace, trace_field, None)
                expected = strategic_payloads[payload_key]
                if recorded is not None and recorded != expected:
                    raise GameTheoryReplayMismatchError(
                        f"trace {trace_field} differs for {trace.company_id}"
                    )
                if recorded is not None:
                    trace_bindings += 1
            chosen = getattr(trace, "chosen_action", None)
            if chosen is not None and chosen != trace.final_action:
                raise GameTheoryReplayMismatchError(
                    f"chosen action differs for {trace.company_id}"
                )
            counterfactuals = getattr(trace, "counterfactual_results", None)
            advice = observation.get("game_theory_advice")
            if counterfactuals is not None and isinstance(advice, dict):
                expected_candidates = advice.get(
                    "candidate_actions", advice.get("candidates", [])
                )
                if counterfactuals.get("candidate_actions") != expected_candidates:
                    raise GameTheoryReplayMismatchError(
                        f"counterfactual candidates differ for {trace.company_id}"
                    )
    if hidden_leaks:
        raise GameTheoryReplayMismatchError(
            f"strategic payloads contain {hidden_leaks} hidden-state keys"
        )
    payload = {
        "event_count": len(events),
        "opponent_model_view_count": len(opponent_views),
        "utility_inference_view_count": len(utility_views),
        "advisor_view_count": len(advisor_views),
        "repeated_game_view_count": len(repeated_views),
        "trace_binding_count": trace_bindings,
        "hidden_state_leak_count": hidden_leaks,
    }
    return GameTheoryReplayReport(
        **payload,
        replay_hash=sha256_hash(
            {
                "protocol": "game-theory-replay-v1.0.0",
                **payload,
            }
        ),
    )
