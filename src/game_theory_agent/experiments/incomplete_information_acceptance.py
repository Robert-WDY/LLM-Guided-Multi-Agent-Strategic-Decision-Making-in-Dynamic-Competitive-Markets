"""Deterministic P0-P5 incomplete-information acceptance run."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from game_theory_agent.advisor import verify_advisor_replay
from game_theory_agent.agents.context import DecisionContextBuilder
from game_theory_agent.agents.memory import EpisodeMemory
from game_theory_agent.agents.prompt_builder import AgentPromptBuilder
from game_theory_agent.api import CONFIG, SESSIONS, agent_app, app
from game_theory_agent.belief import verify_belief_replay
from game_theory_agent.information import (
    InformationReplayMismatchError,
    ObservationEnvelope,
    ObservationSnapshot,
    verify_information_replay,
    verify_information_snapshot,
)
from game_theory_agent.interaction import verify_interaction_replay
from game_theory_agent.market import MarketEnv, MarketState
from game_theory_agent.market.replay import verify_replay
from game_theory_agent.orchestration.round_event import CommunicationPhaseRecord


COMPANIES = ("company_A", "company_B", "company_C", "company_D")


def run_acceptance(output: Path) -> dict[str, Any]:
    controller_token = "p0-p5-acceptance-controller"
    os.environ["MARKET_CONTROLLER_TOKEN"] = controller_token
    SESSIONS.clear()
    controller = TestClient(app)
    gateway = TestClient(agent_app)
    episode_id = "incomplete-information-p0-p5-acceptance"
    created_response = controller.post(
        "/api/episodes",
        headers={"X-Controller-Token": controller_token},
        json={
            "episode_id": episode_id,
            "episode_seed": 20260821,
            "company_ids": list(COMPANIES),
            "max_rounds": 5,
            "market_model": "balanced",
            "information_mode": "public",
            "communication_mode": "public_private",
            "belief_mode": "public_action_signal_v2",
            "advisor_mode": "bayesian_price_v1",
        },
    )
    if created_response.status_code != 201:
        raise RuntimeError(created_response.text)
    created = created_response.json()
    tokens = created["agent_tokens"]
    initial_state = MarketState.from_dict(created["state"])

    preclose = {
        company_id: gateway.get(
            f"/v1/episodes/{episode_id}/companies/{company_id}/observation",
            headers={"X-Agent-Token": tokens[company_id]},
        ).json()
        for company_id in COMPANIES
    }
    strict_contracts = all(
        ObservationEnvelope.model_validate(item) for item in preclose.values()
    )
    public_states_identical = len(
        {
            json.dumps(item["public_state"], sort_keys=True)
            for item in preclose.values()
        }
    ) == 1
    private_scope_correct = all(
        item["private_state"]["company_id"] == company_id
        and item["private_state"]["company"]["company_id"] == company_id
        for company_id, item in preclose.items()
    )
    forbidden = (
        "cash_balance_cents",
        "round_profit_cents",
        "actual_unit_cost_cents",
        "resilience_ppm",
        "active_incident",
        "persona",
    )
    opponent_private_leak_count = sum(
        serialized.count(field)
        for item in preclose.values()
        for field in forbidden
        for serialized in [json.dumps(item["competitors"], sort_keys=True)]
    )

    submission = {
        "round": initial_state.round,
        "state_version": initial_state.state_version,
        "state_hash": initial_state.state_hash,
        "submission": {
            "messages": [
                {
                    "channel": "private",
                    "recipients": ["company_B"],
                    "speech_act": "promise",
                    "content": "本轮计划把价格降到9000分。",
                    "own_action_claim": {"price_cents": 9000},
                }
            ]
        },
    }
    sent = gateway.post(
        f"/v1/episodes/{episode_id}/companies/company_A/communication/submissions",
        headers={"X-Agent-Token": tokens["company_A"]},
        json=submission,
    )
    if sent.status_code != 202:
        raise RuntimeError(sent.text)
    message_id = sent.json()["message_ids"][0]
    close = controller.post(
        f"/api/v1/controller/episodes/{episode_id}/communication/close",
        headers={"X-Controller-Token": controller_token},
        json={
            key: submission[key]
            for key in ("round", "state_version", "state_hash")
        },
    )
    if close.status_code != 200:
        raise RuntimeError(close.text)
    state_unchanged_after_signal = (
        SESSIONS[episode_id].env.get_state().state_hash
        == initial_state.state_hash
    )
    postclose = {
        company_id: gateway.get(
            f"/v1/episodes/{episode_id}/companies/{company_id}/observation",
            headers={"X-Agent-Token": tokens[company_id]},
        ).json()
        for company_id in COMPANIES
    }
    signal_ids = {
        company_id: [
            signal["message_id"]
            for signal in observation["belief_state"][
                "visible_communication_signals"
            ]
        ]
        for company_id, observation in postclose.items()
    }
    signal_visibility_correct = (
        # The sender sees its private message in CommunicationView, but a
        # BeliefState never models the observer itself as an opponent.
        signal_ids["company_A"] == []
        and signal_ids["company_B"] == [message_id]
        and signal_ids["company_C"] == []
        and signal_ids["company_D"] == []
    )
    b_cut = postclose["company_B"]["belief_state"]["opponent_beliefs"][
        "company_A"
    ]["next_price_direction"]["price_cut_ppm"]
    c_cut = postclose["company_C"]["belief_state"]["opponent_beliefs"][
        "company_A"
    ]["next_price_direction"]["price_cut_ppm"]
    prediction_responded_to_signal = b_cut > c_cut
    b_context = DecisionContextBuilder().build(
        postclose["company_B"], "company_B", EpisodeMemory()
    )
    prompt = AgentPromptBuilder().build(b_context)
    planner_binding = (
        b_context.belief_state == postclose["company_B"]["belief_state"]
        and b_context.game_theory_advice
        == postclose["company_B"]["game_theory_advice"]
        and "Approximate Bayesian Price Response" in prompt
        and "未验证信号" in prompt
    )
    advisor_non_binding = all(
        item["game_theory_advice"]["recommendation_is_non_binding"]
        and not item["game_theory_advice"]["uses_hidden_opponent_state"]
        and "final_action" not in item["game_theory_advice"]
        for item in postclose.values()
    )

    settle = controller.post(
        f"/api/v1/controller/episodes/{episode_id}/settle-agent-round",
        headers={"X-Controller-Token": controller_token},
        json={
            "step_id": f"{episode_id}:1:0",
            "intent_ids": {},
            "fallback": "rule",
        },
    )
    if settle.status_code != 200:
        raise RuntimeError(settle.text)
    session = SESSIONS[episode_id]
    joint_action = {
        company_id: item["action"]
        for company_id, item in settle.json()["decision_resolutions"].items()
    }
    phase = CommunicationPhaseRecord.model_validate(
        close.json()["communication_phase"]
    )
    event = SimpleNamespace(
        event_schema_version="agent-round-event-v1.8.0",
        state_before=initial_state.to_dict(),
        joint_action=joint_action,
        communication_phase=phase,
        traces=[
            SimpleNamespace(
                company_id=company_id,
                observation=postclose[company_id],
                observation_hash=postclose[company_id]["observation_hash"],
                decision_context=None,
                information_snapshot=ObservationSnapshot.from_observation(
                    postclose[company_id], company_id
                ),
            )
            for company_id in COMPANIES
        ],
    )
    information_views = verify_information_replay([event], session.manifest)
    belief_views = verify_belief_replay([event], session.manifest)
    advisor_views = verify_advisor_replay([event])
    interaction_views = verify_interaction_replay([phase])
    market_states = verify_replay(
        MarketEnv(CONFIG), session.manifest, session.transitions
    )

    next_b = gateway.get(
        f"/v1/episodes/{episode_id}/companies/company_B/observation",
        headers={"X-Agent-Token": tokens["company_B"]},
    ).json()
    reliability_after_betrayal = next_b["belief_state"]["opponent_beliefs"][
        "company_A"
    ]["signal_reliability_ppm"]

    tampered_replay_rejected = False
    tampered = deepcopy(event.traces[0].information_snapshot)
    tampered.observation["public_state"]["rounds_remaining"] += 1
    try:
        verify_information_snapshot(initial_state, tampered)
    except (InformationReplayMismatchError, ValueError):
        tampered_replay_rejected = True

    checks = {
        "p0_strict_contracts": bool(strict_contracts),
        "p0_public_states_identical": public_states_identical,
        "p0_private_scope_correct": private_scope_correct,
        "p0_opponent_private_leak_zero": opponent_private_leak_count == 0,
        "p0_tampered_observation_rejected": tampered_replay_rejected,
        "p1_belief_hash_present": all(
            item["belief_hash"].startswith("sha256:")
            for item in postclose.values()
        ),
        "p2_prediction_responded_to_signal": prediction_responded_to_signal,
        "p3_belief_and_advisor_entered_planner": planner_binding,
        "p4_signal_visibility_correct": signal_visibility_correct,
        "p4_signal_did_not_change_market": state_unchanged_after_signal,
        "p4_betrayal_lowered_reliability": reliability_after_betrayal == 333_333,
        "p5_advisor_non_binding": advisor_non_binding,
        "economic_replay_100_percent": len(market_states) == 2,
        "interaction_replay_100_percent": len(interaction_views) == 1,
        "information_replay_100_percent": len(information_views) == 4,
        "belief_replay_100_percent": len(belief_views) == 4,
        "advisor_replay_100_percent": len(advisor_views) == 4,
    }
    summary = {
        "acceptance_schema_version": "incomplete-information-p0-p5-v1.0.0",
        "episode_id": episode_id,
        "episode_seed": 20260821,
        "market_state_before_signal_hash": initial_state.state_hash,
        "market_state_after_signal_hash": initial_state.state_hash,
        "market_state_after_settlement_hash": market_states[-1].state_hash,
        "private_message_id": message_id,
        "signal_visibility": signal_ids,
        "company_B_cut_probability_ppm": b_cut,
        "company_C_cut_probability_ppm": c_cut,
        "company_A_reliability_after_false_claim_ppm": (
            reliability_after_betrayal
        ),
        "replay_counts": {
            "economic_states": len(market_states),
            "interaction_rounds": len(interaction_views),
            "information_views": len(information_views),
            "belief_views": len(belief_views),
            "advisor_views": len(advisor_views),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "research_boundary": (
            "deterministic engineering proof only; no claim of LLM behavioral "
            "effect, equilibrium convergence, or profit improvement"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_acceptance(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
