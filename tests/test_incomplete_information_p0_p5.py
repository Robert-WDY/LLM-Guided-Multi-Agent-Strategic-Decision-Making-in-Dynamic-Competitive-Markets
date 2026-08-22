from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from game_theory_agent.advisor import (
    AdvisorReplayMismatchError,
    BayesianGameAdvisor,
    verify_advisor_replay,
)
from game_theory_agent.agents.context import DecisionContextBuilder
from game_theory_agent.agents.memory import EpisodeMemory
from game_theory_agent.agents.prompt_builder import AgentPromptBuilder
from game_theory_agent.api import CONFIG, SESSIONS, agent_app, app
from game_theory_agent.belief import (
    BeliefLedger,
    BeliefReplayMismatchError,
    verify_belief_replay,
)
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.information import (
    ObservationEnvelope,
    ObservationSnapshot,
)
from game_theory_agent.interaction import (
    CommunicationRoundLedger,
    CommunicationSubmission,
)
from game_theory_agent.market import MarketEnv


COMPANIES = ("company_A", "company_B", "company_C", "company_D")


def _state(episode_id: str = "p0-p5"):
    env = MarketEnv(CONFIG)
    state = env.reset(
        list(COMPANIES),
        episode_id=episode_id,
        episode_seed=20260821,
        market_model="balanced",
        max_rounds=5,
    )
    return env, state


def test_strict_public_private_and_observation_contracts():
    from game_theory_agent.agents.observation import ObservationBuilder

    _env, state = _state("strict-contracts")
    views = {
        company_id: ObservationBuilder().build(state, company_id, "public")
        for company_id in COMPANIES
    }
    assert len({str(item["public_state"]) for item in views.values()}) == 1
    assert {
        item["private_state"]["company_id"] for item in views.values()
    } == set(COMPANIES)
    assert all(
        item["private_state"]["company"]["company_id"]
        == item["private_state"]["company_id"]
        for item in views.values()
    )

    # Validate a real Gateway envelope and prove that the top-level contract is
    # closed rather than accepting an unregistered visibility bypass.
    import os

    os.environ["MARKET_CONTROLLER_TOKEN"] = "p0-p5-token"
    SESSIONS.clear()
    created = TestClient(app).post(
        "/api/episodes",
        json={"episode_id": "strict-envelope", "episode_seed": 1, "max_rounds": 5},
    )
    assert created.status_code == 201
    observation = TestClient(agent_app).get(
        "/v1/episodes/strict-envelope/companies/company_A/observation"
    ).json()
    envelope = ObservationEnvelope.model_validate(observation)
    assert envelope.private_state.company_id == "company_A"
    forged = deepcopy(observation)
    forged["opponent_cash_cents"] = 99_999_999
    with pytest.raises(ValidationError):
        ObservationEnvelope.model_validate(forged)


def test_signal_belief_is_scoped_non_binding_and_reliability_updates():
    _env, state = _state("signal-ledger")
    communication = CommunicationRoundLedger(
        episode_id=state.episode_id,
        round_number=1,
        state_version=0,
        state_hash=state.state_hash,
        company_ids=state.company_ids,
        mode="public_private",
    )
    delivered = communication.submit(
        sender_company_id="company_A",
        submission=CommunicationSubmission.model_validate(
            {
                "messages": [
                    {
                        "channel": "private",
                        "recipients": ["company_B"],
                        "speech_act": "promise",
                        "content": "我计划把价格降到9000分。",
                        "own_action_claim": {"price_cents": 9000},
                    }
                ]
            }
        ),
    )[0]
    closure = communication.close()
    ledger = BeliefLedger(
        episode_id=state.episode_id,
        company_ids=state.company_ids,
        mode="public_action_signal_v2",
    )
    prices = {
        item.company_id: item.commercial.price_cents for item in state.companies
    }
    b_view, _ = ledger.company_view(
        observer_company_id="company_B",
        round_number=1,
        state_version=0,
        visible_messages=closure.views["company_B"].visible_messages,
        public_prices=prices,
    )
    c_view, _ = ledger.company_view(
        observer_company_id="company_C",
        round_number=1,
        state_version=0,
        visible_messages=closure.views["company_C"].visible_messages,
        public_prices=prices,
    )
    assert [item.message_id for item in b_view.visible_communication_signals] == [
        delivered.message_id
    ]
    assert c_view.visible_communication_signals == []
    assert b_view.visible_communication_signals[0].verified_fact is False
    assert b_view.opponent_beliefs["company_A"].next_price_direction.price_cut_ppm > 333_334
    assert c_view.opponent_beliefs["company_A"].next_price_direction.price_cut_ppm == 333_334

    actions = {
        company_id: build_rule_action(CONFIG, state, company_id)
        for company_id in state.company_ids
    }
    # Rule A does not honor the 9000 claim, so Beta(1,1) becomes Beta(1,2).
    ledger.update_after_settlement(
        state, actions, communication_messages=closure.all_messages
    )
    assert ledger.claim_reliability_ppm("company_A") == 333_333


def test_bayesian_advisor_is_deterministic_bounded_and_non_executable():
    env, state = _state("advisor")
    ledger = BeliefLedger(
        episode_id=state.episode_id, company_ids=state.company_ids
    )
    belief, belief_hash = ledger.company_view(
        observer_company_id="company_A", round_number=1, state_version=0
    )
    constraints = env.get_action_constraints("company_A", 0)
    advice = BayesianGameAdvisor().advise(
        belief_state=belief,
        own_company=state.company("company_A").to_dict(),
        action_constraints=constraints,
    )
    repeated = BayesianGameAdvisor().advise(
        belief_state=belief.model_dump(mode="json"),
        own_company=state.company("company_A").to_dict(),
        action_constraints=constraints,
    )
    assert advice == repeated
    assert advice.belief_hash == belief_hash
    assert advice.recommendation_is_non_binding is True
    assert advice.uses_hidden_opponent_state is False
    assert constraints["bounds"]["price_cents"]["min"] <= advice.recommended_price_cents
    assert advice.recommended_price_cents <= constraints["bounds"]["price_cents"]["max"]
    assert "final_action" not in advice.model_dump(mode="json")


def _create_signal_episode(monkeypatch: pytest.MonkeyPatch):
    token = "p0-p5-controller"
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", token)
    SESSIONS.clear()
    controller = TestClient(app)
    gateway = TestClient(agent_app)
    response = controller.post(
        "/api/episodes",
        headers={"X-Controller-Token": token},
        json={
            "episode_id": "p0-p5-api",
            "episode_seed": 77,
            "max_rounds": 5,
            "information_mode": "public",
            "communication_mode": "public_private",
            "belief_mode": "public_action_signal_v2",
            "advisor_mode": "bayesian_price_v1",
        },
    )
    assert response.status_code == 201, response.text
    return controller, gateway, token, response.json()


def test_p0_p5_api_pipeline_and_replays(monkeypatch):
    controller, gateway, controller_token, created = _create_signal_episode(monkeypatch)
    tokens = created["agent_tokens"]
    state = created["state"]
    before_hash = state["state_hash"]
    request = {
        "round": state["round"],
        "state_version": state["state_version"],
        "state_hash": state["state_hash"],
        "submission": {
            "messages": [
                {
                    "channel": "private",
                    "recipients": ["company_B"],
                    "speech_act": "promise",
                    "content": "本轮计划降到9000分。",
                    "own_action_claim": {"price_cents": 9000},
                }
            ]
        },
    }
    sent = gateway.post(
        "/v1/episodes/p0-p5-api/companies/company_A/communication/submissions",
        headers={"X-Agent-Token": tokens["company_A"]},
        json=request,
    )
    assert sent.status_code == 202, sent.text
    closed = controller.post(
        "/api/v1/controller/episodes/p0-p5-api/communication/close",
        headers={"X-Controller-Token": controller_token},
        json={key: request[key] for key in ("round", "state_version", "state_hash")},
    )
    assert closed.status_code == 200, closed.text
    assert SESSIONS["p0-p5-api"].env.get_state().state_hash == before_hash

    observations = {
        company_id: gateway.get(
            f"/v1/episodes/p0-p5-api/companies/{company_id}/observation",
            headers={"X-Agent-Token": tokens[company_id]},
        ).json()
        for company_id in COMPANIES
    }
    message_id = sent.json()["message_ids"][0]
    assert [
        item["message_id"]
        for item in observations["company_B"]["belief_state"][
            "visible_communication_signals"
        ]
    ] == [message_id]
    assert observations["company_C"]["belief_state"][
        "visible_communication_signals"
    ] == []
    assert all(
        observation["game_theory_advice"]["recommendation_is_non_binding"]
        for observation in observations.values()
    )
    b_context = DecisionContextBuilder().build(
        observations["company_B"], "company_B", EpisodeMemory()
    )
    prompt = AgentPromptBuilder().build(b_context)
    assert "Approximate Bayesian Response" in prompt
    assert "未验证信号" in prompt

    session = SESSIONS["p0-p5-api"]
    true_state = session.env.get_state()
    actions = {
        company_id: build_rule_action(CONFIG, true_state, company_id).to_dict()
        for company_id in COMPANIES
    }
    event = SimpleNamespace(
        event_schema_version="agent-round-event-v1.8.0",
        state_before=true_state.to_dict(),
        joint_action=actions,
        communication_phase=SimpleNamespace(
            closure=session.communication_ledgers[
                (true_state.round, true_state.state_version, true_state.state_hash)
            ].close(),
            generation_traces=[],
        ),
        traces=[
            SimpleNamespace(
                company_id=company_id,
                information_snapshot=ObservationSnapshot.from_observation(
                    observations[company_id], company_id
                ),
            )
            for company_id in COMPANIES
        ],
    )
    assert len(verify_belief_replay([event], session.manifest)) == 4
    assert len(verify_advisor_replay([event])) == 4

    tampered = deepcopy(event)
    tampered.traces[0].information_snapshot.observation["belief_state"][
        "opponent_beliefs"
    ]["company_B"]["next_price_direction"]["price_cut_ppm"] += 1
    with pytest.raises((BeliefReplayMismatchError, ValueError)):
        verify_belief_replay([tampered], session.manifest)

    tampered_advice = deepcopy(event)
    raw_advice = tampered_advice.traces[0].information_snapshot.observation[
        "game_theory_advice"
    ]
    raw_advice["recommended_price_cents"] += 1
    with pytest.raises((AdvisorReplayMismatchError, ValueError)):
        verify_advisor_replay([tampered_advice])
