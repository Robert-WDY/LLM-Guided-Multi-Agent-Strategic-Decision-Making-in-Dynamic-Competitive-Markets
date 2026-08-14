from fastapi.testclient import TestClient

from game_theory_agent.api import SESSIONS, agent_app, app


client = TestClient(app)
agent_client = TestClient(agent_app)


def test_api_creates_and_steps_v4_episode():
    SESSIONS.clear()
    created = client.post(
        "/api/episodes",
        json={
            "episode_id": "api-test",
            "episode_seed": 42,
            "company_ids": ["company_A", "company_B"],
        },
    )
    assert created.status_code == 201
    body = created.json()
    state = body["state"]
    assert state["round"] == 1
    assert body["manifest"]["environment_version"] == "market-env-v4.1.0"
    assert set(body["action_constraints"]) == {"company_A", "company_B"}

    joint_action = {}
    for company_id in ("company_A", "company_B"):
        joint_action[company_id] = {
            "action_id": f"api-test:1:{company_id}",
            "episode_id": "api-test",
            "agent_id": company_id,
            "round": 1,
            "state_version": 0,
            "price_cents": 10000,
            "advertising_budget_cents": 800000,
            "service_budget_cents": 800000,
            "capacity_investment_cents": 0,
            "resilience_budget_cents": 0,
            "incident_response": {"mode": "wait", "repair_budget_cents": 0},
            "strategy_summary": "api test",
        }
    stepped = client.post(
        "/api/episodes/api-test/steps",
        json={"step_id": "api-test:1:0", "joint_action": joint_action},
    )
    assert stepped.status_code == 200
    next_state = stepped.json()["state"]
    assert next_state["round"] == 2
    assert next_state["state_version"] == 1
    assert next_state["market"]["realized_demand_orders"] > 0
    assert next_state["state_hash"].startswith("sha256:")


def test_agent_can_discover_episode_choices_without_creation_access():
    response = agent_client.get("/v1/episode-options")
    assert response.status_code == 200
    body = response.json()
    assert body["round_options"] == [5, 10, 15, 20]
    assert body["seed"]["min"] == 0
    assert body["seed"]["max"] == (1 << 64) - 1
    assert set(body["market_models"]) == {
        "random",
        "balanced",
        "value_oriented",
        "quality_oriented",
        "service_oriented",
    }
    assert body["creation_boundary"]["agent_gateway_can_create"] is False


def test_controller_accepts_round_seed_and_market_choices():
    SESSIONS.clear()
    created = client.post(
        "/api/episodes",
        json={
            "episode_id": "configured-episode",
            "episode_seed": 123_456,
            "company_ids": ["company_A", "company_B"],
            "market_model": "service_oriented",
            "max_rounds": 5,
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert body["state"]["episode_seed"] == 123_456
    assert body["state"]["max_rounds"] == 5
    assert body["state"]["market"]["market_model_id"] == "service_oriented"
    assert body["episode_options"]["round_options"] == [5, 10, 15, 20]


def test_api_rejects_unknown_episode():
    response = client.get("/api/episodes/missing/state")
    assert response.status_code == 404


def test_single_company_mode_runs_rule_opponents_and_retrospective():
    SESSIONS.clear()
    created = client.post(
        "/api/episodes",
        json={
            "episode_id": "player-test",
            "episode_seed": 42,
            "company_ids": [
                "company_A",
                "company_B",
                "company_C",
                "company_D",
            ],
            "game_mode": "single_company",
            "player_company_id": "company_A",
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert body["game_mode"] == "single_company"
    assert body["player_company_id"] == "company_A"
    assert body["company_analysis"]["health_score"] >= 0
    assert body["company_analysis"]["recommendations"]
    assert body["state"]["market"]["market_model_id"] in body["market_model_options"]

    state = body["state"]
    final_body = None
    for round_number in range(1, 11):
        player_action = {
            "action_id": f"player-test:{round_number}:company_A",
            "episode_id": "player-test",
            "agent_id": "company_A",
            "round": state["round"],
            "state_version": state["state_version"],
            "price_cents": 9800,
            "advertising_budget_cents": 600000,
            "service_budget_cents": 700000,
            "capacity_investment_cents": 0,
            "resilience_budget_cents": 0,
            "incident_response": {"mode": "wait", "repair_budget_cents": 0},
            "strategy_summary": "human player test",
        }
        stepped = client.post(
            "/api/episodes/player-test/player-steps",
            json={
                "step_id": (f"player-test:{state['round']}:{state['state_version']}"),
                "player_action": player_action,
            },
        )
        assert stepped.status_code == 200, stepped.text
        final_body = stepped.json()
        assert set(final_body["opponent_actions"]) == {
            "company_B",
            "company_C",
            "company_D",
        }
        assert all(
            action["strategy_summary"].startswith("rule-opponent")
            for action in final_body["opponent_actions"].values()
        )
        state = final_body["state"]

    assert final_body is not None
    assert state["terminal"]
    retrospective = final_body["retrospective"]
    assert retrospective["status"] == "complete"
    assert retrospective["outcome"] in {"成功", "部分成功", "失败"}
    assert len(retrospective["rounds"]) == 10
    assert retrospective["rounds"][0]["market_demand_orders"] > 0
    assert "supply_cost_delta_ppm" in retrospective["rounds"][0]
    assert "outside_option_orders" in retrospective["rounds"][0]
    assert retrospective["methodology"]
    assert len(retrospective["rankings"]["composite"]) == 4
    assert len(retrospective["rankings"]["total_assets"]) == 4
    assert len(retrospective["trend_series"]) == 4
    assert all(len(series["points"]) == 11 for series in retrospective["trend_series"])
    assert retrospective["rank_explanation"]
    assert retrospective["component_comparison"]
    fetched = client.get("/api/episodes/player-test/retrospective")
    assert fetched.status_code == 200
    assert fetched.json()["rounds"] == retrospective["rounds"]
    agent_terminal = agent_client.get(
        "/v1/episodes/player-test/companies/company_A/observation"
    )
    assert agent_terminal.status_code == 200
    terminal_body = agent_terminal.json()
    assert terminal_body["decision_round"] is None
    assert terminal_body["last_settled_round"] == 10
    assert terminal_body["terminal_summary"]["own_rank"] in {1, 2, 3, 4}
    assert len(terminal_body["terminal_summary"]["ranking"]) == 4
    assert len(terminal_body["terminal_summary"]["rankings"]["total_assets"]) == 4
    assert len(terminal_body["public_history"]) == 10
    assert (
        terminal_body["public_history"][0]["own_result"][
            "round_operating_cost_cents"
        ]
        > 0
    )
    resolved_signals = [
        outcome
        for item in terminal_body["public_history"]
        for outcome in item["resolved_signal_outcomes"]
    ]
    assert resolved_signals
    assert {item["outcome"] for item in resolved_signals} <= {
        "realized",
        "not_realized",
    }
    for history_item in terminal_body["public_history"]:
        expected_supply = history_item["market"]["base_supply_cost_index_ppm"]
        for event in history_item["active_events_during_round"]:
            expected_supply = round(
                expected_supply * event["supply_cost_multiplier_ppm"] / 1_000_000
            )
        assert history_item["market"]["actual_supply_cost_index_ppm"] == expected_supply


def test_player_step_rejects_market_mode_episode():
    SESSIONS.clear()
    client.post(
        "/api/episodes",
        json={
            "episode_id": "market-only",
            "company_ids": ["company_A", "company_B"],
        },
    )
    response = client.post(
        "/api/episodes/market-only/player-steps",
        json={"step_id": "market-only:1:0", "player_action": {}},
    )
    assert response.status_code == 409


def test_agent_gateway_accepts_intent_without_mutating_and_controller_executes(
    monkeypatch,
):
    SESSIONS.clear()
    created = client.post(
        "/api/episodes",
        json={
            "episode_id": "agent-api-test",
            "episode_seed": 17,
            "company_ids": ["company_A", "company_B"],
        },
    )
    assert created.status_code == 201
    initial_hash = created.json()["state"]["state_hash"]

    capabilities = agent_client.get("/v1/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["can_execute_action"] is False

    observation = agent_client.get(
        "/v1/episodes/agent-api-test/companies/company_A/observation"
    )
    assert observation.status_code == 200
    observed = observation.json()
    assert observed["state_hash"] == initial_hash
    assert "cash_balance_cents" not in observed["public_companies"][1]
    assert "financial" in observed["own_company"]

    submitted = agent_client.post(
        "/v1/episodes/agent-api-test/intents",
        json={
            "agent_id": "test-planner",
            "company_id": "company_A",
            "round": 1,
            "state_version": 0,
            "requested_action": {
                "price_cents": 10_000,
                "advertising_budget_cents": 600_000,
                "service_budget_cents": 700_000,
                "capacity_investment_cents": 900_000,
                "resilience_budget_cents": 0,
            },
            "rationale": "test",
        },
    )
    assert submitted.status_code == 202, submitted.text
    intent = submitted.json()
    assert intent["executed"] is False
    capacity_resolution = intent["resolution"]["action"][
        "capacity_investment_cents"
    ]
    assert capacity_resolution == 0
    unchanged = client.get("/api/episodes/agent-api-test/state")
    assert unchanged.json()["state"]["state_hash"] == initial_hash

    disabled = client.post(
        "/api/v1/controller/episodes/agent-api-test/settle-agent-round",
        json={
            "step_id": "agent-api-test:1:0",
            "intent_ids": {"company_A": intent["intent_id"]},
            "fallback": "rule",
        },
    )
    assert disabled.status_code == 503

    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", "unit-test-controller-token")
    settled = client.post(
        "/api/v1/controller/episodes/agent-api-test/settle-agent-round",
        headers={"X-Controller-Token": "unit-test-controller-token"},
        json={
            "step_id": "agent-api-test:1:0",
            "intent_ids": {"company_A": intent["intent_id"]},
            "fallback": "rule",
        },
    )
    assert settled.status_code == 200, settled.text
    assert settled.json()["state"]["state_version"] == 1
    assert settled.json()["executed_intent_ids"] == [intent["intent_id"]]
    next_observation = agent_client.get(
        "/v1/episodes/agent-api-test/companies/company_A/observation"
    ).json()
    assert len(next_observation["public_history"]) == 1
    history = next_observation["public_history"][0]
    assert history["own_result"]["round_operating_cost_cents"] > 0
    assert "event_impact_explanation" in history
    fetched = agent_client.get(
        f"/v1/episodes/agent-api-test/intents/{intent['intent_id']}"
    )
    assert fetched.json()["status"] == "executed"


def test_agent_gateway_rejects_stale_intent():
    SESSIONS.clear()
    client.post(
        "/api/episodes",
        json={
            "episode_id": "stale-agent-test",
            "company_ids": ["company_A", "company_B"],
        },
    )
    response = agent_client.post(
        "/v1/episodes/stale-agent-test/intents",
        json={
            "agent_id": "test-planner",
            "company_id": "company_A",
            "round": 1,
            "state_version": 99,
            "requested_action": {"price_cents": 10_000},
        },
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "STALE_OBSERVATION"
