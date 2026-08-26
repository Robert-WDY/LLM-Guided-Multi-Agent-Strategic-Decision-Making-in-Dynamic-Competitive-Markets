from fastapi.testclient import TestClient

from game_theory_agent.api import SESSIONS, app
from game_theory_agent.orchestration import COMPANY_IDS
from game_theory_agent.orchestration.coordinator import RoundCoordinator
from game_theory_agent.orchestration.local import LocalControllerClient, LocalGatewayClient
from game_theory_agent.orchestration.local import run_coroutine_sync


def test_100_rule_episodes_complete_twenty_rounds_without_partial_settlement(
    monkeypatch,
):
    token = "stability-controller-token"
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", token)
    SESSIONS.clear()
    client = TestClient(app)
    coordinator = RoundCoordinator(
        LocalControllerClient(token),
        LocalGatewayClient(),
        {},
    )
    missing_actions = 0
    duplicate_actions = 0
    version_conflicts = 0
    for seed in range(100):
        episode_id = f"stability-{seed}"
        created = client.post(
            "/api/episodes",
            json={
                "episode_id": episode_id,
                "episode_seed": seed,
                "company_ids": ["company_A", "company_B", "company_C", "company_D"],
                "max_rounds": 20,
                "market_model": "balanced",
            },
            headers={"X-Controller-Token": token},
        )
        assert created.status_code == 201, created.text
        rounds = run_coroutine_sync(coordinator.run_episode(episode_id))
        assert len(rounds) == 20
        assert rounds[-1].settlement["state"]["terminal"] is True
        expected_before = 0
        seen_step_ids = []
        for index, coordinated in enumerate(rounds, start=1):
            event = coordinated.event
            assert event.settled_round == index
            assert event.state_before["state_version"] == expected_before
            if event.state_after["state_version"] != expected_before + 1:
                version_conflicts += 1
            expected_before = event.state_after["state_version"]
            joint = coordinated.event.joint_action
            assert set(joint) == {
                "company_A",
                "company_B",
                "company_C",
                "company_D",
            }
            if len(joint) != 4:
                missing_actions += 1
            step_id = f"{episode_id}:{event.state_before['round']}:{event.state_before['state_version']}"
            if step_id in seen_step_ids:
                duplicate_actions += 1
            seen_step_ids.append(step_id)
            if event.state_after["state_version"] != event.state_before["state_version"] + 1:
                version_conflicts += 1
    assert missing_actions == 0
    assert duplicate_actions == 0
    assert version_conflicts == 0


def test_eight_rule_agents_complete_twenty_rounds(monkeypatch):
    token = "eight-company-controller-token"
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", token)
    SESSIONS.clear()
    company_ids = list(COMPANY_IDS[:8])
    created = TestClient(app).post(
        "/api/episodes",
        json={
            "episode_id": "eight-rule-20",
            "episode_seed": 8,
            "company_ids": company_ids,
            "max_rounds": 20,
            "market_model": "balanced",
        },
        headers={"X-Controller-Token": token},
    )
    assert created.status_code == 201, created.text
    coordinator = RoundCoordinator(
        LocalControllerClient(token),
        LocalGatewayClient(),
        {},
    )
    rounds = run_coroutine_sync(coordinator.run_episode("eight-rule-20"))
    assert len(rounds) == 20
    assert rounds[-1].settlement["state"]["terminal"] is True
    assert set(rounds[0].event.joint_action) == set(company_ids)
    for coordinated in rounds:
        traces = {trace.company_id: trace for trace in coordinated.event.traces}
        assert all(
            traces[company_id].resolution_source == "controller-rule-fallback"
            for company_id in company_ids
        )
