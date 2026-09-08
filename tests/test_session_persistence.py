from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from game_theory_agent import api
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import load_market_config
from game_theory_agent.persistence import CheckpointError, CODEC
from game_theory_agent.orchestration.episode_runtimes import requested_action_from_payload, sync_episode_runtimes


@pytest.fixture
def saved_market(monkeypatch, isolated_persistence):
    monkeypatch.setenv("MARKET_PERSISTENCE", "1")
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN", "checkpoint-test-token")
    api.SESSIONS.clear()
    return TestClient(api.app), {"X-Controller-Token":"checkpoint-test-token"}


def create(client, headers, **extra):
    response = client.post("/api/episodes", headers=headers, json={"episode_id":"durable-test", "episode_seed":77, **extra})
    assert response.status_code == 201, response.text
    return response.json()


def test_restart_restores_market_and_same_step_id_does_not_settle_twice(saved_market):
    client, headers = saved_market
    create(client, headers)
    session = api.SESSIONS["durable-test"]
    before = session.env.get_state()
    joint = {company:build_rule_action(api.CONFIG,before,company).to_dict() for company in before.company_ids}
    request = {"step_id":"durable-test:1:0", "joint_action":joint}
    first = client.post("/api/episodes/durable-test/steps",json=request)
    assert first.status_code == 200, first.text
    stored_hash = first.json()["state"]["state_hash"]
    api.SESSIONS.clear()
    restored = client.post("/api/v1/controller/episodes/durable-test/restore",headers=headers)
    assert restored.status_code == 200, restored.text
    assert restored.json()["state"]["state_hash"] == stored_hash
    assert len(api.SESSIONS["durable-test"].transitions)==1
    retried = client.post("/api/episodes/durable-test/steps", json=request)
    assert retried.status_code == 200, retried.text
    assert retried.json() == first.json()
    assert len(api.SESSIONS["durable-test"].transitions) == 1
    assert client.post("/api/episodes", json={"episode_id":"durable-test"}).status_code == 409
    # MarketEnv's authoritative cache survives restart, even when an action
    # retry arrives via another adapter.
    env = api.SESSIONS["durable-test"].env
    from game_theory_agent.market import CompanyAction
    result = env.step("durable-test:1:0",{k:CompanyAction.from_dict(v) for k,v in joint.items()})
    assert result.state_after.state_hash == stored_hash
    assert env.get_state().state_version == 1


def test_restore_preserves_agent_observation_memory_and_tokens(saved_market):
    client, headers = saved_market
    created = create(client,headers,information_mode="public",belief_mode="public_action_v1",opponent_model_mode="public_strategy_v1",agent_configs={"company_A":{"provider":"mock","persona":{"persona_id":"risk_guarded_v1"}}})
    current = created["state"]
    request = {"run_id":"mock-first", "expected_round":current["round"],"expected_state_version":current["state_version"],"expected_state_hash":current["state_hash"],"max_rounds":1}
    response = client.post("/api/v1/controller/episodes/durable-test/coordinator-run",headers=headers,json=request)
    assert response.status_code == 200, response.text
    gateway = TestClient(api.agent_app)
    agent_headers={"X-Agent-Token":created["agent_tokens"]["company_A"]}
    url="/v1/episodes/durable-test/companies/company_A/observation"
    observation = gateway.get(url,headers=agent_headers).json()
    memory = api.SESSIONS["durable-test"].agent_runtimes["company_A"].memory.snapshot()
    api.SESSIONS.clear()
    assert gateway.get(url,headers=agent_headers).json()==observation
    restored = api.SESSIONS["durable-test"]
    assert sync_episode_runtimes(restored)["company_A"].memory.snapshot()==memory
    repeated = client.post("/api/v1/controller/episodes/durable-test/coordinator-run",headers=headers,json=request)
    assert repeated.json()==response.json()


def test_saved_configuration_excludes_ui_credentials_and_detects_corruption(saved_market):
    client,headers=saved_market
    create(client,headers)
    response=client.post("/api/v1/controller/episodes/durable-test/save",headers=headers,json={"ui_state":{"config":{"controllerToken":"do-not-save","seed":77},"agent_tokens":{"a":"never"}}})
    assert response.status_code==200,response.text
    text=api.SESSION_STORE.path.read_bytes()
    assert b"do-not-save" not in text
    assert client.get("/api/v1/controller/saved-episodes").status_code==401
    with api.SESSION_STORE._connect() as db:
        db.execute("UPDATE checkpoints SET document='{}'")
    api.SESSIONS.clear()
    response=client.post("/api/v1/controller/episodes/durable-test/restore",headers=headers)
    assert response.status_code==409
    with pytest.raises(CheckpointError):
        CODEC.decode({"kind":"ledger","type":"os:system","data":{}})


def test_interrupted_run_requires_explicit_recovery_without_model_calls(saved_market):
    client,headers=saved_market
    create(client,headers,information_mode="public",belief_mode="public_action_v1")
    session=api.SESSIONS["durable-test"]
    session.coordinator_active_run_id="interrupted-paid-run"
    api.SESSION_STORE.save(session)
    api.SESSIONS.clear()
    restored=client.post("/api/v1/controller/episodes/durable-test/restore",headers=headers)
    assert restored.json()["checkpoint"]["recovery_required"]
    response=client.post("/api/v1/controller/episodes/durable-test/recover",headers=headers)
    assert response.status_code==200,response.text
    assert response.json()["recovery"]["model_calls"]==0
    assert response.json()["state"]["round"]==2
    assert not response.json()["checkpoint"]["recovery_required"]


def test_v10_human_procurement_and_runtime_persona_follow_episode_config(monkeypatch, saved_market):
    config=load_market_config(api.PROJECT_ROOT/"configs/market_v10_multi_objective.yaml")
    from game_theory_agent.market import MarketEnv
    from game_theory_agent.market.replay import EpisodeManifest
    env=MarketEnv(config)
    state=env.reset(episode_id="v10-runtime")
    manifest=EpisodeManifest.create(env,state,agent_configs={"company_A":{"provider":"mock","persona":{"persona_id":"public_service"}}})
    session=api.EpisodeSession(env=env,manifest=manifest)
    runtime=sync_episode_runtimes(session)["company_A"]
    assert runtime.context_builder.persona_profile.persona_id=="public_service"
    requested=requested_action_from_payload({"price_cents":10000,"primary_supplier_id":"resilient_supplier","backup_supplier_id":"economy_supplier","primary_supplier_share_ppm":600000})
    assert requested.primary_supplier_id=="resilient_supplier"
    assert requested.backup_supplier_id=="economy_supplier"
    assert requested.primary_supplier_share_ppm==600000
