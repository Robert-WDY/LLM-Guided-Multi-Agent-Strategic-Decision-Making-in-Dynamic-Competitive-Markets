from dataclasses import replace
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market.cooperation_personas import apply_cooperation_persona

CONFIG = load_market_config(Path(__file__).parents[1]/"configs/market_v14_complete.yaml")

def action(state, persona, company="company_A"):
    return apply_cooperation_persona(CONFIG,state,company,build_rule_action(CONFIG,state,company),persona)

def test_contribution_withholding_forgiveness_and_cash_constraints():
    env=MarketEnv(CONFIG);s=env.reset(episode_id="formal",episode_seed=21,max_rounds=5,cooperation_mode="combined_v1")
    assert action(s,"cooperator").shared_resilience_contribution_cents>0
    assert action(s,"free_rider").shared_resilience_contribution_cents==0
    assert action(s,"retaliator").shared_resilience_contribution_cents>0
    actions={cid:action(s,"free_rider",cid) for cid in s.company_ids}
    s=env.step("formal:1:0",actions).state_after
    assert action(s,"retaliator").shared_resilience_contribution_cents==0
    actions={cid:action(s,"cooperator",cid) for cid in s.company_ids}
    s=env.step("formal:2:1",actions).state_after
    assert action(s,"retaliator").shared_resilience_contribution_cents>0
    terminal=replace(s,rounds_remaining=1)
    assert action(terminal,"cooperator").shared_resilience_contribution_cents==0
    poor=replace(s,companies=tuple(replace(c,financial=replace(c.financial,cash_balance_cents=0)) if c.company_id=="company_A" else c for c in s.companies))
    assert action(poor,"cooperator").shared_resilience_contribution_cents==0

def test_api_catalog_and_rule_seats_use_formal_profile(monkeypatch,isolated_persistence):
    from game_theory_agent import api
    from game_theory_agent.agents.personas import PersonaRegistry
    monkeypatch.setattr(api,"CONFIG",CONFIG);monkeypatch.setattr(api,"PERSONA_REGISTRY",PersonaRegistry.from_market_config(CONFIG))
    monkeypatch.setenv("MARKET_CONTROLLER_TOKEN","formal")
    api.SESSIONS.clear();client=TestClient(api.app);headers={"X-Controller-Token":"formal"}
    catalog=client.get("/api/v1/capabilities/personas").json()
    assert catalog["demo_only_personas"]==[]
    profiles={x["persona_id"]:x for x in catalog["profiles"]}
    assert len({profiles[p]["profile_hash"] for p in ("cooperator","free_rider","retaliator")})==3
    created=client.post("/api/episodes",headers=headers,json={"episode_id":"formal-api","max_rounds":5,"cooperation_mode":"combined_v1","agent_configs":{
        "company_A":{"provider":"rule","persona_name":"cooperator"},"company_B":{"provider":"rule","persona_name":"free_rider"}}})
    assert created.status_code==201,created.text
    s=created.json()["state"]
    body={"run_id":"one","expected_round":s["round"],"expected_state_version":s["state_version"],"expected_state_hash":s["state_hash"],"max_rounds":1}
    reply=client.post("/api/v1/controller/episodes/formal-api/coordinator-run",headers=headers,json=body)
    assert reply.status_code==200,reply.text
    view=client.get("/api/v1/controller/episodes/formal-api/research-view",headers=headers).json()
    actions=view["detail"]["final_actions"]
    assert actions["company_A"]["shared_resilience_contribution_cents"]>0
    assert actions["company_B"]["shared_resilience_contribution_cents"]==0
