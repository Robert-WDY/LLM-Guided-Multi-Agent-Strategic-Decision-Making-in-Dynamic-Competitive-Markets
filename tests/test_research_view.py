from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from game_theory_agent import api
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.market import load_market_config

@pytest.fixture
def client(monkeypatch,isolated_persistence):
    config=load_market_config(Path(__file__).parents[1]/'configs/market_v13_1_local.yaml')
    monkeypatch.setattr(api,'CONFIG',config)
    monkeypatch.setattr(api,'PERSONA_REGISTRY',PersonaRegistry.from_market_config(config))
    monkeypatch.setenv('MARKET_CONTROLLER_TOKEN','research-test')
    monkeypatch.setenv('MARKET_PERSISTENCE','1')
    api.SESSIONS.clear()
    return TestClient(api.app)

def test_research_trace_human_actions_messages_and_cold_restore(client):
    headers={'X-Controller-Token':'research-test'}
    p='/api/v1/controller/episodes/research-test'
    created=client.post('/api/episodes',headers=headers,json={'episode_id':'research-test','episode_seed':210131,'max_rounds':5,'information_mode':'public','communication_mode':'public_private','cooperation_mode':'combined_v1','repeated_game_mode':'reciprocity_v1','agent_configs':{'company_A':{'provider':'human','agent_type':'human','model':'Human'}}})
    assert created.status_code==201,created.text
    state=created.json()['state']
    body={'run_id':'human-one','expected_round':state['round'],'expected_state_version':state['state_version'],'expected_state_hash':state['state_hash'],'max_rounds':1,'player_action':{'agent_id':'company_A','price_cents':9800,'service_budget_cents':100000,'capacity_investment_cents':100000,'procurement_quantity_orders':1234,'primary_supplier_id':'resilient_supplier','backup_supplier_id':'economy_supplier','primary_supplier_share_ppm':600000,'threshold_project_contribution_cents':100000},'player_communication':{'messages':[{'channel':'public','recipients':[],'content':'本轮计划提高服务质量'},{'channel':'private','recipients':['company_B'],'content':'希望下一轮互助'}]}}
    response=client.post(p+'/coordinator-run',headers=headers,json=body)
    assert response.status_code==200,response.text
    view=client.get(p+'/research-view',headers=headers)
    assert view.status_code==200,view.text
    detail=view.json()['detail']
    assert detail['trace_available'] and len(detail['traces'])==4
    action=detail['final_actions']['company_A']
    assert action['procurement_quantity_orders']==1234
    assert action['service_budget_cents']==100000
    assert action['threshold_project_contribution_cents']==100000
    assert len(detail['messages'])==2
    assert detail['messages'][1]['recipients']==['company_B'] or detail['messages'][0]['recipients']==['company_B']
    assert client.get(p+'/research-view').status_code==401
    assert client.get(p+'/research-view?round_number=5',headers=headers).status_code==422
    api.SESSIONS.clear()
    restored=client.get(p+'/research-view',headers=headers)
    assert restored.json()==view.json()
    assert client.post(p+'/coordinator-run',headers=headers,json=body).json()==response.json()

def test_empty_and_old_rounds_do_not_invent_traces(client):
    headers={'X-Controller-Token':'research-test'}
    created=client.post('/api/episodes',headers=headers,json={'episode_id':'old','max_rounds':5})
    assert created.status_code==201,created.text
    path='/api/v1/controller/episodes/old/research-view'
    assert client.get(path,headers=headers).json()['detail'] is None
    from game_theory_agent.gameplay import build_rule_action
    session=api.SESSIONS['old'];state=session.env.get_state()
    joint={i:build_rule_action(api.CONFIG,state,i).to_dict() for i in state.company_ids}
    step=client.post('/api/episodes/old/steps',json={'step_id':'old:1:0','joint_action':joint})
    assert step.status_code==200,step.text
    detail=client.get(path,headers=headers).json()['detail']
    assert not detail['trace_available'] and not detail['traces']
