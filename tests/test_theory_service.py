from pathlib import Path
import pytest
from fastapi import FastAPI,HTTPException
from fastapi.testclient import TestClient
from game_theory_agent.market import load_market_config
from game_theory_agent.game_theory.service import TheoryService,ExperimentRequest,router

CONFIG=load_market_config(Path(__file__).parents[1]/'configs/market_v14_local.yaml')


def test_persist_integrity_idempotency_and_authorization(tmp_path):
    service=TheoryService(tmp_path,CONFIG);request=ExperimentRequest(request_id='one',kind='matrix')
    record=service.execute(request);assert service.execute(request)==record
    assert TheoryService(tmp_path,CONFIG).get(record['id'])==record
    with pytest.raises(ValueError):service.execute(request.model_copy(update={'kind':'information'}))
    def auth(token):
        if token!='test':raise HTTPException(403,'controller required')
    app=FastAPI();app.include_router(router(tmp_path,CONFIG,auth));client=TestClient(app)
    assert client.get('/api/v1/controller/theory-lab/catalog').status_code==403
    assert client.get('/api/v1/controller/theory-lab/experiments',headers={'X-Controller-Token':'test'}).json()['experiments'][0]['id']==record['id']
    assert client.post('/api/v1/controller/theory-lab/experiments',headers={'X-Controller-Token':'test'},json={'request_id':'bad','kind':'learning','parameters':{'rounds':1000000}}).status_code==422
    p=tmp_path/(record['id']+'.json');p.write_text(p.read_text(encoding='utf-8').replace('prisoners_dilemma','matching_pennies'),encoding='utf-8')
    with pytest.raises(ValueError,match='checksum'):service.get(record['id'])


def test_all_history_pages_survive_isolated_corrupt_records(tmp_path):
    import json
    service=TheoryService(tmp_path,CONFIG)
    records=[service.execute(ExperimentRequest(request_id=f'history-{i}',kind='matrix')) for i in range(105)]
    broken=records[50]['id'];(tmp_path/(broken+'.json')).write_text('{}',encoding='utf-8')
    truncated=records[51]['id'];(tmp_path/(truncated+'.json')).write_text('{',encoding='utf-8')
    app=FastAPI();app.include_router(router(tmp_path,CONFIG,lambda token:None));client=TestClient(app)
    ids=[];issues=[];offset=0
    while offset is not None:
        response=client.get('/api/v1/controller/theory-lab/experiments',params={'offset':offset,'limit':30})
        assert response.status_code==200
        page=response.json();ids.extend(v['id'] for v in page['experiments']);issues.extend(i['id'] for i in page['issues']);offset=page['next_offset']
    assert len(ids)==len(set(ids))==103
    assert set(issues)=={broken,truncated}
    assert set(ids)=={r['id'] for r in records}-{broken,truncated}
    assert (tmp_path/(broken+'.json')).read_text()=='{}'
    assert client.get('/api/v1/controller/theory-lab/experiments/'+broken).status_code==422
    assert client.get('/api/v1/controller/theory-lab/experiments',params={'limit':1000}).status_code==422
    assert client.post('/api/v1/controller/theory-lab/experiments',json={'request_id':'bool','kind':'repeated','parameters':{'rounds':True}}).status_code==422
    assert len(list(tmp_path.glob('*.json')))==105
