"""No paid operation is reachable before acceptance or after an unknown result."""
import asyncio
import importlib
import sys
from pathlib import Path
import pytest

@pytest.fixture
def runner(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    mod=importlib.import_module('real_reliable_v19')
    monkeypatch.setattr(mod,'OUT',tmp_path)
    monkeypatch.setattr(mod,'REAL',tmp_path/'real')
    monkeypatch.setattr(mod,'freeze_market',lambda:None)
    monkeypatch.setattr(mod,'status',lambda:dict(total_cny=30,reserved_cny=9.993565,new_calls=186))
    async def forbidden(*args,**kwargs):raise AssertionError('paid call must remain unreachable')
    monkeypatch.setattr(mod,'choose_option',forbidden)
    return mod

def test_missing_acceptance_blocks_all_model_calls(runner):
    asyncio.run(runner.main())
    assert runner.read_json(runner.REAL/'status.json')['new_calls']==0
    assert runner.read_json(runner.REAL/'status.json')['status']=='blocked_by_preregistered_gate'

def test_failed_economic_gate_preserves_reason_and_blocks_calls(runner):
    runner.write_json(runner.OUT/'acceptance.json',dict(engineering_passed=True,paid_gate_passed=False))
    runner.write_json(runner.OUT/'analysis.json',dict(paid_gate_reasons=['negative out-of-sample gains']))
    asyncio.run(runner.main())
    assert runner.read_json(runner.REAL/'status.json')['reasons']==['negative out-of-sample gains']

def test_unknown_draft_and_adoption_are_not_retried(runner,tmp_path):
    p=tmp_path/'pending.json';runner.write_json(p,dict(status='pending_unknown'))
    with pytest.raises(RuntimeError,match='no automatic retry'):asyncio.run(runner.draft_call('test-model',None,None,p))
    with pytest.raises(RuntimeError,match='no retry'):asyncio.run(runner.adoption_call('test-model',None,None,{}, {},p))

def test_unchanged_advice_needs_no_paid_adoption(runner,tmp_path):
    import json
    value=json.loads((Path(__file__).resolve().parents[1]/'frontend/tests/fixtures/advisor-reliable.json').read_text(encoding='utf-8'))
    draft=value['draft_action'];p=tmp_path/'unchanged.json'
    r=asyncio.run(runner.adoption_call('test-model',None,None,draft,dict(action=draft),p))
    assert r['status']=='not_needed' and r['selected']=='draft'

@pytest.mark.parametrize('model',['deepseek-v4-flash','doubao-seed-2-0-lite-260215'])
@pytest.mark.parametrize('market,companies,seed',[(m,(2,5,10)[i%3],526001+i) for i,m in enumerate(('normal','recession','tight_supply','price_sensitive','project_feasible','scaled'))])
def test_real_draft_envelope_offline_for_each_size(runner,tmp_path,monkeypatch,model,market,companies,seed):
    from types import SimpleNamespace
    import json,openai
    c,s,_,_,_=runner.source(market,companies,seed)
    captured=[]
    class OfflineClient:
        def __init__(self,**kwargs):self.chat=SimpleNamespace(completions=object())
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    async def reply(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(dict(action=dict(price_cents=12000),reason='offline schema fixture'))))],usage=None)
    monkeypatch.setattr(openai,'AsyncOpenAI',OfflineClient)
    monkeypatch.setattr(runner,'GuardedCompletions',lambda *args,**kwargs:SimpleNamespace(create=reply))
    monkeypatch.setenv('DEEPSEEK_API_KEY','offline-fixture');monkeypatch.setenv('ARK_API_KEY','offline-fixture')
    r=asyncio.run(runner.draft_call(model,c,s,tmp_path/'draft.json'))
    assert r['status']=='complete' and len(captured)==1
    assert len(json.dumps(captured[0]['messages'],ensure_ascii=False).encode())<=8000
