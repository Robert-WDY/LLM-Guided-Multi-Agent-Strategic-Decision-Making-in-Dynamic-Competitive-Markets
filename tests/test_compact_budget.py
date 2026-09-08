import asyncio
from types import SimpleNamespace
import pytest
from game_theory_agent import local_budget as b

def test_compact_reserves_request_bound_without_refunding(tmp_path):
    p=tmp_path/"budget";b.initialize(p)
    class Provider:
        async def create(self,**kwargs): return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100,completion_tokens=50))
    args=dict(model="deepseek-v4-flash",stream=False,max_tokens=512,extra_body={"thinking":{"type":"disabled"}},messages=[{"role":"user","content":"小决策"}])
    asyncio.run(b.GuardedCompletions(Provider(),p,compact=True).create(**args))
    status=b.status(p)
    assert 7.008<status["reserved_cny"]<7.04 and status["new_calls"]==1
    args["messages"][0]["content"]="x"*8001
    with pytest.raises(b.LocalBudgetError):asyncio.run(b.GuardedCompletions(None,p,compact=True).create(**args))
    assert b.status(p)==status

def test_compact_unknown_failure_remains_reserved(tmp_path):
    p=tmp_path/"budget";b.initialize(p)
    class Provider:
        async def create(self,**kwargs):raise TimeoutError
    with pytest.raises(TimeoutError):
        asyncio.run(b.GuardedCompletions(Provider(),p,compact=True).create(model="deepseek-v4-flash",stream=False,max_tokens=512,extra_body={"thinking":{"type":"disabled"}},messages=[{"role":"user","content":"x"}]))
    assert b.status(p)["reserved_cny"]>7.008
