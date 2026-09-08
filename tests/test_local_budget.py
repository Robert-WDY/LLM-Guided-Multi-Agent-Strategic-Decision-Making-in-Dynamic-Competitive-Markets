import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from game_theory_agent import local_budget as budget

@pytest.fixture
def ledger(tmp_path):
    path = tmp_path/'budget.sqlite3'
    budget.initialize(path)
    return path

def test_shared_budget_is_durable_and_race_safe(ledger):
    def call(_):
        try: return budget.reserve(ledger)
        except budget.LocalBudgetError: return None
    with ThreadPoolExecutor(max_workers=12) as pool: accepted=list(pool.map(call,range(20)))
    assert len([a for a in accepted if a]) == 7
    assert budget.status(ledger)['remaining_cny'] == .052
    assert budget.initialize(ledger)['new_calls'] == 7
    with pytest.raises(budget.LocalBudgetError): budget.reserve(ledger)

def test_missing_corrupt_and_deleted_ledger_fail_closed(tmp_path, ledger):
    with pytest.raises(budget.LocalBudgetError): budget.reserve(tmp_path/'missing')
    ledger.write_bytes(b'broken ledger')
    assert not budget.status(ledger)['ready']
    with pytest.raises(budget.LocalBudgetError): budget.reserve(ledger)
    ledger.unlink()
    with pytest.raises(budget.LocalBudgetError): budget.initialize(ledger)

def request():
    return dict(model='deepseek-v4-flash', messages=[{'role':'user','content':'test'}], stream=False, max_tokens=4000, extra_body={'thinking':{'type':'disabled'}})

def test_failed_provider_call_keeps_reservation_and_never_retries(ledger):
    class Failing:
        count=0
        async def create(self,**kwargs):
            self.count+=1
            raise TimeoutError('unknown provider outcome')
    provider=Failing()
    with pytest.raises(TimeoutError): asyncio.run(budget.GuardedCompletions(provider,ledger).create(**request()))
    assert provider.count == 1
    assert budget.status(ledger)['reserved_cny'] == 7.428

@pytest.mark.parametrize('field,value',[('model','deepseek-v4-pro'),('max_tokens',4001),('stream',True),('messages',[{'role':'user','content':'x'*124001}]),('extra_body',{})])
def test_unbounded_requests_block_before_cash_or_network(ledger,field,value):
    args=request(); args[field]=value
    with pytest.raises(budget.LocalBudgetError): asyncio.run(budget.GuardedCompletions(None,ledger).create(**args))
    assert budget.status(ledger)['new_calls'] == 0

def test_success_records_usage_but_does_not_refund(ledger):
    class Provider:
        async def create(self,**kwargs): return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100,completion_tokens=20))
    asyncio.run(budget.GuardedCompletions(Provider(),ledger).create(**request()))
    assert budget.status(ledger)['remaining_cny'] == 2.572

def test_unexpected_provider_overage_blocks_future_calls(ledger):
    class Provider:
        async def create(self,**kwargs): return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=200000,completion_tokens=20))
    with pytest.raises(budget.LocalBudgetError): asyncio.run(budget.GuardedCompletions(Provider(),ledger).create(**request()))
    with pytest.raises(budget.LocalBudgetError): budget.reserve(ledger)
