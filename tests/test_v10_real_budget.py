import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from game_theory_agent.experiments.v10_beta_real import BudgetLedger, MeteredCompletions, RESERVATION, MODEL


def test_durable_budget_counts_failed_attempts_and_refuses_repeated_call(tmp_path):
    path=tmp_path/"cost.sqlite3"
    ledger=BudgetLedger(path,"plan",cap=RESERVATION*2,max_calls=3)
    ledger.reserve("failed")
    resumed=BudgetLedger(path,"plan",cap=RESERVATION*2,max_calls=3)
    with pytest.raises(sqlite3.IntegrityError): resumed.reserve("failed")
    resumed.reserve("second")
    with pytest.raises(RuntimeError,match="budget exhausted"): resumed.reserve("third")
    assert sum(e["reserved"] for e in resumed.entries())==2*RESERVATION
    with pytest.raises(RuntimeError,match="another preregistration"): BudgetLedger(path,"different")


def test_oversized_request_is_refused_before_any_network_or_reservation(tmp_path):
    calls=[]
    async def network(**kwargs): calls.append(kwargs)
    ledger=BudgetLedger(tmp_path/"cost.sqlite3","plan")
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=network)))
    meter=MeteredCompletions(client,ledger,"one")
    with pytest.raises(RuntimeError,match="preflight"):
        asyncio.run(meter.create(model=MODEL,max_tokens=4000,stream=False,messages=[{"role":"user","content":"中"*64000}]))
    assert not calls and not ledger.entries()


def test_usage_is_recorded_before_downstream_json_validation(tmp_path):
    ledger=BudgetLedger(tmp_path/"cost.sqlite3","plan")
    ledger.reserve("invalid-json")
    response=SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100,completion_tokens=50),model_dump_json=lambda:'{"choices":[{"message":{"content":"invalid"}}]}')
    ledger.record("invalid-json",response)
    assert ledger.entries()[0]["actual"] == 750
    assert ledger.entries()[0]["reserved"] == RESERVATION
