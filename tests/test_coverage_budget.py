import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from game_theory_agent.experiments.advisor_coverage_real_v11 import CoverageBudgetLedger, MeteredCompletions, MODEL, RESERVATION


def test_shared_budget_and_interrupted_call_cannot_be_retried(tmp_path):
    path=tmp_path/"budget.sqlite3"
    ledger=CoverageBudgetLedger(path,"spec",cap=RESERVATION*2,max_calls=2)
    ledger.reserve("interrupted")
    restarted=CoverageBudgetLedger(path,"spec",cap=RESERVATION*2,max_calls=2)
    with pytest.raises(sqlite3.IntegrityError):restarted.reserve("interrupted")
    restarted.reserve("second")
    with pytest.raises(RuntimeError):restarted.reserve("third")
    assert sum(r["reserved"] for r in restarted.entries())==2*RESERVATION
    assert all(r["actual"] is None for r in restarted.entries())


def test_oversized_prompt_never_reserves_or_calls_provider(tmp_path):
    ledger=CoverageBudgetLedger(tmp_path/"budget.sqlite3","spec")
    client=SimpleNamespace()  # any network access would fail the test
    metered=MeteredCompletions(client,ledger,"oversized")
    with pytest.raises(RuntimeError,match="preflight"):
        asyncio.run(metered.create(model=MODEL,max_tokens=4000,stream=False,messages=[{"content":"x"*128000}]))
    assert ledger.entries()==[]


def test_schema_failure_still_has_durable_provider_usage(tmp_path):
    ledger=CoverageBudgetLedger(tmp_path/"budget.sqlite3","spec")
    ledger.reserve("invalid-output")
    response=SimpleNamespace(usage=SimpleNamespace(prompt_tokens=30000,completion_tokens=1000),model_dump_json=lambda:'{"invalid":true}')
    ledger.record("invalid-output",response)
    assert ledger.entries()[0]["actual"]==99000
