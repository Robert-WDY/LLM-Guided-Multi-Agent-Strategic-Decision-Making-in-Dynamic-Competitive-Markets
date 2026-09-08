import asyncio
from types import SimpleNamespace
import pytest
from game_theory_agent import local_budget as b
from game_theory_agent.market.actor_model import choose_option


def test_second_provider_compact_and_responses_share_durable_ledger(tmp_path):
    path=tmp_path/"budget.sqlite3";b.initialize(path)
    class Completions:
        async def create(self,**kwargs):
            assert kwargs["model"]=="doubao-seed-2-0-lite-260215"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"option_id":"reserve","reason":"preserve budget"}'))],
                usage=SimpleNamespace(prompt_tokens=100,completion_tokens=20))
    result=asyncio.run(choose_option(actor_id="government",objective="test",observation={},options=[{"id":"reserve"}],
        provider=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),ledger=path,model_name="doubao-seed-2-0-lite-260215",max_output_tokens=128))
    assert result.provider=="doubao" and b.status(path)["new_calls"]==1
    class Responses:
        async def create(self,**kwargs):return SimpleNamespace(usage=SimpleNamespace(input_tokens=100,output_tokens=20),output_text="test")
    response=asyncio.run(b.GuardedResponses(Responses(),path).create(model="doubao-seed-2-0-lite-260215",input="test",max_output_tokens=4000,extra_body={"thinking":{"type":"disabled"}}))
    assert response.output_text=="test" and b.status(path)["new_calls"]==2
    with b._connect(path) as db:assert all(r[0]==132 for r in db.execute("SELECT actual FROM calls"))


def test_unpriced_tools_or_schema_never_reach_either_provider(tmp_path):
    path=tmp_path/"budget.sqlite3";b.initialize(path)
    for model in ("deepseek-v4-flash","doubao-seed-2-0-lite-260215"):
        with pytest.raises(b.LocalBudgetError):
            asyncio.run(b.GuardedCompletions(None,path,compact=True,model_name=model).create(model=model,
                stream=False,max_tokens=128,extra_body={"thinking":{"type":"disabled"}},
                messages=[{"role":"user","content":"test"}],n=2))
    assert b.status(path)["new_calls"]==0
