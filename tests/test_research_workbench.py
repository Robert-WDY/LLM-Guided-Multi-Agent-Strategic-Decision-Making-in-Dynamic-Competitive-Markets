from pathlib import Path
import pytest
from game_theory_agent.market import load_market_config
from game_theory_agent.research_workbench import Workbench,BatchRequest
CONFIG=load_market_config(Path(__file__).parents[1]/"configs/market_v14_complete.yaml")


def request(**kwargs):
    return BatchRequest(request_id="one",seeds=[260201,260202],rounds=5,
        variants=[dict(label="baseline"),dict(label="learning",modes={"consumers":"learning"})],**kwargs)


def test_durable_queue_idempotency_paired_results_and_export(tmp_path):
    q=Workbench(tmp_path,CONFIG,start_worker=False);j=q.submit(request())
    assert q.submit(request())["id"]==j["id"]
    q.execute(j["id"]);j=q.get(j["id"])
    assert j["status"]=="complete" and len(j["results"])==4
    assert j["statistics"]["paired_comparisons"][0]["paired_n"]==2
    export=q.export(j["id"]);assert len(export["cases"])==4 and not export["contains_credentials"]
    restarted=Workbench(tmp_path,CONFIG,start_worker=False)
    assert restarted.get(j["id"])["results"]==j["results"]
    with pytest.raises(ValueError):q.submit(request().model_copy(update={"rounds":10}))


def test_queue_cancel_and_recovery_and_auth_validation(tmp_path):
    q=Workbench(tmp_path,CONFIG,start_worker=False);j=q.submit(request())
    assert q.control(j["id"],"cancel")["status"]=="cancelled"
    q.control(j["id"],"resume");q.execute(j["id"]);assert q.get(j["id"])["status"]=="complete"
    with pytest.raises(ValueError,match="authorization"):
        q.submit(BatchRequest(request_id="paid",seeds=[1],variants=[dict(label="model",modes={"government":"model"})]))
    with pytest.raises(ValueError):
        q.submit(BatchRequest(request_id="bad",seeds=[1],variants=[dict(label="bad",overrides={"arbitrary":2})]))


def test_corrupt_free_case_rebuild_preserves_evidence_and_rejects_paid(tmp_path):
    q=Workbench(tmp_path,CONFIG,start_worker=False);j=q.submit(request())
    broken=tmp_path/j["id"]/"case-0-260201";broken.mkdir(parents=True)
    (broken/"checkpoint.json").write_bytes(b"\x00"*32)
    q.execute(j["id"]);assert q.get(j["id"])["status"]=="failed"
    q.control(j["id"],"rebuild-free")
    assert list((tmp_path/"recovery-evidence").rglob("checkpoint.json"))[0].read_bytes()==b"\x00"*32
    q.execute(j["id"]);assert q.get(j["id"])["status"]=="complete"
    import json
    with q.connect() as db:
        payload=q.get(j["id"])["payload"];payload["variants"][0]["modes"]={"government":"model"}
        db.execute("UPDATE jobs SET status='failed',payload=? WHERE id=?",(json.dumps(payload),j["id"]))
    with pytest.raises(ValueError,match="without any real model"):
        q.control(j["id"],"rebuild-free")
