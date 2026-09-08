import importlib.util
from pathlib import Path
import sqlite3
import hashlib
import json
import pytest

spec=importlib.util.spec_from_file_location('local_storage',Path(__file__).parents[1]/'scripts/local_storage.py')
storage=importlib.util.module_from_spec(spec); spec.loader.exec_module(storage)


def test_v14_backup_restores_queue_and_checks_config_before_mutation(tmp_path):
    from game_theory_agent.market import load_market_config
    from game_theory_agent.research_workbench import Workbench,BatchRequest
    config_path=Path(__file__).parents[1]/"configs/market_v14_stage24.yaml"
    data=tmp_path/"data";data.mkdir()
    with sqlite3.connect(data/"sessions.sqlite3") as db:
        db.execute("CREATE TABLE checkpoints(document TEXT,sha256 TEXT)")
    db.close()
    q=Workbench(data/"workbench",load_market_config(config_path),start_worker=False)
    job=q.submit(BatchRequest(request_id="backup",seeds=[280001],rounds=5,variants=[dict(label="rule")]))
    q.control(job["id"],"cancel");q.execute(job["id"])
    backup=tmp_path/"backup";m=storage.backup(data,backup,config_path=config_path)
    assert m["schema"]=="local-market-backup-v2" and m["workbench_files"]
    result=storage.restore(data,backup,config_path=config_path)
    assert not result["budget_restored"] and q.get(job["id"])["status"]=="cancelled"
    with pytest.raises(ValueError):
        storage.restore(data,backup,config_path=Path(__file__).parents[1]/"configs/market_v13_1_local.yaml")

def test_backup_restore_preserves_existing_copy_and_budget(tmp_path):
    data=tmp_path/'data';data.mkdir()
    source=data/'sessions.sqlite3'
    db=sqlite3.connect(source)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE checkpoints(document TEXT,sha256 TEXT)')
    text=json.dumps({'episode':'one'})
    db.execute('INSERT INTO checkpoints VALUES(?,?)',(text,hashlib.sha256(text.encode()).hexdigest()));db.commit()
    target=tmp_path/'backup'
    storage.backup(data,target) # Includes a live committed WAL transaction.
    db.execute('DELETE FROM checkpoints');db.commit();db.close()
    budget=data/'model-budget.sqlite3';budget.write_bytes(b'never roll back spending')
    result=storage.restore(data,target)
    assert storage.verify(source)==1
    assert budget.read_bytes()==b'never roll back spending'
    assert storage.verify(Path(result['previous_copy'])/'sessions.sqlite3')==0
    assert result['budget_restored'] is False
    (target/'sessions.sqlite3').write_bytes(b'corrupt')
    with pytest.raises(ValueError): storage.restore(data,target)
    assert storage.verify(source)==1
