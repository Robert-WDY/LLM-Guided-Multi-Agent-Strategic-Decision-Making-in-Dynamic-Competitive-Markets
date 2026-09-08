"""Verify installed Windows manager against the actual local production services."""
from pathlib import Path
import hashlib
import json
import subprocess
import time
import httpx

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/local-release-acceptance-r2'
DATA=ROOT/'.local-state'
BASE='http://127.0.0.1:8010'

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    def manager(action, *args, success=True):
        # Windows background children can inherit PIPE handles and prevent EOF.
        # Real files let the manager exit independently of its long-lived services.
        with (OUT/'manager.log').open('ab') as log:
            result=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/local_market.ps1'),'-Action',action,*map(str,args)],cwd=ROOT,stdout=log,stderr=log,timeout=90)
        assert (result.returncode==0)==success, (action,'see manager.log')
    def req(method,path,**kwargs):
        result=httpx.request(method,BASE+path,timeout=120,**kwargs)
        assert result.is_success,(path,result.status_code,result.text[:500])
        return result.json()
    def run(payload):
        s=payload['state']
        body=dict(run_id=f"release:{s['round']}",expected_round=s['round'],expected_state_version=s['state_version'],expected_state_hash=s['state_hash'],max_rounds=1)
        return req('POST',path+'/coordinator-run',json=body),body
    manager('Start'); manager('Status')
    health=req('GET','/api/health')
    assert health['config_id']=='market-v13-1-autonomous-local'
    assert health['local_model_budget']['new_calls']==0
    assert httpx.get('http://localhost:3210/').status_code==200
    ledger=DATA/'model-budget.sqlite3'
    cash_hash=hashlib.sha256(ledger.read_bytes()).hexdigest()
    # A second manager must fail on occupied ports, leaving the real services alive.
    manager('Start','-DataDirectory',OUT/'collision',success=False)
    assert req('GET','/api/health')['status']=='ok'
    # Deliberately wrong ownership metadata must also refuse to kill a process.
    records=json.loads((DATA/'processes.json').read_text(encoding='utf-8-sig'))
    tampered=OUT/'wrong-owner';tampered.mkdir()
    records[0]['commandLine']='not our process'
    (tampered/'processes.json').write_text(json.dumps(records),encoding='utf-8')
    manager('Stop','-DataDirectory',tampered,success=False)
    assert req('GET','/api/health')['status']=='ok'
    episode='local-release-verification-r2'
    path='/api/v1/controller/episodes/'+episode
    payload=req('POST','/api/episodes',json=dict(episode_id=episode,episode_seed=210101,max_rounds=5,information_mode='public',communication_mode='public_private',cooperation_mode='combined_v1'))
    payload,_=run(payload);payload,body=run(payload)
    at_backup=payload
    req('POST',path+'/save',json={'ui_state':{'entryMode':'research','config':{'rounds':5},'agents':[]}})
    manager('Backup','-BackupPath',OUT/'backup')
    payload,_=run(payload)
    manager('Stop')
    manager('Restore','-BackupPath',OUT/'backup')
    assert hashlib.sha256(ledger.read_bytes()).hexdigest()==cash_hash
    manager('Start')
    restored=req('POST',path+'/restore',json={})
    assert restored['state']==at_backup['state']
    assert req('POST',path+'/coordinator-run',json=body)==at_backup
    payload=restored
    while not payload['state']['terminal']:payload,_=run(payload)
    export=req('GET',path+'/export')
    assert len(export['transitions'])==5
    manager('Status')
    assert req('GET','/api/health')['local_model_budget']['new_calls']==0
    result=dict(passed=True,model_calls=0,production_http=True,windows_powershell=True,port_collision_refused=True,wrong_owner_not_killed=True,backup_restore=True,budget_not_rolled_back=True,repeat_idempotent=True,terminal_hash=payload['state']['state_hash'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
