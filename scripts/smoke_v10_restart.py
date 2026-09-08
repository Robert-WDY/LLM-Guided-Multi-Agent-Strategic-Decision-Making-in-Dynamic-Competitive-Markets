"""Exercise a real local HTTP process restart without provider calls."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/".tmp/v10-restart-smoke"
OUTPUT.mkdir(parents=True,exist_ok=True)
env={**os.environ,"MARKET_CONFIG_PATH":str(ROOT/"configs/market_v10_multi_objective.yaml"),"MARKET_SESSION_DB":str(OUTPUT/"sessions.sqlite3"),"MARKET_PERSISTENCE":"1","MARKET_CONTROLLER_TOKEN":"local-smoke-only","MARKET_LOCAL_DEV_UNAUTHENTICATED_CONTROLLER":"0"}
url="http://127.0.0.1:18019"
headers={"X-Controller-Token":"local-smoke-only"}


def start():
    process=subprocess.Popen([sys.executable,"-m","uvicorn","game_theory_agent.api:app","--host","127.0.0.1","--port","18019"],cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    for _ in range(80):
        if process.poll() is not None: raise RuntimeError("smoke server failed to start")
        try:
            if httpx.get(url+"/api/health",timeout=1).status_code==200: return process
        except httpx.HTTPError: pass
        time.sleep(.1)
    process.terminate(); process.wait(timeout=10)
    raise RuntimeError("smoke server startup timed out")


def checked(response):
    assert response.is_success, response.text
    response.raise_for_status()
    return response.json()


process=start()
try:
    created=checked(httpx.post(url+"/api/episodes",headers=headers,json={"episode_seed":120999,"max_rounds":5},timeout=15))
    state=created["state"]; eid=state["episode_id"]
    actions={c:{"action_id":f"{eid}:1:{c}","episode_id":eid,"agent_id":c,"round":1,"state_version":0,"price_cents":10000,"primary_supplier_id":"economy_supplier","backup_supplier_id":"resilient_supplier","primary_supplier_share_ppm":500000} for c in state["companies"]}
    request={"step_id":f"{eid}:1:0","joint_action":actions}
    settled=checked(httpx.post(url+f"/api/episodes/{eid}/steps",json=request,timeout=20))
    process.terminate(); process.wait(timeout=10)
    process=start()
    restored=checked(httpx.post(url+f"/api/v1/controller/episodes/{eid}/restore",headers=headers,timeout=15))
    retried=checked(httpx.post(url+f"/api/episodes/{eid}/steps",json=request,timeout=15))
    exported=checked(httpx.get(url+f"/api/v1/controller/episodes/{eid}/export",headers=headers,timeout=15))
    assert restored["state"]["state_hash"]==settled["state"]["state_hash"]
    assert retried==settled and len(exported["transitions"])==1
    assert all(o["primary_supplier_share_ppm"]==500000 for o in restored["state"]["supply_chain"]["last_procurement_outcomes"].values())
    evidence={"passed":True,"actual_process_restart":True,"idempotent_http_retry":True,"v10_procurement_preserved":True,"config_id":restored["manifest"]["config_id"] if "config_id" in restored["manifest"] else "market-v10-multi-objective","state_hash":restored["state"]["state_hash"],"provider_calls":0}
    (OUTPUT/"result.json").write_text(json.dumps(evidence,indent=2),encoding="utf-8")
    print(json.dumps(evidence))
finally:
    process.terminate(); process.wait(timeout=10)
