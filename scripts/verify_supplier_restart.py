"""Cold HTTP restart smoke on isolated ports/storage, with zero model calls."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx

from game_theory_agent.market import MarketEnv, MarketState, load_market_config
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.decisioning import resolve_action_request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/supplier-autonomy-v11/http-restart-final"
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/"sessions.sqlite3").exists(): raise RuntimeError("use a fresh smoke directory")
with socket.socket() as socket_handle:
    socket_handle.bind(("127.0.0.1",0));port=socket_handle.getsockname()[1]
base=f"http://127.0.0.1:{port}"
config=load_market_config(ROOT/"configs/market_v11_supplier.yaml")
environment=os.environ.copy()
environment.update(MARKET_CONFIG_PATH=str(ROOT/"configs/market_v11_supplier.yaml"),MARKET_SESSION_DB=str(OUT/"sessions.sqlite3"),MARKET_API_HOST="127.0.0.1",MARKET_API_PORT=str(port),MARKET_AGENT_GATEWAY_ENABLED="0",MARKET_PERSISTENCE="1",MARKET_LOCAL_DEV_UNAUTHENTICATED_CONTROLLER="1",PYTHONPATH=os.pathsep.join((str(ROOT/"src"),str(ROOT/".venv/Lib/site-packages"))))
process=None


def start():
    global process
    # Base interpreter avoids the Windows venv redirector leaving an orphan.
    process=subprocess.Popen([sys._base_executable,"-m","game_theory_agent.api"],cwd=ROOT,env=environment,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    for _ in range(100):
        if process.poll() is not None:raise RuntimeError("smoke server exited")
        try:
            if httpx.get(base+"/api/health").status_code==200:return
        except httpx.TransportError:pass
        time.sleep(.1)
    raise RuntimeError("smoke server did not become ready")


def stop():
    if process is not None and process.poll() is None:
        process.terminate();process.wait(timeout=10)


def request(method,path,**kwargs):
    response=httpx.request(method,base+path,timeout=30,**kwargs)
    response.raise_for_status();return response.json()


try:
    start()
    created=request("POST","/api/episodes",json={"episode_id":"supplier-http-restart","episode_seed":140051,"max_rounds":10})
    state=MarketState.from_dict(created["state"])
    for _ in range(3):
        joint={i:build_rule_action(config,state,i).to_dict() for i in state.company_ids}
        body={"step_id":f"{state.episode_id}:{state.round}:{state.state_version}","joint_action":joint}
        settled=request("POST",f"/api/episodes/{state.episode_id}/steps",json=body)
        state=MarketState.from_dict(settled["state"])
    stop();start()
    restored=request("POST",f"/api/v1/controller/episodes/{state.episode_id}/restore")
    assert restored["state"]==state.to_dict()
    repeated=request("POST",f"/api/episodes/{state.episode_id}/steps",json=body)
    assert repeated==settled
    expected=MarketEnv(config);expected.load_state(state)
    actions={i:build_rule_action(config,state,i) for i in state.company_ids}
    step_id=f"{state.episode_id}:{state.round}:{state.state_version}"
    # HTTP goes through the intent resolver before the authoritative engine.
    # Use that same path for the uninterrupted reference continuation.
    resolved={i:resolve_action_request(config,state,i,a.to_dict(),source="control-api",action_id=a.action_id).action for i,a in actions.items()}
    next_expected=expected.step(step_id,resolved).state_after
    next_actual=request("POST",f"/api/episodes/{state.episode_id}/steps",json={"step_id":step_id,"joint_action":{i:a.to_dict() for i,a in actions.items()}})
    assert next_actual["state"]==next_expected.to_dict()
    result={"passed":True,"cold_http_restart":True,"quote_and_settlement_price_preserved":True,"repeat_step_idempotent":True,"next_settlement_matches":True,"model_calls":0,"restored_hash":state.state_hash,"next_hash":next_expected.state_hash}
    (OUT/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))
finally:stop()
