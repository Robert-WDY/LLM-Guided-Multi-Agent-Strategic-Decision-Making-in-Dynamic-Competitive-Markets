"""One paid hosted decision, cold-persisted research evidence; no retry."""
import hashlib
import json
from pathlib import Path
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/stage27-doubao-full"
BASE = "http://127.0.0.1:8010"

def request(path, body=None):
    req = urllib.request.Request(BASE+path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)

def write(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    budget = request("/api/health")["local_model_budget"]
    files = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"src").rglob("*.py")}
    write("preregistration.json", {"scope":"One hosted real-model decision and durable trace; not a profitability claim",
          "seed":270001,"maximum_calls":1,"maximum_additional_cny":0.42,"original_total_cny":10,
          "budget_before":budget,"source_hashes":files,"gates":["one paid call", "model succeeded", "trace persisted", "duplicate run does not call twice"]})
    episode = request("/api/episodes",{"episode_id":"stage27-doubao-full-270001","episode_seed":270001,"max_rounds":5,
        "information_mode":"public","communication_mode":"off","cooperation_mode":"combined_v1",
        "agent_configs":{"company_A":{"provider":"doubao","agent_type":"model","model":"doubao-seed-2-0-lite-260215","persona_name":"balanced_v1"}}})
    state=episode["state"]; prefix="/api/v1/controller/episodes/"+state["episode_id"]
    body={"run_id":"stage27-doubao-full-one","max_rounds":1,"expected_round":state["round"],"expected_state_version":state["state_version"],
          "expected_state_hash":state["state_hash"],"authorize_real_model":True,"maximum_model_calls":1}
    try:
        result=request(prefix+"/coordinator-run",body)
        write("response.json",result)
        view=request(prefix+"/research-view")
        write("research-view.json",view)
        repeated=request(prefix+"/coordinator-run",body)
        after=request("/api/health")["local_model_budget"]
        trace=next(t for t in view["detail"]["traces"] if t["company_id"]=="company_A")
        summary={"passed":trace["decision_status"]=="submitted" and result==repeated and after["new_calls"]-budget["new_calls"]==1,
                 "decision_status":trace["decision_status"],"model":trace.get("model_name"),"errors":trace.get("errors"),
                 "duplicate_equal":result==repeated,"budget_after":after,"resolution":trace.get("resolution_adjustments"),
                 "conclusion":"Tests real model transport, action settlement and research trace only; one sample cannot establish strategy superiority."}
        write("summary.json",summary);print(json.dumps(summary,ensure_ascii=False))
    except Exception as exc:
        write("failure.json",{"type":type(exc).__name__,"error":str(exc),"no_automatic_retry":True})
        raise

if __name__=="__main__": main()
