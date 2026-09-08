"""Frozen v10 paired pilot with a durable 10-CNY budget, no retries.

Every request is reserved on disk before network I/O. Failed or interrupted
requests retain their full reservation; restart cannot issue the same call.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from dataclasses import replace
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from dotenv import load_dotenv
from game_theory_agent.agents import AgentRuntime, EpisodeMemory
from game_theory_agent.agents.personas import PersonaRegistry, PersonaUtilityTracker
from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.belief import BeliefLedger
from game_theory_agent.opponent import OpponentModelLedger
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.experiments.final_market_advisor_v9_validation import _build_window, _responsive_first_round, _enterprise_value, FOCAL
from game_theory_agent.experiments.final_market_v9_real_llm_smoke import _complete_observation
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.protocols import sha256_hash, state_hash
from game_theory_agent.model_clients import DeepSeekModelClient
from game_theory_agent.strategic_reliability import PublicMarketRolloutAdvisor

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "experiment-specs/v10-beta-real/PREREGISTRATION_v2.json"
OUT = ROOT / "runs/v10-beta-real"
FREEZE = ROOT / "releases/v10-internal-beta-1/manifest.json"
MODEL = "deepseek-v4-flash"
INPUT_LIMIT, OUTPUT_LIMIT = 64000, 4000
INPUT_PRICE, OUTPUT_PRICE = 3, 9  # micro-CNY/token; official peak, cache miss
RESERVATION = INPUT_LIMIT * INPUT_PRICE + OUTPUT_LIMIT * OUTPUT_PRICE
CAP = 10_000_000
MAX_CALLS = 32
SEEDS = (120801, 120802)
WINDOWS = ("normal_supply", "economy_outage")
PROFILES = ("balanced_v1", "stakeholder_balanced", "public_service", "resilience_steward")
CONDITIONS = ("baseline", "advice")


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)


class BudgetLedger:
    def __init__(self, path: Path, spec_hash: str, cap=CAP, max_calls=MAX_CALLS):
        self.path, self.cap, self.max_calls = path, cap, max_calls
        path.parent.mkdir(parents=True,exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY, spec_hash TEXT)")
            db.execute("INSERT OR IGNORE INTO identity VALUES (1,?)",(spec_hash,))
            if db.execute("SELECT spec_hash FROM identity WHERE id=1").fetchone()[0] != spec_hash:
                raise RuntimeError("budget ledger belongs to another preregistration")
            db.execute("CREATE TABLE IF NOT EXISTS calls (call_key TEXT PRIMARY KEY, reserved INTEGER NOT NULL, status TEXT NOT NULL, response TEXT, input_tokens INTEGER, output_tokens INTEGER, actual INTEGER)")

    def reserve(self, key):
        with closing(sqlite3.connect(self.path,timeout=30)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            count,total=db.execute("SELECT COUNT(*),COALESCE(SUM(reserved),0) FROM calls").fetchone()
            if count >= self.max_calls or total+RESERVATION > self.cap:
                raise RuntimeError("10-CNY/call budget exhausted before request")
            db.execute("INSERT INTO calls (call_key,reserved,status) VALUES (?,?,'reserved')",(key,RESERVATION))

    def record(self, key, response):
        usage=getattr(response,"usage",None)
        inp,out=getattr(usage,"prompt_tokens",None),getattr(usage,"completion_tokens",None)
        cost=None if inp is None or out is None else int(inp)*INPUT_PRICE+int(out)*OUTPUT_PRICE
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE calls SET status='response_received',response=?,input_tokens=?,output_tokens=?,actual=? WHERE call_key=?",(response.model_dump_json(),inp,out,cost,key))
        if inp is not None and (inp>INPUT_LIMIT or out>OUTPUT_LIMIT):
            raise RuntimeError("provider usage exceeds enforced request bound")

    def entries(self):
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory=sqlite3.Row
            return [dict(row) for row in db.execute("SELECT call_key,reserved,status,input_tokens,output_tokens,actual FROM calls ORDER BY rowid")]


class MeteredCompletions:
    def __init__(self, client, ledger, key): self.client,self.ledger,self.key=client,ledger,key
    async def create(self, **kwargs):
        if kwargs.get("model") != MODEL or kwargs.get("max_tokens") != OUTPUT_LIMIT or kwargs.get("stream") is not False:
            raise RuntimeError("request differs from frozen budget contract")
        # UTF-8 bytes bound text-token count conservatively; reserve room for
        # chat framing. Refuse oversized prompts before reserving or sending.
        prompt_bytes=len(json.dumps(kwargs["messages"],ensure_ascii=False).encode("utf-8"))+1024
        if prompt_bytes > INPUT_LIMIT:
            raise RuntimeError("prompt exceeds preflight byte/token budget")
        self.ledger.reserve(self.key)
        response=await self.client.chat.completions.create(**kwargs)
        self.ledger.record(self.key,response)
        return response


def v10_window(config, seed, window):
    env=MarketEnv(config)
    state=env.reset(episode_id=f"v10-pilot-{seed}-{window}",episode_seed=seed,max_rounds=20,cooperation_mode="combined_v1")
    beliefs=BeliefLedger(episode_id=state.episode_id,company_ids=state.company_ids)
    opponents=OpponentModelLedger(episode_id=state.episode_id,company_ids=state.company_ids)
    for _ in range(4):
        actions={c:replace(build_rule_action(config,state,c),primary_supplier_id="economy_supplier",backup_supplier_id="resilient_supplier",primary_supplier_share_ppm=500000) for c in state.company_ids}
        after=env.step(f"{state.episode_id}:{state.round}:{state.state_version}",actions).state_after
        beliefs.update_after_settlement(state,actions)
        opponents.update_after_settlement(state,after,actions)
        state=after
    if window=="economy_outage":
        supply=replace(state.supply_chain,suppliers=tuple(replace(s,disrupted=True,available_capacity_orders=s.base_capacity_orders*3//10) if s.supplier_id=="economy_supplier" else s for s in state.supply_chain.suppliers))
        state=replace(state,supply_chain=supply,state_hash="")
        state=replace(state,state_hash=state_hash(state.to_dict()))
        env.load_state(state)
    belief,belief_hash=beliefs.company_view(observer_company_id=FOCAL,round_number=state.round,state_version=state.state_version)
    opponent,opponent_hash=opponents.company_view(observer_company_id=FOCAL,round_number=state.round,state_version=state.state_version)
    observation=ObservationBuilder().build(state,FOCAL,"public",belief_state=belief.model_dump(mode="json"),belief_hash=belief_hash,belief_schema_version=belief.belief_schema_version)
    observation.update(opponent_model_state=opponent.model_dump(mode="json"),opponent_model_hash=opponent_hash)
    return state,belief,opponent,observation


def build(config, registry, seed, window, persona):
    state,belief,opponent,observation=v10_window(config,seed,window)
    advice=PublicMarketRolloutAdvisor(config).advise(observation=observation,company_id=FOCAL,persona_profile=registry.get(persona),belief_state=belief,opponent_model=opponent,horizon_rounds=3,scenario_count=5,advisor_mode="strategic_market_v9").model_dump(mode="json")
    return state,observation,advice


def prepare():
    if SPEC.exists(): raise RuntimeError("preregistration already exists")
    config=load_market_config(ROOT/"configs/market_v10_multi_objective.yaml")
    registry=PersonaRegistry.from_market_config(config)
    cells=[]
    for seed in SEEDS:
        for window in WINDOWS:
            for index,persona in enumerate(PROFILES):
                state,observation,advice=build(config,registry,seed,window,persona)
                eligible=FOCAL in state.strategic_market.active_company_ids
                observations={c:_complete_observation(config,state,observation,advice=advice if c=="advice" else None) for c in CONDITIONS}
                exposed=bool(observations["advice"].get("game_theory_advice"))
                order=list(CONDITIONS if (seed+index+WINDOWS.index(window))%2==0 else reversed(CONDITIONS)) if exposed else ["baseline"]
                cells.append({"seed":seed,"window":window,"persona":persona,"eligible":eligible,"state_hash":state.state_hash,"advice_hash":advice["advice_hash"],"disposition":advice["execution_disposition"],"exposed":exposed,"observation_hashes":{c:o["observation_hash"] for c,o in observations.items()},"order":order})
                print(f"prepare {seed}/{window}/{persona}: {advice['execution_disposition']}",flush=True)
    spec={"version":"v10-beta-real-v1","registered_at":datetime.now(UTC).isoformat(),"model":MODEL,"config_hash":config.config_sha256,"authorization_cny":10,"max_calls":MAX_CALLS,"maximum_reserved_microunits":MAX_CALLS*RESERVATION,"price_snapshot":{"url":"https://api-docs.deepseek.com/zh-cn/quick_start/pricing/","verified_date":"2026-09-07","input_cny_per_million":3,"output_cny_per_million":9,"basis":"peak cache-miss; deliberately do not discount cached/off-peak usage"},"design":"Same frozen v10 state/model/persona, baseline vs public v9 advice. 3-round responsive rule continuation. Four objective profiles, two engineered states, two new seeds. If advice abstains, run the baseline only to validate persona/market execution; do not spend on an identical unexposed treatment. Effect estimation requires exposed complete pairs. Legacy v6 mature_shortage was excluded before registration because the focal company exited under v10 in all first 10 inspected seeds. No LLM outcomes used to select cells.","primary_metrics":["paired_discounted_persona_utility","paired_total_economic_welfare","paired_focal_enterprise_value","worst_pair","actual_exposure","failed_and_interrupted_calls"],"retry_policy":"no schema repair, no transport retries, no automatic retry after interruption; full reservation retained on all failures","cells":cells}
    spec["hash"]=sha256_hash(spec)
    write(SPEC,spec)
    return spec


def outcome(config, registry, state, action, persona):
    env=MarketEnv(config); env.load_state(state)
    tracker=PersonaUtilityTracker(registry.evaluator(registry.get(persona)))
    initial=state.welfare_accounting.cumulative_total_economic_welfare_cents
    for offset in range(min(3,state.rounds_remaining)):
        before=env.get_state()
        actions=_responsive_first_round(config,before,action) if offset==0 else {c:build_rule_action(config,before,c) for c in before.company_ids}
        after=env.step(f"{before.episode_id}:{before.round}:{before.state_version}",actions).state_after
        tracker.record(before,after,FOCAL)
    return {"utility_ppm":tracker.cumulative_discounted_utility_ppm,"economic_welfare_cents":after.welfare_accounting.cumulative_total_economic_welfare_cents-initial,"enterprise_value_cents":_enterprise_value(after,FOCAL,config),"final_state_hash":after.state_hash}


def verify_freeze():
    release=json.loads(FREEZE.read_text(encoding="utf-8"))
    for name,expected in release["files"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f"frozen source changed: {name}")
    return release["source_sha256"]


async def run():
    source_hash=verify_freeze()
    spec=json.loads(SPEC.read_text(encoding="utf-8")); identity=spec.pop("hash")
    assert sha256_hash(spec)==identity
    config=load_market_config(ROOT/"configs/market_v10_multi_objective.yaml")
    assert config.config_sha256==spec["config_hash"]
    load_dotenv(ROOT/".env",override=False)
    if not os.getenv("DEEPSEEK_API_KEY"): raise RuntimeError("DEEPSEEK_API_KEY missing; zero requests sent")
    from openai import AsyncOpenAI
    provider=AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],base_url="https://api.deepseek.com",timeout=90,max_retries=0)
    ledger=BudgetLedger(OUT/"budget.sqlite3",identity)
    registry=PersonaRegistry.from_market_config(config)
    rows=json.loads((OUT/"rows.json").read_text(encoding="utf-8")) if (OUT/"rows.json").exists() else []
    completed={r["key"] for r in rows}
    try:
        for cell in spec["cells"]:
            if not cell["eligible"]: continue
            state,observation,advice=build(config,registry,cell["seed"],cell["window"],cell["persona"])
            assert state.state_hash==cell["state_hash"] and advice["advice_hash"]==cell["advice_hash"]
            for condition in cell["order"]:
                key=f"{cell['seed']}:{cell['window']}:{cell['persona']}:{condition}"
                if key in completed: continue
                row={"key":key,"seed":cell["seed"],"window":cell["window"],"persona":cell["persona"],"condition":condition,"exposed":condition=="advice" and cell["exposed"],"success":False}
                if key in {entry["call_key"] for entry in ledger.entries()}:
                    row["error_code"]="interrupted_no_automatic_retry"
                else:
                    view=_complete_observation(config,state,observation,advice=advice if condition=="advice" else None)
                    assert view["observation_hash"]==cell["observation_hashes"][condition]
                    proxy=SimpleNamespace(chat=SimpleNamespace(completions=MeteredCompletions(provider,ledger,key)))
                    client=DeepSeekModelClient(model=MODEL,client=proxy,max_schema_attempts=1,max_transport_retries=0,temperature=0,top_p=0.1)
                    runtime=AgentRuntime(agent_id=key,company_id=FOCAL,model_client=client,memory=EpisodeMemory(),persona_profile=registry.get(cell["persona"]),persona_registry=registry)
                    result=await runtime.decide(view,timeout_seconds=95)
                    row.update({"success":result.success,"error_code":result.error_code,"model_name":result.model_name,"input_tokens":result.input_tokens,"output_tokens":result.output_tokens,"provider_audit":result.provider_audit.model_dump(mode="json") if result.provider_audit else None,"raw_response":result.raw_response})
                    if result.success and result.decision:
                        requested=result.decision.requested_action.model_dump(mode="json")
                        resolution=resolve_action_request(config,state,FOCAL,requested,source="v10-beta-real")
                        row["requested_action"]=requested
                        row["final_action"]=resolution.action.to_dict()
                        row["outcome"]=outcome(config,registry,state,resolution.action,cell["persona"])
                rows.append(row); write(OUT/"rows.json",rows)
                print(f"{len(rows)}/{MAX_CALLS} {key} success={row['success']} exposed={row['exposed']}",flush=True)
    finally:
        await provider.close()
    pairs=[]
    for cell in spec["cells"]:
        selected={r["condition"]:r for r in rows if all(r[k]==cell[k] for k in ("seed","window","persona"))}
        if len(selected)!=2 or not all(r["success"] for r in selected.values()): continue
        a,b=selected["baseline"],selected["advice"]
        pairs.append({"seed":cell["seed"],"window":cell["window"],"persona":cell["persona"],"exposed":b["exposed"],**{k:b["outcome"][k]-a["outcome"][k] for k in ("utility_ppm","economic_welfare_cents","enterprise_value_cents")},"action_changed":a["requested_action"]!=b["requested_action"]})
    entries=ledger.entries()
    summary={"source_sha256":source_hash,"spec_hash":identity,"logical_results":len(rows),"provider_attempts":len(entries),"successful_results":sum(r["success"] for r in rows),"complete_pairs":len(pairs),"exposed_pairs":sum(p["exposed"] for p in pairs),"pairs":pairs,"budget_authorized_cny":10,"reservation_cny":sum(e["reserved"] for e in entries)/1e6,"usage_estimate_cny":sum(e["actual"] or 0 for e in entries)/1e6,"unknown_usage_calls":sum(e["actual"] is None for e in entries),"conservative_cost_bound_cny":sum(e["actual"] if e["actual"] is not None else e["reserved"] for e in entries)/1e6,"estimate_basis":"official peak/cache-miss pricing; not provider invoice","utility_positive":sum(p["utility_ppm"]>0 for p in pairs),"utility_equal":sum(p["utility_ppm"]==0 for p in pairs),"utility_negative":sum(p["utility_ppm"]<0 for p in pairs),"worst_utility_delta_ppm":min((p["utility_ppm"] for p in pairs),default=None),"mean_welfare_delta_cents":mean(p["economic_welfare_cents"] for p in pairs) if pairs else None,"claim_boundary":"Small directional pilot, engineered states, one provider, not a significance or natural multi-round effectiveness claim."}
    write(OUT/"summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="pairs"},ensure_ascii=False),flush=True)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--prepare",action="store_true"); args=parser.parse_args()
    prepare() if args.prepare else asyncio.run(run())
