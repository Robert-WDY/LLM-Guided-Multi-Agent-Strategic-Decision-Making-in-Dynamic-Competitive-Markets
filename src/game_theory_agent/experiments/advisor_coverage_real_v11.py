"""New-seed released-advice pilot within the original shared 10-CNY allowance."""
import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from game_theory_agent.agents import AgentRuntime, EpisodeMemory
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.agents.prompt_builder import AgentPromptBuilder
from game_theory_agent.agents.context import DecisionContextBuilder
from game_theory_agent.belief import BeliefLedger
from game_theory_agent.opponent import OpponentModelLedger
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import MarketEnv, MarketState, load_market_config
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.model_clients import DeepSeekModelClient
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.strategic_reliability import PublicMarketRolloutAdvisor
from game_theory_agent.experiments.advisor_coverage_v11 import ROOT,CONFIG,observe,write
from game_theory_agent.experiments.v10_beta_real import BudgetLedger, MODEL, outcome
from game_theory_agent.experiments.final_market_v9_real_llm_smoke import _complete_observation

SPEC=ROOT/"experiment-specs/advisor-coverage-real-v11/PREREGISTRATION.json"
OUT=ROOT/"runs/advisor-coverage-real-v11"
SEEDS=(130901,130902)
PROFILES=("balanced_v1","public_service")
INPUT_LIMIT=128000
RESERVATION=INPUT_LIMIT*3+4000*9


class CoverageBudgetLedger(BudgetLedger):
    def reserve(self,key):
        with closing(sqlite3.connect(self.path,timeout=30)) as db,db:
            db.execute("BEGIN IMMEDIATE")
            count,total=db.execute("SELECT COUNT(*),COALESCE(SUM(reserved),0) FROM calls").fetchone()
            if count>=self.max_calls or total+RESERVATION>self.cap:
                raise RuntimeError("shared budget exhausted before request")
            db.execute("INSERT INTO calls (call_key,reserved,status) VALUES (?,?,'reserved')",(key,RESERVATION))

    def record(self,key,response):
        usage=getattr(response,"usage",None)
        inp,out=getattr(usage,"prompt_tokens",None),getattr(usage,"completion_tokens",None)
        cost=None if inp is None or out is None else int(inp)*3+int(out)*9
        with closing(sqlite3.connect(self.path)) as db,db:
            db.execute("UPDATE calls SET status='response_received',response=?,input_tokens=?,output_tokens=?,actual=? WHERE call_key=?",(response.model_dump_json(),inp,out,cost,key))
        if (inp is not None and inp>INPUT_LIMIT) or (out is not None and out>4000):
            raise RuntimeError("provider usage exceeded enforced bound")


class MeteredCompletions:
    def __init__(self,client,ledger,key):self.client,self.ledger,self.key=client,ledger,key

    async def create(self,**kwargs):
        if kwargs.get("model")!=MODEL or kwargs.get("max_tokens")!=4000 or kwargs.get("stream") is not False:
            raise RuntimeError("request differs from budget contract")
        if len(json.dumps(kwargs["messages"],ensure_ascii=False).encode("utf-8"))+1024>INPUT_LIMIT:
            raise RuntimeError("prompt exceeds preflight byte/token budget")
        self.ledger.reserve(self.key)
        response=await self.client.chat.completions.create(**kwargs)
        self.ledger.record(self.key,response)
        return response


def source_hash():
    files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"src").rglob("*.py")}
    return sha256_hash(files)


def view_for(config,state,observation,advice,condition):
    return _complete_observation(config,state,observation,advice=advice if condition=="advice" else None)


def prepare():
    if SPEC.exists(): raise RuntimeError("preregistration already exists")
    assert json.loads((ROOT/"runs/advisor-coverage-v11/summary.json").read_text())["passed"]
    config=load_market_config(CONFIG);registry=PersonaRegistry.from_market_config(config)
    advisor=PublicMarketRolloutAdvisor(config);cells=[]
    for seed in SEEDS:
        env=MarketEnv(config);state=env.reset(episode_id=f"coverage-real-{seed}",episode_seed=seed,max_rounds=20,cooperation_mode="combined_v1")
        beliefs=BeliefLedger(episode_id=state.episode_id,company_ids=state.company_ids)
        opponents=OpponentModelLedger(episode_id=state.episode_id,company_ids=state.company_ids)
        found=set()
        while not state.terminal and len(found)<len(PROFILES):
            if state.state_version>=7 and "company_A" in state.strategic_market.active_company_ids:
                observation,b,o=observe(config,state,beliefs,opponents)
                for persona in PROFILES:
                    if persona in found:continue
                    advice=advisor.advise(observation=observation,company_id="company_A",persona_profile=registry.get(persona),belief_state=b,opponent_model=o,horizon_rounds=3,scenario_count=5,advisor_mode="strategic_market_v9").model_dump(mode="json")
                    if advice["execution_disposition"]!="recommend":continue
                    cells.append({"seed":seed,"persona":persona,"state":state.to_dict(),"observation":observation,"advice":advice,"order":["baseline","advice"] if (seed+PROFILES.index(persona))%2==0 else ["advice","baseline"]})
                    found.add(persona)
            actions={c:build_rule_action(config,state,c) for c in state.company_ids}
            after=env.step(f"{state.episode_id}:{state.round}:{state.state_version}",actions).state_after
            beliefs.update_after_settlement(state,actions);opponents.update_after_settlement(state,after,actions);state=after
    prior=BudgetLedger(ROOT/"runs/v10-beta-real/budget.sqlite3",json.loads((ROOT/"runs/v10-beta-real/summary.json").read_text())["spec_hash"]).entries()
    prior_reserved=sum(e["reserved"] for e in prior)
    assert prior_reserved+len(cells)*2*RESERVATION<=10000000
    for cell in cells:
        state=MarketState.from_dict(cell["state"])
        builder=DecisionContextBuilder(persona_profile=registry.get(cell["persona"]),persona_registry=registry)
        cell["prompt_byte_bounds"]={}
        for condition in cell["order"]:
            context=builder.build(view_for(config,state,cell["observation"],cell["advice"],condition),"company_A",EpisodeMemory())
            messages=[{"role":"system","content":"你是受约束的市场经营规划器。只输出合法 JSON，不得调用工具或输出额外说明。"},{"role":"user","content":AgentPromptBuilder().build(context)}]
            bound=len(json.dumps(messages,ensure_ascii=False).encode("utf-8"))+1024
            cell["prompt_byte_bounds"][condition]=bound
            if bound>INPUT_LIMIT: raise RuntimeError(f"prompt preflight too large: {bound}; zero calls sent")
    spec={"source_hash":source_hash(),"config_hash":config.config_sha256,"seeds":SEEDS,"profiles":PROFILES,"selection":"First actually released recommendation after >=7 naturally simulated rounds per new seed/profile. Selected before LLM outcomes; conditional on release, not population-unbiased.","price_input_microcny":3,"price_output_microcny":9,"original_total_budget_microcny":10000000,"prior_reserved_microcny":prior_reserved,"remaining_cap_microcny":10000000-prior_reserved,"maximum_calls":len(cells)*2,"cells":cells}
    spec["hash"]=sha256_hash(spec);write(SPEC,spec)
    print("registered released pairs",len(cells),"reserved maximum CNY",len(cells)*2*RESERVATION/1e6,flush=True)


async def run():
    spec=json.loads(SPEC.read_text(encoding="utf-8"));identity=spec.pop("hash")
    assert sha256_hash(spec)==identity and source_hash()==spec["source_hash"]
    config=load_market_config(CONFIG);assert config.config_sha256==spec["config_hash"]
    registry=PersonaRegistry.from_market_config(config)
    load_dotenv(ROOT/".env",override=False)
    from openai import AsyncOpenAI
    provider=AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],base_url="https://api.deepseek.com",max_retries=0,timeout=90)
    ledger=CoverageBudgetLedger(OUT/"budget.sqlite3",identity,cap=spec["remaining_cap_microcny"],max_calls=spec["maximum_calls"])
    rows=json.loads((OUT/"rows.json").read_text(encoding="utf-8")) if (OUT/"rows.json").exists() else []
    completed={r["key"] for r in rows}
    try:
        for cell in spec["cells"]:
            state=MarketState.from_dict(cell["state"])
            for condition in cell["order"]:
                key=f"{cell['seed']}:{cell['persona']}:{condition}"
                if key in completed:continue
                row={"key":key,"seed":cell["seed"],"persona":cell["persona"],"condition":condition,"round":state.round,"success":False}
                if key in {e["call_key"] for e in ledger.entries()}:
                    row["error_code"]="interrupted_no_retry"
                else:
                    proxy=SimpleNamespace(chat=SimpleNamespace(completions=MeteredCompletions(provider,ledger,key)))
                    client=DeepSeekModelClient(model=MODEL,client=proxy,temperature=0,top_p=.1,max_schema_attempts=1,max_transport_retries=0)
                    runtime=AgentRuntime(agent_id=key,company_id="company_A",model_client=client,memory=EpisodeMemory(),persona_profile=registry.get(cell["persona"]),persona_registry=registry)
                    view=view_for(config,state,cell["observation"],cell["advice"],condition)
                    result=await runtime.decide(view,timeout_seconds=95)
                    row.update(success=result.success,error_code=result.error_code,input_tokens=result.input_tokens,output_tokens=result.output_tokens,raw_response=result.raw_response)
                    if result.success and result.decision:
                        requested=result.decision.requested_action.model_dump(mode="json")
                        resolution=resolve_action_request(config,state,"company_A",requested,source="coverage-real-v11")
                        row.update(requested_action=requested,final_action=resolution.action.to_dict(),outcome=outcome(config,registry,state,resolution.action,cell["persona"]))
                rows.append(row);write(OUT/"rows.json",rows)
                print(key,"success",row["success"],flush=True)
    finally:await provider.close()
    pairs=[]
    for cell in spec["cells"]:
        indexed={r["condition"]:r for r in rows if r["seed"]==cell["seed"] and r["persona"]==cell["persona"]}
        if len(indexed)!=2 or not all(r["success"] for r in indexed.values()):continue
        pairs.append({"seed":cell["seed"],"persona":cell["persona"],**{k:indexed["advice"]["outcome"][k]-indexed["baseline"]["outcome"][k] for k in ("utility_ppm","economic_welfare_cents","enterprise_value_cents")}})
    entries=ledger.entries()
    cost=sum(e["actual"] if e["actual"] is not None else e["reserved"] for e in entries)
    summary={"spec_hash":identity,"source_hash":spec["source_hash"],"calls":len(entries),"success":sum(r["success"] for r in rows),"exposed_complete_pairs":len(pairs),"pairs":pairs,"positive_utility":sum(p["utility_ppm"]>0 for p in pairs),"zero_utility":sum(p["utility_ppm"]==0 for p in pairs),"negative_utility":sum(p["utility_ppm"]<0 for p in pairs),"conservative_cost_cny":cost/1e6,"original_budget_conservative_reservation_cny":(spec["prior_reserved_microcny"]+sum(e["reserved"] for e in entries))/1e6,"unknown_usage_calls":sum(e["actual"] is None for e in entries),"boundary":"Small release-conditioned real-LLM pilot; closed-loop 20-round evidence is synthetic, not LLM causal proof."}
    write(OUT/"summary.json",summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");args=parser.parse_args()
    prepare() if args.prepare else asyncio.run(run())
