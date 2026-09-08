import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from dotenv import load_dotenv
from game_theory_agent.market import MarketEnv,load_market_config
from game_theory_agent.market.actor_model import choose_option
from game_theory_agent.market.cooperation_personas import apply_cooperation_persona
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.local_budget import status

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/stage21-real"
def write(name,v): (OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding="utf-8")
async def main():
    OUT.mkdir(parents=True,exist_ok=False);load_dotenv(ROOT/".env",override=False)
    config=load_market_config(ROOT/"configs/market_v14_stage21.yaml");registry=PersonaRegistry.from_market_config(config)
    write("preregistration.json",{"config_hash":config.config_sha256,"maximum_calls":2,"maximum_additional_cny":0.082,"seed":211010,"personas":["cooperator","free_rider"],"budget_before":status(),"claim":"bounded-choice real-model objective adherence, not unrestricted dialogue or general personality validation"})
    rows=[]
    for persona in ("cooperator","free_rider"):
        env=MarketEnv(config);s=env.reset(episode_id=f"stage21-real-{persona}",episode_seed=211010,max_rounds=5,cooperation_mode="combined_v1")
        base=build_rule_action(config,s,"company_A")
        choices={p:apply_cooperation_persona(config,s,"company_A",base,p) for p in ("cooperator","free_rider")}
        options=[{"id":p,"shared_contribution_cents":a.shared_resilience_contribution_cents,"threshold_contribution_cents":a.threshold_project_contribution_cents,"fixed_spend_cents":a.fixed_spend_cents} for p,a in choices.items()]
        try:
            decision=await choose_option(actor_id="company_A",objective=registry.get(persona).objective,
                observation={"round":1,"rounds_remaining":5,"own_cash_cents":s.company("company_A").financial.cash_balance_cents,"public_history":"none yet"},options=options)
            actions={cid:build_rule_action(config,s,cid) for cid in s.company_ids};actions["company_A"]=choices[decision.option_id]
            after=env.step(f"{s.episode_id}:1:0",actions).state_after
            rows.append({"persona":persona,"choice":asdict(decision),"matched_declared_rule_direction":decision.option_id==persona,"settled_action":actions["company_A"].to_dict(),"state_after":after.to_dict()})
        except Exception as exc:
            rows.append({"persona":persona,"failure":type(exc).__name__,"message":str(exc),"retry":False})
        write("rows.json",rows)
    write("summary.json",{"passed":all(r.get("matched_declared_rule_direction") for r in rows),"rows":len(rows),"budget_after":status(),"conclusion":"Two constrained real-model choices and settlements; no statistical superiority or unrestricted persona claim."})
    print(json.dumps(json.loads((OUT/"summary.json").read_text(encoding="utf-8")),ensure_ascii=False))
if __name__=="__main__":asyncio.run(main())
