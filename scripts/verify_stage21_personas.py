"""Pre-registered synthetic 3x3 persona compositions on held seeds."""
import json
from pathlib import Path
from statistics import mean
from game_theory_agent.market import MarketEnv,load_market_config
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market.cooperation_personas import PERSONAS,apply_cooperation_persona

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/stage21-personas-r2"
def write(name,v): (OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding="utf-8")
def main():
    OUT.mkdir(parents=True,exist_ok=False);config=load_market_config(ROOT/"configs/market_v14_stage21.yaml")
    seeds=list(range(211001,211009))
    write("preregistration.json",{"config_hash":config.config_sha256,"seeds":seeds,"rounds":20,"focal_personas":PERSONAS,"resident_personas":PERSONAS,"comparison":"same seed, rotate focal A against three residents; no LLM quality claim"})
    rows=[]
    for resident in PERSONAS:
        for focal in PERSONAS:
            for seed in seeds:
                env=MarketEnv(config);s=env.reset(episode_id=f"stage21-common-{seed}",episode_seed=seed,max_rounds=20,cooperation_mode="combined_v1")
                contrib=0
                while not s.terminal:
                    actions={cid:apply_cooperation_persona(config,s,cid,build_rule_action(config,s,cid),focal if cid=="company_A" else resident) for cid in s.company_ids}
                    contrib+=(actions["company_A"].shared_resilience_contribution_cents or 0)+(actions["company_A"].threshold_project_contribution_cents or 0)
                    s=env.step(f"{s.episode_id}:{s.round}:{s.state_version}",actions).state_after
                rows.append({"resident":resident,"focal":focal,"seed":seed,"focal_profit_cents":s.company("company_A").financial.cumulative_profit_cents,"focal_contribution_cents":contrib,
                             "welfare_cents":s.welfare_accounting.cumulative_total_economic_welfare_cents,"surviving":s.strategic_market.active_company_count})
    write("rows.json",rows)
    groups=[{"resident":r,"focal":f,"n":8,"mean_profit_cents":mean(x["focal_profit_cents"] for x in rows if x["resident"]==r and x["focal"]==f),
             "mean_welfare_cents":mean(x["welfare_cents"] for x in rows if x["resident"]==r and x["focal"]==f)} for r in PERSONAS for f in PERSONAS]
    write("summary.json",{"passed":all(x["focal_contribution_cents"]==0 for x in rows if x["focal"]=="free_rider"),"episodes":len(rows),"rounds":len(rows)*20,"groups":groups,"conclusion":"Profiles have distinct public-contribution behavior. Payoffs depend on resident mix; contribution is not guaranteed individually optimal. Synthetic engineering and directional evidence only."})
    print(json.dumps({"episodes":len(rows),"rounds":len(rows)*20,"groups":groups},ensure_ascii=False))
if __name__=="__main__":main()
