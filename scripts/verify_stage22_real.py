import asyncio,json
from dataclasses import asdict
from pathlib import Path
from dotenv import load_dotenv
from game_theory_agent.market import MarketEnv,load_market_config
from game_theory_agent.market.replay import EpisodeManifest,MarketTransition,verify_replay
from game_theory_agent.market.actor_model import choose_option
from game_theory_agent.market.supplier_strategy import decode
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.local_budget import status
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/stage22-real"
def write(name,v):(OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding="utf-8")
async def main():
    OUT.mkdir(parents=True,exist_ok=False);load_dotenv(ROOT/".env",override=False)
    config=load_market_config(ROOT/"configs/market_v14_stage22.yaml")
    write("preregistration.json",{"config":config.to_dict(),"config_hash":config.config_sha256,"maximum_calls":2,"maximum_additional_cny":0.082,"budget_before":status(),
        "seeds":[220301,220302],"selection":"one supplier's own-state choice after six natural rule rounds; all options evaluated after the model response"})
    rows=[]
    for seed in (220301,220302):
        env=MarketEnv(config);s=env.reset(episode_id=f"real-s22-{seed}",episode_seed=seed,max_rounds=20,cooperation_mode="combined_v1")
        manifest=EpisodeManifest.create(env,s,cooperation_mode="combined_v1");transitions=[]
        for _ in range(6):
            actions={cid:build_rule_action(config,s,cid) for cid in s.company_ids}
            result=env.step(f"{s.episode_id}:{s.round}:{s.state_version}",actions);transitions.append(MarketTransition.create(s,actions,result));s=result.state_after
        supplier=s.supply_chain.suppliers[seed%2];d=decode(supplier)
        options=[{"id":"balanced","effect":"default inventory and reliability budget"},{"id":"inventory","effect":"double target inventory fraction, capped at 2000 units"},{"id":"reliability","effect":"double reliability investment budget only if expected own margin repays cost"},{"id":"no_credit","effect":"decline new borrowing, preserve all other choices"}]
        try:
            decision=await choose_option(actor_id=supplier.supplier_id,objective="Maximize remaining own operating value, remain solvent, respect contracts; no access to rivals' private cash.",
                 observation={"round":s.round,"rounds_remaining":s.rounds_remaining,"own_cash_cents":supplier.account.cash_cents,"own_debt_cents":d["debt_cents"],
                 "own_inventory_orders":d["inventory_orders"],"own_sales":supplier.round_sales_orders,"own_orders":supplier.round_requested_orders,
                 "own_reliability_ppm":supplier.reliability_ppm,"own_unit_margin_cents":supplier.unit_price_cents-supplier.unit_cost_cents},options=options)
            actions={cid:build_rule_action(config,s,cid) for cid in s.company_ids};counterfactuals={}
            for option in options:
                shadow=MarketEnv(config);shadow.load_state(s)
                result=shadow.step(f"{s.episode_id}:{s.round}:{s.state_version}",actions,actor_choices={supplier.supplier_id:option["id"]})
                counterfactuals[option["id"]]={"profit_cents":result.state_after.supply_chain.supplier(supplier.supplier_id).round_profit_cents,"welfare_cents":result.state_after.welfare_accounting.round_total_economic_welfare_cents}
            result=env.step(f"{s.episode_id}:{s.round}:{s.state_version}",actions,actor_choices={supplier.supplier_id:decision.option_id})
            transitions.append(MarketTransition.create(s,actions,result))
            verify_replay(MarketEnv(config),manifest,[MarketTransition.from_dict(t.to_dict()) for t in transitions])
            row={"seed":seed,"choice":asdict(decision),"counterfactuals":counterfactuals,"trace":transitions[-1].to_dict(),"replay_passed":True}
        except Exception as exc:row={"seed":seed,"failure":type(exc).__name__,"message":str(exc),"retry":False}
        rows.append(row);write("rows.json",rows)
    summary={"passed":all(r.get("replay_passed") for r in rows),"budget_after":status(),"outcomes":[{"seed":r["seed"],"option":r.get("choice",{}).get("option_id"),"failure":r.get("failure")} for r in rows],
             "conclusion":"Two real supplier decisions in the full market, with identical-state alternatives and replay. Short-run regret is diagnostic, not a long-run optimum claim."}
    write("summary.json",summary);print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__":asyncio.run(main())
