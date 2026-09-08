import json
from pathlib import Path
from statistics import mean
from game_theory_agent.market import MarketEnv,MarketConfig,MarketState,load_market_config
from game_theory_agent.market.replay import EpisodeManifest,MarketTransition,verify_replay
from game_theory_agent.market.supplier_strategy import decode
from game_theory_agent.gameplay import build_rule_action

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/stage22-suppliers-r3"
def write(name,v):(OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding="utf-8")
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    base=load_market_config(ROOT/"configs/market_v14_stage22.yaml")
    seeds=list(range(220201,220209));options=("balanced","inventory","reliability","no_credit")
    write("preregistration.json",{"config":base.to_dict(),"config_hash":base.config_sha256,"seeds":seeds,"options":options,
        "regimes":["normal","low_capital"],"rounds":20,"gates":["every cash/material/liability invariant","cold resume rounds 5 and 12","full replay representative episodes"],"comparison":"same seed and regime; no cherry-picked passing seeds"})
    rows=[]
    for regime in ("normal","low_capital"):
        raw=base.to_dict()
        if regime=="low_capital":raw["supply_chain"]["transaction_accounting"]["initial_supplier_cash_cents"]=100000
        config=MarketConfig.from_mapping(raw)
        for option in options:
            for seed in seeds:
                env=MarketEnv(config);s=env.reset(episode_id=f"s22-{regime}-common-{seed}",episode_seed=seed,max_rounds=20,cooperation_mode="combined_v1")
                manifest=EpisodeManifest.create(env,s,cooperation_mode="combined_v1")
                transitions=[];metrics=dict(borrowed_cents=0,defaulted_cents=0,spoilage_orders=0,reliability_gain_ppm=0,contracts=0,max_inventory=0)
                while not s.terminal:
                    if s.state_version in (5,12):
                        env=MarketEnv(config);env.load_state(MarketState.from_dict(s.to_dict()))
                    actions={cid:build_rule_action(config,s,cid) for cid in s.company_ids}
                    result=env.step(f"{s.episode_id}:{s.round}:{s.state_version}",actions,actor_choices={sid:option for sid in s.supply_chain.supplier_ids})
                    transitions.append(MarketTransition.create(s,actions,result));s=result.state_after
                    for supplier in s.supply_chain.suppliers:
                        d=decode(supplier);a=d["audit"]
                        for k in ("borrowed_cents","defaulted_cents","spoilage_orders","reliability_gain_ppm"):metrics[k]+=a[k]
                        metrics["contracts"]+=sum(n["accepted"] for n in a["negotiations"])
                        metrics["max_inventory"]=max(metrics["max_inventory"],d["inventory_orders"])
                if seed==seeds[0]:
                    restored=[MarketTransition.from_dict(t.to_dict()) for t in transitions]
                    verify_replay(MarketEnv(config),manifest,restored)
                    write(f"{regime}-{option}-replay.json",{"manifest":manifest.to_dict(),"transitions":[t.to_dict() for t in transitions]})
                rows.append(dict(regime=regime,option=option,seed=seed,welfare_cents=s.welfare_accounting.cumulative_total_economic_welfare_cents,
                    supplier_profit_cents=sum(x.cumulative_profit_cents for x in s.supply_chain.suppliers),
                    bankrupt_suppliers=sum(decode(x)["bankrupt"] for x in s.supply_chain.suppliers),**metrics))
                write("rows.json",rows)
    summary={"passed":len(rows)==64,"episodes":len(rows),"rounds":len(rows)*20,"groups":[dict(regime=r,option=o,n=8,
             mean_welfare_cents=mean(x["welfare_cents"] for x in rows if x["regime"]==r and x["option"]==o),
             bankrupt_suppliers=sum(x["bankrupt_suppliers"] for x in rows if x["regime"]==r and x["option"]==o),
             borrowed_cents=sum(x["borrowed_cents"] for x in rows if x["regime"]==r and x["option"]==o)) for r in ("normal","low_capital") for o in options],
             "limits":"Synthetic scenarios and constrained policy grid; no proof of real-world profit or welfare optimality."}
    write("summary.json",summary);print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__":main()
