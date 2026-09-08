"""Pre-registered supplier pricing ablations; engineering, not optimality proof."""
import argparse
import json
from collections import Counter
from dataclasses import replace
from statistics import median

from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition, verify_replay
from game_theory_agent.market.supplier_policy import price_bounds
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.persistence import CODEC
from game_theory_agent.experiments.advisor_coverage_v11 import ROOT, write
from game_theory_agent.experiments.v10_beta_validation import invariant, _information_audit

CONFIG=ROOT/"configs/market_v11_supplier.yaml"
OUT=ROOT/"runs/supplier-autonomy-v11"
SPEC=ROOT/"experiment-specs/supplier-autonomy-v11/PREREGISTRATION.json"
SEEDS=tuple(range(140101,140109))
VARIANTS=("base","fast_price_step","capacity_half","low_reliability")
BUYERS=("fixed_economy","cheapest_quote","diverse")


def configuration(variant,enabled):
    data=load_market_config(CONFIG).to_dict()
    supply=data["supply_chain"]
    supply["autonomous_pricing"]["enabled"]=enabled
    if variant=="fast_price_step":supply["autonomous_pricing"]["step_ppm"]=200000
    if variant=="capacity_half":
        for supplier in supply["suppliers"].values():supplier["base_capacity_orders"]//=2
    if variant=="low_reliability":
        for supplier in supply["suppliers"].values():supplier["reliability_ppm"]=500000
    return MarketConfig.from_mapping(data)


def actions_for(config,state,buyer):
    cheapest=min(state.supply_chain.suppliers,key=lambda s:(s.unit_price_cents,s.supplier_id)).supplier_id
    actions={i:build_rule_action(config,state,i) for i in state.company_ids}
    return {i:replace(a,primary_supplier_id=cheapest if buyer=="cheapest_quote" else "economy_supplier",
                      backup_supplier_id="resilient_supplier" if buyer=="diverse" else None,
                      primary_supplier_share_ppm=500000 if buyer=="diverse" else 1000000)
            if i in state.strategic_market.active_company_ids else a for i,a in actions.items()}


def episode(config,seed,buyer,enabled,variant):
    env=MarketEnv(config);state=env.reset(episode_id=f"supplier-{seed}",episode_seed=seed,max_rounds=20,cooperation_mode="combined_v1")
    manifest=EpisodeManifest.create(env,state,experiment_id="supplier-autonomy-v11")
    transitions=[];trace=[];gates=True;missed=0;changes=0
    prices={s.supplier_id:[] for s in state.supply_chain.suppliers}
    while not state.terminal:
        actions=actions_for(config,state,buyer)
        leaks,consistent=_information_audit(state);gates &= leaks==0 and consistent
        result=env.step(f"{state.episode_id}:{state.round}:{state.state_version}",actions);after=result.state_after
        transitions.append(MarketTransition.create(state,actions,result))
        gates &= all(invariant(after).values())
        for supplier in after.supply_chain.suppliers:
            prices[supplier.supplier_id].append(supplier.unit_price_cents)
            changes += supplier.unit_price_cents!=state.supply_chain.supplier(supplier.supplier_id).unit_price_cents
            if enabled:
                lo,hi=price_bounds(supplier,config.mapping("supply_chain"));gates &= lo<=supplier.unit_price_cents<=hi
        trace.append({s.supplier_id:s.disrupted for s in after.supply_chain.suppliers})
        missed+=sum(o.unfulfilled_orders for o in after.supply_chain.last_procurement_outcomes)
        state=after
        if state.state_version==8:
            restored=CODEC.decode(CODEC.encode(state))
            replacement=MarketEnv(config);replacement.load_state(restored);env=replacement
    gates &= verify_replay(MarketEnv(config),manifest,tuple(transitions))[-1].state_hash==state.state_hash
    return {"variant":variant,"seed":seed,"buyer":buyer,"autonomous":enabled,"config_hash":config.config_sha256,"gates":gates,"quote_changes":changes,"quote_paths":prices,"exogenous_disruptions":trace,"unfulfilled_orders":missed,"welfare_cents":state.welfare_accounting.cumulative_total_economic_welfare_cents,"supplier_profit_cents":{s.supplier_id:s.cumulative_profit_cents for s in state.supply_chain.suppliers},"exits":len(state.company_ids)-len(state.strategic_market.active_company_ids),"state_hash":state.state_hash}


def prepare():
    if SPEC.exists():raise RuntimeError("preregistration exists")
    spec={"seeds":SEEDS,"variants":VARIANTS,"buyers":BUYERS,"conditions":[False,True],"rounds":20,"model_calls":0,"base_config_hash":load_market_config(CONFIG).config_sha256,"design":"8 seeds base plus first4 seeds per3 stress variants; paired exogenous disruptions, rule buyer operating actions, either fixed economy/all cheapest published/diverse50:50 procurement. No advisor/LLM. Preserve quote/profit/welfare/shortage paths; no requirement that autonomous prices increase welfare or all supplier profits.","acceptance":["all_engineering_and_replay","same_paired_supplier_shocks","autonomous_quote_changes_present","disabled_quotes_fixed"]}
    spec["hash"]=sha256_hash(spec);write(SPEC,spec)


def run():
    spec=json.loads(SPEC.read_text(encoding="utf-8"));identity=spec.pop("hash")
    assert sha256_hash(spec)==identity and load_market_config(CONFIG).config_sha256==spec["base_config_hash"]
    if (OUT/"summary.json").exists():raise RuntimeError("results exist")
    rows=[];pairs=[]
    for variant in VARIANTS:
        for seed in SEEDS if variant=="base" else SEEDS[:4]:
            for buyer in BUYERS:
                pair=[episode(configuration(variant,enabled),seed,buyer,enabled,variant) for enabled in (False,True)]
                rows.extend(pair);baseline,autonomous=pair
                pairs.append({"variant":variant,"seed":seed,"buyer":buyer,"same_shocks":baseline["exogenous_disruptions"]==autonomous["exogenous_disruptions"],**{k:autonomous[k]-baseline[k] for k in ("welfare_cents","unfulfilled_orders","exits")},"supplier_profit_delta_cents":{s:autonomous["supplier_profit_cents"][s]-baseline["supplier_profit_cents"][s] for s in baseline["supplier_profit_cents"]}})
                write(OUT/"rows.json",rows)
            print(variant,seed,"finished",flush=True)
    gates={"engineering_and_replay":all(r["gates"] for r in rows),"paired_supplier_shocks":all(p["same_shocks"] for p in pairs),"autonomous_actions":all(r["quote_changes"]>0 for r in rows if r["autonomous"]),"fixed_control":all(r["quote_changes"]==0 for r in rows if not r["autonomous"])}
    groups={}
    for buyer in BUYERS:
        selected=[p for p in pairs if p["buyer"]==buyer and p["variant"]=="base"]
        groups[buyer]={"pairs":len(selected),"welfare_positive":sum(p["welfare_cents"]>0 for p in selected),"median_welfare_delta_cents":median(p["welfare_cents"] for p in selected),"worst_welfare_delta_cents":min(p["welfare_cents"] for p in selected),"median_unfulfilled_delta":median(p["unfulfilled_orders"] for p in selected),"supplier_median_profit_delta_cents":{s:median(p["supplier_profit_delta_cents"][s] for p in selected) for s in selected[0]["supplier_profit_delta_cents"]}}
    summary={"spec_hash":identity,"episodes":len(rows),"rounds":20*len(rows),"model_calls":0,"gates":gates,"passed":all(gates.values()),"base_groups":groups,"pairs":pairs}
    write(OUT/"summary.json",summary);print(json.dumps({k:v for k,v in summary.items() if k!="pairs"},ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");args=parser.parse_args()
    prepare() if args.prepare else run()
