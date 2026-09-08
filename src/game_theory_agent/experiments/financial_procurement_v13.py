"""Stage15: new held-out regimes for a finance-aware procurement gate."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from math import ceil
from statistics import mean, median
import zipfile

from game_theory_agent.experiments.procurement_response_v12 import ROOT, configuration as old_config, source_files, write
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import MarketEnv
from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.financial_procurement import finance_view, guarded_procurement
from game_theory_agent.market.procurement_policy import apply_procurement, procurement_view, select_procurement
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition, verify_replay
from game_theory_agent.experiments.v10_beta_validation import invariant, _information_audit
from game_theory_agent.persistence import CODEC

OUT=ROOT/"runs/financial-procurement-v13"
SPEC=ROOT/"experiment-specs/financial-procurement-v13/PREREGISTRATION.json"
REGIMES=("material45","material62","compound75","cash16","asymmetric","low_demand")
SEEDS=tuple(range(160201,160213))
POLICIES=("diverse","capacity","guarded")


def configuration(regime):
    if regime=="old_failure":return old_config("compound")
    data=old_config("base").to_dict();sc=data["supply_chain"];p=sc["autonomous_pricing"]
    a,b=sc["suppliers"]["economy_supplier"],sc["suppliers"]["resilient_supplier"]
    if regime=="material45":sc["downstream_input_cost_share_ppm"]=450000;p["step_ppm"]=70000
    elif regime=="material62":
        sc["downstream_input_cost_share_ppm"]=620000;p["step_ppm"]=80000
        a["base_capacity_orders"],b["base_capacity_orders"]=8200,12400
    elif regime=="compound75":
        sc["downstream_input_cost_share_ppm"]=750000;p["step_ppm"]=180000
        a.update(unit_price_cents=2100,unit_cost_cents=1750,base_capacity_orders=7800,reliability_ppm=720000)
        b.update(unit_price_cents=2900,unit_cost_cents=2100,base_capacity_orders=9500,reliability_ppm=840000)
    elif regime=="cash16":
        data["company_initial"]["cash_balance_cents"]=16000000
        data["operating_costs"]["fixed_overhead_cents"]=4200000
        sc["downstream_input_cost_share_ppm"]=650000
    elif regime=="asymmetric":
        a.update(base_capacity_orders=14000,reliability_ppm=900000)
        b.update(base_capacity_orders=6500,reliability_ppm=750000)
    elif regime=="low_demand":data["market"]["base_demand_orders"]=9000
    else:raise ValueError(regime)
    return MarketConfig.from_mapping(data)


def worker(task):
    regime,seed,policy=task;config=configuration(regime)
    env=MarketEnv(config);state=env.reset(episode_id=f"finance-{seed}",episode_seed=seed,max_rounds=20,cooperation_mode="combined_v1")
    manifest=EpisodeManifest.create(env,state,experiment_id="financial-procurement-v13")
    initial_cash=sum(c.financial.cash_balance_cents for c in state.companies)
    transitions=[];audits=[];shocks=[];gates=True;released=eligible=0;missed=0
    while not state.terminal:
        actions={c:build_rule_action(config,state,c) for c in state.company_ids};records={}
        for company in state.strategic_market.active_company_ids:
            if policy=="guarded":
                plan,audit=guarded_procurement(procurement_view(state,company),finance_view(config,state,company,actions[company]))
                actions[company]=apply_procurement(actions[company],plan)
                records[company]=audit;released+=audit["released"];eligible+=1
            elif policy=="capacity":actions[company]=apply_procurement(actions[company],select_procurement(procurement_view(state,company)))
            else:actions[company]=replace(actions[company],primary_supplier_id="economy_supplier",backup_supplier_id="resilient_supplier",primary_supplier_share_ppm=500000)
        leaks,consistent=_information_audit(state);gates &= leaks==0 and consistent
        result=env.step(f"{state.episode_id}:{state.round}:{state.state_version}",actions)
        after=result.state_after;gates &= all(invariant(after).values())
        transitions.append(MarketTransition.create(state,actions,result))
        missed+=sum(o.unfulfilled_orders for o in after.supply_chain.last_procurement_outcomes)
        shocks.append({k:v for k,v in result.random_draw_summary if k.startswith("supplier_disruption:")})
        audits.append({"round":state.round,"decisions":records,"state_hash":after.state_hash})
        state=after
        if state.state_version==8:
            restored=CODEC.decode(CODEC.encode(state));copy=MarketEnv(config);copy.load_state(restored);env=copy
    gates &= verify_replay(MarketEnv(config),manifest,tuple(transitions))[-1].state_hash==state.state_hash
    return {"regime":regime,"seed":seed,"policy":policy,"initial_total_cash_cents":initial_cash,"welfare_cents":state.welfare_accounting.cumulative_total_economic_welfare_cents,"exits":len(state.company_ids)-len(state.strategic_market.active_company_ids),"shortage":missed,"released":released,"eligible":eligible,"engineering":gates,"shocks":shocks,"audits":audits,"final_hash":state.state_hash}


def prepare():
    if SPEC.exists():raise RuntimeError("already registered")
    spec={"regimes":REGIMES,"seeds":SEEDS,"policies":POLICIES,"diagnostic_old_seeds":list(range(150205,150213)),"rounds":20,"model_calls":0,"config_hashes":{r:configuration(r).config_sha256 for r in (*REGIMES,"old_failure")},"source_hash":sha256_hash(source_files()),"gates":{"median_delta_ppm_min":0,"worst_delta_ppm_min":-50000,"cvar10_delta_ppm_min":-20000,"extra_exit_pairs_max":0,"minimum_released_ppm":10000},"design":"Old failure group is diagnostic only. Financial guard evaluated on 6 unseen regimes and 12 independent seeds; primary diverse baseline, unguarded capacity secondary. No selection/tuning using new holdout. All families are synthetic, not industry calibration."}
    spec["hash"]=sha256_hash(spec);write(SPEC,spec)
    OUT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(OUT/"registered-python-source.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source_files():archive.write(ROOT/name,name)


def run():
    spec=json.loads(SPEC.read_text(encoding="utf-8"));identity=spec.pop("hash")
    assert sha256_hash(spec)==identity and spec["source_hash"]==sha256_hash(source_files())
    if (OUT/"summary.json").exists():raise RuntimeError("results exist")
    tasks=[(r,s,p) for r in REGIMES for s in SEEDS for p in POLICIES]+[("old_failure",s,p) for s in range(150205,150213) for p in POLICIES]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(worker,task) for task in tasks]):
            row=future.result();rows.append(row)
            write(OUT/"cells"/f"{row['regime']}-{row['seed']}-{row['policy']}.json",row)
            if len(rows)%40==0:print("completed",len(rows),"/",len(tasks),flush=True)
    indexed={(r["regime"],r["seed"],r["policy"]):r for r in rows};pairs=[]
    for r in rows:
        if r["policy"]=="diverse":continue
        b=indexed[r["regime"],r["seed"],"diverse"]
        pairs.append({"regime":r["regime"],"seed":r["seed"],"policy":r["policy"],"delta_ppm":(r["welfare_cents"]-b["welfare_cents"])*1000000//max(abs(b["welfare_cents"]),b["initial_total_cash_cents"]),"extra_exits":r["exits"]-b["exits"],"same_shocks":r["shocks"]==b["shocks"]})
    hold=[p for p in pairs if p["policy"]=="guarded" and p["regime"]!="old_failure"]
    values=sorted(p["delta_ppm"] for p in hold)
    eligible=sum(r["eligible"] for r in rows if r["regime"]!="old_failure")
    released=sum(r["released"] for r in rows if r["regime"]!="old_failure")
    metrics={"pairs":len(hold),"positive":sum(v>0 for v in values),"negative":sum(v<0 for v in values),"median_delta_ppm":median(values),"worst_delta_ppm":values[0],"cvar10_delta_ppm":mean(values[:ceil(len(values)*.1)]),"extra_exit_pairs":sum(p["extra_exits"]>0 for p in hold),"released":released,"eligible":eligible}
    gates={"engineering":all(r["engineering"] for r in rows) and all(p["same_shocks"] for p in pairs),"median":metrics["median_delta_ppm"]>=0,"worst":metrics["worst_delta_ppm"]>=-50000,"tail":metrics["cvar10_delta_ppm"]>=-20000,"exits":metrics["extra_exit_pairs"]==0,"nontrivial_release":released*1000000>=10000*eligible}
    summary={"spec_hash":identity,"source_hash":spec["source_hash"],"episodes":len(rows),"rounds":len(rows)*20,"model_calls":0,"holdout":metrics,"gates":gates,"passed":all(gates.values())}
    write(OUT/"paired-results.json",pairs);write(OUT/"summary.json",summary);print(json.dumps(summary),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");args=parser.parse_args()
    prepare() if args.prepare else run()
