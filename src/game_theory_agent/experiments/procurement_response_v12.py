"""New-regime procurement response research, with preregistered promotion gates."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from math import ceil
from pathlib import Path
from statistics import mean, median
import zipfile

from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.procurement_policy import procurement_view, select_procurement, apply_procurement
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition, verify_replay
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.persistence import CODEC
from game_theory_agent.experiments.v10_beta_validation import invariant, _information_audit

ROOT=Path(__file__).resolve().parents[3]
CONFIG=ROOT/"configs/market_v11_supplier.yaml"
OUT=ROOT/"runs/procurement-response-v12"
SPEC=ROOT/"experiment-specs/procurement-response-v12/PREREGISTRATION.json"
DEV_SEEDS=tuple(range(150101,150109))
HOLDOUT_SEEDS=tuple(range(150201,150213))
DEV_REGIMES=("base","step10","capacity_tilt","cap120","fixed_quotes","input_share55")
HOLDOUT_REGIMES=("step03","step15","reverse_capacity","reliability_reversal","thin_cash","compound")
POLICIES=("fixed_economy","cheapest","diverse","capacity","gradual")


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    temporary.replace(path)


def configuration(regime):
    data=load_market_config(CONFIG).to_dict();sc=data["supply_chain"];pricing=sc["autonomous_pricing"]
    a,b=sc["suppliers"]["economy_supplier"],sc["suppliers"]["resilient_supplier"]
    if regime=="step10":pricing["step_ppm"]=100000
    elif regime=="capacity_tilt":a["base_capacity_orders"],b["base_capacity_orders"]=8250,13750
    elif regime=="cap120":pricing["max_base_price_ppm"]=1200000
    elif regime=="fixed_quotes":pricing["enabled"]=False
    elif regime=="input_share55":sc["downstream_input_cost_share_ppm"]=550000
    elif regime=="step03":pricing["step_ppm"]=30000
    elif regime=="step15":pricing["step_ppm"]=150000
    elif regime=="reverse_capacity":a["base_capacity_orders"],b["base_capacity_orders"]=16000,9000
    elif regime=="reliability_reversal":a["reliability_ppm"],b["reliability_ppm"]=990000,650000
    elif regime=="thin_cash":
        data["company_initial"]["cash_balance_cents"]=20000000
        data["operating_costs"]["fixed_overhead_cents"]=4000000
    elif regime=="compound":
        pricing["step_ppm"]=120000
        sc["downstream_input_cost_share_ppm"]=700000
        a.update(base_capacity_orders=7000,unit_price_cents=2000,unit_cost_cents=1650,reliability_ppm=650000)
        b.update(base_capacity_orders=10000,unit_price_cents=2800,unit_cost_cents=2000,reliability_ppm=880000)
    elif regime!="base":raise ValueError(regime)
    return MarketConfig.from_mapping(data)


def source_files():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"src").rglob("*.py")}


def prepare():
    if SPEC.exists():raise RuntimeError("preregistration already exists")
    hashes={r:configuration(r).config_sha256 for r in (*DEV_REGIMES,*HOLDOUT_REGIMES)}
    spec={"dev_seeds":DEV_SEEDS,"holdout_seeds":HOLDOUT_SEEDS,"dev_regimes":DEV_REGIMES,"holdout_regimes":HOLDOUT_REGIMES,"policies":POLICIES,"rounds":20,"model_calls":0,"config_hashes":hashes,"source_hash":sha256_hash(source_files()),
          "design":"All firms use the same tested procurement policy; operating rule unchanged, current quote/capacity known, proportional demand congestion approximated by equal capacity per active buyer. Compare diverse 50:50 as primary control, cheapest and fixed economy as secondary. Development selects one of capacity/gradual by extra-exit pairs ascending, normalized lower-tail CVaR descending, median welfare descending. Choice frozen before holdout; all holdout candidates retained for transparency but cannot replace the selection.",
          "promotion":{"engineering":True,"development_and_holdout_required":True,"median_welfare_delta_ppm_min":0,"worst_welfare_delta_ppm_min":-50000,"cvar10_welfare_delta_ppm_min":-20000,"extra_exit_pairs_max":0},
          "normalization":"paired welfare delta / max(abs(diverse cumulative welfare), initial total buyer cash), scaled ppm; empirical CVaR10 is mean of worst ceil(n*0.1) cells. No confidence interval or real-world safety claim; cells share seeds across regimes."}
    spec["hash"]=sha256_hash(spec);write(SPEC,spec)
    OUT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(OUT/"registered-python-source.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source_files():archive.write(ROOT/name,name)


def worker(task):
    regime,seed,policy=task;config=configuration(regime)
    env=MarketEnv(config);state=env.reset(episode_id=f"procurement-{seed}",episode_seed=seed,max_rounds=20,cooperation_mode="combined_v1")
    manifest=EpisodeManifest.create(env,state,experiment_id="procurement-response-v12")
    initial_cash=sum(c.financial.cash_balance_cents for c in state.companies)
    transitions=[];audits=[];shocks=[];gates=True;missed=0;switches=0;previous={};restored_env=None
    while not state.terminal:
        actions={c:build_rule_action(config,state,c) for c in state.company_ids}
        cheapest=min(state.supply_chain.suppliers,key=lambda s:(s.unit_price_cents,s.supplier_id)).supplier_id
        round_plans={}
        for company in state.strategic_market.active_company_ids:
            if policy in {"capacity","gradual"}:
                plan=select_procurement(procurement_view(state,company),policy)
                actions[company]=apply_procurement(actions[company],plan)
                round_plans[company]=plan.to_dict()
            else:
                actions[company]=replace(actions[company],primary_supplier_id=cheapest if policy=="cheapest" else "economy_supplier",backup_supplier_id="resilient_supplier" if policy=="diverse" else None,primary_supplier_share_ppm=500000 if policy=="diverse" else 1000000)
            action=actions[company]
            share=action.primary_supplier_share_ppm if action.primary_supplier_id=="economy_supplier" else 1000000-action.primary_supplier_share_ppm
            if company in previous:switches+=share!=previous[company]
            previous[company]=share
        leaks,consistent=_information_audit(state);gates &= leaks==0 and consistent
        step_id=f"{state.episode_id}:{state.round}:{state.state_version}"
        result=env.step(step_id,actions);after=result.state_after
        if restored_env is not None:
            gates &= restored_env.step(step_id,actions).state_after.state_hash==after.state_hash
            restored_env=None
        transitions.append(MarketTransition.create(state,actions,result))
        gates &= all(invariant(after).values())
        missed+=sum(o.unfulfilled_orders for o in after.supply_chain.last_procurement_outcomes)
        shocks.append(dict(result.random_draw_summary))
        audits.append({"round":state.round,"plans":round_plans,"quotes":{s.supplier_id:s.unit_price_cents for s in state.supply_chain.suppliers},"welfare_cents":after.welfare_accounting.round_total_economic_welfare_cents,"exits":len(after.company_ids)-len(after.strategic_market.active_company_ids)})
        state=after
        if state.state_version==8:
            restored_env=MarketEnv(config);restored_env.load_state(CODEC.decode(CODEC.encode(state)))
    gates &= verify_replay(MarketEnv(config),manifest,tuple(transitions))[-1].state_hash==state.state_hash
    row={"regime":regime,"seed":seed,"policy":policy,"config_hash":config.config_sha256,"engineering":gates,"initial_total_cash_cents":initial_cash,"welfare_cents":state.welfare_accounting.cumulative_total_economic_welfare_cents,"exits":len(state.company_ids)-len(state.strategic_market.active_company_ids),"unfulfilled_orders":missed,"switches":switches,"supplier_profit_cents":{s.supplier_id:s.cumulative_profit_cents for s in state.supply_chain.suppliers},"round_audit":audits,"random_draws":shocks,"final_hash":state.state_hash}
    row["hash"]=sha256_hash(row)
    return row


def metrics(pairs):
    values=sorted(p["welfare_delta_ppm"] for p in pairs)
    return {"pairs":len(pairs),"median_welfare_delta_ppm":median(values),"worst_welfare_delta_ppm":values[0],"cvar10_welfare_delta_ppm":mean(values[:ceil(len(values)*.1)]),"extra_exit_pairs":sum(p["exits_delta"]>0 for p in pairs),"positive_welfare_pairs":sum(v>0 for v in values),"median_shortage_delta":median(p["shortage_delta"] for p in pairs),"maximum_extra_exits":max(p["exits_delta"] for p in pairs)}


def compare(rows,policy):
    by_key={(r["regime"],r["seed"],r["policy"]):r for r in rows}
    pairs=[]
    for r in rows:
        if r["policy"]!=policy:continue
        b=by_key[r["regime"],r["seed"],"diverse"]
        pairs.append({"regime":r["regime"],"seed":r["seed"],"welfare_delta_ppm":(r["welfare_cents"]-b["welfare_cents"])*1000000//max(abs(b["welfare_cents"]),b["initial_total_cash_cents"]),"welfare_delta_cents":r["welfare_cents"]-b["welfare_cents"],"exits_delta":r["exits"]-b["exits"],"shortage_delta":r["unfulfilled_orders"]-b["unfulfilled_orders"],"supplier_shocks_match":all({k:v for k,v in x.items() if k.startswith("supplier_disruption:")}=={k:v for k,v in y.items() if k.startswith("supplier_disruption:")} for x,y in zip(r["random_draws"],b["random_draws"]))})
    return pairs


def safe(metrics):
    return metrics["median_welfare_delta_ppm"]>=0 and metrics["worst_welfare_delta_ppm"]>=-50000 and metrics["cvar10_welfare_delta_ppm"]>=-20000 and metrics["extra_exit_pairs"]==0


def run():
    spec=json.loads(SPEC.read_text(encoding="utf-8"));identity=spec.pop("hash")
    assert sha256_hash(spec)==identity and sha256_hash(source_files())==spec["source_hash"]
    if (OUT/"summary.json").exists():raise RuntimeError("research already completed")
    reports={};all_pairs={}
    for split,regimes,seeds in (("development",DEV_REGIMES,DEV_SEEDS),("holdout",HOLDOUT_REGIMES,HOLDOUT_SEEDS)):
        tasks=[(regime,seed,policy) for regime in regimes for seed in seeds for policy in POLICIES]
        rows=[];pending=[]
        for task in tasks:
            path=OUT/"cells"/("-".join(map(str,task))+".json")
            if path.exists():
                row=json.loads(path.read_text(encoding="utf-8"));row_hash=row.pop("hash");assert sha256_hash(row)==row_hash;row["hash"]=row_hash;rows.append(row)
            else:pending.append(task)
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(worker,task):task for task in pending}
            for future in as_completed(futures):
                row=future.result();assert row["config_hash"]==spec["config_hashes"][row["regime"]]
                rows.append(row);write(OUT/"cells"/("-".join(map(str,futures[future]))+".json"),row)
                if len(rows)%40==0:print(split,len(rows),"/",len(tasks),flush=True)
        by_policy={policy:compare(rows,policy) for policy in POLICIES if policy!="diverse"}
        all_pairs[split]=by_policy
        report={"episodes":len(rows),"engineering":all(r["engineering"] for r in rows),"same_supplier_shocks":all(p["supplier_shocks_match"] for pairs in by_policy.values() for p in pairs),"policies":{p:metrics(v) for p,v in by_policy.items()}}
        reports[split]=report;write(OUT/f"{split}-summary.json",report)
        if split=="development":
            selected=min(("capacity","gradual"),key=lambda p:(report["policies"][p]["extra_exit_pairs"],-report["policies"][p]["cvar10_welfare_delta_ppm"],-report["policies"][p]["median_welfare_delta_ppm"],p))
            selection={"spec_hash":identity,"policy":selected,"development_metrics":report["policies"][selected],"development_risk_pass":safe(report["policies"][selected])}
            selection_path=OUT/"selection-before-holdout.json"
            if selection_path.exists():assert json.loads(selection_path.read_text(encoding="utf-8"))==selection
            else:write(selection_path,selection)
            print("selected before holdout",selected,"development risk pass",selection["development_risk_pass"],flush=True)
    engineering=all(r["engineering"] and r["same_supplier_shocks"] for r in reports.values())
    summary={"spec_hash":identity,"source_hash":spec["source_hash"],"episodes":sum(r["episodes"] for r in reports.values()),"rounds":20*sum(r["episodes"] for r in reports.values()),"model_calls":0,"selected_policy":selected,"engineering_pass":engineering,"promote_default":engineering and selection["development_risk_pass"] and safe(reports["holdout"]["policies"][selected]),"reports":reports}
    write(OUT/"paired-results.json",all_pairs);write(OUT/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");args=parser.parse_args()
    prepare() if args.prepare else run()
