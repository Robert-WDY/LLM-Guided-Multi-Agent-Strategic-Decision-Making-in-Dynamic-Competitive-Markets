"""Fresh-seed factorial validation after demand/payback supplier guard."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import zipfile
from statistics import median

from game_theory_agent.experiments.procurement_response_v12 import ROOT, source_files, write
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition, verify_replay
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.persistence import CODEC
from game_theory_agent.agents.observation import ObservationBuilder

OUT=ROOT/"runs/supplier-payback-v16"
SPEC=ROOT/"experiment-specs/supplier-payback-v16/PREREGISTRATION.json"
POLICIES=tuple(f"{x:03b}" for x in range(8))
REGIMES=("base","drought","cash_stress")


def configuration(regime,policy):
    data=load_market_config(ROOT/"configs/market_v13_1_local.yaml").to_dict()
    p=data["autonomous_market"]
    p["company_procurement"]=policy[0]=="1"
    p["supplier_investment"]["enabled"]=policy[1]=="1"
    p["government"]["enabled"]=policy[2]=="1"
    if regime=="drought":
        for s in data["supply_chain"]["suppliers"].values():s["reliability_ppm"]=550000
        data["supply_chain"]["disruption_capacity_ppm"]=150000
    elif regime=="cash_stress":
        data["company_initial"]["cash_balance_cents"]=12000000
        data["operating_costs"]["fixed_overhead_cents"]=4200000
    elif regime=="constrained_long":
        for supplier in data["supply_chain"]["suppliers"].values():
            supplier["base_capacity_orders"]=5000
            supplier["reliability_ppm"]=1000000
    return MarketConfig.from_mapping(data)


def tasks():
    return [(r,s,p) for r in REGIMES for s in range(190201,190209) for p in POLICIES]+[(r,s,p) for r in ("long","constrained_long") for s in range(190301,190305) for p in POLICIES]


def worker(task):
    regime,seed,policy=task;config=configuration(regime,policy);env=MarketEnv(config)
    state=env.reset(episode_id=f"payback-{regime}-{seed}",episode_seed=seed,max_rounds=60 if regime.endswith("long") else 20,cooperation_mode="combined_v1")
    manifest=EpisodeManifest.create(env,state,experiment_id="supplier-payback-v16")
    transitions=[];evidence=[];investment=support=inspection=refunds=waste=orders=0
    while not state.terminal:
        actions={cid:build_rule_action(config,state,cid) for cid in state.company_ids}
        for cid in state.company_ids:
            view=ObservationBuilder().build(state,cid,"public")
            assert all("account" not in s and "investment_decision" not in s for s in view["market"]["supply_chain"]["suppliers"].values())
            assert all("account" not in s and "investment_decision" not in s for s in env.get_action_constraints(cid,state.state_version)["eligible_suppliers"])
        result=env.step(f"{state.episode_id}:{state.round}:{state.state_version}",actions);after=result.state_after
        invest=sum(s.account.investment_cents or 0 for s in after.supply_chain.suppliers)
        spend=sum(v for _,v in after.government.last_decision.support_by_company_cents)
        for s in after.supply_chain.suppliers:
            assert s.account.opening_cash_cents==state.supply_chain.supplier(s.supplier_id).account.cash_cents
        assert after.government.last_decision.opening_cash_cents==state.government.cash_cents
        for c in after.companies:
            old=state.company(c.company_id)
            residual=c.financial.cash_balance_cents-old.financial.cash_balance_cents-c.financial.round_profit_cents
            if after.strategic_market.lifecycle(c.company_id).exit_round==state.round:
                book=(old.financial.capacity_book_value_cents*(1000000-config.integer("capacity","book_value_depreciation_ppm"))+500000)//1000000+actions[c.company_id].capacity_investment_cents
                assert residual==(book*config.integer("strategic_market","liquidation_recovery_ppm")+500000)//1000000
            else:assert residual==0
        investment+=invest;support+=spend;inspection+=after.government.last_decision.inspection_cost_cents
        refunds+=sum(d.refund_cents for d in after.consumer_decisions)
        waste+=sum(o.material.wasted_orders for o in after.supply_chain.last_procurement_outcomes)
        orders+=sum(c.commercial.sales_orders for c in after.companies)
        evidence.append({"round":state.round,"hash":after.state_hash,"investment_cents":invest,"support_cents":spend,
            "supplier_shocks":{k:v for k,v in result.random_draw_summary if k.startswith("supplier_disruption:")}})
        transitions.append(MarketTransition.create(state,actions,result));state=after
        if state.state_version in (6,16,36):
            copied=CODEC.decode(CODEC.encode(state));env=MarketEnv(config);env.load_state(copied)
    assert verify_replay(MarketEnv(config),manifest,tuple(transitions))[-1].state_hash==state.state_hash
    row={"regime":regime,"seed":seed,"policy":policy,"rounds":len(evidence),"engineering":True,
        "welfare_cents":state.welfare_accounting.cumulative_total_economic_welfare_cents,
        "exits":len(state.company_ids)-len(state.strategic_market.active_company_ids),"supplier_investment_cents":investment,
        "support_cents":support,"inspection_cents":inspection,"refunds_cents":refunds,"waste_orders":waste,"sales_orders":orders,
        "final_hash":state.state_hash,"evidence":evidence,"manifest":manifest.to_dict(),"transitions":[t.to_dict() for t in transitions]}
    return row


def prepare():
    if SPEC.exists():raise RuntimeError("already registered")
    spec={"tasks":tasks(),"policy_bits":["company_procurement","supplier_investment","government"],
        "source_hash":sha256_hash(source_files()),"config_hashes":{r+"-"+p:configuration(r,p).config_sha256 for r in (*REGIMES,"long","constrained_long") for p in POLICIES},
        "gates":["all cash/material/consumer/government invariants","legal public inputs","full replay and multiple restore points","supplier investment/government support/inspection exercised","common supplier shock draws for each seed/regime"],
        "model_calls":0,"promotion_gates":{"full_extra_exit_pairs_max":0,"full_median_welfare_delta_min":0},
        "boundary":"v15 failure data are development only. New seeds, reused three stress families, and one new constrained long-run regime. Report negative incremental policy effects. Full candidate gate is zero extra exits and nonnegative median versus all-off; no universal individual-policy profit claim. Supplier capex requires own excess demand, no current disruption, 70% margin realization and 150% capital coverage before horizon."}
    spec["hash"]=sha256_hash(spec);write(SPEC,spec);OUT.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(OUT/"registered-python-source.zip","x",zipfile.ZIP_DEFLATED) as archive:
        for name in source_files():archive.write(ROOT/name,name)


def run():
    spec=json.loads(SPEC.read_text(encoding="utf-8"));identity=spec.pop("hash")
    assert sha256_hash(spec)==identity and sha256_hash(source_files())==spec["source_hash"]
    if (OUT/"summary.json").exists():raise RuntimeError("results exist")
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,task) for task in tasks()]
        for future in as_completed(futures):
            row=future.result();write(OUT/"cells"/f"{row['regime']}-{row['seed']}-{row['policy']}.json",row)
            rows.append({k:v for k,v in row.items() if k not in {"manifest","transitions"}})
            if len(rows)%32==0:print("completed",len(rows),"/",len(futures),flush=True)
    index={(r["regime"],r["seed"],r["policy"]):r for r in rows};pairs=[]
    for r in rows:
        b=index[r["regime"],r["seed"],"000"]
        assert [e["supplier_shocks"] for e in r["evidence"]]==[e["supplier_shocks"] for e in b["evidence"]]
        if r["policy"]!="000":pairs.append({"regime":r["regime"],"seed":r["seed"],"policy":r["policy"],"welfare_delta_cents":r["welfare_cents"]-b["welfare_cents"],"extra_exits":r["exits"]-b["exits"]})
    metrics={p:{"pairs":len(v:=[r for r in pairs if r["policy"]==p]),"positive":sum(r["welfare_delta_cents"]>0 for r in v),"median_welfare_delta_cents":median(r["welfare_delta_cents"] for r in v),"worst_welfare_delta_cents":min(r["welfare_delta_cents"] for r in v),"extra_exit_pairs":sum(r["extra_exits"]>0 for r in v)} for p in POLICIES if p!="000"}
    exercised={k:sum(r[k] for r in rows) for k in ("supplier_investment_cents","support_cents","inspection_cents","refunds_cents")}
    result={"passed":all(r["engineering"] for r in rows) and all(v>0 for v in exercised.values()),"source_hash":spec["source_hash"],"spec_hash":identity,"episodes":len(rows),"rounds":sum(r["rounds"] for r in rows),"model_calls":0,"mechanisms_exercised":exercised,"paired_metrics":metrics}
    result["promotion_passed"]=result["passed"] and metrics["111"]["extra_exit_pairs"]==0 and metrics["111"]["median_welfare_delta_cents"]>=0
    incremental=[]
    for r in rows:
        if r["policy"]=="111":
            baseline=index[r["regime"],r["seed"],"101"]
            incremental.append({"regime":r["regime"],"seed":r["seed"],"welfare_delta_cents":r["welfare_cents"]-baseline["welfare_cents"],"extra_exits":r["exits"]-baseline["exits"]})
    write(OUT/"incremental-investment.json",incremental)
    write(OUT/"paired-results.json",pairs);write(OUT/"summary.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");args=parser.parse_args()
    prepare() if args.prepare else run()
