"""Reproduce failed holdout cells exactly and expose settlement accounting."""
from dataclasses import replace
import json

from game_theory_agent.experiments.procurement_response_v12 import OUT, configuration, write
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import MarketEnv
from game_theory_agent.market.procurement_policy import apply_procurement, procurement_view, select_procurement


def run(regime,seed,policy):
    config=configuration(regime);env=MarketEnv(config)
    state=env.reset(episode_id=f"procurement-{seed}",episode_seed=seed,max_rounds=20,cooperation_mode="combined_v1")
    rounds=[]
    while not state.terminal:
        actions={i:build_rule_action(config,state,i) for i in state.company_ids}
        for company in state.strategic_market.active_company_ids:
            if policy=="capacity":actions[company]=apply_procurement(actions[company],select_procurement(procurement_view(state,company)))
            else:actions[company]=replace(actions[company],primary_supplier_id="economy_supplier",backup_supplier_id="resilient_supplier",primary_supplier_share_ppm=500000)
        after=env.step(f"{state.episode_id}:{state.round}:{state.state_version}",actions).state_after
        rounds.append({"round":state.round,"posted_quotes":{s.supplier_id:s.unit_price_cents for s in state.supply_chain.suppliers},"companies":{i:{"financial":after.company(i).financial.to_dict(),"operations":after.company(i).operations.to_dict(),"commercial":after.company(i).commercial.to_dict(),"lifecycle":after.strategic_market.lifecycle(i).to_dict(),"cash_before_cents":state.company(i).financial.cash_balance_cents,"primary_share_ppm":actions[i].primary_supplier_share_ppm,"unit_contribution_before_fixed_cents":after.company(i).commercial.price_cents-after.company(i).operations.actual_unit_cost_cents-config.integer("operating_costs","fulfillment_cost_per_order_cents")} for i in state.company_ids}})
        state=after
    registered=json.loads((OUT/"cells"/f"{regime}-{seed}-{policy}.json").read_text(encoding="utf-8"))
    assert registered["final_hash"]==state.state_hash
    return {"regime":regime,"seed":seed,"policy":policy,"registered_hash_matches":True,"rounds":rounds}


if __name__=="__main__":
    pairs=json.loads((OUT/"paired-results.json").read_text(encoding="utf-8"))["holdout"]["capacity"]
    failures=[p for p in pairs if p["exits_delta"]>0 or p["welfare_delta_ppm"]<0]
    results=[]
    for pair in failures:
        for policy in ("diverse","capacity"):
            result=run(pair["regime"],pair["seed"],policy)
            write(OUT/"forensics"/f"{pair['regime']}-{pair['seed']}-{policy}.json",result)
            exits=[]
            for row in result["rounds"]:
                for company,data in row["companies"].items():
                    if data["lifecycle"]["exit_round"]==row["round"]:
                        exits.append({"company":company,"round":row["round"],"reason":data["lifecycle"]["exit_reason"],"unit_contribution_before_fixed_cents":data["unit_contribution_before_fixed_cents"],"cash_before_cents":data["cash_before_cents"],"input_price_cents":data["operations"]["procurement_unit_input_price_cents"]})
            results.append({"regime":pair["regime"],"seed":pair["seed"],"policy":policy,"hash_matches":True,"exits":exits})
    write(OUT/"forensics/summary.json",{"extra_diagnostic_replays":len(results),"all_registered_final_hashes_match":True,"results":results})
    print(json.dumps(results,ensure_ascii=False),flush=True)
