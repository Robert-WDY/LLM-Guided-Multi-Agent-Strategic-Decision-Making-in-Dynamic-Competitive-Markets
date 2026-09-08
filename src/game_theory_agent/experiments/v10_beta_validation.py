"""Pre-registered synthetic integration and sensitivity gates for v10 beta.

Policy selection here is an authoritative simulation oracle, not an LLM or a
public-information advisor. Historical experiment artifacts are never changed.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean

from game_theory_agent.agents.personas import PersonaRegistry, PersonaUtilityTracker
from game_theory_agent.experiments.combined_market_v6_validation import _initial_state, _actions, _information_audit
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition, verify_replay

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/market_v10_multi_objective.yaml"
OUT = ROOT / "runs/v10-beta-validation"
SPEC = ROOT / "experiment-specs/v10-beta-validation/PREREGISTRATION.json"
SEEDS = tuple(range(120701, 120713))
PROFILES = ("none", "stakeholder_balanced", "public_service", "resilience_steward")
POLICIES = ("inactive_cheap", "combined_cheap", "combined_diverse", "combined_resilient")
VARIANTS = ("base", "weak_reliability", "strong_reliability", "expensive_resilient", "capacity_squeeze", "no_externality", "high_externality", "low_welfare_weight", "high_welfare_weight")


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configuration(name):
    data = load_market_config(CONFIG).to_dict()
    suppliers = data["supply_chain"]["suppliers"]
    if name == "weak_reliability":
        for s in suppliers.values(): s["reliability_ppm"] = 500_000
    elif name == "strong_reliability":
        for s in suppliers.values(): s["reliability_ppm"] = 990_000
    elif name == "expensive_resilient":
        suppliers["resilient_supplier"]["unit_price_cents"] = 3600
    elif name == "capacity_squeeze":
        for s in suppliers.values(): s["base_capacity_orders"] = 5500
    elif name in {"no_externality", "high_externality"}:
        data["welfare_accounting"]["stockout_externality_per_order_cents"] = 0 if name == "no_externality" else 10000
    elif name in {"low_welfare_weight", "high_welfare_weight"}:
        target = 100_000 if name == "low_welfare_weight" else 800_000
        for profile in PROFILES[1:]:
            weights = data["persona_utilities"]["weights_ppm"][profile]
            denominator = 1_000_000 - weights["social_welfare"]
            weights.update({k:v * (1_000_000-target) // denominator for k,v in weights.items() if k != "social_welfare"})
            weights["social_welfare"] = target
            weights["profit"] += 1_000_000 - sum(weights.values())
    return MarketConfig.from_mapping(data)


def procurement(action, policy):
    return replace(action, primary_supplier_id="resilient_supplier" if policy.endswith("resilient") else "economy_supplier", backup_supplier_id="resilient_supplier" if policy.endswith("diverse") else None, primary_supplier_share_ppm=500_000 if policy.endswith("diverse") else 1_000_000)


def invariant(state):
    w = state.welfare_accounting
    return {
        "demand_closure": state.market.no_purchase_orders + state.market.lost_after_stockout_orders + sum(c.commercial.sales_orders for c in state.companies) == state.market.realized_demand_orders,
        "cash_nonnegative": all(c.financial.cash_balance_cents >= 0 for c in state.companies),
        "welfare_closure": w.round_total_economic_welfare_cents == w.round_consumer_surplus_cents + w.round_downstream_producer_surplus_cents + w.round_upstream_producer_surplus_cents + w.round_government_net_budget_cents - w.round_stockout_externality_cents - w.round_business_exit_externality_cents,
        "upstream_accounted": w.round_upstream_producer_surplus_cents == state.supply_chain.round_upstream_producer_surplus_cents,
    }


def episode(config, seed, policy, *, rounds, combined):
    if combined:
        env, state = _initial_state(config, seed)
    else:
        env = MarketEnv(config)
        state = env.reset(episode_id=f"sensitivity-{seed}", episode_seed=seed, max_rounds=20, cooperation_mode="combined_v1")
    manifest = EpisodeManifest.create(env, state, experiment_id="v10-beta-validation")
    registry = PersonaRegistry.from_market_config(config)
    trackers = {p:PersonaUtilityTracker(registry.evaluator(registry.get(p))) for p in PROFILES}
    transitions = []
    gates = {"demand_closure":True,"cash_nonnegative":True,"welfare_closure":True,"upstream_accounted":True,"public_privacy":True}
    missed, aid, detected = 0, 0, 0
    for _ in range(rounds):
        leaks, consistent = _information_audit(state)
        gates["public_privacy"] &= leaks == 0 and consistent
        actions = _actions(config, state, "inactive_controls" if policy.startswith("inactive") else "combined_strategies") if combined else {c:build_rule_action(config,state,c) for c in state.company_ids}
        active = state.strategic_market.active_company_ids
        actions = {c:procurement(a,policy) if c in active else a for c,a in actions.items()}
        result = env.step(f"{state.episode_id}:{state.round}:{state.state_version}", actions)
        transitions.append(MarketTransition.create(state, actions, result))
        after = result.state_after
        for tracker in trackers.values(): tracker.record(state, after, "company_A")
        state = after
        for key,value in invariant(state).items(): gates[key] &= value
        missed += sum(o.unfulfilled_orders for o in state.supply_chain.last_procurement_outcomes)
        aid += sum(t.fulfilled_orders for t in state.strategic_market.last_mutual_aid_transfers)
        detected += sum(t.detected for t in state.strategic_market.last_price_coordination_outcomes)
    replayed = verify_replay(MarketEnv(config),manifest,tuple(transitions))[-1]
    gates["replay"] = replayed.state_hash == state.state_hash
    w = state.welfare_accounting
    return {"seed":seed,"split":"development" if seed < 120707 else "holdout","policy":policy,"rounds":rounds,"state_hash":state.state_hash,"config_hash":config.config_sha256,"gates":gates,"cumulative_welfare_cents":w.cumulative_total_economic_welfare_cents,"cumulative_focal_profit_cents":state.company("company_A").financial.cumulative_profit_cents,"cumulative_downstream_profit_cents":w.cumulative_downstream_producer_surplus_cents,"unfulfilled_procurement_orders":missed,"project_status":state.strategic_market.threshold_project.status.value,"mutual_aid_orders":aid,"detected_coordination":detected,"exited_count":sum(l.status.value == "exited" for l in state.strategic_market.company_lifecycle),"persona_utility_ppm":{p:t.cumulative_discounted_utility_ppm for p,t in trackers.items()}}


def prepare():
    if SPEC.exists(): raise RuntimeError("preregistration already exists")
    spec = {"experiment":"v10-beta-validation","seeds":SEEDS,"holdout_seeds":SEEDS[6:],"policies":POLICIES,"variants":VARIANTS,"profiles":PROFILES,"combination_rounds":20,"sensitivity_rounds":8,"zero_model_calls":True,"design":"12 seeds; 4 integrated policies; 9 OAT variants x 3 sourcing policies; full deterministic replay. Synthetic oracle ranks predeclared policies by each objective, not an LLM/advisor result.","gates":["demand_closure","cash_nonnegative","welfare_closure","upstream_accounted","public_privacy","replay"],"base_config_hash":load_market_config(CONFIG).config_sha256,"config_hashes":{v:configuration(v).config_sha256 for v in VARIANTS}}
    spec["hash"] = sha256_hash(spec)
    write(SPEC,spec)
    return spec


def run():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    expected_hash=spec.pop("hash")
    assert sha256_hash(spec)==expected_hash
    assert spec["base_config_hash"]==load_market_config(CONFIG).config_sha256
    if (OUT/"summary.json").exists(): raise RuntimeError("results already exist")
    combos=[]
    for seed in SEEDS:
        for policy in POLICIES: combos.append(episode(configuration("base"),seed,policy,rounds=20,combined=True))
        print(f"combination {seed} complete",flush=True)
    write(OUT/"combination.json",combos)
    rows=[]
    for variant in VARIANTS:
        config=configuration(variant)
        assert config.config_sha256 == spec["config_hashes"][variant]
        for seed in SEEDS:
            for policy in POLICIES[1:]:
                row=episode(config,seed,policy,rounds=8,combined=False)
                row["variant"]=variant
                rows.append(row)
        print(f"sensitivity {variant} complete",flush=True)
    write(OUT/"sensitivity.json",rows)
    pairs=[]
    for variant in VARIANTS:
        for policy in POLICIES[2:]:
            values=[next(r for r in rows if r["seed"]==seed and r["variant"]==variant and r["policy"]==policy)["cumulative_welfare_cents"]-next(r for r in rows if r["seed"]==seed and r["variant"]==variant and r["policy"]=="combined_cheap")["cumulative_welfare_cents"] for seed in SEEDS[6:]]
            pairs.append({"variant":variant,"policy":policy,"holdout_n":len(values),"mean_welfare_delta_cents":mean(values),"worst_welfare_delta_cents":min(values),"positive_count":sum(v>0 for v in values)})
    choices=[]
    for variant in VARIANTS:
        for profile in PROFILES:
            counts={p:0 for p in POLICIES[1:]}
            for seed in SEEDS[6:]:
                options=[r for r in rows if r["seed"]==seed and r["variant"]==variant]
                best=max(options,key=lambda r:(r["persona_utility_ppm"][profile],r["policy"]))
                counts[best["policy"]]+=1
            choices.append({"variant":variant,"profile":profile,"holdout_policy_choices":counts})
    gates={key:all(row["gates"][key] for row in combos+rows) for key in combos[0]["gates"]}
    summary={"preregistration_hash":expected_hash,"model_calls":0,"episodes":len(combos+rows),"settled_rounds":sum(r["rounds"] for r in combos+rows),"gates":gates,"engineering_gate_passed":all(gates.values()),"holdout_sourcing_pairs":pairs,"objective_selections":choices,"claim_boundary":"Deterministic engineering proof and synthetic directional sensitivity; no real economy calibration or population significance.","combination_sha256":sha256_hash(combos),"sensitivity_sha256":sha256_hash(rows)}
    write(OUT/"summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in {"holdout_sourcing_pairs","objective_selections"}},ensure_ascii=False),flush=True)
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args()
    prepare() if args.prepare else run()
