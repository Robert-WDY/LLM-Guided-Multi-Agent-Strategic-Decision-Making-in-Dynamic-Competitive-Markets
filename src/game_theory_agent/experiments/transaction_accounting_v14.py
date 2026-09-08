"""Registered zero-token cash/material conservation and replay stress suite."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import zipfile

from game_theory_agent.experiments.procurement_response_v12 import ROOT, source_files, write
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.config import MarketConfig
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition, verify_replay
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.persistence import CODEC

OUT = ROOT / "runs/transaction-accounting-v14-r2"
SPEC = ROOT / "experiment-specs/transaction-accounting-v14-r2/PREREGISTRATION.json"
REGIMES = ("base", "thin_cash", "expensive", "low_demand", "outage", "long")
SEEDS = tuple(range(170101, 170109))
POLICIES = ("diverse", "price_stress")


def configuration(regime):
    data = load_market_config(ROOT / "configs/market_v12_accounting.yaml").to_dict()
    if regime == "long": data["episode_options"]["round_options"] = [5,10,15,20,60]
    if regime == "thin_cash": data["company_initial"]["cash_balance_cents"] = 8000000
    elif regime == "expensive":
        data["company_initial"]["base_unit_cost_cents"] *= 2
        data["action"]["bounds"]["price_cents"]["max"] = 40000
    elif regime == "low_demand": data["market"]["base_demand_orders"] = 5000
    elif regime == "outage":
        for supplier in data["supply_chain"]["suppliers"].values(): supplier["reliability_ppm"] = 300000
        data["supply_chain"]["disruption_capacity_ppm"] = 0
    return MarketConfig.from_mapping(data)


def worker(task):
    regime, seed, policy = task
    config = configuration(regime); env = MarketEnv(config)
    state = env.reset(episode_id=f"cash-{regime}-{seed}-{policy}", episode_seed=seed,
                      max_rounds=60 if regime == "long" else 20, cooperation_mode="combined_v1")
    manifest = EpisodeManifest.create(env, state, experiment_id="transaction-accounting-v14")
    transitions = []; rows = []; invoice_total = waste_total = 0
    while not state.terminal:
        actions = {}
        for cid in state.company_ids:
            action = build_rule_action(config, state, cid)
            if cid in state.strategic_market.active_company_ids:
                action = replace(action, primary_supplier_id="economy_supplier", backup_supplier_id="resilient_supplier", primary_supplier_share_ppm=500000)
                if policy == "price_stress":
                    action = replace(action, price_cents=config.mapping("action", "bounds", "price_cents")["min"])
            actions[cid] = action
        result = env.step(f"{state.episode_id}:{state.round}:{state.state_version}", actions)
        after = result.state_after
        for c in after.companies:
            previous = state.company(c.company_id)
            residual = c.financial.cash_balance_cents - previous.financial.cash_balance_cents - c.financial.round_profit_cents
            newly_exited = after.strategic_market.lifecycle(c.company_id).exit_round == state.round
            expected_external = (previous.financial.capacity_book_value_cents * (1000000-config.integer("capacity", "book_value_depreciation_ppm")) + 500000)//1000000 + actions[c.company_id].capacity_investment_cents
            expected_external = (expected_external * config.integer("strategic_market", "liquidation_recovery_ppm") + 500000)//1000000 if newly_exited else 0
            assert residual == expected_external, ("company cash mismatch", c.company_id, residual, expected_external)
        paid = sum(o.material.payment_cents for o in after.supply_chain.last_procurement_outcomes)
        received = sum(s.account.receipts_cents for s in after.supply_chain.suppliers)
        assert paid == received
        invoice_total += paid
        waste = sum(o.material.wasted_orders for o in after.supply_chain.last_procurement_outcomes); waste_total += waste
        rows.append({"round":state.round, "paid_cents":paid, "received_cents":received, "waste_orders":waste, "hash":after.state_hash})
        transitions.append(MarketTransition.create(state, actions, result)); state = after
        if state.state_version in (7, 17, 37):
            restored = CODEC.decode(CODEC.encode(state)); other = MarketEnv(config); other.load_state(restored); env = other
    assert verify_replay(MarketEnv(config), manifest, tuple(transitions))[-1].state_hash == state.state_hash
    return {"regime":regime, "seed":seed, "policy":policy, "rounds":len(rows), "rows":rows,
            "invoice_cents":invoice_total, "waste_orders":waste_total, "exits":len(state.company_ids)-len(state.strategic_market.active_company_ids),
            "welfare_cents":state.welfare_accounting.cumulative_total_economic_welfare_cents, "final_hash":state.state_hash,
            "engineering":True, "manifest":manifest.to_dict(), "transitions":[t.to_dict() for t in transitions]}


def prepare():
    if SPEC.exists(): raise RuntimeError("already registered")
    spec = {"regimes":REGIMES, "seeds":SEEDS, "policies":POLICIES, "config_hashes":{r:configuration(r).config_sha256 for r in REGIMES},
            "source_hash":sha256_hash(source_files()), "model_calls":0,
            "gates":["cash nonnegative", "invoice receipt equality", "production and material closure", "company cash including explicit liquidation", "restore and full replay"],
            "boundary":"Synthetic open economy. Prepaid perishable inputs; external consumer endowments, operating costs and liquidation. No welfare superiority gate between different accounting models."}
    spec["hash"] = sha256_hash(spec); write(SPEC, spec); OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT/"registered-python-source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name in source_files(): archive.write(ROOT/name, name)


def run():
    spec = json.loads(SPEC.read_text(encoding="utf-8")); identity = spec.pop("hash")
    assert sha256_hash(spec) == identity and sha256_hash(source_files()) == spec["source_hash"]
    if (OUT/"summary.json").exists(): raise RuntimeError("results already exist")
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, (r,s,p)) for r in REGIMES for s in SEEDS for p in POLICIES]
        for future in as_completed(futures):
            row = future.result(); write(OUT/"cells"/f"{row['regime']}-{row['seed']}-{row['policy']}.json", row)
            rows.append({k:v for k,v in row.items() if k not in {"manifest", "transitions", "rows"}})
            if len(rows)%16 == 0: print("completed", len(rows), "/", len(futures), flush=True)
    result = {"spec_hash":identity, "source_hash":spec["source_hash"], "episodes":len(rows), "rounds":sum(r["rounds"] for r in rows),
              "model_calls":0, "passed":all(r["engineering"] for r in rows), "invoice_cents":sum(r["invoice_cents"] for r in rows),
              "waste_orders":sum(r["waste_orders"] for r in rows), "results":rows}
    write(OUT/"summary.json", result); print(json.dumps({k:v for k,v in result.items() if k!="results"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare", action="store_true"); args = parser.parse_args()
    prepare() if args.prepare else run()
