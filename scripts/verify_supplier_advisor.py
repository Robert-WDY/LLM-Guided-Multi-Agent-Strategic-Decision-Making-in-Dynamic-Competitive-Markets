"""New-seed integration of endogenous quotes with public advice, no LLM."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from game_theory_agent.experiments.advisor_coverage_v11 import run_episode, write
from game_theory_agent.market import load_market_config
from game_theory_agent.market.protocols import sha256_hash

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/supplier-advisor-integration-v11"
SPEC=ROOT/"experiment-specs/supplier-advisor-integration-v11/PREREGISTRATION.json"
CONFIG=ROOT/"configs/market_v11_supplier.yaml"
SEEDS=(140301,140302)
PERSONAS=("balanced_v1","public_service")


def worker(task):return run_episode(load_market_config(CONFIG),*task)


if __name__=="__main__":
    spec={"seeds":SEEDS,"personas":PERSONAS,"modes":["rule","paired"],"rounds":20,"config_hash":load_market_config(CONFIG).config_sha256,"acceptance":["all_engineering_gates","at_least_one_release"],"boundary":"Small new-seed integration smoke; v10.1 holdout effect estimates do not transfer automatically to endogenous supplier prices."}
    spec["hash"]=sha256_hash(spec)
    if SPEC.exists():
        assert json.loads(SPEC.read_text(encoding="utf-8"))==json.loads(json.dumps(spec))
    else: write(SPEC,spec)
    if (OUT/"summary.json").exists():raise RuntimeError("integration already completed")
    tasks=[(seed,persona,mode) for seed in SEEDS for persona in PERSONAS for mode in ("rule","paired")]
    with ProcessPoolExecutor(max_workers=2) as pool: rows=list(pool.map(worker,tasks))
    pairs=[]
    for seed in SEEDS:
        for persona in PERSONAS:
            by_mode={r["mode"]:r for r in rows if r["seed"]==seed and r["persona"]==persona}
            pairs.append({"seed":seed,"persona":persona,**{key:by_mode["paired"][key]-by_mode["rule"][key] for key in ("utility_ppm","enterprise_value_cents","welfare_cents")}})
    summary={"spec_hash":spec["hash"],"episodes":len(rows),"rounds":len(rows)*20,"model_calls":0,"released":sum(r["release_count"] for r in rows),"eligible":sum(r["eligible_count"] for r in rows),"engineering":all(r["gates"] for r in rows),"pairs":pairs}
    summary["passed"]=summary["engineering"] and summary["released"]>0
    write(OUT/"rows.json",rows);write(OUT/"summary.json",summary)
    print(json.dumps(summary),flush=True)
