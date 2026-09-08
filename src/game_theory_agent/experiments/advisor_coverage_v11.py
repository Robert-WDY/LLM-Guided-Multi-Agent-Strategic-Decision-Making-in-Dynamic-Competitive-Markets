"""Pre-registered closed-loop public-advice validation against legacy and rules."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from concurrent.futures import ProcessPoolExecutor, as_completed

from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.agents.personas import PersonaRegistry, PersonaUtilityTracker
from game_theory_agent.belief import BeliefLedger
from game_theory_agent.opponent import OpponentModelLedger
from game_theory_agent.gameplay import build_rule_action
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.replay import EpisodeManifest, MarketTransition, verify_replay
from game_theory_agent.market.protocols import sha256_hash
from game_theory_agent.strategic_reliability import PublicMarketRolloutAdvisor
from game_theory_agent.strategic_reliability.contracts import StrategicActionCandidate
from game_theory_agent.strategic_reliability.rollout import _candidate_to_action
from game_theory_agent.strategic_reliability.paired_gate import PAIRED_MARKET_GATE_POLICY
from game_theory_agent.experiments.v10_beta_validation import invariant, _information_audit
from game_theory_agent.experiments.final_market_advisor_v9_validation import _enterprise_value
from game_theory_agent.experiments.final_market_v9_real_llm_smoke import _complete_observation

ROOT=Path(__file__).resolve().parents[3]
CONFIG=ROOT/"configs/market_v10_1_advisor.yaml"
OUT=ROOT/"runs/advisor-coverage-v11"
SPEC=ROOT/"experiment-specs/advisor-coverage-v11/PREREGISTRATION.json"
SEEDS=tuple(range(130701,130709))
PERSONAS=("balanced_v1","public_service")
MODES=("rule","legacy","paired")


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def observe(config,state,beliefs,opponents):
    b,bh=beliefs.company_view(observer_company_id="company_A",round_number=state.round,state_version=state.state_version)
    o,oh=opponents.company_view(observer_company_id="company_A",round_number=state.round,state_version=state.state_version)
    view=ObservationBuilder().build(state,"company_A","public",belief_state=b.model_dump(mode="json"),belief_hash=bh,belief_schema_version=b.belief_schema_version)
    view.update(opponent_model_state=o.model_dump(mode="json"),opponent_model_hash=oh)
    return _complete_observation(config,state,view,advice=None),b,o


def run_episode(config,seed,persona,mode):
    env=MarketEnv(config)
    state=env.reset(episode_id=f"coverage-{seed}",episode_seed=seed,max_rounds=20,cooperation_mode="combined_v1")
    manifest=EpisodeManifest.create(env,state,experiment_id="advisor-coverage-v11")
    registry=PersonaRegistry.from_market_config(config)
    tracker=PersonaUtilityTracker(registry.evaluator(registry.get(persona)))
    beliefs=BeliefLedger(episode_id=state.episode_id,company_ids=state.company_ids)
    opponents=OpponentModelLedger(episode_id=state.episode_id,company_ids=state.company_ids)
    advisor=PublicMarketRolloutAdvisor(config,gate_policy="legacy" if mode=="legacy" else PAIRED_MARKET_GATE_POLICY["policy_version"])
    transitions=[]; decisions=[]; gates=True
    while not state.terminal:
        actions={c:build_rule_action(config,state,c) for c in state.company_ids}
        leaks,consistent=_information_audit(state); gates &= leaks==0 and consistent
        if mode!="rule" and state.state_version>=7 and "company_A" in state.strategic_market.active_company_ids:
            view,b,o=observe(config,state,beliefs,opponents)
            advice=advisor.advise(observation=view,company_id="company_A",persona_profile=registry.get(persona),belief_state=b,opponent_model=o,horizon_rounds=3,scenario_count=5,advisor_mode="strategic_market_v9")
            released=advice.execution_disposition=="recommend"
            decisions.append({"round":state.round,"released":released,"candidate":advice.recommended_candidate_id,"confidence":advice.reliability_gate["opponent_model_confidence_ppm"],"reasons":advice.reliability_gate["abstain_reason_codes"],"advice_hash":advice.advice_hash})
            if released:
                candidate=next(c.candidate for c in advice.candidate_actions if c.candidate.candidate_id==advice.recommended_candidate_id)
                actions["company_A"]=_candidate_to_action(candidate,state,"company_A")
        result=env.step(f"{state.episode_id}:{state.round}:{state.state_version}",actions)
        after=result.state_after
        transitions.append(MarketTransition.create(state,actions,result))
        beliefs.update_after_settlement(state,actions); opponents.update_after_settlement(state,after,actions)
        tracker.record(state,after,"company_A")
        gates &= all(invariant(after).values())
        state=after
    gates &= verify_replay(MarketEnv(config),manifest,tuple(transitions))[-1].state_hash==state.state_hash
    return {"seed":seed,"split":"development" if seed<130705 else "holdout","persona":persona,"mode":mode,"gates":gates,"decisions":decisions,"release_count":sum(d["released"] for d in decisions),"eligible_count":len(decisions),"utility_ppm":tracker.cumulative_discounted_utility_ppm,"enterprise_value_cents":_enterprise_value(state,"company_A",config),"welfare_cents":state.welfare_accounting.cumulative_total_economic_welfare_cents,"exited": "company_A" not in state.strategic_market.active_company_ids,"state_hash":state.state_hash}


def prepare():
    if SPEC.exists(): raise RuntimeError("preregistration exists")
    spec={"seeds":SEEDS,"holdout":SEEDS[4:],"personas":PERSONAS,"modes":MODES,"rounds":20,"history_min":7,"scenarios":5,"horizon":3,"model_calls":0,"config_hash":load_market_config(CONFIG).config_sha256,"gate_policy":PAIRED_MARKET_GATE_POLICY,"acceptance":{"minimum_holdout_eligible_coverage_ppm":100000,"median_holdout_utility_delta_nonnegative":True,"worst_holdout_ev_delta_floor_cents":-2500000,"no_extra_focal_exits":True,"all_engineering_gates":True},"design":"Natural evolving 20-round trajectories, no resets or injected future state. Holdout seeds fixed before running. Rule baseline and legacy vs paired gate; 7-round warmup is required by unchanged n/(n+3)>=0.7 confidence gate. Loss floor is 5% of config initial cash (50m cents), not a claim of no possible losses."}
    spec["hash"]=sha256_hash(spec);write(SPEC,spec)


def worker(task):
    return run_episode(load_market_config(CONFIG),*task)


def run():
    spec=json.loads(SPEC.read_text(encoding="utf-8")); identity=spec.pop("hash")
    assert sha256_hash(spec)==identity
    config=load_market_config(CONFIG); assert config.config_sha256==spec["config_hash"]
    if (OUT/"summary.json").exists(): raise RuntimeError("results exist")
    rows=json.loads((OUT/"rows.json").read_text(encoding="utf-8")) if (OUT/"rows.json").exists() else []
    completed={(r["seed"],r["persona"],r["mode"]) for r in rows}
    tasks=[(seed,persona,mode) for seed in SEEDS for persona in PERSONAS for mode in MODES if (seed,persona,mode) not in completed]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,task) for task in tasks]
        for future in as_completed(futures):
            row=future.result(); rows.append(row)
            rows.sort(key=lambda r:(r["seed"],r["persona"],r["mode"]))
            write(OUT/"rows.json",rows)
            print(row["seed"],row["persona"],row["mode"],row["release_count"],"/",row["eligible_count"],flush=True)
    pairs=[]
    for seed in SEEDS:
        for persona in PERSONAS:
            indexed={r["mode"]:r for r in rows if r["seed"]==seed and r["persona"]==persona}
            for comparator in ("rule","legacy"):
                a,b=indexed[comparator],indexed["paired"]
                pairs.append({"seed":seed,"persona":persona,"split":b["split"],"comparator":comparator,"extra_exit":b["exited"] and not a["exited"],**{key:b[key]-a[key] for key in ("utility_ppm","enterprise_value_cents","welfare_cents")}})
    hold=[p for p in pairs if p["split"]=="holdout" and p["comparator"]=="rule"]
    by_mode={m:{"released":sum(r["release_count"] for r in rows if r["mode"]==m and r["split"]=="holdout"),"eligible":sum(r["eligible_count"] for r in rows if r["mode"]==m and r["split"]=="holdout")} for m in MODES}
    coverage=by_mode["paired"]["released"]*1000000//max(1,by_mode["paired"]["eligible"])
    gates={"engineering":all(r["gates"] for r in rows),"holdout_coverage":coverage>=100000,"holdout_median_utility":median(p["utility_ppm"] for p in hold)>=0,"holdout_tail_floor":min(p["enterprise_value_cents"] for p in hold)>=-2500000,"no_extra_exits":not any(p["extra_exit"] for p in hold)}
    summary={"spec_hash":identity,"episodes":len(rows),"rounds":len(rows)*20,"model_calls":0,"gates":gates,"passed":all(gates.values()),"holdout_coverage_ppm":coverage,"holdout_by_mode":by_mode,"holdout_median_utility_delta":median(p["utility_ppm"] for p in hold),"holdout_worst_ev_delta_cents":min(p["enterprise_value_cents"] for p in hold),"pairs":pairs,"abstention_reasons":dict(Counter(reason for r in rows if r["mode"]=="paired" for d in r["decisions"] for reason in d["reasons"]))}
    write(OUT/"summary.json",summary); print(json.dumps({k:v for k,v in summary.items() if k not in {"pairs","abstention_reasons"}},ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");args=parser.parse_args()
    prepare() if args.prepare else run()
