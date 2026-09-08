"""Pre-registered paired ten-round, two-seed, two-provider four-role experiments."""
import asyncio,json,sys
from pathlib import Path
from statistics import mean
from dotenv import load_dotenv
from game_theory_agent.market import MarketEnv,MarketState,load_market_config
from game_theory_agent.market.actor_experiments import run_episode,read_json,write_json
from game_theory_agent.market.actor_policies import options,actions_for,reward
from game_theory_agent.local_budget import status
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/stage27-long-real"
async def main():
    OUT.mkdir(parents=True,exist_ok="--resume-after-review" in sys.argv);load_dotenv(ROOT/".env",override=False)
    c=load_market_config(ROOT/"configs/market_v14_stage24.yaml")
    actors=["company_A","economy_supplier","consumers","government"];seeds=[270101,270102]
    models=["doubao-seed-2-0-lite-260215","deepseek-v4-flash"]
    if "--resume-after-review" not in sys.argv:write_json(OUT/"preregistration.json",dict(config=c.to_dict(),config_hash=c.config_sha256,seeds=seeds,rounds=10,
        model_actors=actors,other_actors="autonomous rules",models=models,maximum_new_calls=160,budget_before=status(),
        protocol="Eight market actors settle together; four role representatives call independently each round. Identical episode_id/seed across treatments. Strict schema, no retries, shared original 10 CNY ceiling.",
        metrics=["full replay","per-role model choice execution","own three-round conditional regret","total welfare","profit","choice switches"],
        limits="Constrained option selection, no unrestricted multi-turn reasoning or Nash optimality. n=2 seeds is directional only."))
    rows=[]
    for model in ("rule",*models):
        for seed in seeds:
            directory=OUT/f"{model}-{seed}"
            try:
                r=await run_episode(c,directory=directory,seed=seed,rounds=10,
                    modes={a:"model" for a in actors} if model!="rule" else {},authorize_real=model!="rule",
                    model_name=model if model!="rule" else models[0])
                rows.append({k:v for k,v in r.items() if k!="memory"}|dict(model=model))
            except Exception as exc:
                write_json(OUT/"failure.json",dict(model=model,seed=seed,error=type(exc).__name__,message=str(exc),budget=status(),no_retry=True))
                raise
            write_json(OUT/"episodes.json",rows);print(json.dumps(dict(model=model,seed=seed,status=r["status"],budget=status()["remaining_cny"])),flush=True)
    diagnostics=[]
    for model in models:
        for seed in seeds:
            directory=OUT/f"{model}-{seed}";checkpoint=read_json(directory/"checkpoint.json")
            for transition in checkpoint["transitions"]:
                s=MarketState.from_dict(transition["state_before"]);journal=read_json(directory/f"round-{s.round:03d}.json")
                chosen={a:v["option"] for a,v in journal["decisions"].items()}
                for actor in actors:
                    returns={}
                    for alternative in options(actor):
                        choices={**chosen,actor:alternative};e=MarketEnv(c);e.load_state(s);v=s;total=0
                        for n in range(min(3,s.rounds_remaining)):
                            aa=actions_for(c,v,choices)
                            v=e.step(f"{v.episode_id}:{v.round}:{v.state_version}",aa,
                                actor_choices={a:o for a,o in choices.items() if a not in v.company_ids}).state_after
                            total+=(.95**n)*reward(v,actor)
                        returns[alternative]=total
                    executed=(transition["final_actions"][actor]==actions_for(c,s,chosen)[actor].to_dict()) if actor in s.company_ids else transition["step_result"]["actor_choices"].get(actor)==chosen[actor]
                    diagnostics.append(dict(model=model,seed=seed,round=s.round,actor=actor,chosen=chosen[actor],
                        executed=executed,three_round_returns=returns,regret_cents=max(returns.values())-returns[chosen[actor]]))
    write_json(OUT/"diagnostics.json",diagnostics)
    summary=dict(passed=len(rows)==6 and all(r["replay_passed"] for r in rows) and all(d["executed"] for d in diagnostics),
        episodes=6,rounds=60,real_model_calls=len(diagnostics),budget_after=status(),
        groups=[dict(model=m,n=2,mean_welfare_cents=mean(r["welfare_cents"] for r in rows if r["model"]==m)) for m in ("rule",*models)],
        role_diagnostics=[dict(model=m,actor=a,n=20,execution_rate=mean(d["executed"] for d in diagnostics if d["model"]==m and d["actor"]==a),
            mean_conditional_three_round_regret_cents=mean(d["regret_cents"] for d in diagnostics if d["model"]==m and d["actor"]==a)) for m in models for a in actors],
        limits="Two paired seeds. Regret uses a simulator teacher holding others' current portfolios fixed for three rounds, not an achievable hidden-information optimum. Model output is a finite menu choice.")
    write_json(OUT/"summary.json",summary);print(json.dumps(summary))
if __name__=="__main__":asyncio.run(main())
