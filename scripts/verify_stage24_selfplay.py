import asyncio,json
from pathlib import Path
from statistics import mean
from game_theory_agent.market import MarketEnv,load_market_config
from game_theory_agent.market.actor_experiments import run_episode,write_json
from game_theory_agent.market.actor_policies import actor_ids
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/stage24-selfplay"
async def main():
    OUT.mkdir(parents=True,exist_ok=False);c=load_market_config(ROOT/"configs/market_v14_stage24.yaml")
    env=MarketEnv(c);ids=actor_ids(env.reset(episode_id="ids"));modes={a:"learning" for a in ids}
    write_json(OUT/"preregistration.json",dict(config=c.to_dict(),training_seeds=list(range(240201,240217)),
        holdout_seeds=list(range(240301,240309)),rounds=20,actors=list(ids),
        design="All role populations update own realized reward. Freeze best mean portfolio per actor before held-out comparison. Common episode_id per seed."))
    memory={};training=[]
    for seed in range(240201,240217):
        r=await run_episode(c,directory=OUT/f"train-{seed}",seed=seed,rounds=20,modes=modes,memory=memory)
        memory=r["memory"];training.append(dict(seed=seed,welfare_cents=r["welfare_cents"]))
        write_json(OUT/"population.json",memory)
    fixed={a:max(v["means"],key=v["means"].get) for a,v in memory.items()}
    write_json(OUT/"frozen-policies.json",fixed);rows=[]
    for seed in range(240301,240309):
        for mode in ("rule","online_learning","frozen_population"):
            r=await run_episode(c,directory=OUT/f"{mode}-{seed}",seed=seed,rounds=20,
                modes=modes if mode=="online_learning" else {},fixed=fixed if mode=="frozen_population" else {})
            rows.append({k:v for k,v in r.items() if k!="memory"}|{"treatment":mode});write_json(OUT/"holdout.json",rows)
    summary=dict(training_episodes=16,holdout_episodes=24,rounds=800,all_replays_passed=all(r["replay_passed"] for r in rows),
        groups=[dict(treatment=m,n=8,mean_welfare_cents=mean(r["welfare_cents"] for r in rows if r["treatment"]==m)) for m in ("rule","online_learning","frozen_population")],
        limits="Synthetic jointly adaptive finite populations. Observed returns are confounded by co-adaptation; no Nash convergence or universal gain is claimed.")
    write_json(OUT/"summary.json",summary);print(json.dumps(summary))
if __name__=="__main__":asyncio.run(main())
