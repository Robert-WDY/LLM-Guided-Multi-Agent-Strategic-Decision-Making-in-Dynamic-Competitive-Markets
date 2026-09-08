import asyncio,json,sys
from pathlib import Path
from statistics import mean
from game_theory_agent.market import MarketEnv,MarketState,load_market_config
from game_theory_agent.market.actor_policies import actor_ids,options,observation,reward,actions_for
from game_theory_agent.market.actor_experiments import read_json,write_json,run_episode
from game_theory_agent.market.neural_policy import fit,features,role,FEATURES,physics_hash,MODEL,infer,artifact_hash
from game_theory_agent.market.protocols import sha256_hash
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/stage25-training"
async def main():
    OUT.mkdir(parents=True,exist_ok="--resume-training" in sys.argv);c=load_market_config(ROOT/"configs/market_v14_stage24.yaml")
    write_json(OUT/"preregistration.json",dict(config_hash=c.config_sha256,training_seeds=list(range(240201,240217)),
        state_versions=[2,6,10,14,18],label_horizon=3,discount=0.95,holdout_seeds=list(range(250301,250309)),
        teacher="synthetic counterfactual simulator with full state",student="only explicit own/public feature allowlist",
        objective="each role own discounted outcome; government total welfare",architecture="18-16-tanh-role-softmax",epochs=400))
    if "--resume-training" in sys.argv:
        rows=read_json(OUT/"training-data.json")
        if len(rows)!=640:raise ValueError("incomplete teacher dataset")
    else:
        rows=[]
        for seed in range(240201,240217):
            saved=read_json(ROOT/f"runs/stage24-selfplay/train-{seed}/checkpoint.json")
            for transition in saved["transitions"]:
                s=MarketState.from_dict(transition["state_after"])
                if s.state_version not in (2,6,10,14,18):continue
                for actor in actor_ids(s):
                    scores=[]
                    for o in options(actor):
                        env=MarketEnv(c);env.load_state(s);v=s;score=0
                        for h in range(min(3,s.rounds_remaining)):
                            choices={actor:o};aa=actions_for(c,v,choices)
                            v=env.step(f"{v.episode_id}:{v.round}:{v.state_version}",aa,actor_choices={actor:o} if actor not in v.company_ids else {}).state_after
                            score+=(0.95**h)*reward(v,actor)
                        scores.append(score)
                    rows.append(dict(seed=seed,state_version=s.state_version,actor=actor,role=role(actor),
                        x=features(observation(s,actor)).tolist(),options=list(options(actor)),returns=scores,label=max(range(len(scores)),key=scores.__getitem__)))
            write_json(OUT/"training-data.json",rows)
    models={r:fit([x for x in rows if x["role"]==r]) for r in ("company","supplier","consumers","government")}
    artifact=dict(version="four-actor-neural-selector-v1",physics_hash=physics_hash(c),features=list(FEATURES),
        models=models,training_seeds=list(range(240201,240217)),teacher_horizon=3,limits="Synthetic simulator teacher; small imitation selector, not an LLM fine-tune or real-world calibrated policy.")
    artifact["artifact_sha256"]=artifact_hash(artifact);write_json(MODEL,artifact);write_json(OUT/"model.json",artifact)
    env=MarketEnv(c);ids=actor_ids(env.reset(episode_id="ids"));results=[]
    frozen=read_json(ROOT/"runs/stage24-selfplay/frozen-policies.json")
    for seed in range(250301,250309):
        for treatment in ("rule","neural","frozen_population"):
            v=await run_episode(c,directory=OUT/f"{treatment}-{seed}",seed=seed,rounds=20,
                    modes={a:"neural" for a in ids} if treatment=="neural" else {},fixed=frozen if treatment=="frozen_population" else {})
            results.append({k:w for k,w in v.items() if k!="memory"}|dict(treatment=treatment));write_json(OUT/"holdout.json",results)
    summary=dict(training_examples=len(rows),artifact_sha256=artifact["artifact_sha256"],holdout_episodes=len(results),
        losses={r:{k:m[k] for k in ("loss_initial","loss_final","training_rows")} for r,m in models.items()},
        groups=[dict(treatment=t,n=8,mean_welfare_cents=mean(x["welfare_cents"] for x in results if x["treatment"]==t)) for t in ("rule","neural","frozen_population")],
        all_replay_passed=all(v["replay_passed"] for v in results))
    write_json(OUT/"summary.json",summary);print(json.dumps(summary))
if __name__=="__main__":asyncio.run(main())
