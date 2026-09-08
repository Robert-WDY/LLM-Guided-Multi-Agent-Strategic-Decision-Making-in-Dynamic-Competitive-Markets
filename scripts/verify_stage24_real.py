import asyncio,json
from pathlib import Path
from dataclasses import asdict
from dotenv import load_dotenv
from game_theory_agent.market import MarketEnv,load_market_config
from game_theory_agent.market.actor_policies import actor_ids,observation,descriptions,objective,actions_for,reward
from game_theory_agent.market.actor_model import choose_option
from game_theory_agent.market.actor_experiments import write_json
from game_theory_agent.market.replay import EpisodeManifest,MarketTransition,verify_replay
from game_theory_agent.local_budget import status
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/stage24-real"
async def main():
    OUT.mkdir(parents=True,exist_ok=False);load_dotenv(ROOT/".env",override=False)
    c=load_market_config(ROOT/"configs/market_v14_stage24.yaml");e=MarketEnv(c)
    s=e.reset(episode_id="s24-real-240401",episode_seed=240401,max_rounds=10,cooperation_mode="combined_v1")
    manifest=EpisodeManifest.create(e,s,cooperation_mode="combined_v1");tr=[];records=[]
    targets=["company_A",s.supply_chain.supplier_ids[0],"consumers","government"]
    write_json(OUT/"preregistration.json",dict(config=c.to_dict(),rounds=10,real_decision_round=4,actors=targets,maximum_calls=4,budget_before=status()))
    while not s.terminal:
        choices={}
        if s.round==4:
            for actor in targets:
                obs=observation(s,actor)
                write_json(OUT/f"{actor}-pending.json",dict(state_hash=s.state_hash,observation=obs,status="pending_unknown"))
                d=await choose_option(actor_id=actor,objective=objective(actor),observation=obs,options=descriptions(actor),max_output_tokens=128)
                choices[actor]=d.option_id;records.append(dict(observation=obs,decision=asdict(d)))
                write_json(OUT/"real-decisions.json",records)
        aa=actions_for(c,s,choices);result=e.step(f"{s.episode_id}:{s.round}:{s.state_version}",aa,actor_choices={a:o for a,o in choices.items() if a not in s.company_ids})
        tr.append(MarketTransition.create(s,aa,result));s=result.state_after
    verify_replay(MarketEnv(c),manifest,[MarketTransition.from_dict(t.to_dict()) for t in tr])
    write_json(OUT/"replay.json",dict(manifest=manifest.to_dict(),transitions=[t.to_dict() for t in tr]))
    write_json(OUT/"summary.json",dict(passed=True,real_calls=len(records),choices={r["decision"]["actor_id"]:r["decision"]["option_id"] for r in records},
        budget_after=status(),limits="One live decision per role, completed in a ten-round market. Not evidence of sustained model quality."))
    print(json.dumps(dict(passed=True,budget=status())))
if __name__=="__main__":asyncio.run(main())
