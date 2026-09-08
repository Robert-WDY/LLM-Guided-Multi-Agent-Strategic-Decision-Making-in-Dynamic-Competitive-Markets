import asyncio
from dataclasses import asdict
from pathlib import Path
from dotenv import load_dotenv
from game_theory_agent.market import MarketEnv,MarketState,load_market_config
from game_theory_agent.market.actor_policies import observation,objective,descriptions,actions_for,reward,options
from game_theory_agent.market.actor_model import choose_option
from game_theory_agent.market.actor_experiments import read_json,write_json
from game_theory_agent.market.neural_policy import predict
from game_theory_agent.local_budget import status
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/stage25-real"
async def main():
    OUT.mkdir(parents=True,exist_ok=False);load_dotenv(ROOT/".env",override=False)
    c=load_market_config(ROOT/"configs/market_v14_stage24.yaml")
    saved=read_json(ROOT/"runs/stage24-real/replay.json");s=MarketState.from_dict(saved["transitions"][3]["state_before"])
    write_json(OUT/"preregistration.json",dict(state_hash=s.state_hash,actors=["company_A","consumers"],maximum_calls=2,
        comparison="trained selector and new real choices on identical held-out state; evaluate all three-round returns",budget_before=status()))
    rows=[]
    for actor in ("company_A","consumers"):
        obs=observation(s,actor);neural=predict(actor,obs,config=c)
        d=await choose_option(actor_id=actor,objective=objective(actor),observation=obs,options=descriptions(actor),max_output_tokens=128)
        row=dict(actor=actor,neural=neural,real=asdict(d),returns={})
        for option in options(actor):
            e=MarketEnv(c);e.load_state(s);v=s;total=0
            for n in range(3):
                aa=actions_for(c,v,{actor:option})
                v=e.step(f"{v.episode_id}:{v.round}:{v.state_version}",aa,actor_choices={actor:option} if actor not in v.company_ids else {}).state_after
                total+=(.95**n)*reward(v,actor)
            row["returns"][option]=total
        rows.append(row);write_json(OUT/"rows.json",rows)
    write_json(OUT/"summary.json",dict(passed=True,budget_after=status(),choices=[dict(actor=r["actor"],neural=r["neural"],real=r["real"]["option_id"]) for r in rows],
        conclusion="Actual real-model choices and trained selector compared on same state. Small diagnostic, not proof of general superiority."))
if __name__=="__main__":asyncio.run(main())
