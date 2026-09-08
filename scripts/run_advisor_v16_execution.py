"""Additional independent deviation audit and rolling execution acceptance."""
import asyncio,json,time
from statistics import mean
from pathlib import Path
from run_advisor_v16_acceptance import ROOT,DEST,CONFIG,initial
from game_theory_agent.game_theory.objectives import AdviceRequest,objective,score
from game_theory_agent.game_theory.advisor_market import forecast,Rollouts,RECIPES
from game_theory_agent.market.actor_experiments import run_episode,read_json,write_json
from game_theory_agent.market.actor_policies import actor_ids,options
from game_theory_agent.local_budget import status

async def main():
 path=DEST/'execution-acceptance.json'
 if path.exists():print(json.dumps(read_json(path)));return
 before=status();start=time.perf_counter();checks=[];rows=[]
 write_json(DEST/'execution-preregistration.json',dict(seeds=[369601,369602],variants=['rule','advisor_company','advisor_profit','advisor_welfare'],rounds=5,real_calls=0,criteria='Replay and action identity; measure paired gains without requiring benefit in every case. Recompute all reported unilateral gaps independently.'))
 for count in (2,5,10):
  result=read_json(DEST/f'multiplayer-{count}.json');req=AdviceRequest.model_validate(result['request']);state,_=forecast(CONFIG,initial(count),req.company_id);spec=objective(req,state);engine=Rollouts(CONFIG,state,req.model_copy(update={'step_budget':5000}));choices={a:['baseline','price_down','price_up','shared'] if a in state.company_ids else list(options(a)) for a in actor_ids(state)};choices[req.company_id].append('recommended')
  def payoff(profile,actor):
   decoded={a:(result['recipe'] if v=='recommended' else RECIPES[v]) if a in state.company_ids else v for a,v in profile.items()}
   out=[engine.evaluate(decoded,s) for s in (2000,2001)]
   return mean(score(v,spec)['total'] if actor==req.company_id else v['role_utilities'][actor]/10000000 for v in out)
  for name in ('recommendation_check','equilibrium'):
   d=result['diagnostics'][name]
   if d['complete']:
    for actor in choices:
     gap=max(0,max(payoff({**d['profile'],actor:v},actor) for v in choices[actor])-payoff(d['profile'],actor));assert abs(gap-d['deviation_gains'][actor])<1e-8
    checks.append(dict(companies=count,kind=name,actors=len(choices),all_gaps_match=True))
 for seed in (369601,369602):
  baseline=None
  for mode in ('rule','advisor_company','advisor_profit','advisor_welfare'):
   directory=DEST/'closed-loop'/f'{seed}-{mode}';summary=await run_episode(CONFIG,directory=directory,seed=seed,rounds=5,modes={'company_A':mode});cp=read_json(directory/'checkpoint.json');company=cp['state']['companies']['company_A'] if isinstance(cp['state']['companies'],dict) else next(c for c in cp['state']['companies'] if c['company_id']=='company_A');profit=company['financial']['cumulative_profit_cents'];row=dict(seed=seed,mode=mode,profit=profit,welfare=summary['welfare_cents'],replay=summary['replay_passed'],rounds=5)
   if baseline is None:baseline=row
   row.update(profit_gain_cents=profit-baseline['profit'],welfare_gain_cents=row['welfare']-baseline['welfare'])
   if mode!='rule':
    for i,trace in enumerate(cp['transitions'],1):
     d=read_json(directory/f'round-{i:03d}.json')['decisions']['company_A'];assert d['advisor_decision']['action']==trace['final_actions']['company_A']
   rows.append(row);print(json.dumps(row),flush=True)
 assert status()==before
 result=dict(passed=True,independent_deviation_checks=checks,episodes=len(rows),rounds=40,results=rows,elapsed_seconds=time.perf_counter()-start,new_real_calls=0);write_json(path,result);print(json.dumps({k:v for k,v in result.items() if k!='results'}))
if __name__=='__main__':asyncio.run(main())
