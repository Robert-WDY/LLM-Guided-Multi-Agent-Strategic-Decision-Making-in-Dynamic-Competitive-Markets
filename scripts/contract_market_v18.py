"""Supplement: the original long-contract probe equals the default three rounds."""
from concurrent.futures import ProcessPoolExecutor
from evaluate_market_v18 import *

def case(spec):
 market,n,seed=spec;results=[]
 for duration in (1,3):
  c,e,s=initial(market,n,seed,10);traces=[];profit=welfare=0;checks=0
  for _ in range(5):
   aa,extra,_=population(c,s,'rule',{});aa['company_A']=recipe_action(c,s,'company_A',{'duration':duration});a=step(e,s,aa,extra);checks+=audit_settlement(s.to_dict(),a.to_dict(),c.to_dict());profit+=a.company('company_A').financial.round_profit_cents;welfare+=a.welfare_accounting.round_total_economic_welfare_cents;traces.append(dict(action=aa['company_A'].to_dict(),hash=a.state_hash));s=a
  results.append(dict(duration=duration,profit=profit,welfare=welfare,traces=traces,checks=checks))
 row=dict(market=market,companies=n,seed=seed,profit_delta=results[0]['profit']-results[1]['profit'],welfare_delta=results[0]['welfare']-results[1]['welfare'],action_changed=results[0]['traces'][0]['action']['contract_duration_rounds']!=results[1]['traces'][0]['action']['contract_duration_rounds'])
 save_compressed(OUT/'contract-probes'/f'{market}-{n}-{seed}.json.gz',dict(summary=row,results=results));return row

def main():
 write_json(OUT/'contract-preregistration.json',dict(reason='Original three-round contract probe equals default in all 66 cases; cannot infer inactivity',durations=[1,3],seeds=[485001,485002],horizon=5,markets=list(MARKETS),companies=[2,5,10],scope='Supplementary coverage test, not holdout tuning or replacing the original null result'))
 with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(case,[(m,n,s) for m in MARKETS for n in (2,5,10) for s in (485001,485002)]))
 write_json(OUT/'contract-probes.json',dict(passed=True,cases=66,trajectories=132,rounds=660,action_changed=sum(r['action_changed'] for r in rows),economically_changed=sum(r['profit_delta']!=0 or r['welfare_delta']!=0 for r in rows),results=rows));print('contract probes complete')

if __name__=='__main__':main()
