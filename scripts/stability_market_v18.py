"""Determinism and local-input sensitivity, distinct from economic reliability."""
from concurrent.futures import ProcessPoolExecutor
from evaluate_market_v18 import *

def check(case):
 market,n,seed=case;c,e,s=initial(market,n,seed,20);req=RobustRequest(goal='profit',horizon=1,scenarios=2,max_candidates=8,finalists=1,step_budget=max(500,len(actor_ids(s))*50),diagnostics=False,backtest=False,seed=seed)
 a=advise(c,s,req);b=advise(c,s,req)
 aa=deepcopy(a);bb=deepcopy(b);aa['search'].pop('elapsed_seconds');bb['search'].pop('elapsed_seconds');assert aa==bb
 rows=[]
 for field in ('cash','price'):
  for ppm in (990000,1010000):
   own=s.company('company_A');new=replace(own,financial=replace(own.financial,cash_balance_cents=own.financial.cash_balance_cents*ppm//1000000)) if field=='cash' else replace(own,commercial=replace(own.commercial,price_cents=own.commercial.price_cents*ppm//1000000))
   changed=replace(s,companies=(new,*s.companies[1:]));changed=replace(changed,state_hash=state_hash(changed.to_dict()));e.load_state(changed);r=advise(c,changed,req);rows.append(dict(field=field,ppm=ppm,recipe_changed=r['recipe']!=a['recipe'],price_change=r['action']['price_cents']-a['action']['price_cents'],complete=r['v17']['response_complete']))
 result=dict(market=market,companies=n,seed=seed,deterministic=True,perturbations=rows)
 write_json(OUT/'stability'/f'{market}-{n}-{seed}.json',result);return result

def main():
 spec=dict(markets=['normal','tight_supply','scaled'],companies=[2,5,10],seeds=[486001,486002],perturbations='own cash and posted price each +/-1%',version='fast v17',scope='Repeat byte equality excluding elapsed time; local recipe changes are measured, never required to be zero. Does not prove smooth behavior at phase/cash boundaries.')
 write_json(OUT/'stability-preregistration.json',spec)
 with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(check,[(m,n,s) for m in spec['markets'] for n in spec['companies'] for s in spec['seeds']]))
 write_json(OUT/'stability.json',dict(passed=True,cases=len(rows),advisor_calls=len(rows)*6,perturbations=len(rows)*4,recipe_changes=sum(x['recipe_changed'] for r in rows for x in r['perturbations']),results=rows));print('stability complete',len(rows))

if __name__=='__main__':main()
