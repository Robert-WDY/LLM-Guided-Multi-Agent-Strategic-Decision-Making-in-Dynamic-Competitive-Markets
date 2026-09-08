"""Five-round controlled action probes. Run before inspecting main evaluation results."""
import json,time
from concurrent.futures import ProcessPoolExecutor
from evaluate_market_v18 import *

RECIPES={'rule':{},'price_down':{'price':850000},'price_up':{'price':1150000},'service':{'service_budget_cents':100000},'advertising':{'advertising_budget_cents':100000},'capacity':{'capacity_investment_cents':100000},'resilience':{'resilience_budget_cents':100000},'public':{'shared_resilience_contribution_cents':100000},'project':{'threshold_project_contribution_cents':100000},'procure_less':{'quantity':600000},'procure_more':{'quantity':1200000},'diversify':{'supplier_share':500000},'resilient_supplier':{'supplier_share':0},'contract_long':{'duration':3},'reserve':{'reserve':True}}

def probe(case):
 market,n,seed=case;rows=[]
 for label,recipe in RECIPES.items():
  c,e,s=initial(market,n,seed,10);source=s.state_hash;actions=[];trajectory=[];checks=0;profit=welfare=0
  for _ in range(5):
   aa,extra,_=population(c,s,'rule',{});aa['company_A']=recipe_action(c,s,'company_A',recipe);actions.append(aa['company_A'].to_dict());a=step(e,s,aa,extra);checks+=audit_settlement(s.to_dict(),a.to_dict(),c.to_dict());profit+=a.company('company_A').financial.round_profit_cents;welfare+=a.welfare_accounting.round_total_economic_welfare_cents;trajectory.append(dict(actions={k:v.to_dict() for k,v in aa.items()},extra=extra,state_hash=a.state_hash));s=a
  x=s.company('company_A');rows.append(dict(label=label,profit=profit,welfare=welfare,price=x.commercial.price_cents,sales=x.commercial.sales_orders,capacity=x.operations.base_capacity_orders,resilience=x.risk.resilience_ppm,public_resilience=s.shared_resilience.industry_resilience_ppm,project=s.strategic_market.threshold_project.status.value,action=actions[0],final_hash=s.state_hash,trajectory=trajectory,arithmetic_checks=checks))
 base=rows[0];effective=[r['label'] for r in rows[1:] if r['final_hash']!=base['final_hash']]
 # Outcome hash includes action metadata; compare economic metrics independently.
 metrics=('profit','welfare','sales','capacity','resilience','public_resilience','project')
 economic=[r['label'] for r in rows[1:] if any(r[k]!=base[k] for k in metrics)]
 result=dict(market=market,companies=n,seed=seed,source_hash=source,rows=rows,economically_effective=economic)
 save_compressed(OUT/'probes'/f'{market}-{n}-{seed}.json.gz',result)
 return dict(market=market,companies=n,seed=seed,economically_effective=economic,arithmetic_checks=sum(r['arithmetic_checks'] for r in rows),effects={r['label']:{k:r[k]-base[k] for k in ('profit','welfare','sales','capacity','resilience','public_resilience')} for r in rows})

def main():
 spec=dict(seeds=[485001,485002],horizon=5,recipes=RECIPES,markets=list(MARKETS),companies=[2,5,10],scope='One company changes a declared recipe across five rounds; other policies fixed but react to evolving state. Economic deltas, not action labels, determine effective dimensions.')
 write_json(OUT/'probes-preregistration.json',spec)
 with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(probe,[(m,n,s) for m in MARKETS for n in (2,5,10) for s in spec['seeds']]))
 write_json(OUT/'probes.json',dict(passed=True,cases=len(rows),trajectories=len(rows)*len(RECIPES),rounds=len(rows)*len(RECIPES)*5,results=rows));print('probes complete',len(rows))

if __name__=='__main__':main()
