"""Measure actual full UI advisor defaults separately from the fast benchmark."""
from evaluate_market_v18 import *

def main():
 write_json(OUT/'ui-latency-preregistration.json',dict(companies=[2,5,10],seed=487001,request='RobustRequest defaults: company goal, H3/S3/16 candidates/1800 steps, diagnostics and backtest enabled',scope='One warm-process observation per company count, not a latency distribution or SLA; sequential after all experiments'))
 rows=[]
 for n in (2,5,10):
  c,e,s=initial('normal',n,487001,20);start=time.perf_counter();r=advise(c,s,RobustRequest(seed=487001));seconds=time.perf_counter()-start
  row=dict(companies=n,seconds=seconds,steps=r['search']['used_steps'],complete=r['v17']['response_complete'],recommended=r['recommended_id']);rows.append(row);write_json(OUT/f'ui-default-{n}.json',r);print(json.dumps(row),flush=True)
 write_json(OUT/'ui-default-latency.json',dict(passed=True,results=rows,scope='Measured actual default request; no production latency guarantee'))

if __name__=='__main__':main()
