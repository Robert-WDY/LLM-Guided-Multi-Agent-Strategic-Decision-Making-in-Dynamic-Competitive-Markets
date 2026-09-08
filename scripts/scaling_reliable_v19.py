"""Fixed-total versus per-firm scaled resources, independently seeded long episodes."""
from statistics import pstdev
from reliable_v19_common import *

def run(case):
    regime,n,seed=case;c,e,s=make(regime,n,seed);initial=s.to_dict();memory={};rows=[];revenue=stockout=0;dispersion=[]
    for _ in range(20):
        aa,memory=actions(c,s,memory,seed);after=step(e,s,aa);audit_settlement(s.to_dict(),after.to_dict(),c.to_dict());revenue+=sum(x.financial.round_revenue_cents for x in after.companies);stockout+=after.market.lost_after_stockout_orders;dispersion.append(pstdev(x.commercial.price_cents for x in after.companies))
        rows.append(dict(round=s.round,actions={a:v.to_dict() for a,v in aa.items()},state_hash=after.state_hash));s=after
    profit=sum(x.financial.cumulative_profit_cents for x in s.companies);summary=dict(regime=regime,companies=n,seed=seed,margin=profit/revenue if revenue else None,profit=profit,price_dispersion=mean(dispersion),exits=n-s.strategic_market.active_company_count,stockout=stockout,consumer_welfare=s.welfare_accounting.cumulative_consumer_surplus_cents,welfare=s.welfare_accounting.cumulative_total_economic_welfare_cents,checks=replay(c,MarketState.from_dict(initial),rows))
    save(OUT/'scaling'/f'{regime}-{n}-{seed}.json.gz',dict(summary=summary,config=c.to_dict(),initial_state=initial,transitions=rows));return summary
def main():
    freeze_market();cases=[(m,n,s) for m in ('normal','scaled') for n in (2,5,10) for s in SPEC['scaling_seeds']]
    with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,cases))
    write_json(OUT/'scaling.json',dict(passed=True,cases=len(rows),rounds=2400,results=rows,advisor_gain_source='Primary advisor experiment: paired Draft versus Advisor normal/scaled cells use separate 521001..521020 seeds. Do not pair raw episodes across these different seed cohorts.'));print('scaling complete 120',flush=True)
if __name__=='__main__':main()
