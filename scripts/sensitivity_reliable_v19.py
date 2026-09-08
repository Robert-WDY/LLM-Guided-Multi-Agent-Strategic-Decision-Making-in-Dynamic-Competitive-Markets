"""Small parameter sensitivity and an exact sequential-game response sanity check."""
import numpy as np
from reliable_v19_common import *
from game_theory_agent.game_theory.lab import sequential

def main():
    freeze_market();rows=[]
    for parameter in ('price_elasticity','service_effect','shock_frequency','inventory_spoilage','supplier_overhead'):
        for factor in (.5,1.,2.):
            for seed in (525001,525002,525003):
                c,_,_=make('normal',5,seed,10);raw=c.to_dict()
                if parameter in ('price_elasticity','service_effect'):
                    key='price' if parameter=='price_elasticity' else 'service'
                    for seg in raw['consumer_choice']['segments'].values():seg['coefficients_ppm'][key]=round(seg['coefficients_ppm'][key]*factor)
                elif parameter=='shock_frequency':
                    for event in raw['events']['definitions'].values():event['signal_generation_probability_ppm']=min(1000000,round(event['signal_generation_probability_ppm']*factor))
                elif parameter=='inventory_spoilage':raw['supply_chain']['strategic_policy']['inventory_spoilage_ppm']=round(raw['supply_chain']['strategic_policy']['inventory_spoilage_ppm']*factor)
                else:raw['supply_chain']['strategic_policy']['fixed_overhead_cents']=round(raw['supply_chain']['strategic_policy']['fixed_overhead_cents']*factor)
                c=MarketConfig.from_mapping(raw);e=MarketEnv(c);s=e.reset(company_ids=[f'company_{chr(65+i)}' for i in range(5)],episode_id=f'sensitivity-{seed}',episode_seed=seed,max_rounds=10,cooperation_mode='combined_v1');initial=s.to_dict();memory={};traces=[];revenue=0
                for _ in range(10):
                    aa,memory=actions(c,s,memory,seed);a=step(e,s,aa);audit_settlement(s.to_dict(),a.to_dict(),c.to_dict());revenue+=sum(x.financial.round_revenue_cents for x in a.companies);traces.append(dict(round=s.round,actions={k:v.to_dict() for k,v in aa.items()},state_hash=a.state_hash));s=a
                profit=sum(x.financial.cumulative_profit_cents for x in s.companies);row=dict(parameter=parameter,factor=factor,seed=seed,profit=profit,margin=profit/revenue if revenue else None,welfare=s.welfare_accounting.cumulative_total_economic_welfare_cents,checks=replay(c,MarketState.from_dict(initial),traces));rows.append(row);save(OUT/'sensitivity'/f'{parameter}-{factor}-{seed}.json.gz',dict(summary=row,config=c.to_dict(),initial_state=initial,transitions=traces))
    game=sequential(intercept=30,cost=6,maximum=24);tree=game['tree'];train=[r for r in tree if r['leader_quantity']%2==0];test=[r for r in tree if r['leader_quantity']%2==1]
    # Learn the continuous response from even quantities; integer ties use the documented follower-best-response set.
    x=np.array([[1,r['leader_quantity']] for r in train]);y=np.array([mean(r['follower_responses']) for r in train]);coef=np.linalg.lstsq(x,y,rcond=None)[0]
    errors=[abs(float(coef@[1,r['leader_quantity']])-mean(r['follower_responses'])) for r in test]
    write_json(OUT/'sensitivity.json',dict(passed=True,cases=len(rows),results=rows,calibrated=False,scope='3-seed one-parameter stress scan. Not fitted to external industry data. Operating margins are outputs, not directly tuned to incomparable Census after-tax margins.'))
    write_json(OUT/'sequential-response.json',dict(passed=max(errors)<1e-8,training='even leader quantities',holdout='odd leader quantities',coefficients=coef.tolist(),max_mean_best_response_error=max(errors),commitment_solutions=game['commitment_solutions'],scope='Exact linear Cournot/Stackelberg diagnostic; does not prove the market histogram response model correct.'))
    print('sensitivity 45 cases; sequential max error',max(errors),flush=True)
if __name__=='__main__':main()
