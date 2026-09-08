"""Eight common seeds, company A treatment vs unchanged autonomous counterparts."""
import asyncio
import json
from pathlib import Path
from statistics import mean,stdev
from game_theory_agent.market import load_market_config
from game_theory_agent.market.actor_experiments import run_episode,write_json,read_json
from game_theory_agent.game_theory.market_strategies import MODES

ROOT=Path(__file__).resolve().parents[1]


async def main():
    config=load_market_config(ROOT/'configs/market_v14_local.yaml');dest=ROOT/'runs/theory-stage33';dest.mkdir(parents=True,exist_ok=True)
    seeds=list(range(330101,330109));variants=['rule',*MODES]
    write_json(dest/'preregistration.json',dict(seeds=seeds,variants=variants,rounds=20,focal='company_A',others='autonomous rule',paired_episode_ids=True,config_sha256=config.config_sha256,real_calls=0))
    rows=[]
    for variant in variants:
        for seed in seeds:
            path=dest/f'{variant}-{seed}'
            r=await run_episode(config,directory=path,seed=seed,rounds=20,modes={'company_A':variant})
            cp=read_json(path/'checkpoint.json');companies=cp['state']['companies']
            company=next(c for c in companies if c['company_id']=='company_A') if isinstance(companies,list) else companies['company_A']
            rows.append({k:v for k,v in r.items() if k!='memory'}|dict(variant=variant,focal_profit_cents=company['financial']['cumulative_profit_cents']))
        print(json.dumps(dict(completed_variant=variant,episodes=len(rows))),flush=True)
    baseline={r['seed']:r for r in rows if r['variant']=='rule'}
    groups=[]
    for variant in variants:
        group=[r for r in rows if r['variant']==variant]
        gains=[r['focal_profit_cents']-baseline[r['seed']]['focal_profit_cents'] for r in group]
        welfare=[r['welfare_cents']-baseline[r['seed']]['welfare_cents'] for r in group]
        groups.append(dict(variant=variant,n=len(group),mean_focal_profit_cents=mean(r['focal_profit_cents'] for r in group),paired_profit_gain_cents=mean(gains),paired_profit_sd_cents=stdev(gains),profit_wins=sum(g>0 for g in gains),paired_welfare_gain_cents=mean(welfare)))
    report=dict(passed=all(r['replay_passed'] for r in rows),episodes=len(rows),rounds=len(rows)*20,groups=groups,results=rows,scope='One focal company, eight fixed seeds, same initial state and random stream; public forecast is approximate and one-step. No dynamic Nash or universal superiority claim.')
    write_json(dest/'summary.json',report);print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)


if __name__=='__main__':asyncio.run(main())
