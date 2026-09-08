"""Predeclared zero-token concept experiments, stage by stage."""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from game_theory_agent.game_theory.lab import GAMES,STRATEGIES,analyze,repeated,learning,sequential,information,bargaining,public_goods
from game_theory_agent.market.actor_experiments import write_json

ROOT=Path(__file__).resolve().parents[1]


def run(stage):
    dest=ROOT/'runs'/f'theory-stage{stage}';dest.mkdir(parents=True,exist_ok=True)
    summary=dest/'summary.json'
    if summary.exists():return json.loads(summary.read_text(encoding='utf-8'))
    seeds=list(range(290101,290109))
    write_json(dest/'preregistration.json',dict(stage=stage,seeds=seeds,repeated_rounds=60,noise=[0,.1],learning_rounds=3000,real_calls=0,selection='all listed combinations; no best-seed selection'))
    results=[]
    if stage==29:
        results=[dict(game=g,**analyze(v['payoffs'])) for g,v in GAMES.items()]
        assert results[0]['pure_nash']==[[1,1]] and results[-1]['interior_mixed_nash'][0]['exact']==['1/2','1/2']
        report=dict(stage=stage,passed=True,conditions=4,conclusion='A Nash outcome need not be Pareto efficient; matching pennies requires mixing.')
    elif stage==30:
        for noise in (0,.1):
            for row in STRATEGIES:
                for column in STRATEGIES:
                    for seed in seeds:
                        r=repeated(row=row,column=column,noise=noise,rounds=60,seed=seed)
                        write_json(dest/f'{noise}-{row}-{column}-{seed}.json',r)
                        results.append({k:v for k,v in r.items() if k!='history'})
        groups=[dict(noise=n,strategy=s,n=sum(r['noise']==n and r['row']==s for r in results),mean_payoff=mean(r['average_payoffs'][0] for r in results if r['noise']==n and r['row']==s),cooperation=mean(r['cooperation_rate'][0] for r in results if r['noise']==n and r['row']==s)) for n in (0,.1) for s in STRATEGIES]
        report=dict(stage=stage,passed=True,episodes=len(results),rounds=len(results)*60,groups=groups,conclusion='Tournament scores depend on this opponent population and noise; not an equilibrium certificate.')
    elif stage==31:
        for game in GAMES:
            for algorithm in ('regret_matching','fictitious_play'):
                for seed in seeds:
                    r=learning(game=game,algorithm=algorithm,rounds=3000,seed=seed)
                    assert max(abs(a-b) for a,b in zip(r['cce_deviation_gains'],r['history'][-1]['external_regret_per_round']))<1e-8
                    results.append(r)
        groups=[dict(game=g,algorithm=a,n=8,mean_cce_gap=mean(r['cce_gap'] for r in results if r['game']==g and r['algorithm']==a),maximum_cce_gap=max(r['cce_gap'] for r in results if r['game']==g and r['algorithm']==a)) for g in GAMES for a in ('regret_matching','fictitious_play')]
        report=dict(stage=stage,passed=True,runs=len(results),rounds=len(results)*3000,groups=groups,conclusion='Report measured external regret/CCE gaps separately from marginal Nash gaps; no universal convergence claim.')
    elif stage==32:
        for p in range(11):
            for a in range(5,11):
                r=information(weak_prior=p/10,accuracy=a/10);assert -.000001<=r['value_of_information']<=r['perfect_information_bound']+.000001;results.append(r)
        for cost in (3,6,9):results.append(sequential(cost=cost))
        for w in (.25,.5,.75):
            for d in ((20,10),(0,0),(80,30)):results.append(bargaining(weight=w,disagreement_row=d[0],disagreement_column=d[1]))
        for n in (2,4,8):results.append(public_goods(players=n))
        assert bargaining()['solutions'][0]['allocation']==[55,45]
        report=dict(stage=stage,passed=True,conditions=len(results),default_stackelberg=sequential()['commitment_solutions'],default_information=information(),conclusion='Commitment, credible signals and disagreement options change incentives under stated assumptions; transfers do not create surplus.')
    else:raise ValueError('stage must be 29–32')
    write_json(dest/'results.json',results);write_json(summary,report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=int,required=True,choices=[29,30,31,32]);args=parser.parse_args()
    print(json.dumps(run(args.stage),ensure_ascii=True))
