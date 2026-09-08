"""Report all paired holdout contrasts without tuning or removing adverse cases."""
import json
from pathlib import Path
from statistics import mean,median
from game_theory_agent.market.actor_experiments import read_json,write_json
from game_theory_agent.game_theory.advisor_audit import paired_summary
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'runs/advisor-v17'

def main():
 stages=[read_json(DEST/f'stage{i}.json') for i in range(1,6)];assert all(s['passed'] for s in stages)
 def selected_action(row):
  r=read_json(DEST/'holdout'/f"{row['seed']}-{row['pressure']}-{row['variant']}.json");return {k:v for k,v in r['action'].items() if k not in ('action_id','strategy_summary')}
 rows=stages[4]['results'];index={(r['seed'],r['pressure'],r['variant']):r for r in rows};assert len(index)==len(rows)==232
 contrasts={};pressures={}
 for goal in ('profit','welfare'):
  for pressure in ('normal','supplier','government'):
   group=[r for r in rows if r['goal']==goal and r['pressure']==pressure and r['variant']=='v17'];assert len(group)==16
   for comparator in ('v16','strong','no_response','no_learning'):
    paired=[(r,index[(r['seed'],pressure,comparator)]) for r in group if (r['seed'],pressure,comparator) in index]
    if not paired:continue
    contrasts[f'{goal}-{pressure}-minus-{comparator}']=dict(utility=paired_summary([a['gain']-b['gain'] for a,b in paired]),profit_cents=paired_summary([a['profit_gain']-b['profit_gain'] for a,b in paired]),welfare_cents=paired_summary([a['welfare_gain']-b['welfare_gain'] for a,b in paired]),same_executed_action=sum(selected_action(a)==selected_action(b) for a,b in paired),median_runtime_ratio=median(b['elapsed']/max(.000001,a['elapsed']) for a,b in paired))
   if pressure!='normal':
    pressures[f'{goal}-{pressure}-minus-normal']=paired_summary([r['gain']-index[(r['seed'],'normal','v17')]['gain'] for r in group])
 variants={}
 for variant in ('v16','v17','strong','no_response','no_learning'):
  group=[r for r in rows if r['variant']==variant]
  variants[variant]=dict(cases=len(group),negative_gain_cases=sum(r['gain']<0 for r in group),zero_gain_cases=sum(r['gain']==0 for r in group),positive_gain_cases=sum(r['gain']>0 for r in group),negative_profit_gain_cases=sum(r['profit_gain']<0 for r in group),negative_welfare_gain_cases=sum(r['welfare_gain']<0 for r in group),minimum_cash=min(r['minimum_cash'] for r in group),median_runtime=median(r['elapsed'] for r in group),max_runtime=max(r['elapsed'] for r in group),complete_responses=sum(r['response_complete'] is True for r in group),mean_profit_relative_error=mean(r['relative_error'] for r in group),mean_choice_stability=mean(r['sensitivity']['same_choice_fraction'] for r in group if r['sensitivity']) if variant!='v16' else None)
 multiplayer=[]
 for r in stages[2]['results']:
  if r['mode']!='robust_profit':continue
  for comparator in ('rule','advisor_profit'):
   b=next(b for b in stages[2]['results'] if (b['companies'],b['seed'],b['rounds'],b['mode'])==(r['companies'],r['seed'],r['rounds'],comparator))
   multiplayer.append(dict(companies=r['companies'],seed=r['seed'],rounds=r['rounds'],comparator=comparator,profit_delta=r['profit']-b['profit'],welfare_delta=r['welfare']-b['welfare'],volatility_delta=r['price_volatility']-b['price_volatility'],exits=r['closed_companies']))
 adverse=[{k:r[k] for k in ('seed','goal','pressure','variant','selected','gain','profit_gain','welfare_gain')} for r in rows if r['gain']<0 or r['profit_gain']<0 or r['welfare_gain']<0]
 report=dict(passed=True,variants=variants,paired_contrasts=contrasts,matched_pressures=pressures,multiplayer=multiplayer,adverse_cases=adverse,arithmetic_checks=sum(r['arithmetic_checks'] for r in rows)+sum(r['arithmetic_checks'] for r in stages[2]['results']),scope='32 independent seeds total, 16 per goal; pressure variants are paired repeats, not 96 independent seeds. Strong/ablation groups have only four seeds per goal. Bootstrap intervals are exploratory and unadjusted for multiple comparisons. Runtime under four concurrent workers is not an isolated latency benchmark. All losses retained; no holdout-based retuning.')
 write_json(DEST/'analysis.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('adverse_cases','paired_contrasts','multiplayer')}))

if __name__=='__main__':main()
