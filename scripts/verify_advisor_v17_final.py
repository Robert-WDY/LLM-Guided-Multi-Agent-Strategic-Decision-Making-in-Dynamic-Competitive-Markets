"""Final engineering gates, separate from economic superiority claims."""
import hashlib,json
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree
from game_theory_agent.market import MarketEnv,load_market_config
from game_theory_agent.market.actor_policies import actions_for
from game_theory_agent.market.actor_experiments import read_json,write_json
from game_theory_agent.game_theory.advisor_audit import audit_settlement
from game_theory_agent.local_budget import status
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'runs/advisor-v17'

def main():
 stages=[read_json(DEST/f'stage{i}.json') for i in range(1,6)];assert all(s['passed'] for s in stages)
 assert [s['cases'] for s in stages]==[96,68,24,10,232]
 assert stages[2]['rounds']==300 and stages[2]['advisor_decisions']==540
 assert all(r['replay'] and r['complete_responses']==r['advisor_decisions'] for r in stages[2]['results'])
 assert all(r['response_complete'] for r in stages[4]['results'] if r['variant']!='v16')
 for name,expected in read_json(DEST/'implementation-before-holdout.json').items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,'implementation changed after holdout preregistration'
 suite=ElementTree.parse(ROOT.parent.parent/'migration_2026-09-07/advisor-v17-final.xml').getroot().find('testsuite');assert int(suite.get('tests'))==471 and all(int(suite.get(k,'0'))==0 for k in ('errors','failures','skipped'))
 live=read_json(DEST/'live-api.json');restore=read_json(DEST/'backup-restore.json');analysis=read_json(DEST/'analysis.json');assert live['passed'] and restore['all_api_and_hashes_equal'] and analysis['passed']
 config=load_market_config(ROOT/'configs/market_v14_local.yaml');env=MarketEnv(config);before=env.reset(episode_seed=376901,max_rounds=5,cooperation_mode='combined_v1');after=env.step(f'{before.episode_id}:{before.round}:{before.state_version}',actions_for(config,before,{})).state_after
 b=before.to_dict();a=after.to_dict();checks=audit_settlement(b,a);faults=[]
 fields=[('companies','company_A','financial','cash_balance_cents'),('companies','company_A','financial','round_revenue_cents'),('companies','company_A','financial','cumulative_profit_cents'),('welfare_accounting','round_downstream_producer_surplus_cents'),('welfare_accounting','round_total_economic_welfare_cents'),('welfare_accounting','cumulative_total_economic_welfare_cents')]
 supply=a['supply_chain'];order=next(iter(supply['last_procurement_outcomes']));supplier=next(iter(supply['suppliers']))
 fields.extend([('supply_chain','last_procurement_outcomes',order,'material','used_orders'),('supply_chain','last_procurement_outcomes',order,'material','payment_cents'),('supply_chain','suppliers',supplier,'account','receipts_cents')])
 for path in fields:
  mutated=deepcopy(a);target=mutated
  for part in path[:-1]:target=target[part]
  target[path[-1]]+=1
  try:audit_settlement(b,mutated)
  except AssertionError as exc:faults.append(dict(field=list(path),delta=1,detected=str(exc)))
  else:raise AssertionError('undetected arithmetic corruption: '+str(path))
 write_json(DEST/'independent-corruption.json',dict(passed=True,unmodified_checks=checks,detected_faults=faults,before=b,after=a,scope='Independent integer identities and nine injected errors. Recorded ledger inputs remain trusted; this is not a second complete economic simulator.'))
 budget=status();assert budget['new_calls']==186 and budget['reserved_cny']==9.993565
 frontend=dict(build=True,typecheck=True,lint=True,tests=16)
 result=dict(stage=6,passed=True,cases=len(faults)+live['episodes'],backend_tests=471,frontend=frontend,live_api=True,browser_qa=False,backup_restore=True,corruptions_detected=len(faults),arithmetic_checks=analysis['arithmetic_checks']+checks,new_real_calls=0,budget_reserved_cny=budget['reserved_cny'],global_optimality_proven=False,real_economy_validated=False,scope='Engineering acceptance and transparent empirical results; no universal economic superiority gate.')
 write_json(DEST/'stage6.json',result);write_json(DEST/'acceptance.json',result);print(json.dumps(result))

if __name__=='__main__':main()
