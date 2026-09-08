"""Read-only replay to locate a concrete pre-guard portfolio rejection."""
import inspect
from evaluate_market_v18 import *
from game_theory_agent.market import actor_policies

def main():
 source=inspect.getsource(actor_policies.company_action)
 guard='    if cid not in state.strategic_market.active_company_ids:\n        return base\n'
 assert guard in source
 scope=dict(vars(actor_policies));exec(source.replace(guard,'').replace('def company_action(', 'def old_action('),scope)
 old=scope['old_action'];checked=0
 for row in read_json(OUT/'tournament.json')['results']:
  if not row['exits']:continue
  key=f"{row['market']}-{row['companies']}-{row['seed']}-{row['rounds']}-{row['policy']}"
  v=read_compressed(OUT/'episodes'/(key+'.json.gz'));c=MarketConfig.from_mapping(v['config']);s=MarketState.from_dict(v['initial_state']);e=MarketEnv(c);e.load_state(s)
  for t in v['transitions']:
   for cid in s.company_ids:
    if cid in s.strategic_market.active_company_ids:continue
    for option in actor_policies.COMPANY_OPTIONS:
     checked+=1
     try:old(c,s,cid,option)
     except Exception as exc:
      write_json(OUT/'exit-guard-reproduction.json',dict(reproduced=True,episode=key,round=s.round,company=cid,option=option,old_error=str(exc),config=c.to_dict(),state=s.to_dict(),patched=actor_policies.company_action(c,s,cid,option).to_dict(),checks=checked))
      print(key,s.round,cid,option,str(exc),flush=True);return
   s=step(e,s,{a:CompanyAction.from_dict(x) for a,x in t['actions'].items()},t['extra'])
 print('No rejection reproduced',checked)
 write_json(OUT/'exit-guard-reproduction.json',dict(reproduced=False,checks=checked,scope='No rejection found on stored post-fix trajectories; do not claim exact original failure reproduced.'))

if __name__=='__main__':main()
