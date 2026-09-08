"""Retain exact states reproducing pre-fix audit/exit defects and patched outcomes."""
from evaluate_market_v18 import *

def main():
 records=[]
 c,e,s=initial('recession',2,480001,5)
 for _ in range(5):
  aa,extra,_=population(c,s,'mutual_aid',{});a=step(e,s,aa,extra)
  if a.to_dict()['strategic_market']['mutual_aid']['last_transfers']:
   x=a.company('company_B');assert x.financial.round_revenue_cents!=x.commercial.price_cents*x.commercial.sales_orders
   checks=audit_settlement(s.to_dict(),a.to_dict(),c.to_dict());records.append(dict(defect='audit_omitted_aid_fee',before=s.to_dict(),after=a.to_dict(),config=c.to_dict(),patched_checks=checks));break
  s=a
 c,e,s=initial('normal',5,481001,20)
 for _ in range(20):
  aa,extra,_=population(c,s,'coordination',{});a=step(e,s,aa,extra)
  if a.strategic_market.active_company_count<s.strategic_market.active_company_count:
   checks=audit_settlement(s.to_dict(),a.to_dict(),c.to_dict());records.append(dict(defect='audit_omitted_liquidation',before=s.to_dict(),after=a.to_dict(),config=c.to_dict(),patched_checks=checks))
   break
  s=a
 from game_theory_agent.market.actor_policies import company_action,COMPANY_OPTIONS
 v=read_json(OUT/'exit-guard-reproduction.json');assert v['reproduced']
 c=MarketConfig.from_mapping(v['config']);s=MarketState.from_dict(v['state']);cid=v['company'];frozen=company_action(c,s,cid,'balanced');old_rejected=False
 # Even a balanced request re-enters the active-company price-floor resolver.
 # Preserve the actual failed input, rather than inventing a changed price.
 try:resolve_action_request(c,s,cid,frozen.to_dict(),source='four-actor-portfolio',action_id=f'portfolio:{s.round}:{cid}:balanced')
 except Exception as exc:old_rejected=True;error=str(exc)
 assert old_rejected and 'frozen price' in error
 assert all(company_action(c,s,cid,o)==frozen for o in COMPANY_OPTIONS)
 records.append(dict(defect='exited_portfolio_reentered_active_price_guard',source=s.to_dict(),config=c.to_dict(),old_request=frozen.to_dict(),old_error=error,patched_action=frozen.to_dict(),episode=v['episode'],round=s.round))
 assert len(records)==3
 write_json(OUT/'regression-reproductions.json',dict(passed=True,scope='Exact recreated synthetic inputs reproduce the old formula/portfolio failures; current code passes. Not deletion or reinterpretation of the original failed runs.',records=records));print('three exact regression reproductions passed')

if __name__=='__main__':main()
