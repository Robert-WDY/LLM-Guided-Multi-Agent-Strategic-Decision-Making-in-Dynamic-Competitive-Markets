"""Standalone arithmetic oracle: imports no market engine or its validators."""
import random
from statistics import mean
from fractions import Fraction

def audit_settlement(before,after,config=None):
 checks=0
 transfers=after['strategic_market']['mutual_aid']['last_transfers'];aid_income={cid:0 for cid in after['companies']}
 for transfer in transfers:
  assert transfer['total_transfer_fee_cents']==transfer['fulfilled_orders']*transfer['fee_per_order_cents'],'mutual aid invoice'
  aid_income[transfer['donor_company_id']]+=transfer['total_transfer_fee_cents'];checks+=1
 for cid,c in after['companies'].items():
  old=before['companies'][cid];f=c['financial'];commercial=c['commercial']
  previous=next(x for x in before['strategic_market']['company_lifecycle'] if x['company_id']==cid)
  current=next(x for x in after['strategic_market']['company_lifecycle'] if x['company_id']==cid)
  proceeds=0
  if previous['status']!='exited' and current['status']=='exited':
   assert config is not None,'configuration required to audit new liquidation'
   # Independent rational rounding: depreciation uses ties-to-even, liquidation ties-up.
   book=round(Fraction(old['financial']['capacity_book_value_cents']*(1000000-config['capacity']['book_value_depreciation_ppm']),1000000))+after['last_joint_action'][cid]['capacity_investment_cents']
   proceeds=(book*config['strategic_market']['liquidation_recovery_ppm']+500000)//1000000
   assert f['capacity_book_value_cents']==0,'liquidated asset book';checks+=1
  assert f['cash_balance_cents']-old['financial']['cash_balance_cents']==f['round_profit_cents']+proceeds,'company cash including liquidation'
  assert f['round_revenue_cents']==commercial['price_cents']*commercial['sales_orders']+aid_income[cid],'sales revenue plus mutual aid income'
  assert f['cumulative_profit_cents']-old['financial']['cumulative_profit_cents']==f['round_profit_cents'],'profit accumulation';checks+=3
 supply=after['supply_chain'];receipts={a:0 for a in supply['suppliers']}
 for o in supply['last_procurement_outcomes'].values():
  m=o['material'];assert m['used_orders']+m['wasted_orders']==o['fulfilled_orders'],'physical material'
  assert sum(m['payments_by_supplier'].values())==m['payment_cents']<=m['budget_cents'],'invoice budget'
  for supplier,quantity in o['supplier_allocation_orders'].items():
   s=supply['suppliers'][supplier];ledger=s.get('strategic_ledger');price=ledger['audit']['settlement_prices_cents'][o['company_id']] if ledger else s['account']['settlement_price_cents']
   assert m['payments_by_supplier'].get(supplier,0)==quantity*price,'invoice price times quantity';checks+=1
  for a,v in m['payments_by_supplier'].items():receipts[a]+=v
  checks+=2
 for a,s in supply['suppliers'].items():assert receipts[a]==s['account']['receipts_cents'],'supplier receipts';checks+=1
 w=after['welfare_accounting'];old=before['welfare_accounting'];profits=sum(c['financial']['round_profit_cents'] for c in after['companies'].values());assert profits==w['round_downstream_producer_surplus_cents'],'downstream surplus'
 expected=profits+w['round_consumer_surplus_cents']+w['round_upstream_producer_surplus_cents']+w['round_government_net_budget_cents']-w['round_stockout_externality_cents']-w['round_business_exit_externality_cents']
 assert expected==w['round_total_economic_welfare_cents'],'welfare sum'
 assert w['cumulative_total_economic_welfare_cents']-old['cumulative_total_economic_welfare_cents']==expected,'welfare accumulation'
 return checks+3

def paired_summary(values,seed=379900,draws=2000):
 if not values:return dict(n=0,mean=None,interval=None)
 rng=random.Random(seed);n=len(values);samples=sorted(mean(rng.choices(values,k=n)) for _ in range(draws))
 return dict(n=n,mean=mean(values),interval=[samples[int(.025*draws)],samples[int(.975*draws)]],negative=sum(v<0 for v in values),zero=sum(v==0 for v in values),positive=sum(v>0 for v in values),worst=min(values),scope='按独立种子重采样的95%百分位bootstrap区间；只对指定合成环境，探索性比较未校正多重检验。')
