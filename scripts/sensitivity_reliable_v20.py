"""Offline component intervention after the 2x2 diagnosis; no production truth."""
from diagnose_reliable_v20 import *

def case(row):
    p=DEST/'state-sensitivity-extended'/(row['case_id']+'.json.gz')
    if p.exists():return read(p)['summary']
    c,s,memory,_,_=source(row['market'],row['companies'],row['seed']);public,_=forecast(c,s,'company_A')
    saved=read(OUT/'advice'/(row['case_id']+'.json.gz'));raw=saved['draft'];selected=saved['details']['conditional']['action'];req=ReliableRequest(draft_action=raw,goal=row['goal'],mode='research',seed=row['seed'],public_history=saved['history']);model=load_model()
    ref=read(DEST/'diagnosis'/(row['case_id']+'.json.gz'));target=mean(x['gain'] for x in ref['groups']['D'][:4]);details={}
    for field in ('financial','operations','brand','risk','history','market','supply_cash'):
        def corrected(x):
            if x.company_id=='company_A':return x
            actual=s.company(x.company_id)
            # Cash-only avoids importing realized financial accounting without its matching sales/cost ledger.
            if field in ('market','supply_cash'):return x
            value=replace(x.financial,cash_balance_cents=actual.financial.cash_balance_cents) if field=='financial' else replace(x.operations,base_capacity_orders=actual.operations.base_capacity_orders,effective_capacity_orders=actual.operations.effective_capacity_orders,capacity_utilization_ppm=min(1000000,x.commercial.sales_orders*1000000//max(1,actual.operations.effective_capacity_orders))) if field=='operations' else getattr(actual,field)
            return replace(x,**{field:value})
        peers=tuple(corrected(x) for x in public.companies)
        initial=replace(public,companies=peers)
        if field=='market':initial=replace(initial,market=s.market)
        if field=='supply_cash':initial=replace(initial,supply_chain=replace(public.supply_chain,suppliers=tuple(replace(x,account=replace(x.account,cash_cents=s.supply_chain.supplier(x.supplier_id).account.cash_cents)) for x in public.supply_chain.suppliers)))
        initial=replace(initial,state_hash=state_hash(initial.to_dict()));gains=[]
        for scenario in range(2000,2004):
            seed=int(hashlib.sha256(f'v20-diagnostic:{row["seed"]}:{scenario}'.encode()).hexdigest()[:15],16)
            b=simulate(c,initial,memory,row['seed'],req,model,raw,raw,scenario,5,'model',seed)
            a=simulate(c,initial,memory,row['seed'],req,model,selected,raw,scenario,5,'model',seed)
            gains.append(a[row['goal']]-b[row['goal']])
        details[field]=dict(gains=gains,error=mean(gains)-target)
    summary=dict(case_id=row['case_id'],goal=row['goal'],original_error=mean(x['gain'] for x in ref['groups']['A'][:4])-target,errors={k:v['error'] for k,v in details.items()})
    save(p,dict(summary=summary,details=details));return summary

if __name__=='__main__':
    rows=read_json(DEST/'diagnostic-selection.json')['cases'];results=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for r in pool.map(case,rows):
            results.append(r);print('state sensitivity',len(results),'/',len(rows),flush=True)
    write_json(DEST/'state-sensitivity-extended.json',dict(results=results,mean_absolute_error_cents={k:mean(abs(r['original_error'] if k=='original' else r['errors'][k]) for r in results) for k in ('original','financial','operations','brand','risk','history','market','supply_cash')}))
