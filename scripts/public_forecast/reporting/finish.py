"""Final evidence inventory including both prospectively scored real cohorts."""
import hashlib
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sqlite3
import numpy as np
from collect import OUT,RAW,ROOT,cases,save,read


def main():
    extra={};fallbacks={}
    for m in ('econ','native'):
        earlier=cases('rtest',m);summary=[]
        for r in earlier:
            j=r['opportunity']['t']-1;before=r['history'][j];alt=r['alternative'][0]
            summary.append((abs(before[0]-alt[0])<1.e-8,abs(before[2]-alt[2])<1.e-8,before[5]>0))
        extra[m]=dict(equal_executed_price=sum(x for x,y,z in summary),
                      changed_price_equal_sales=sum(not x and y for x,y,z in summary),
                      changed_price_equal_sales_with_market_stockout=sum(not x and y and z for x,y,z in summary))
        for phase in ('rtrain','rcal','rtest','offer'):
            rows=cases(phase,m)
            fallbacks[f'{phase}-{m}']=sum(len(r['fallbacks']) for r in rows)
    dr=cases('direction','econ')
    assert len(dr)==80 and len(list((RAW/'direction-prospective').glob('*.json')))==80
    for r in dr:
        op=r['opportunity'];j=op['t']-1
        assert op==read(RAW/'direction-prospective'/f"{r['provider']}-{r['seed']}.json")
        assert len(r['history'])==60
        for key,factor in [('up',1.04),('down',.96)]:
            assert len(r['branches'][key])==60-j
            low,high=r['history'][j][7:9]
            np.testing.assert_allclose(np.asarray(r['branches'][key])[:4,6],np.clip(op['draft']*factor,low,high))
    fallbacks['direction-econ']=sum(len(r['fallbacks']) for r in dr)
    sig=read(OUT/'direction-freeze.json')
    assert sig['code']==hashlib.sha256((ROOT/'scripts/public_forecast/reporting/direction.py').read_bytes()).hexdigest()
    assert sig['design']==hashlib.sha256((ROOT/'experiment-specs/public-forecast-v1/followup/DIRECTION.md').read_bytes()).hexdigest()
    initial=read(OUT/'freeze.json');follow=read(OUT/'offer-freeze.json')
    for name,digest in initial['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert follow['code']==hashlib.sha256((ROOT/'scripts/public_forecast/reporting/challenger.py').read_bytes()).hexdigest()
    assert follow['design']==hashlib.sha256((ROOT/'experiment-specs/public-forecast-v1/followup/FORECAST-OFFER.md').read_bytes()).hexdigest()
    assert len(list((RAW/'prospective').glob('*.json')))==192 and len(list((RAW/'offer-prospective').glob('*.json')))==128
    with sqlite3.connect(ROOT/'runs/pricing-feedback-v2/model-ledger.sqlite') as db:
        original=db.execute('select sum(cost) from calls where id<=12460').fetchone()[0]
        ledger=db.execute('select provider,status,count(*),sum(cost),sum(input),sum(output) from calls where id>12460 group by provider,status').fetchall()
    assert original==23808940 and all(v[1]!='reserved' for v in ledger)
    cost=sum(v[3] for v in ledger)
    assert cost+original<=50000000
    save(OUT/'delivery.json',dict(prior_yuan=original/1e6,additional_yuan=cost/1e6,total_yuan=(original+cost)/1e6,
         remaining_yuan=(50000000-original-cost)/1e6,ledger_columns=['provider','status','attempts','micro_yuan','input','output'],
         rows=ledger,attempts=sum(v[2] for v in ledger),valid=sum(v[2] for v in ledger if v[1]=='ok'),
         input_tokens=sum(v[4] or 0 for v in ledger),output_tokens=sum(v[5] or 0 for v in ledger),fallback_events=fallbacks,
         submitted_heldout_before_settlement=400,marginal_diagnostics=extra))
    paths=[*sorted(RAW.rglob('*.gz')),*sorted((RAW/'prospective').glob('*.json')),*sorted((RAW/'offer-prospective').glob('*.json')),*sorted((RAW/'direction-prospective').glob('*.json')),
           *sorted(OUT.iterdir()),*sorted((ROOT/'experiment-specs/public-forecast-v1').rglob('*.md')),
           *sorted((ROOT/'scripts/public_forecast').rglob('*.py')),ROOT/'src/game_theory_agent/game_theory/public_forecast.py',
           ROOT/'tests/test_public_forecast.py']
    save(OUT/'manifest.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in paths if p.is_file() and p.name!='manifest.json'})


if __name__=='__main__':main()

