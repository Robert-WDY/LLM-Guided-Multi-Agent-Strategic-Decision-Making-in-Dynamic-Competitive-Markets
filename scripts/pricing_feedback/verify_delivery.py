"""Package paired episodes, validate end-to-end real branches, and inventory raw evidence."""
from pathlib import Path
import hashlib
from collections import Counter
import sqlite3
from study import RAW,OUT,ROOT,read,save
from paid import connect, LIMIT
from prompt_audit import check


def main():
    audit=read(OUT/'audit.json')
    assert not check()['affected']
    before=read(OUT/'prompt-audit.json')
    assert before['after']['multi_draft_histories']==0
    compact=[];once=[];counts=Counter()
    for pattern,stage in [('long-confirm-*.gz','terminal'),('units-*.gz','units'),('real2-*.gz','real'),
                          ('realunits-*.gz','real_units'),('once-confirm-*.gz','real_once')]:
        for f in sorted(RAW.glob(pattern)):
            data=read(f)
            if stage=='real_once':
                v=data['labels'][0];t=v['t'];baseline=data['rows'][t-1:];alt=data['alternate']
                assert len(alt)==len(baseline)==61-t
                delta=sum(.99**i*(x[3]-y[3]) for i,(x,y) in enumerate(zip(alt,baseline)))
                assert abs(delta-v['delta'])<1.e-6
                # A binding price floor can lift on a later round of the same contract.
                if alt[:4]==baseline[:4]:assert alt==baseline
                once.append(dict(market=data['market'],provider=data['provider'],seed=data['seed'],condition=data['condition'],first_price_changed=alt[0][0]!=baseline[0][0],
                                 term_price_changed=any(x[0]!=y[0] for x,y in zip(alt[:4],baseline[:4])),**v))
                counts[stage]+=2
                continue
            for r in data if isinstance(data,list) else [data]:
                assert len(r['rows'])==60
                compact.append(dict(stage=stage,seed=r['seed'],market=r['market'],condition=r['condition'],
                      arm=r['arm'],provider=f.name.split('-')[1] if stage.startswith('real') else None,
                      profit=sum(.99**i*x[3] for i,x in enumerate(r['rows']))/(100 if r['market']=='native' else 1)))
                counts[stage]+=1
    assert dict(counts)==dict(terminal=3072,units=1920,real=512,real_units=256,real_once=384)
    assert len(once)==192
    save(OUT/'additional-episodes.json.gz',compact);save(OUT/'real-once-opportunities.json.gz',once)
    with connect() as db:
        rows=db.execute('select provider,status,count(*),coalesce(sum(input),0),coalesce(sum(output),0),sum(cost) from calls group by provider,status').fetchall()
    reserved=sum(x[5] for x in rows)
    assert reserved<=LIMIT and all(r[1]!='reserved' for r in rows)
    save(OUT/'delivery-audit.json',dict(secondary_episodes=dict(counts),first_confirmation_rounds=audit['rounds'],
         total_attempts=sum(r[2] for r in rows),input_tokens=sum(r[3] for r in rows),
         output_tokens=sum(r[4] for r in rows),conservative_yuan=reserved/1.e6,
         model_drafts_after_reconciliation=before['after'],ledger_columns=['provider','status','attempts','input','output','micro_yuan'],
         ledger=rows))
    paths=[*sorted(RAW.rglob('*.gz')),*sorted(OUT.iterdir()),*sorted((ROOT/'experiment-specs/pricing-feedback-v2').glob('*.md')),
           *sorted((ROOT/'scripts/pricing_feedback').glob('*.py')),ROOT/'src/game_theory_agent/game_theory/cycle_pricing.py']
    save(OUT/'manifest.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
                              if p.is_file() and p.name!='manifest.json'})


if __name__=='__main__':main()

