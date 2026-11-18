"""Invoke upstream's actual batch episode runner with a provider/trace transport."""
import ast
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from common import OUT, RAW, market, read, save
from pricing import instance
from econ_evals.experiments.pricing import run_pricing_experiment as runner


def check(mode,n,seed):
    args=instance(seed,n)
    reference=read(RAW/f'pricing-feedback-{n}-{seed}.json.gz')
    period=0
    def call_llm(**request):
        nonlocal period
        messages=request['messages']
        if len(messages)==1:
            name='get_previous_pricing_data'; arguments={}
        else:
            name='set_prices'
            if mode=='real':
                from provider import decision
                hi=float(messages[0]['content'].split('above ')[-1].strip().rstrip('.'))
                history=messages[-1]['content'][0]['content']
                assert str(hi) in messages[0]['content']
                prompt=request['system']+'\n'+messages[0]['content']+'\n'+history+'\nOnly provide the numeric price, between 0 and '+str(hi)+'.'
                prices=[decision('deepseek',f'official:{seed}:{period}',prompt,0,hi)]
            else:
                hi=reference[period]['hint']
                prices=[hi*.48]*n
                if period:
                    best=max(reference[max(0,period-12):period],key=lambda r:sum(r['profit']))
                    prices=list(best['p'])
                    j=period%n
                    prices[j]=min(hi,max(.01,prices[j]+hi*(.06 if (period//n)%2 else -.06)))
            arguments={'prices_dict_str':str(dict(zip(args.product_ids,prices)))}
            period+=1
        completion={'content':[{'type':'tool_use','id':'tool-'+str(len(messages)),'name':name,'input':arguments}]}
        return {'model':mode}, {}, completion
    with tempfile.TemporaryDirectory(dir=RAW) as work:
        base=Path(work)
        with patch.object(runner,'call_llm',call_llm),patch.object(runner,'get_base_dir_path',lambda:base):
            runner.run(args,str(n),'stage')
        log=next(base.rglob('logs.csv'))
        df=pd.read_csv(log).dropna(subset=['prices'])
        assert len(df)==period==100
        values=[sum(ast.literal_eval(v).values()) for v in df['profits']]
        # Feedback transport sends prices before rounding; recompute below against logged actual prices.
        if mode=='real':
            rows=df[['attempt_num','prices','quantities','profits']].to_dict('records')
            save(RAW/f'official-real-{seed}.json.gz',rows)
        differences=[abs(x-sum(y['profit'])) for x,y in zip(values,reference)]
        if mode=='replay':assert max(differences)<1e-8
        ratio=sum(values[50:])/sum(r['opt'] for r in reference[50:])
        return dict(seed=seed,products=n,mode=mode,periods=period,score=ratio,
                    max_profit_difference=max(differences) if mode=='replay' else None)


if __name__=='__main__':
    mode=sys.argv[1]
    cases=[(1,0)] if mode=='real' else [(n,s) for n in (1,4,10) for s in (0,1)]
    rows=[check(mode,n,s) for n,s in cases]
    save(OUT/f'official-{mode}.json',rows)
    print(rows,flush=True)

