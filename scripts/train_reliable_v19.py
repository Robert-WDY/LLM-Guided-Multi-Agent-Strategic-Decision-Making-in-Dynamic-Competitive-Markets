"""Randomized synthetic training followed by untouched response-calibration seeds."""
import random
from reliable_v19_common import *

def collect(seeds):
    rows=[]
    for seed in seeds:
        n=(2,5,10)[seed%3];m=REGIMES[seed%len(REGIMES)];c,e,s=make(m,n,seed);memory={};previous=None;previous_service=0;pending=None;rng=random.Random(seed)
        for t in range(14):
            aa,memory=actions(c,s,memory,seed)
            if pending:
                for peer in s.company_ids[1:]:
                    rows.append(dict(seed=seed,round=s.round,peer=peer,features=pending[peer],change=aa[peer].price_cents/s.company(peer).commercial.price_cents-1))
            base=aa['company_A'];pool=local_candidates(c,s,'company_A',base,12);aa['company_A']=CompanyAction.from_dict(rng.choice(pool)['action'])
            a=step(e,s,aa);audit_settlement(s.to_dict(),a.to_dict(),c.to_dict());pending={peer:features(frame(s),frame(a),peer,aa['company_A'].service_budget_cents) for peer in s.company_ids[1:]};s=a
    return rows
def main():
    freeze_market();train=collect(SPEC['training_seeds']);model=fit(train);write_json(MODEL_PATH,model);save(OUT/'training-public-rows.json.gz',train)
    test=collect(SPEC['calibration_seeds']);result={}
    for mode in ('generic','conditional'):
        values=[]
        for r in test:
            p=predict(model,r['features'],mode);y=bucket(r['change'])+1;values.append(dict(seed=r['seed'],brier=sum((v-int(i==y))**2 for i,v in enumerate(p['probabilities'])),absolute_error=abs(sum(a*b for a,b in zip(p['probabilities'],p['changes']))-r['change']),probabilities=p['probabilities'],label=y))
        result[mode]=dict(brier=mean(x['brier'] for x in values),mae=mean(x['absolute_error'] for x in values),rows=values)
    save(OUT/'calibration-public-rows.json.gz',test);write_json(OUT/'response-calibration.json',dict(training_rows=len(train),test_rows=len(test),model_sha256=model['sha256'],results=result,conditional_no_worse=result['conditional']['brier']<=result['generic']['brier'],scope='Unseen seed predictive calibration for existing scripted opponents, not LLM response calibration. No private actor state in model rows.'))
    print({k:(v['brier'],v['mae']) for k,v in result.items()},flush=True)
if __name__=='__main__':main()
