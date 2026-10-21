"""Independent saved-action replay and accounting audit for both repair attempts."""
from evaluate_reliable_v201 import *

def audit_file(path):
    v=read(path);c=MarketConfig.from_mapping(v['config']);s=MarketState.from_dict(v['source']);checks=steps=0
    if 'branches' not in v:
        for label in ('actual','baseline'):
            record=v[label];e=MarketEnv(c);e.load_state(s);before=s;profit=welfare=0.
            for t,row in enumerate(record['transitions']):
                after=step(e,before,{k:CompanyAction.from_dict(a) for k,a in row['actions'].items()})
                assert after.state_hash==row['state_hash']
                checks+=audit_settlement(before.to_dict(),after.to_dict(),c.to_dict());steps+=1
                profit+=.95**t*after.company('company_A').financial.round_profit_cents;welfare+=.95**t*after.welfare_accounting.round_total_economic_welfare_cents
                m=record['metrics'][str(t+1)];assert abs(m['profit']-profit)<1e-5 and abs(m['welfare']-welfare)<1e-5
                before=after
        return dict(file=str(path.relative_to(ROOT)),steps=steps,checks=checks)
    first=[]
    for label,branch in v['branches'].items():
        e=MarketEnv(c);e.load_state(s);before=s;mem=deepcopy(v['initial_memory']);total=0.;count=0
        for t,row in enumerate(branch['trace']):
            fresh,mem=actions(c,before,mem,v['summary']['seed']);assert fresh['company_A'].to_dict()==row['draft'],'draft was not regenerated from branch state'
            changed=economic_change(row['actions']['company_A'],row['draft']);assert changed==row['intervention'];count+=changed
            if label=='none' or (label=='once' and t>0):assert not changed
            after=step(e,before,{k:CompanyAction.from_dict(a) for k,a in row['actions'].items()})
            assert after.state_hash==row['state_after_hash'];assert before.state_hash==row['state_before_hash']
            checks+=audit_settlement(before.to_dict(),after.to_dict(),c.to_dict());steps+=1
            value=after.company('company_A').financial.round_profit_cents if v['summary']['goal']=='profit' else after.welfare_accounting.round_total_economic_welfare_cents
            total+=.95**t*value;assert value==row['round_goal_cents'] and abs(total-row['discounted_cumulative_cents'])<1e-5
            before=after
        assert abs(total-branch['total'])<1e-5 and count==branch['interventions']
        first.append({k:a for k,a in branch['trace'][0]['actions'].items() if k!='company_A'})
    assert first[0]==first[1]==first[2],'same-round peers responded to unseen candidate'
    return dict(file=str(path.relative_to(ROOT)),steps=steps,checks=checks)

if __name__=='__main__':
    target=DEST201 if '201' in sys.argv[1:] else DEST
    paths=[p for name in ('calibration','holdout','rolling') for p in sorted((target/name).glob('*.gz'))];results=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for r in pool.map(audit_file,paths):results.append(r)
    result=dict(passed=True,files=len(results),steps=sum(r['steps'] for r in results),accounting_checks=sum(r['checks'] for r in results),results=results)
    write_json(target/'independent-replay.json',result);print({k:v for k,v in result.items() if k!='results'},flush=True)
