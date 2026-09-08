"""Export complete recorded evidence and expanded negative-case simulator traces.

No model calls; no frozen evaluator changes. Instrumentation wraps the same
frozen implementation in memory and checks its result against saved evidence.
"""
import json,hashlib,sys,time,zipfile
from pathlib import Path
from collections import Counter
from copy import deepcopy
from reliable_v19_common import *
import game_theory_agent.game_theory.reliable_advisor as core

DEST=ROOT/'artifacts/reliable-v19-trace'

def full_actual(c,state,trace):
    env=MarketEnv(c);env.load_state(state);before=state;rows=[]
    for item in trace:
        after=step(env,before,{k:CompanyAction.from_dict(v) for k,v in item['actions'].items()})
        assert after.state_hash==item['state_hash']
        audit_settlement(before.to_dict(),after.to_dict(),c.to_dict())
        rows.append(dict(round=before.round,actions=item['actions'],state_before_hash=before.state_hash,state_after=after.to_dict()))
        before=after
    return rows

def expand(row):
    key=f"{row['market']}-{row['companies']}-{row['seed']}-{row['goal']}"
    source=OUT/'advice'/(key+'.json.gz');saved=read(source);c=MarketConfig.from_mapping(saved['config']);state=MarketState.from_dict(saved['source_state'])
    request=ReliableRequest(draft_action=saved['draft'],goal=row['goal'],mode='research',response_model='conditional',max_candidates=12,step_budget=1800,seed=row['seed'],public_history=saved['history'])
    model_runs=[];context={};candidate_ids={economic_signature(v['action']):v['id'] for v in saved['details']['conditional']['candidates']}
    original_env,original_rollouts=core.MarketEnv,core.DraftRollouts
    class TracedEnv(original_env):
        def load_state(self,state,*args,**kwargs):
            result=super().load_state(state,*args,**kwargs)
            self.export_trace=dict(context=deepcopy(context),initial_state=state.to_dict(),steps=[])
            model_runs.append(self.export_trace)
            return result
        def step(self,*args,**kwargs):
            result=super().step(*args,**kwargs)
            if hasattr(self,'export_trace'):
                self.export_trace['steps'].append(dict(actions={k:v.to_dict() for k,v in args[1].items()},state_after=result.state_after.to_dict()))
            return result
    class TracedRollouts(original_rollouts):
        def evaluate(self,raw,draft,scenario,horizon):
            context.clear();context.update(candidate=candidate_ids.get(economic_signature(raw),'unknown'),scenario=scenario,horizon=horizon,phase='discovery' if scenario<1000 else 'validation')
            return super().evaluate(raw,draft,scenario,horizon)
    try:
        core.MarketEnv,core.DraftRollouts=TracedEnv,TracedRollouts
        reproduced=core.advise(c,state,request)
    finally:core.MarketEnv,core.DraftRollouts=original_env,original_rollouts
    expected=deepcopy(saved['details']['conditional']);actual=deepcopy(reproduced)
    expected['search'].pop('elapsed_seconds');actual['search'].pop('elapsed_seconds')
    assert actual==expected,'instrumented advisor changed the saved result'
    candidates={name:full_actual(c,state,data['transitions']) for name,data in saved['actual'].items()}
    selected=reproduced['recommended_id'];base=candidates['agent_draft'];chosen=candidates[selected];deltas=[]
    for t,(b,a) in enumerate(zip(base,chosen)):
        bs,ss=b['state_after'],a['state_after'];bw,sw=bs['welfare_accounting'],ss['welfare_accounting']
        dw={k:sw[k]-bw[k] for k in sw if k.startswith('round_') and isinstance(sw[k],(int,float))}
        profit=ss['companies']['company_A']['financial']['round_profit_cents']-bs['companies']['company_A']['financial']['round_profit_cents']
        deltas.append(dict(round=a['round'],discount=.95**t,profit_delta_cents=profit,welfare_delta_cents=dw['round_total_economic_welfare_cents'],welfare_components_delta=dw,baseline_actions=b['actions'],advice_actions=a['actions']))
    target='profit_delta_cents' if row['goal']=='profit' else 'welfare_delta_cents'
    assert abs(sum(x['discount']*x[target] for x in deltas)-row['actual_5round_cents'])<1e-5
    record=dict(source_file=str(source.relative_to(ROOT)),source_sha256=sha(source),source=saved,actual_full_states=candidates,
                predicted_full_states=model_runs,round_deltas=deltas,advisor_reproduction_equal_except_runtime=True,
                scope='Authoritative states appear only in evaluator traces. The advisor uses its separately reconstructed legal forecast state. No LLM call or hidden model reasoning.')
    save(DEST/'negative-cases'/(key+'.json.gz'),record)
    return dict(**row,case_id=key,model_paths=len(model_runs),model_steps=sum(len(x['steps']) for x in model_runs),actual_paths=len(candidates),round_deltas=deltas)

def economic_signature(raw):
    return core.economic_key(CompanyAction.from_dict(raw))

def main():
    freeze_market()
    for p,digest in read_json(OUT/'implementation-before-holdout.json').items():assert sha(ROOT/p)==digest
    DEST.mkdir(parents=True,exist_ok=True)
    rows=read_json(OUT/'advice.json')['results'];index=[]
    for row in rows:
        key=f"{row['market']}-{row['companies']}-{row['seed']}-{row['goal']}";p=OUT/'advice'/(key+'.json.gz')
        saved=read(p)
        index.append(dict(case_id=key,market=row['market'],companies=row['companies'],goal=row['goal'],seed=row['seed'],
                          input_hash=saved['source_state']['state_hash'],source_file=str(p.relative_to(ROOT)),sha256=sha(p),
                          candidate_count=len(saved['details']['conditional']['candidates']),decisions=row['results']))
    write_json(DEST/'all-720-cases-index.json',index)
    failures=read_json(OUT/'negative-case-validation-mismatch.json');expanded=[]
    for i,row in enumerate(sorted(failures,key=lambda x:x['actual_5round_cents'])):
        expanded.append(expand(row));print('expanded',i+1,'/',len(failures),flush=True)
    write_json(DEST/'negative-cases-summary.json',expanded)
    write_json(DEST/'trace-verification.json',dict(passed=True,primary_states=720,advisor_decisions=2160,expanded_negative_cases=len(expanded),reproduced_model_steps=sum(x['model_steps'] for x in expanded),actual_paths=sum(x['actual_paths'] for x in expanded),source_hashes_unchanged=True,new_paid_calls=0,scope='All original cases remain in the existing 4158-file evidence.zip. Additional full forecast and settlement states are re-executed for the 28 negative cases.'))
    print('Trace export complete')

if __name__=='__main__':main()
