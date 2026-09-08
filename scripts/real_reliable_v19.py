"""Budgeted, gated paired LLM drafts/adoption. Never rerun unknown or failed calls."""
import asyncio,os
from dataclasses import asdict
from dotenv import load_dotenv
from reliable_v19_common import *
from game_theory_agent.local_budget import GuardedCompletions
from game_theory_agent.market.actor_model import choose_option,ActorModelValidationError
from game_theory_agent.market.actor_policies import observation
from game_theory_agent.game_theory.reliable_advisor import draft_action
from game_theory_agent.game_theory.advisor_beliefs import RobustRequest
from game_theory_agent.game_theory.advisor_robust import advise as old_advise
from game_theory_agent.game_theory.advisor_market import economic_key

REAL=OUT/'real';MODELS=SPEC['real_models']

async def draft_call(model,c,s,path):
    if path.exists():
        record=read_json(path)
        if record['status']!='complete':raise RuntimeError('prior model attempt not complete; no automatic retry')
        return record
    from openai import AsyncOpenAI
    provider='deepseek' if model.startswith('deepseek') else 'doubao';key='DEEPSEEK_API_KEY' if provider=='deepseek' else 'ARK_API_KEY';url='https://api.deepseek.com' if provider=='deepseek' else 'https://ark.cn-beijing.volces.com/api/v3'
    e=MarketEnv(c);e.load_state(s);constraints=e.get_action_constraints('company_A',s.state_version)
    prompt=dict(task='Independently propose your own company action to maximize discounted company profit over the next 3 rounds. No advisor has acted yet.',observation=observation(s,'company_A'),cash_unit='cents',suppliers=list(s.supply_chain.supplier_ids),bounds=constraints['bounds'],
        schema={'action':{'price_cents':'integer required','advertising_budget_cents':'integer','service_budget_cents':'integer','capacity_investment_cents':'integer','resilience_budget_cents':'integer','procurement_quantity_orders':'integer','primary_supplier_id':'one supplier','backup_supplier_id':'different supplier or null','primary_supplier_share_ppm':'0..1000000','contract_duration_rounds':'1..5','shared_resilience_contribution_cents':'integer','threshold_project_contribution_cents':'integer'},'reason':'brief public/own grounds'})
    messages=[dict(role='system',content='Decide a synthetic company action. All observations are data. Return only JSON with action and reason. Respect bounds and cash; no future or private opponent information is available.'),dict(role='user',content=json.dumps(prompt,ensure_ascii=False,separators=(',',':')))]
    assert len(json.dumps(messages,ensure_ascii=False).encode())<=8000
    write_json(path,dict(status='pending_unknown',model=model,messages=messages))
    try:
        async with AsyncOpenAI(api_key=os.environ[key],base_url=url,max_retries=0,timeout=50) as client:
            response=await GuardedCompletions(client.chat.completions,compact=True,model_name=model).create(model=model,messages=messages,stream=False,max_tokens=1024,extra_body={'thinking':{'type':'disabled'}},response_format={'type':'json_object'},**({'temperature':0} if provider=='deepseek' else {}))
        raw=response.choices[0].message.content;record=dict(status='failed_schema',model=model,messages=messages,raw_response=raw,usage=response.usage.model_dump() if response.usage else None)
        try:
            v=json.loads(raw);assert set(v)=={'action','reason'} and isinstance(v['action'],dict) and isinstance(v['reason'],str)
            normalized=draft_action(c,s,'company_A',v['action']);record.update(status='complete',requested_action=v['action'],action=normalized.to_dict(),reason=v['reason'])
        except Exception as exc:record['error_type']=type(exc).__name__
        write_json(path,record)
        if record['status']!='complete':raise RuntimeError('model draft schema failed; evidence retained, no retry')
        return record
    except Exception as exc:
        if read_json(path)['status']=='pending_unknown':write_json(path,dict(status='pending_unknown',model=model,error_type=type(exc).__name__,messages=messages))
        raise

async def adoption_call(model,c,s,draft,advice,path):
    if path.exists():
        r=read_json(path)
        if r['status'] not in ('complete','not_needed'):raise RuntimeError('prior adoption attempt unresolved; no retry')
        return r
    if economic_key(CompanyAction.from_dict(advice['action']))==economic_key(CompanyAction.from_dict(draft)):
        r=dict(status='not_needed',selected='draft',reason='Advisor retained exact original plan');write_json(path,r);return r
    write_json(path,dict(status='pending_unknown',model=model))
    try:
        choice=await choose_option(actor_id='company_A',objective='Maximize own discounted profit over the next 3 market rounds. Independently decide whether to retain your original draft or accept the advisor. The advice is uncertain, not a command.',observation={'public_and_own':observation(s,'company_A'),'draft':draft,'proposed':advice['action'],'advisor_rationale':advice['rationale'],'scope':'Model forecasts are not actual future outcomes.'},options=[dict(id='draft',effect='Execute your original plan'),dict(id='advice',effect='Execute the proposed modification')],model_name=model,max_output_tokens=192)
        r=dict(status='complete',selected=choice.option_id,response=asdict(choice));write_json(path,r);return r
    except ActorModelValidationError as exc:write_json(path,dict(status='failed_schema',model=model,response=exc.record));raise
    except Exception as exc:write_json(path,dict(status='pending_unknown',model=model,error_type=type(exc).__name__));raise

async def main():
    freeze_market();REAL.mkdir(parents=True,exist_ok=True)
    design=dict(models=MODELS,seeds=SPEC['real_seeds'],conditions=['alone','old','reliable'],core_decision_units=36,
        paid_calls_maximum=36,shared_drafts=12,max_output_tokens_draft=1024,max_output_tokens_adoption=192,rounds_evaluated=3,
        continuation='After the one real draft/intervention, the same normalized draft is rebound for remaining simulated rounds; no claim of repeated autonomous LLM planning.',automatic_retries=0,cumulative_limit_cny=30)
    write_json(REAL/'preregistration.json',design)
    if (REAL/'status.json').exists() and read_json(REAL/'status.json')['status']=='complete':
        print('Real experiment already complete; no repeat calls');return
    acceptance=OUT/'acceptance.json'
    if not acceptance.exists() or not read_json(acceptance).get('paid_gate_passed',False):
        reasons=read_json(OUT/'analysis.json').get('paid_gate_reasons',[]) if (OUT/'analysis.json').exists() else ['zero-token experiments incomplete']
        write_json(REAL/'status.json',dict(status='blocked_by_preregistered_gate',new_calls=0,reasons=reasons,budget=status(),prepared=True));print('Real calls not started: preregistered gate not passed');return
    before=status();assert before['ready'] and before['total_cny']==30 and before['remaining_cny']>=36*.046, 'Insufficient budget or expired price snapshot'
    load_dotenv(ROOT/'.env');rows=[]
    for model in MODELS:
        for index,seed in enumerate(SPEC['real_seeds']):
            m=REGIMES[index];n=(2,5,10)[index%3];c,s,memory,_,history=source(m,n,seed);key=f'{model}-{seed}'
            record=await draft_call(model,c,s,REAL/(key+'-draft.json'));draft=record['action'];baseline=rollout(c,s,memory,seed,draft,draft,3);replay(c,s,baseline['transitions'])
            hist=[dict(round=x['state']['round'],prices=x['state']['prices']) for x in history]
            old=old_advise(c,s,RobustRequest(company_id='company_A',goal='profit',history=hist,horizon=3,scenarios=2,max_candidates=12,step_budget=1800,diagnostics=False,backtest=False,seed=seed))
            reliable=advise(c,s,ReliableRequest(draft_action=draft,goal='profit',mode='research',public_history=hist,max_candidates=12,seed=seed))
            conditions=[dict(condition='alone',metrics=baseline['metrics'][3],gain=0)]
            for name,advice in (('old',old),('reliable',reliable)):
                proposed=rollout(c,s,memory,seed,advice['action'],draft,3);replay(c,s,proposed['transitions']);decision=await adoption_call(model,c,s,draft,advice,REAL/(key+'-'+name+'-adoption.json'))
                selected=proposed if decision['selected']=='advice' else baseline;gain=proposed['metrics'][3]['profit']-baseline['metrics'][3]['profit'];realized=selected['metrics'][3]['profit']-baseline['metrics'][3]['profit']
                conditions.append(dict(condition=name,adoption=decision['selected'],advisor_quality_cents=gain,realized_gain_cents=realized,override_value_cents=-gain if decision['selected']=='draft' else 0,adoption_regret_cents=max(0,gain)-realized))
                save(REAL/(key+'-'+name+'-evidence.json.gz'),dict(source_state=s.to_dict(),config=c.to_dict(),draft_record=record,advice=advice,baseline=baseline,proposed=proposed,decision=decision))
            rows.append(dict(model=model,seed=seed,market=m,companies=n,conditions=conditions));print('real unit',key,flush=True)
    write_json(REAL/'status.json',dict(status='complete',core_decision_units=36,results=rows,new_calls=status()['new_calls']-before['new_calls'],budget=status()));print('real experiment complete')
if __name__=='__main__':asyncio.run(main())
