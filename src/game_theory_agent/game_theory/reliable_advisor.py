"""Agent draft improvement layer with lagged responses and model-conditional abstention."""
import hashlib
import math
from copy import deepcopy
from dataclasses import replace
from statistics import mean,stdev
from time import perf_counter
from typing import Literal
from pydantic import Field
from game_theory_agent.market import CompanyAction,MarketEnv
from game_theory_agent.market.actor_policies import company_action,actor_ids
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.market.validation import ActionValidator
from game_theory_agent.market.exceptions import ActionValidationError
from game_theory_agent.decisioning import resolve_action_request
from .objectives import AdviceRequest,objective,score,aggregate
from .advisor_market import forecast,economic_key,BudgetExhausted
from .reliable_response import frame,features,predict,sample,load_model
from .advisor_beliefs import PriceFrame

class ReliableRequest(AdviceRequest):
    draft_action:dict
    mode:Literal['interactive','research']='interactive'
    response_model:Literal['none','generic','conditional']='conditional'
    max_candidates:int=Field(default=12,ge=2,le=20)
    step_budget:int=Field(default=1800,ge=1,le=12000)
    diagnostics:bool=False
    backtest:bool=False
    public_history:list[PriceFrame]=Field(default_factory=list,max_length=60)

def draft_action(config,state,cid,raw):
    if 'price_cents' not in raw:raise ValueError('agent draft must explicitly include price_cents')
    if 'episode_id' in raw:
        action=CompanyAction.from_dict(raw)
        try:ActionValidator(config).validate(action,state=state,company_id=cid).require_valid()
        except ActionValidationError as exc:raise ValueError(str(exc)) from exc
        return action
    return resolve_action_request(config,state,cid,raw,source='agent-draft',action_id=f'draft:{state.round}:{cid}').action

def local_candidates(config,state,cid,draft,limit=10):
    pool=[dict(id='agent_draft',label='Agent原计划',action=draft.to_dict(),change={})];seen={economic_key(draft)}
    if cid not in state.strategic_market.active_company_ids:return pool
    # All changes are relative to the exact normalized draft, never a rule replacement.
    mutations=[('price_down',{'price_cents':draft.price_cents*95//100}),('price_up',{'price_cents':draft.price_cents*105//100})]
    raw=draft.to_dict()
    for k in ('service_budget_cents','advertising_budget_cents','capacity_investment_cents','resilience_budget_cents'):
        mutations.append((k+'_adjust',{k:max(0,raw[k]-30000) if raw[k]>0 else 30000}))
    q=raw.get('procurement_quantity_orders')
    if q is not None:mutations.extend([('procurement_down',{'procurement_quantity_orders':q*9//10}),('procurement_up',{'procurement_quantity_orders':q*11//10})])
    mutations.extend([('supplier_mix',{'primary_supplier_share_ppm':500000}),('contract_adjust',{'contract_duration_rounds':1 if raw.get('contract_duration_rounds',1)!=1 else 3}),('cooperation_adjust',{'shared_resilience_contribution_cents':0 if raw.get('shared_resilience_contribution_cents',0) else 30000})])
    for label,change in mutations:
        try:a=resolve_action_request(config,state,cid,{**raw,**change},source='reliable-local',action_id=f'local:{state.round}:{cid}:{label}').action
        except (ValueError,ActionValidationError):continue
        key=economic_key(a)
        if key in seen:continue
        seen.add(key);pool.append(dict(id=label,label=label,action=a.to_dict(),change={k:a.to_dict()[k] for k in change}))
        if len(pool)>=limit:break
    return pool

def rebind(config,state,cid,raw):
    if cid not in state.strategic_market.active_company_ids:return company_action(config,state,cid,'balanced')
    return resolve_action_request(config,state,cid,raw,source='draft-continuation',action_id=f'continuation:{state.round}:{cid}').action

def lower_bound(gains):
    if len(gains)<2:return -float('inf')
    # One-sided Student-t 95%: finite MC sample, conditional on this simulator/model.
    t={2:6.314,3:2.920,4:2.353,5:2.132,6:2.015,7:1.943,8:1.895}.get(len(gains),1.895)
    return mean(gains)-t*stdev(gains)/math.sqrt(len(gains))

class DraftRollouts:
    def __init__(self,config,predicted,request,model):
        self.config=config;self.initial=predicted;self.request=request;self.model=model;self.used=0;self.cache={}
        self.history=[x.model_dump() for x in request.public_history]+[frame(predicted)]
    def response(self,x,mode,peer):
        p=predict(self.model,x,mode)
        # Separate posterior for each named peer, using only earlier settled prices.
        counts=[8*v for v in p['probabilities']];sums=[c*v for c,v in zip(counts,p['changes'])];n=0
        for a,b,c in zip(self.history,self.history[1:],self.history[2:]):
            prior={**a,'shortage':0};current={**b,'shortage':0}
            if mode=='conditional' and features(prior,current,peer)[0]!=x[0]:continue
            change=max(-.2,min(.2,c['prices'][peer]/b['prices'][peer]-1));k=0 if change<-.01 else 2 if change>.01 else 1
            counts[k]+=1;sums[k]+=change;n+=1
        return dict(probabilities=[v/sum(counts) for v in counts],changes=[v/c for v,c in zip(sums,counts)],peer_observations=n)
    def evaluate(self,raw,draft,scenario,horizon):
        key=(economic_key(CompanyAction.from_dict(raw)),scenario,horizon)
        if key in self.cache:return self.cache[key]
        h=min(horizon,self.initial.rounds_remaining)
        if self.used+h>self.request.step_budget:raise BudgetExhausted()
        seed=int(hashlib.sha256(f'{self.initial.episode_seed}:{self.request.seed}:{scenario}'.encode()).hexdigest()[:15],16)
        s=replace(self.initial,episode_seed=seed);s=replace(s,state_hash=state_hash(s.to_dict()));e=MarketEnv(self.config);e.load_state(s)
        cid=self.request.company_id;start=s.company(cid);previous=frame(s);profit=welfare=0.;minimum=start.financial.cash_balance_cents;trace=[];own_service=0
        for t in range(h):
            current=frame(s);aa={a:company_action(self.config,s,a,'balanced') for a in s.company_ids}
            for peer in s.company_ids:
                if peer==cid or peer not in s.strategic_market.active_company_ids:continue
                # Candidate is not visible until after first settlement. Current-round distribution is shared.
                mode='generic' if t==0 else self.request.response_model
                if self.request.response_model!='none':
                    p=self.response(features(previous,current,peer,own_service),mode,peer)
                    request=aa[peer].to_dict();request['price_cents']=round(s.company(peer).commercial.price_cents*(1+sample(p,self.request.seed,scenario,s.round,peer)))
                    aa[peer]=rebind(self.config,s,peer,request)
            aa[cid]=rebind(self.config,s,cid,raw if t==0 else draft)
            after=e.step(f'{s.episode_id}:{s.round}:{s.state_version}',aa).state_after;self.used+=1
            c=after.company(cid);profit+=self.request.discount**t*c.financial.round_profit_cents;welfare+=self.request.discount**t*after.welfare_accounting.round_total_economic_welfare_cents;minimum=min(minimum,c.financial.cash_balance_cents)
            trace.append(dict(round=s.round,actions={a:x.to_dict() for a,x in aa.items()},state_hash=after.state_hash));previous=current;own_service=aa[cid].service_budget_cents;s=after
        c=s.company(cid);out=dict(profit=profit,welfare=welfare,cash=c.financial.cash_balance_cents-start.financial.cash_balance_cents,share=c.commercial.market_share_ppm-start.commercial.market_share_ppm,capacity=c.operations.base_capacity_orders-start.operations.base_capacity_orders,resilience=c.risk.resilience_ppm-start.risk.resilience_ppm,minimum_cash=minimum,exited=cid not in s.strategic_market.active_company_ids,trace=trace)
        self.cache[key]=out;return out

def advise(config,state,request,*,model=None):
    req=request if isinstance(request,ReliableRequest) else ReliableRequest.model_validate(request);cid=req.company_id;start=perf_counter()
    if state.terminal or cid not in state.strategic_market.active_company_ids:raise ValueError('live nonterminal company required')
    frames=[x.model_dump() for x in req.public_history]
    if any(x['round']>=state.round or set(x['prices'])!=set(state.company_ids) or any(v<=0 for v in x['prices'].values()) for x in frames) or any(a['round']>=b['round'] for a,b in zip(frames,frames[1:])):raise ValueError('history must be public, complete and strictly before source round')
    draft=draft_action(config,state,cid,req.draft_action);pool=local_candidates(config,state,cid,draft,req.max_candidates)
    predicted,record=forecast(config,state,cid);spec=objective(req,predicted);model=load_model() if model is None else model
    runner=DraftRollouts(config,predicted,req,model);h=1 if req.mode=='interactive' else 3;horizons=[h] if h==1 else [3,5]
    train=range(3);validation=range(1000,1008);rankings=[];checks=[];complete=True;selected='agent_draft';reason='没有足够证据优于原计划，保留原计划。'
    try:
        for c in pool:
            values=[runner.evaluate(c['action'],draft.to_dict(),s,h) for s in train]
            rankings.append(dict(candidate=c,**aggregate(values,spec)))
        finalist=max((r for r in rankings if r['candidate']['id']!='agent_draft'),key=lambda r:(r['safe'],r['score']),default=None)
        if finalist:
            for horizon in horizons:
                a=[runner.evaluate(finalist['candidate']['action'],draft.to_dict(),s,horizon) for s in validation]
                b=[runner.evaluate(draft.to_dict(),draft.to_dict(),s,horizon) for s in validation]
                gains=[score(x,spec)['total']-score(y,spec)['total'] for x,y in zip(a,b)]
                safe=all(x['minimum_cash']>=req.cash_floor_cents and not x['exited'] for x in a)
                checks.append(dict(horizon=min(horizon,state.rounds_remaining),paired_gains=gains,mean_gain=mean(gains),lower_bound=lower_bound(gains),negative_samples=sum(x<0 for x in gains),safe=safe,pass_gate=safe and lower_bound(gains)>0 and all(x>=0 for x in gains)))
            if all(c['pass_gate'] for c in checks):selected=finalist['candidate']['id'];reason='所选候选相对原计划在独立模拟验证中，各期限下界为正且满足样本约束。'
    except BudgetExhausted:complete=False;selected='agent_draft';reason='验证预算不足，未完成检查，原计划不变。'
    choice=next(c for c in pool if c['id']==selected)
    return dict(version='reliable-strategic-advisor-v1',source_state_hash=state.state_hash,source_round=state.round,company_count=len(state.company_ids),actor_count=len(actor_ids(state)),mode=req.mode,response_model=req.response_model,model_sha256=model['sha256'],objective=spec,draft_action=draft.to_dict(),action=choice['action'],recommended_id=selected,recommended_label=choice['label'],disposition='abstain' if selected=='agent_draft' else 'recommend',rationale=reason,candidates=pool,rankings=rankings,validation=checks,forecast=record.model_dump(mode='json'),search=dict(elapsed_seconds=perf_counter()-start,used_steps=runner.used,step_budget=req.step_budget,evaluated_candidates=len(rankings),complete=complete),
                continuation='候选仅改变当前一步，随后回到同一原计划并按新状态约束修正；不是后续每轮重新调用LLM。',
                limitations=['主基线为已规范化Agent原计划；弃权不保证原计划安全。','下界只针对声明模型的配对模拟，未知对手/分布变化不受保证。','本轮对手不观察候选；条件响应从下一轮开始。','供应商/消费者/政府继续原生规则；尚未学习其条件响应。','有限候选与3/5轮代理值，不是全球最优、完整企业估值或动态纳什证明。'])
