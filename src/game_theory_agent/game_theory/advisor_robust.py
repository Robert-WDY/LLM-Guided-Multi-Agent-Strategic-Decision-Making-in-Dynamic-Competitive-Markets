"""v17: public learning, finite response-aware search, honest abstention and sensitivity."""
from statistics import mean
from time import perf_counter
from .objectives import objective,aggregate,COMPONENTS
from .advisor_beliefs import RobustRequest,learn,sampled_recipe
from .advisor_candidates import expanded
from .advisor_market import Rollouts,BudgetExhausted,forecast,recipe_action,mechanism_coverage
from game_theory_agent.market.actor_policies import actor_ids,options

ROBUST_MODES={'robust_company':'company','robust_profit':'profit','robust_welfare':'welfare'}

class LearnedRollouts(Rollouts):
 def __init__(self,config,state,request,beliefs):super().__init__(config,state,request);self.beliefs=beliefs
 def evaluate(self,profile,scenario):
  other={a:sampled_recipe(b,self.request.seed,scenario,a) for a,b in self.beliefs.items() if a not in profile}
  return super().evaluate({**other,**profile},scenario)

def select_validated(rows,baseline_id='baseline'):
 approved=[r for r in rows if r['id']!=baseline_id and r['safe'] and r['paired_gains'] and min(r['paired_gains'])>0]
 return max(approved,key=lambda r:r['score'])['id'] if approved else baseline_id

def sensitivity(outcomes,spec,selected):
 choices=[];base=outcomes['baseline'];weights=spec['weights'];variants=[]
 for k in COMPONENTS:
  for factor in (.8,1.2):
   w=dict(weights);w[k]=w[k]*factor if w[k]>0 else .02;total=sum(w.values());w={a:b/total for a,b in w.items()};variants.append((k,factor,w))
 for key,factor,w in variants:
  s={**spec,'weights':w};b=aggregate(base,s);rows=[]
  for cid,values in outcomes.items():
   a=aggregate(values,s);rows.append(dict(id=cid,**a,paired_gains=[x-y for x,y in zip(a['sample_scores'],b['sample_scores'])]))
  choices.append(dict(component=key,factor=factor,selected=select_validated(rows)))
 metrics={k:aggregate(v,spec)['metrics'] for k,v in outcomes.items()};dominators=[k for k,v in metrics.items() if k!=selected and all(v[c]>=metrics[selected][c] for c in COMPONENTS) and any(v[c]>metrics[selected][c] for c in COMPONENTS)]
 return dict(perturbations=choices,same_choice_fraction=mean(r['selected']==selected for r in choices),pareto_dominators=dominators,scope='已验证候选范围；非零权重±20%，零权重加2%后归一化。只报告敏感性，不篡改用户目标。')

def advise(config,state,request):
 req=request if isinstance(request,RobustRequest) else RobustRequest.model_validate(request);cid=req.company_id
 if state.terminal or cid not in state.strategic_market.active_company_ids:raise ValueError('a live company and nonterminal state are required')
 start=perf_counter();beliefs=learn(state,cid,req.history);predicted,record=forecast(config,state,cid);spec=objective(req,predicted);h=min(req.horizon,state.rounds_remaining);unit=h*req.scenarios
 if req.step_budget<10*unit:raise ValueError('v17 requires at least ten horizon-by-scenario batches')
 budget=req.step_budget-(4*h if req.backtest else 0);runner=LearnedRollouts(config,predicted,req.model_copy(update={'step_budget':budget}),beliefs if req.learn_opponents else {})
 pool=expanded(config,predicted,cid,req.max_candidates,req.seed,req.search_method);rankings=[];train=list(range(req.scenarios));validation=list(range(1000,1000+req.scenarios));training={};complete=True
 for candidate in pool:
  if runner.used+unit>budget*.3 and rankings:break
  out=[runner.evaluate({cid:candidate['recipe']},s) for s in train];training[candidate['id']]=out;rankings.append(dict(candidate=candidate,**aggregate(out,spec)))
 baseline=rankings[0];rankings.sort(key=lambda r:(r['safe'],r['score']),reverse=True)
 finalists=[baseline]+[r for r in rankings if r['candidate']['id']!='baseline'][:req.finalists]
 nominal_id=max(rankings,key=lambda r:(r['safe'],r['score']))['candidate']['id'];profiles=[{}];response_details=[]
 # Reserve enough fresh validation for all response profiles and all finalists.
 validation_reserve=len(finalists)*(len(finalists)+1)*unit
 response_limit=budget-validation_reserve
 try:
  if req.response_aware:
   for row in finalists:
    own=row['candidate'];profile={a:sampled_recipe(b,req.seed,2000,a) for a,b in beliefs.items()} if req.learn_opponents else {}
    profile.update({a:'balanced' for a in actor_ids(predicted) if a not in predicted.company_ids});profile[cid]=own['recipe'];changed=[]
    for sweep in range(req.response_passes):
     for actor in actor_ids(predicted):
      if actor==cid:continue
      choices=[profile.get(actor,{}),{},dict(price=900000,reserve=True),dict(price=1100000,reserve=True)] if actor in predicted.company_ids else list(options(actor))
      best=-float('inf');choice=choices[0]
      for alternative in choices:
       if runner.used+2*h>response_limit:raise BudgetExhausted()
       out=[runner.evaluate({**profile,actor:alternative},s) for s in (2000,2001)]
       value=mean(v['role_utilities'][actor] for v in out)
       if value>best+1e-8:best=value;choice=alternative
      if choice!=profile.get(actor):changed.append(actor)
      profile[actor]=choice
    response_details.append(dict(id=own['id'],changed_actors=sorted(set(changed)),profile={a:v for a,v in profile.items() if a!=cid}));profiles.append({a:v for a,v in profile.items() if a!=cid})
 except BudgetExhausted:complete=False
 # Fair comparison: every finalist faces the union of discovered reactions with fresh shocks.
 outcomes={};validated=[]
 if complete:
  try:
   for row in finalists:
    c=row['candidate'];outcomes[c['id']]=[runner.evaluate({**p,cid:c['recipe']},s) for p in profiles for s in validation]
  except BudgetExhausted:complete=False;outcomes={}
 if complete:
  base=aggregate(outcomes['baseline'],spec)
  for row in finalists:
   key=row['candidate']['id'];a=aggregate(outcomes[key],spec);g=[x-y for x,y in zip(a['sample_scores'],base['sample_scores'])]
   validated.append(dict(id=key,**a,paired_gains=g,mean_gain=mean(g),sample_range=[min(g),max(g)],positive_samples=sum(x>0 for x in g)))
  selected_id=select_validated(validated)
 else:
  selected_id='baseline';base=aggregate(training['baseline'],spec)
 selected=next(r for r in rankings if r['candidate']['id']==selected_id);recipe=selected['candidate']['recipe'];robustness=sensitivity(outcomes,spec,selected_id) if complete else None
 checks=None
 if req.diagnostics and runner.used+50<budget:
  from .advisor_diagnostics import diagnose
  checks=diagnose(runner,spec,recipe)
 backtest=None;audit_used=0
 if req.backtest:
  audit=Rollouts(config,state,req.model_copy(update={'step_budget':4*h}));actual=[audit.evaluate({cid:recipe},s) for s in (9000,9001)];reference=[audit.evaluate({cid:{}},s) for s in (9000,9001)];g=[x-y for x,y in zip(aggregate(actual,spec)['sample_scores'],aggregate(reference,spec)['sample_scores'])];error=mean(a['profit'] for a in actual)-selected['metrics']['profit']
  backtest=dict(scope='真实状态副本独立执行，未用于选择。对手仍是指定策略；不是现实/真实语言模型证明。',seeds=[9000,9001],paired_utility_gains=g,profit_gain_cents=mean(a['profit']-b['profit'] for a,b in zip(actual,reference)),welfare_gain_cents=mean(a['welfare']-b['welfare'] for a,b in zip(actual,reference)),prediction_profit_error_cents=error,relative_profit_error=abs(error)/max(10000,abs(mean(a['profit'] for a in actual))),outcomes=actual,baseline=reference);audit_used=audit.used
 action=recipe_action(config,state,cid,recipe);current=state.company(cid)
 return dict(version='advisor-v17.0.0',kind='advisor',request=req.model_dump(),objective=spec,source_state_hash=state.state_hash,source_round=state.round,company_count=len(state.company_ids),actor_count=len(actor_ids(state)),facts=dict(cash_cents=current.financial.cash_balance_cents,sales_orders=current.commercial.sales_orders,rounds_remaining=state.rounds_remaining,market_share_ppm=current.commercial.market_share_ppm),forecast=record.model_dump(mode='json'),mechanisms=mechanism_coverage(config,predicted),recommended_id=selected_id,recommended_label=selected['candidate']['label'],recipe=recipe,action=action.to_dict(),disposition='recommend_after_response' if selected_id!='baseline' else 'baseline_due_to_uncertainty',rationale='通过对手反击集合及独立情景筛选，所有配对样本改善且满足样本约束。' if selected_id!='baseline' and req.response_aware else '未发现稳健改善或反击检查未完成，保留基线。' if selected_id=='baseline' else '反击检查关闭：仅通过场景筛选。',baseline_validation=base,validation=[r for r in validated if r['id']!='baseline'],rankings=rankings,diagnostics=checks,backtest=backtest,search=dict(used_steps=runner.used+audit_used,step_budget=req.step_budget,evaluated_candidates=len(rankings),requested_candidates=req.max_candidates,horizon=h,discovery_scenarios=train,validation_scenarios=validation,budget_exhausted=not complete,elapsed_seconds=perf_counter()-start,global_optimality_proven=False),evidence_trace=(outcomes[selected_id][0] if complete else training['baseline'][0])['trace'],v17=dict(beliefs=beliefs,learning_enabled=req.learn_opponents,response_enabled=req.response_aware,response_complete=complete,response_profiles=response_details,response_passes=req.response_passes,nominal_choice=nominal_id,choice_changed=nominal_id!=selected_id,validation_profiles=len(profiles),sensitivity=robustness,candidate_method=req.search_method,scope='有限轮次的私利最佳回应压力测试进入筛选；不是对手理性、反击穷尽或动态均衡的保证。'),candidate_coverage=dict(searched=[r['candidate']['id'] for r in rankings],unsearched_or_equivalent=[r['label'] for r in pool if r['id'] not in training],scope='组合空间未穷尽；显示本次生成但预算未评估的候选。'),limitations=['历史只含公开价格，不能识别隐藏成本/偏好；冷启动仍依赖先验。','反击只在有限候选和声明目标下搜索，未知响应可能改变结论。','现金安全和收益改善仅限检验样本，基线也可能亏损。','权重表达偏好而非客观真理，敏感性不等于偏好正确性。','引擎一致性不构成现实校准，有限网格最优不等于全局最优。'],failure_conditions=['公开价格趋势突然反转或历史失真','多方使用未建模的共同学习/混合策略','收益发生在预测期限之外','偏好、机制或现金状态超出验证条件'])
