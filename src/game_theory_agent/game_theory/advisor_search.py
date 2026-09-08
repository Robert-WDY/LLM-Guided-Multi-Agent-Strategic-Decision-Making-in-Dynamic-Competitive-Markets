"""Budgeted candidate expansion with separate discovery and validation scenarios."""
from statistics import mean
from time import perf_counter
from .objectives import objective,aggregate,AdviceRequest
from .advisor_market import candidates,recipe_action,economic_key,forecast,mechanism_coverage,Rollouts,BudgetExhausted,LABELS,RECIPES

def advise(config,state,request):
 if not isinstance(request,AdviceRequest):request=AdviceRequest.model_validate(request)
 cid=request.company_id
 if request.step_budget<5*request.horizon*request.scenarios:raise ValueError('step budget must cover at least five horizon-by-scenario batches')
 if cid not in state.company_ids or state.terminal or cid not in state.strategic_market.active_company_ids:raise ValueError('a live company and nonterminal state are required')
 start=perf_counter();predicted,forecast_record=forecast(config,state,cid);spec=objective(request,predicted);planning_request=request.model_copy(update={'step_budget':request.step_budget-(4*min(request.horizon,state.rounds_remaining) if request.backtest else 0)})
 runner=Rollouts(config,predicted,planning_request)
 pool=candidates(config,predicted,cid,max(2,request.max_candidates-2));discovery=list(range(request.scenarios));validation=list(range(1000,1000+request.scenarios))
 rankings=[];exhausted=False;reserve=request.horizon*request.scenarios*4
 def evaluate(candidate,seeds):return [runner.evaluate({cid:candidate['recipe']},s) for s in seeds]
 for candidate in pool:
  if runner.used+request.horizon*request.scenarios+reserve>planning_request.step_budget and rankings:exhausted=True;break
  try:outcomes=evaluate(candidate,discovery)
  except BudgetExhausted:exhausted=True;break
  rankings.append(dict(candidate=candidate,**aggregate(outcomes,spec)))
 if not rankings:raise ValueError('step budget too small for baseline discovery and validation')
 # Expand around the best seed candidate instead of claiming fixed templates exhaust actions.
 best=max(rankings,key=lambda row:(row['safe'],row['score']))
 for factor in (975000,1025000):
  if len(rankings)>=request.max_candidates or runner.used+request.horizon*request.scenarios+reserve>planning_request.step_budget:break
  recipe=dict(best['candidate']['recipe']);recipe['price']=recipe.get('price',1000000)*factor//1000000
  action=recipe_action(config,predicted,cid,recipe)
  if any(economic_key(action)==economic_key(recipe_action(config,predicted,cid,row['candidate']['recipe'])) for row in rankings):continue
  candidate=dict(id=f"refine-{factor}-{best['candidate']['id']}",label='最佳候选价格局部细化',recipe=recipe,action=action.to_dict())
  rankings.append(dict(candidate=candidate,**aggregate(evaluate(candidate,discovery),spec)))
 baseline=rankings[0];rankings.sort(key=lambda row:(row['safe'],row['score']),reverse=True)
 shortlist=[r for r in rankings if r['candidate']['id']!='baseline'][:3]
 base_validation=evaluate(baseline['candidate'],validation);base_agg=aggregate(base_validation,spec);validated=[]
 for row in shortlist:
  try:v=evaluate(row['candidate'],validation)
  except BudgetExhausted:exhausted=True;break
  a=aggregate(v,spec);gains=[x-y for x,y in zip(a['sample_scores'],base_agg['sample_scores'])]
  validated.append(dict(id=row['candidate']['id'],**a,paired_gains=gains,positive_samples=sum(x>0 for x in gains),sample_range=[min(gains),max(gains)],mean_gain=mean(gains)))
 approved=[r for r in validated if r['safe'] and min(r['paired_gains'])>0]
 selected_id=max(approved,key=lambda row:row['score'])['id'] if approved else 'baseline'
 selected=next(row for row in rankings if row['candidate']['id']==selected_id)
 action=recipe_action(config,state,cid,selected['candidate']['recipe'])
 checks=None
 if request.diagnostics:
  from .advisor_diagnostics import diagnose
  checks=diagnose(runner,spec,selected['candidate']['recipe'])
 backtest=None;audit_used=0
 if request.backtest:
  audit_request=request.model_copy(update={'step_budget':4*min(request.horizon,state.rounds_remaining)})
  audit=Rollouts(config,state,audit_request)
  actual=[audit.evaluate({cid:selected['candidate']['recipe']},s) for s in (9000,9001)]
  reference=[audit.evaluate({cid:{}},s) for s in (9000,9001)]
  gains=[score_value-spec_value for score_value,spec_value in zip(aggregate(actual,spec)['sample_scores'],aggregate(reference,spec)['sample_scores'])]
  backtest=dict(scope='真实状态副本上的独立事后对照，不参与推荐选择，不推进原市场；两个样本不是现实有效性证明。',seeds=[9000,9001],paired_utility_gains=gains,profit_gain_cents=mean(a['profit']-b['profit'] for a,b in zip(actual,reference)),welfare_gain_cents=mean(a['welfare']-b['welfare'] for a,b in zip(actual,reference)),prediction_profit_error_cents=mean(a['profit'] for a in actual)-selected['metrics']['profit'],outcomes=actual,baseline=reference)
  audit_used=audit.used
 elapsed=perf_counter()-start
 current=state.company(cid)
 return dict(version='advisor-v16.0.0',kind='advisor',request=request.model_dump(),objective=spec,source_state_hash=state.state_hash,source_round=state.round,company_count=len(state.company_ids),actor_count=len(predicted.company_ids)+len(predicted.supply_chain.supplier_ids)+2,
  facts=dict(cash_cents=current.financial.cash_balance_cents,sales_orders=current.commercial.sales_orders,rounds_remaining=state.rounds_remaining,market_share_ppm=current.commercial.market_share_ppm),
  forecast=forecast_record.model_dump(mode='json'),mechanisms=mechanism_coverage(config,predicted),candidate_coverage=dict(searched=[r['candidate']['id'] for r in rankings],unsearched_or_equivalent=[LABELS[k] for k in RECIPES if k not in {r['candidate']['id'] for r in rankings}],scope='未单独搜索的模板包含预算/候选上限限制，以及在当前状态下与已搜索动作完全相同的模板。'),
  recommended_id=selected_id,recommended_label=selected['candidate']['label'],recipe=selected['candidate']['recipe'],action=action.to_dict(),
  disposition='recommend_in_model' if approved else 'baseline_due_to_uncertainty',
  rationale='在独立验证情景中均提高所选目标且满足样本现金/退出约束。' if approved else '未找到在全部独立验证情景中改善目标且满足约束的候选，保留基线；基线自身仍需检查风险。',
  baseline_validation=base_agg,validation=validated,rankings=rankings,diagnostics=checks,
  backtest=backtest,search=dict(used_steps=runner.used+audit_used,planning_steps=runner.used,audit_steps=audit_used,step_budget=request.step_budget,evaluated_candidates=len(rankings),requested_candidates=request.max_candidates,horizon=min(request.horizon,predicted.rounds_remaining),discovery_scenarios=discovery,validation_scenarios=validation,budget_exhausted=exhausted or runner.used>=request.step_budget,elapsed_seconds=elapsed,global_optimality_proven=False),
  evidence_trace=runner.evaluate({cid:selected['candidate']['recipe']},validation[0])['trace'],
  limitations=['效用是固定尺度指标，非额外货币福利；对手偏好与未来策略是建模假设。','验证情景范围是样本范围，不是置信区间；没有现实数据校准。','只在已搜索候选、预测期限和场景内比较，未证明全局最优或完整动态纳什。','产能/韧性仅作为终端公开指标代理；超出预测期限的合同和投资收益存在截断误差。','互助与协调为条件请求，未穷举全部伙伴；预算内未评估的动作或伙伴可能更好。'],
  failure_conditions=['对手采用样本外价格/投入/合作策略','私有供给成本、融资或预算偏离公开先验','事件分布、规则版本、当前状态或目标改变','长期回报主要在预测期限之外'])
