"""Legal recipe actions and bounded full-engine multi-role rollouts."""
from dataclasses import replace
from statistics import mean
import hashlib,json
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.market import MarketEnv
from game_theory_agent.market.actor_policies import company_action,actor_ids,options,observation
from game_theory_agent.decisioning import resolve_action_request
from game_theory_agent.agents.observation import ObservationBuilder
from game_theory_agent.agents.personas import PersonaRegistry
from game_theory_agent.strategic_reliability.public_rollout import build_public_forecast_state

RECIPES={
 'aid_offer':{'aid':'offer'},'aid_request':{'aid':'request'},'coordinate':{'coordinate':True},
 'baseline':{},'price_down':{'price':950000},'price_up':{'price':1050000},
 'price_down_large':{'price':900000},'price_up_large':{'price':1100000},
 'service':{'service_budget_cents':30000},'advertise':{'advertising_budget_cents':30000},
 'capacity':{'capacity_investment_cents':30000},'resilience':{'resilience_budget_cents':30000},
 'reserve':{'reserve':True},'procure_less':{'quantity':750000},'procure_more':{'quantity':1200000},
 'supplier_diversify':{'supplier_share':500000},'short_contract':{'duration':1},'long_contract':{'duration':3},
 'shared':{'shared_resilience_contribution_cents':30000},'project':{'threshold_project_contribution_cents':30000},
 'no_shared':{'shared_resilience_contribution_cents':0,'threshold_project_contribution_cents':0},
 'growth_bundle':{'price':950000,'capacity_investment_cents':20000,'advertising_budget_cents':20000},
 'safe_bundle':{'supplier_share':500000,'resilience_budget_cents':20000},
}
LABELS={'aid_offer':'提供产能互助','aid_request':'请求产能互助','coordinate':'条件价格协调','baseline':'规则基线','price_down':'降价5%','price_up':'提价5%','price_down_large':'降价10%','price_up_large':'提价10%','service':'增加服务投入','advertise':'增加广告投入','capacity':'增加产能投入','resilience':'增加韧性投入','reserve':'现金保全','procure_less':'减少采购','procure_more':'增加采购','supplier_diversify':'分散供应商','short_contract':'短合同','long_contract':'长合同','shared':'公共韧性投入','project':'门槛项目投入','no_shared':'停止新增公共投入','growth_bundle':'增长组合','safe_bundle':'韧性采购组合'}

def recipe_action(config,state,cid,recipe):
 base=company_action(config,state,cid,'balanced')
 if cid not in state.strategic_market.active_company_ids:return base
 p=base.to_dict();c=state.company(cid)
 env=MarketEnv(config);env.load_state(state);limits=env.get_action_constraints(cid,state.state_version);bounds=limits['bounds'];cash=max(0,c.financial.cash_balance_cents)
 if 'price' in recipe:p['price_cents']=p['price_cents']*recipe['price']//1000000
 if 'observed_price_factor' in recipe:p['price_cents']=c.commercial.price_cents*recipe['observed_price_factor']//1000000
 for key,value in recipe.items():
  if key.endswith('_cents') and key in bounds:
   if key in ('capacity_investment_cents','resilience_budget_cents','shared_resilience_contribution_cents') and state.rounds_remaining<=1:continue
   if key=='threshold_project_contribution_cents' and not limits['threshold_project_contribution_enabled']:continue
   p[key]=min(bounds[key]['max'],cash*value//1000000)
 if recipe.get('reserve'):
  for key in ('advertising_budget_cents','service_budget_cents','capacity_investment_cents','resilience_budget_cents','shared_resilience_contribution_cents','threshold_project_contribution_cents'):
   if key in p:p[key]=0
 if 'quantity' in recipe and limits.get('procurement_quantity_enabled'):
  p['procurement_quantity_orders']=min(c.operations.base_capacity_orders,p.get('procurement_quantity_orders',c.operations.base_capacity_orders)*recipe['quantity']//1000000)
 if 'supplier_share' in recipe and limits['supply_chain_enabled']:p['primary_supplier_share_ppm']=recipe['supplier_share']
 partners=[a for a in state.strategic_market.active_company_ids if a!=cid]
 if recipe.get('partner') in partners:partners=[recipe['partner']]+[a for a in partners if a!=recipe['partner']]
 if recipe.get('aid') and limits['mutual_aid_enabled'] and partners:
  p['mutual_aid_partner_company_id']=partners[0];p['mutual_aid_capacity_offer_orders']=max(1,c.operations.base_capacity_orders//10) if recipe['aid']=='offer' else 0;p['mutual_aid_capacity_request_orders']=max(1,c.operations.base_capacity_orders//10) if recipe['aid']=='request' else 0
 if recipe.get('coordinate') and limits['price_coordination_enabled'] and partners:
  p['price_coordination_partner_company_id']=partners[0];p['price_coordination_target_cents']=p['price_cents']
 if 'duration' in recipe and limits.get('supply_contract_enabled'):p['contract_duration_rounds']=min(recipe['duration'],state.rounds_remaining)
 return resolve_action_request(config,state,cid,p,source='advisor-v16',action_id=f'advisor:{state.round}:{cid}').action

def economic_key(action):
 p=action.to_dict();return json.dumps({k:v for k,v in p.items() if k not in ('action_id','strategy_summary')},sort_keys=True)

def candidates(config,state,cid,limit=16):
 rows=[];seen=set()
 # Include investment and cooperation before optional wider price refinements.
 order=['baseline','price_down','price_up','reserve','capacity','resilience','shared','project','no_shared','procure_less','procure_more','supplier_diversify','short_contract','long_contract','service','advertise','growth_bundle','safe_bundle','aid_offer','aid_request','coordinate','price_down_large','price_up_large']
 for key in order:
  action=recipe_action(config,state,cid,RECIPES[key]);identity=economic_key(action)
  if identity in seen:continue
  seen.add(identity);rows.append(dict(id=key,label=LABELS[key],recipe=RECIPES[key],action=action.to_dict()))
  if len(rows)>=limit:break
 return rows

def forecast(config,state,cid):
 legal=ObservationBuilder().build(state,cid,'public')
 return build_public_forecast_state(config=config,observation=legal,company_id=cid,persona_profile=PersonaRegistry.from_market_config(config).get('balanced_v1'))

def mechanism_coverage(config,state):
 return [dict(name=name,enabled=bool(active),simulation='MarketEnv全机制结算' if active else '未启用',search=search,assumption=assumption) for name,active,search,assumption in [
 ('价格/广告/服务/产能',True,'可搜索','对手成本、产能和资金以公开先验估计'),
 ('采购/供应商/合同',state.supply_chain is not None,'采购数量、供应商占比与期限','供应商报价/生产/可靠性/融资由引擎及场景响应策略更新，未读取私有订单簿'),
 ('库存/融资/破产',config.data.get('supply_chain',{}).get('strategic_policy'),'引擎演化和供应商策略响应','借款偿付遵循引擎，未搜索连续贷款合约'),
 ('政府检查/罚款/补贴',state.government is not None,'多种政府策略场景与偏离检查','其他角色目标是声明的模型假设，不是已知真实偏好'),
 ('消费者预算/偏好/退出',bool(config.data.get('autonomous_market')),'多种消费策略场景与偏离检查','固定群体预算体系；没有家庭工资和跨期储蓄'),
 ('公共韧性/门槛项目',state.shared_resilience is not None or bool(state.strategic_market and state.strategic_market.threshold_project),'贡献与不贡献','阈值及跨期收益由引擎结算'),
 ('互助/价格协调',bool(state.strategic_market and (state.strategic_market.mutual_aid_enabled or state.strategic_market.price_coordination_enabled)),'互助申请/提供与条件报价候选','有限预算下可能未搜索全部伙伴；对方是否接受取决于其动作，不能单方面保证协议成立')]]

class BudgetExhausted(Exception):pass
class Rollouts:
 def __init__(self,config,state,request):self.config=config;self.initial=state;self.request=request;self.used=0;self.cache={};self.action_cache={};self.cache_actions=True
 def company_action(self,state,actor,selected):
  # Only called after env.load_state/step validated this immutable state.
  # Cache action construction, never skip settlement or budget accounting.
  key=(state.state_hash,actor,json.dumps(selected,sort_keys=True))
  if self.cache_actions and key in self.action_cache:return self.action_cache[key]
  action=recipe_action(self.config,state,actor,selected) if isinstance(selected,dict) else company_action(self.config,state,actor,selected)
  if self.cache_actions:
   if len(self.action_cache)>=5000:self.action_cache.clear()
   self.action_cache[key]=action
  return action
 def evaluate(self,profile,scenario):
  key=json.dumps([profile,scenario],sort_keys=True)
  if key in self.cache:return self.cache[key]
  horizon=min(self.request.horizon,self.initial.rounds_remaining)
  if self.used+horizon>self.request.step_budget:raise BudgetExhausted('engine step budget exhausted')
  seed=int(hashlib.sha256(f'{self.initial.episode_seed}:{self.request.seed}:{scenario}'.encode()).hexdigest()[:15],16)
  state=replace(self.initial,episode_seed=seed);state=replace(state,state_hash=state_hash(state.to_dict()));env=MarketEnv(self.config);env.load_state(state)
  cid=self.request.company_id;start=state.company(cid);minimum=start.financial.cash_balance_cents;profit=welfare=0.;role_profits={a:0. for a in actor_ids(state)};trace=[]
  for t in range(horizon):
   before=state;actions={};extra={}
   for index,actor in enumerate(actor_ids(state)):
    if actor in profile:
     selected=profile[actor]
     if actor in state.company_ids:actions[actor]=self.company_action(state,actor,selected)
     else:extra[actor]=selected
    elif actor in state.company_ids:
     obs=observation(state,actor);style=(scenario+index)%4
     selected='value' if style==1 and obs['profit']<0 else 'margin' if style==2 and obs['sales']>=obs['capacity']*.8 else 'resilience' if style==3 else 'balanced'
     actions[actor]=self.company_action(state,actor,selected if actor in state.strategic_market.active_company_ids else 'balanced')
    else:
     choices=options(actor);extra[actor]=choices[(scenario+index+t//2)%len(choices)]
   result=env.step(f'{state.episode_id}:{state.round}:{state.state_version}',actions,actor_choices=extra);state=result.state_after;self.used+=1
   c=state.company(cid);minimum=min(minimum,c.financial.cash_balance_cents);profit+=self.request.discount**t*c.financial.round_profit_cents;welfare+=self.request.discount**t*state.welfare_accounting.round_total_economic_welfare_cents
   for actor in role_profits:
    v=state.company(actor).financial.round_profit_cents if actor in state.company_ids else state.supply_chain.supplier(actor).round_profit_cents if actor in state.supply_chain.supplier_ids else state.welfare_accounting.round_consumer_surplus_cents if actor=='consumers' else state.welfare_accounting.round_total_economic_welfare_cents
    role_profits[actor]+=self.request.discount**t*v
   trace.append(dict(round=before.round,own_action=actions[cid].to_dict(),joint_actions={a:v.to_dict() for a,v in actions.items()},other_actor_choices=extra,profit_cents=c.financial.round_profit_cents,welfare_cents=state.welfare_accounting.round_total_economic_welfare_cents,state_hash=state.state_hash))
  end=state.company(cid)
  metrics=dict(profit=profit,welfare=welfare,cash=end.financial.cash_balance_cents-start.financial.cash_balance_cents,share=end.commercial.market_share_ppm-start.commercial.market_share_ppm,capacity=end.operations.base_capacity_orders-start.operations.base_capacity_orders,resilience=end.risk.resilience_ppm-start.risk.resilience_ppm,minimum_cash=minimum,exited=cid not in state.strategic_market.active_company_ids,role_utilities=role_profits,trace=trace,final_state_hash=state.state_hash)
  self.cache[key]=metrics;return metrics
