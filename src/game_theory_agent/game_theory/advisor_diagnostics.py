"""Finite empirical deviation checks. Incomplete searches never certify Nash."""
import json
from copy import deepcopy
from statistics import mean
from .advisor_market import RECIPES,BudgetExhausted
from .objectives import score
from game_theory_agent.market.actor_policies import actor_ids,options

def response_search(payoff,choices,initial,passes=2):
 profile=deepcopy(initial);actors=list(choices);visited=set();cycle=False;evaluations=0
 def value(p):
  nonlocal evaluations
  result=payoff(p);evaluations+=1;return result
 try:
  for _ in range(passes):
   key=json.dumps(profile,sort_keys=True)
   if key in visited:cycle=True;break
   visited.add(key);changed=False
   for actor in actors:
    current=value(profile)[actor];best=current;selection=profile[actor]
    for candidate in choices[actor]:
     other={**profile,actor:candidate};v=value(other)[actor]
     if v>best+1e-9:best=v;selection=candidate
    if selection!=profile[actor]:profile[actor]=selection;changed=True
   if not changed:break
  base=value(profile);gaps={};checked={}
  for actor in actors:
   vals=[]
   for candidate in choices[actor]:vals.append(value({**profile,actor:candidate})[actor])
   gaps[actor]=max(0,max(vals)-base[actor]);checked[actor]=len(vals)
  return dict(profile=profile,deviation_gains=gaps,checked=checked,complete=True,restricted_pure_nash=max(gaps.values(),default=0)<1e-8,cycle=cycle,evaluations=evaluations)
 except BudgetExhausted:
  return dict(profile=profile,deviation_gains={},checked={},complete=False,restricted_pure_nash=None,cycle=cycle,evaluations=evaluations)

def diagnose(runner,spec,recipe):
 state=runner.initial;cid=runner.request.company_id;actors=actor_ids(state);remaining=runner.request.step_budget-runner.used
 if remaining<50:return dict(status='budget_insufficient',scope='未完成多方诊断，不能声称均衡')
 choices={a:['baseline','price_down','price_up','shared'] if a in state.company_ids else list(options(a)) for a in actors}
 initial={a:'baseline' if a in state.company_ids else 'balanced' for a in actors};choices[cid].append('recommended');initial[cid]='recommended'
 def decode(profile):return {a:(recipe if v=='recommended' else RECIPES[v]) if a in state.company_ids else v for a,v in profile.items()}
 limit=runner.used+int(remaining*.3)
 def pay(profile,joint=False):
  if runner.used+2*runner.request.horizon>limit:raise BudgetExhausted()
  outcomes=[runner.evaluate(decode(profile),s) for s in (2000,2001)]
  return {a:mean(o['welfare']/10000000 if joint else score(o,spec)['total'] if a==cid else o['role_utilities'][a]/10000000 for o in outcomes) for a in actors}
 recommendation_check=response_search(pay,choices,initial,passes=0)
 limit=runner.used+int((runner.request.step_budget-runner.used)*.5)
 equilibrium=response_search(pay,choices,initial)
 limit=runner.used+int((runner.request.step_budget-runner.used)*.65)
 joint=response_search(lambda p:pay(p,True),choices,initial,passes=1)
 joint['scope']='研究者联合福利协调方案，改变所有角色的搜索目标；不是各主体会自愿接受的均衡，也不自动执行。'
 # Separately measure unilateral private incentives around the welfare profile.
 incentives={}
 try:
  base=pay(joint['profile'])
  for a in actors:
   incentives[a]=max(0,max(pay({**joint['profile'],a:v})[a]-base[a] for v in choices[a]))
 except BudgetExhausted:pass
 joint['private_deviation_gains']=incentives;joint['participation_check_complete']=len(incentives)==len(actors)
 local=dict(status='not_evaluated',scope='固定其余角色和预测状态，仅用于局部结构诊断')
 try:
  if runner.used+14*runner.request.horizon>runner.request.step_budget:raise BudgetExhausted()
  left,right=state.company_ids[:2];a,b=left,right;matrix=[]
  for x in ('shared','no_shared'):
   row=[]
   for y in ('shared','no_shared'):
    p=decode(initial);p[a]=RECIPES[x];p[b]=RECIPES[y];o=runner.evaluate(p,4000)
    row.append([o['role_utilities'][a],o['role_utilities'][b]])
   matrix.append(row)
  R,S,T,P=matrix[0][0][0],matrix[0][1][0],matrix[1][0][0],matrix[1][1][0]
  r,t,s,p=matrix[0][0][1],matrix[0][1][1],matrix[1][0][1],matrix[1][1][1]
  pd=T>R>P>S and t>r>p>s
  coordination=R>T and P>S and r>t and p>s
  everyone=decode(initial)
  for a in state.company_ids:everyone[a]=RECIPES['shared']
  together=runner.evaluate(everyone,4000);withdraw=runner.evaluate({**everyone,cid:RECIPES['no_shared']},4000)
  gain=withdraw['role_utilities'][cid]-together['role_utilities'][cid];social=withdraw['welfare']-together['welfare']
  local=dict(status='evaluated',actors=[left,right],payoffs=matrix,prisoners_dilemma=pd,coordination=coordination,withdrawal_private_gain_cents=gain,withdrawal_welfare_change_cents=social,free_riding_incentive=gain>0 and social<0,scope='共享韧性贡献/不贡献的单一预测场景、多轮私有利润。不是全市场类型或真实行为动机；门槛项目另作为可执行候选比较。')
 except BudgetExhausted:pass
 return dict(status='evaluated',player_count=len(actors),company_count=len(state.company_ids),recommendation_check=recommendation_check,equilibrium=equilibrium,joint_welfare=joint,local_structure=local,assumed_objectives={a:spec['goal'] if a==cid else 'private_profit' if a in state.company_ids or a in state.supply_chain.supplier_ids else 'consumer_surplus' if a=='consumers' else 'social_welfare' for a in actors},scope='其他公司的利润、消费者剩余、政府福利是模型假设。有限策略/期限/情景的单边偏离检查；未检查策略不能据此获得误差上界或动态纳什证明。')
