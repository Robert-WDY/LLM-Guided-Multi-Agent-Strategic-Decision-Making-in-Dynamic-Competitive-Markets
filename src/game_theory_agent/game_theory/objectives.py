"""Versioned objective contract. Scores are utility indices, never welfare money."""
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator

COMPONENTS=('profit','welfare','cash','share','capacity','resilience')
SCALES=dict(profit=10000000,welfare=10000000,cash=10000000,share=100000,capacity=1000,resilience=100000)
PHASES={
 'survival':dict(profit=.2,cash=.7,resilience=.1),
 'expansion':dict(profit=.35,cash=.1,share=.25,capacity=.2,resilience=.1),
 'mature':dict(profit=.6,cash=.2,resilience=.2),
 'catch_up':dict(profit=.4,cash=.15,share=.3,capacity=.15),
 'closing':dict(profit=.7,cash=.3),
}
class AdviceRequest(BaseModel):
 model_config=ConfigDict(extra='forbid',strict=True)
 company_id:str='company_A'
 goal:Literal['company','profit','welfare','balanced','custom']='company'
 phase:Literal['auto','survival','expansion','mature','catch_up','closing']='auto'
 weights:dict[str,float]|None=None
 horizon:int=Field(default=3,ge=1,le=10)
 scenarios:int=Field(default=3,ge=2,le=8)
 max_candidates:int=Field(default=16,ge=4,le=48)
 step_budget:int=Field(default=1800,ge=100,le=12000)
 risk_aversion:float=Field(default=.25,ge=0,le=2)
 discount:float=Field(default=.95,gt=0,le=1)
 cash_floor_cents:int=Field(default=0,ge=0,le=1000000000)
 seed:int=Field(default=360101,ge=0,le=2147483647)
 diagnostics:bool=True
 backtest:bool=True
 @model_validator(mode='after')
 def valid_weights(self):
  if self.goal=='custom':
   if not self.weights or set(self.weights)-set(COMPONENTS) or any(not 0<=w<=1 for w in self.weights.values()) or abs(sum(self.weights.values())-1)>1e-8:raise ValueError('custom weights must be known, nonnegative and sum to one')
  elif self.weights is not None:raise ValueError('weights require custom goal')
  return self

def objective(request,state):
 c=state.company(request.company_id);cash=c.financial.cash_balance_cents
 if request.phase!='auto':phase=request.phase;reason='用户指定阶段'
 elif cash<max(1000000,c.commercial.price_cents*max(1,c.commercial.sales_orders)//2):phase='survival';reason='现金不足半轮近期销售额或低于1万元现金阈值'
 elif state.rounds_remaining<=3:phase='closing';reason='剩余回合不超过3轮'
 elif c.commercial.market_share_ppm<1000000//len(state.company_ids)//2:phase='catch_up';reason='份额低于均分份额的一半'
 elif state.state_version<3:phase='expansion';reason='处于前3轮且没有现金危机'
 else:phase='mature';reason='现金与份额未触发其他条件'
 w=PHASES[phase] if request.goal=='company' else {'profit':1.} if request.goal=='profit' else {'welfare':1.} if request.goal=='welfare' else {'profit':.5,'welfare':.5} if request.goal=='balanced' else request.weights
 return dict(version='objective-v16.0.0',goal=request.goal,phase=phase,phase_reason=reason,weights={k:w.get(k,0.) for k in COMPONENTS},scales=SCALES,discount=request.discount,risk_aversion=request.risk_aversion,cash_floor_cents=request.cash_floor_cents,scope='公司单边行动；福利目标不授权控制其他主体。现金、产能和韧性为独立效用维度，不与利润相加解释成经济福利。终端资产为公开指标代理，未声称精确长期企业价值。')

def score(metrics,spec):
 contributions={k:metrics.get(k,0)/SCALES[k]*spec['weights'][k] for k in COMPONENTS}
 return dict(total=sum(contributions.values()),contributions=contributions)

def aggregate(outcomes,spec):
 from statistics import mean,pstdev
 values=[score(x,spec)['total'] for x in outcomes];tail=sorted(values)[:max(1,(len(values)+3)//4)]
 expected=mean(values);worst=mean(tail)
 return dict(score=expected-spec['risk_aversion']*(expected-worst),expected=expected,tail_mean=worst,standard_deviation=pstdev(values),safe=all(x['minimum_cash']>=spec['cash_floor_cents'] and not x['exited'] for x in outcomes),metrics={k:mean(x.get(k,0) for x in outcomes) for k in COMPONENTS},contributions={k:mean(score(x,spec)['contributions'][k] for x in outcomes) for k in COMPONENTS},sample_scores=values)
ADVISOR_MODES = {'advisor_company': 'company', 'advisor_profit': 'profit', 'advisor_welfare': 'welfare'}
