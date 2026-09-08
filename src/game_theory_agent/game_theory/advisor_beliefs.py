"""Prequential public-price learning. No competitor accounts or future observations."""
import hashlib, math
from statistics import mean
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field
from .objectives import AdviceRequest

class PriceFrame(BaseModel):
 model_config=ConfigDict(extra='forbid',strict=True)
 round:int=Field(ge=1)
 prices:dict[str,int]

class RobustRequest(AdviceRequest):
 history:list[PriceFrame]=Field(default_factory=list,max_length=60)
 learn_opponents:bool=True
 response_aware:bool=True
 search_method:Literal['expanded','grid']='expanded'
 response_passes:int=Field(default=1,ge=1,le=2)
 finalists:int=Field(default=2,ge=1,le=3)

def public_frame(state):return dict(round=state.round,prices={c.company_id:c.commercial.price_cents for c in state.companies})

def fit_prices(prices):
 counts=[1.,2.,1.];totals=[-.05,0.,.05];weights=[1.,2.,1.]
 for a,b in zip(prices,prices[1:]):
  change=max(-.2,min(.2,b/a-1));k=0 if change<-.01 else 2 if change>.01 else 1
  counts=[1+(c-1)*.85 for c in counts];counts[k]+=1
  totals=[t*.85 for t in totals];weights=[w*.85 for w in weights];totals[k]+=change;weights[k]+=1
 p=[v/sum(counts) for v in counts];changes=[t/w for t,w in zip(totals,weights)]
 return dict(probabilities=p,changes=changes,expected_change=sum(x*y for x,y in zip(p,changes)),observations=max(0,len(prices)-1))

def price_audit(prices):
 errors=[];flat=[];covered=[];widths=[]
 for end in range(1,len(prices)):
  prediction=prices[end-1]*(1+fit_prices(prices[:end])['expected_change']);actual=prices[end]
  if len(errors)>=8:
   # Rolling empirical residual band; report observed coverage, not guaranteed coverage.
   q=sorted(errors[-30:])[min(len(errors[-30:])-1,math.ceil(.9*len(errors[-30:]))-1)]
   covered.append(abs(actual-prediction)/prices[end-1]<=q);widths.append(q)
  errors.append(abs(actual-prediction)/prices[end-1]);flat.append(abs(actual-prices[end-1])/prices[end-1])
 return dict(n=len(errors),mean_relative_error=mean(errors) if errors else None,flat_relative_error=mean(flat) if flat else None,interval_samples=len(covered),empirical_coverage=mean(covered) if covered else None,mean_relative_halfwidth=mean(widths) if widths else None,scope='逐时点只用此前数据；90%历史残差分位带，非分布变化下覆盖保证。')

def learn(state,cid,history):
 frames=[v.model_dump() if isinstance(v,PriceFrame) else v for v in history]
 if any(f['round']>=state.round for f in frames) or any(a['round']>=b['round'] for a,b in zip(frames,frames[1:])):raise ValueError('history must be strictly increasing and before the current round')
 if any(set(f['prices'])!=set(state.company_ids) or any(type(v)!=int or v<=0 for v in f['prices'].values()) for f in frames):raise ValueError('history must contain positive public prices for exactly the current companies')
 frames.append(public_frame(state));result={}
 for actor in state.company_ids:
  if actor==cid:continue
  values=[f['prices'][actor] for f in frames]
  result[actor]={**fit_prices(values),**{'calibration':price_audit(values)}}
 return result

def sampled_recipe(belief,seed,scenario,actor):
 u=int(hashlib.sha256(f'{seed}:{scenario}:{actor}'.encode()).hexdigest()[:12],16)/16**12
 p=belief['probabilities'];k=0 if u<p[0] else 1 if u<p[0]+p[1] else 2
 return {'observed_price_factor':round(1000000*(1+belief['changes'][k]))}
