"""Features available to the focal firm at a price decision, before the market settles."""
import numpy as np


def inputs(history, request, lower, upper, *, native_context=False, quote=False):
    """Rows: own price, rival price, own sales/profit, public demand/stockout,
    own requested price, prior available price floor and ceiling.
    No evaluator clock, shock labels or contemporaneous outcomes are arguments.
    """
    if len(history)<6:
        raise ValueError("six settled rounds required")
    a=np.asarray(history[-24:],dtype=float)
    p,r,q=a[:,0],a[:,1],a[:,2]
    qscale=max(1.,q[-8:].mean())
    # Simultaneous rival choice cannot respond to our unannounced request.
    plan=p[-1] if quote else request
    x=[np.log1p(p[-1]),np.log1p(r[-1]),np.log1p(q[-1]),np.log1p(qscale),
       p[-1]/r[-1]-1,plan/p[-1]-1]
    for lag in (1,2,4,8):
        i=max(0,len(a)-1-lag)
        x.extend((p[-1]/p[i]-1,r[-1]/r[i]-1,(q[-1]-q[i])/qscale))
    for k in (4,12):
        x.extend((np.mean(p[-k:])/p[-1]-1,np.mean(r[-k:])/r[-1]-1,
                  (q[-1]-q[-k:].mean())/qscale,np.std(q[-k:])/qscale,
                  np.std(r[-k:])/r[-1]))
    if native_context:
        demand,stockout=a[:,4],a[:,5]
        ds=max(1.,np.mean(demand[-8:]))
        x.extend((np.log1p(demand[-1]),stockout[-1]/max(demand[-1],1.),
                  lower/p[-1]-1,upper/p[-1]-1,
                  np.clip(plan,lower,upper)/p[-1]-1,
                  (a[-1,6]-p[-1])/p[-1],np.mean(np.abs(a[-4:,6]-p[-4:])/p[-4:]),
                  (demand[-1]-demand[max(0,len(a)-5)])/ds,
                  np.mean(stockout[-4:]/np.maximum(demand[-4:],1.)),
                  np.mean(q[-4:]/np.maximum(demand[-4:],1.))))
    return np.clip(x,-40.,40.), qscale if not quote else r[-1]


def realized_skill(actual, carry, predicted, scale):
    """Trailing relative errors, bounded even on a perfectly constant target."""
    actual,carry,predicted,scale=map(np.asarray,(actual,carry,predicted,scale))
    e0=np.mean(np.abs(actual-carry)/scale,axis=0)
    e1=np.mean(np.abs(actual-predicted)/scale,axis=0)
    return (e0-e1)/(e0+e1+.001)

