"""Development-only fitting; deployment consumes frozen numeric coefficients."""
import numpy as np
from scipy.optimize import minimize


def features(history, decision):
    h=np.asarray(history[-24:],dtype=float)
    p,peer,q,profit=h[:,:4].T
    draft,offer=decision['draft'],decision['proposed']
    recent=max(1.e-8,np.mean(np.abs(profit)))
    sign=np.sign(offer-draft)
    leader=p[np.argmax(profit)]
    return [decision['score'],sign,offer/draft-1,draft/p[-1]-1,peer[-1]/p[-1]-1,
            sign*(leader/draft-1),sign*(peer[-1]/draft-1),np.std(profit)/recent,
            (np.mean(profit[-4:])-np.mean(profit[:4]))/recent,
            np.std(p)/np.mean(p), np.mean(q[-4:])/max(.01,np.mean(q[:4]))-1,
            np.mean(profit[-4:])/max(.01,np.mean(q[-4:])*p[-1]),
            sign*np.mean(profit[-4:])/max(.01,np.mean(q[-4:])*p[-1])]


def fit(rows):
    # Strictly positive improvement is the target. Duplicate prefix rows in shock twins
    # receive a single vote during training; confirmation averages each scenario as specified.
    unique={}
    for r in rows:
        for v in r['labels']:
            x=features(r['rows'][:v['t']-1],v)
            unique[(r['seed'],v['t'],tuple(x),v['delta'])]=(x,v['delta']>1.e-8)
    x,y=map(np.asarray,zip(*unique.values()))
    center=x.mean(axis=0)
    scale=np.maximum(x.std(axis=0),.001)
    z=np.column_stack([np.ones(len(x)),np.clip((x-center)/scale,-5,5)])
    def objective(b):
        a=z@b
        return np.logaddexp(0,a).sum()-y@a+2*np.sum(b[1:]**2)
    res=minimize(objective,np.zeros(z.shape[1]),method='BFGS',tol=1.e-5)
    return dict(center=center.tolist(),scale=scale.tolist(),coef=res.x.tolist(),examples=len(y))


def predict(model, history, decision):
    x=features(history,decision)
    z=np.clip((x-np.asarray(model['center']))/model['scale'],-5,5)
    b=np.asarray(model['coef'])
    return float(1/(1+np.exp(-np.clip(b[0]+z@b[1:],-40,40))))

