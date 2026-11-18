"""Offline estimators and the public, prequential inference path."""
from functools import lru_cache
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from collect import OUT,RAW,ROOT,read,save,cases
from game_theory_agent.game_theory.public_forecast import inputs,realized_skill


def design(h,native,quote,extra=None):
    a=np.asarray(h,dtype=float)
    args=[(a[max(0,t-24):t],a[t,6],a[t,7],a[t,8]) for t in range(6,len(a))]
    if extra is not None:args.extend((a[-24:],p,lo,hi) for p,lo,hi in extra)
    x,sc=zip(*(inputs(*arg,native_context=native,quote=quote) for arg in args))
    return np.asarray(x),np.asarray(sc)


def fit_fc(m):
    result={};errors={}
    for context in (['basic','full'] if m=='native' else ['basic']):
        result[context]={};errors[context]={}
        for j,name in enumerate(('sales','quote')):
            idx=(2,1)[j]
            def bundle(phase):
                x=[];sc=[];y=[];carry=[]
                for r in cases(phase,m):
                    a=np.asarray(r['history']);v,k=design(a,context=='full',bool(j))
                    x.append(v);sc.extend(k);y.extend(a[6:,idx]);carry.extend(a[5:-1,idx])
                return np.concatenate(x),np.asarray(sc),np.asarray(y),np.asarray(carry)
            x,scale,y,base=bundle('train')
            xd,sd,yd,bd=bundle('dev')
            target=np.clip((y-base)/scale,-5,5)
            models={'ridge':make_pipeline(StandardScaler(),Ridge(alpha=60)),
                    'trees':HistGradientBoostingRegressor(max_iter=85,max_leaf_nodes=15,min_samples_leaf=50,
                                                        l2_regularization=4,random_state=41)}
            trials={'carry':np.mean(abs(yd-bd))}
            fitted={}
            for kind,model in models.items():
                model.fit(x,target);fitted[kind]=model
                residual=model.predict(xd)*sd
                for weight in (.5,1.):
                    trials[f'{kind}-{weight}']=np.mean(abs(np.maximum(0,bd+weight*residual)-yd))
            best=min(trials,key=trials.get)
            if best=='carry': result[context][name]=(None,0.)
            else:
                key,weight=best.split('-')
                result[context][name]=(fitted[key],float(weight))
            errors[context][name]=dict(selected=best,dev_mae=trials)
    return result,errors


def forecasts(fc,h,context,extra=None):
    a=np.asarray(h);n=0 if extra is None else len(extra)
    pred=[];scales=[];base=[]
    for j,name in enumerate(('sales','quote')):
        idx=(2,1)[j]
        x,sc=design(a,context=='full',bool(j),extra)
        b=np.r_[a[5:-1,idx],np.repeat(a[-1,idx],n)]
        model,weight=fc[context][name]
        pred.append(b if model is None else np.maximum(0,b+weight*sc*model.predict(x)))
        scales.append(sc);base.append(b)
    return tuple(np.column_stack(z) for z in (pred,scales,base))


def attr_features(h,pred,scale,base,native):
    a=np.asarray(h)[6:]
    observed=a[:,[2,1]]
    error=np.clip((observed-pred)/scale,-4,4)
    change=np.clip((observed-base)/scale,-4,4)
    rows=[]
    for t in range(len(a)):
        f=[]
        for k in (1,4,12):
            for series in (error,change):
                recent=series[max(0,t+1-k):t+1]
                f.extend(recent.mean(axis=0));f.extend(abs(recent).mean(axis=0))
        # Price and sales level compared with the first six public observations.
        initial=np.asarray(h[:6])
        f.extend((a[t,0]/initial[:,0].mean()-1,a[t,1]/initial[:,1].mean()-1,
                  (a[t,2]-initial[:,2].mean())/max(1.,initial[:,2].mean())))
        if native:
            f.extend((a[t,4]/max(1.,np.mean(initial[:,4]))-1,a[t,5]/max(1.,a[t,4]),
                      a[t,6]/a[t,0]-1,a[t,7]/a[t,0]-1))
        rows.append(f)
    return rows


def truth(r):
    return np.array([0 if t<r['when'] or r['condition']=='stable' else 1 if r['condition']=='demand' else 2 for t in range(7,61)])


def fit_attr(m,fc,context):
    x=[];y=[]
    for r in cases('dev',m):
        h=r['history'];p,sc,b=forecasts(fc,h,context)
        x.extend(attr_features(h,p,sc,b,context=='full'));y.extend(truth(r))
    weights=1/np.bincount(y)[y];weights=weights/weights.mean()
    model=HistGradientBoostingClassifier(max_iter=65,max_leaf_nodes=10,min_samples_leaf=45,l2_regularization=4,random_state=42)
    model.fit(x,y,sample_weight=weights)
    return model


def diagnosis(bundle,h,context,extra=None):
    p,sc,b=forecasts(bundle['fc'],h,context,extra)
    n=len(h)-6
    attribution=bundle['attribution'][context].predict_proba(attr_features(h,p[:n],sc[:n],b[:n],context=='full'))
    return p,sc,b,attribution


def flags(probs,cut):
    typ=1+probs[:,1:].argmax(axis=1)
    return np.where((probs.max(axis=1)>=cut)&(probs.argmax(axis=1)!=0),typ,0)


def train(m):
    fc,selection=fit_fc(m)
    bundle=dict(fc=fc,attribution={},alert={})
    for context in fc:
        bundle['attribution'][context]=fit_attr(m,fc,context)
        highs=[]
        for r in cases('cal',m):
            if r['condition']!='stable':continue
            *_,pr=diagnosis(bundle,r['history'],context)
            highs.append(np.max(np.where(pr.argmax(axis=1)==0,0,pr[:,1:].max(axis=1))))
        bundle['alert'][context]=float(np.quantile(highs,.90,method='higher'))+1.e-9
    joblib.dump(bundle,OUT/f'{m}-forecast.joblib')
    save(OUT/f'{m}-development.json',dict(selection=selection,alert=bundle['alert']))


@lru_cache
def bundle(m):
    return joblib.load(OUT/f'{m}-forecast.joblib')


if __name__=='__main__':
    import sys
    train(sys.argv[1])

