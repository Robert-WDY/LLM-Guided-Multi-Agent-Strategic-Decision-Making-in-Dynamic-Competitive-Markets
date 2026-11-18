"""Matched ablations for aggregate demand, stockout and executable-price context."""
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from collect import OUT,cases
from models import design,attr_features,truth

BLOCKS={'demand':[0,7,9],'stockout':[1,8],'limits':[2,3,4,5,6]}


def cols(width,keep):
    start=width-10
    return list(range(start))+[start+i for key in keep for i in BLOCKS[key]]


def fit(m):
    models={}
    if m=='native':
        train=cases('train',m)
        for i,idx in enumerate((2,1)):
            xs=[];ys=[]
            for r in train:
                h=np.asarray(r['history']);x,sc=design(h,True,bool(i))
                xs.append(x);ys.extend(np.clip((h[6:,idx]-h[5:-1,idx])/sc,-5,5))
            x=np.concatenate(xs)
            for key,keep in [('demand',['demand']),('no_stockout',['demand','limits']),('no_limits',['demand','stockout'])]:
                take=cols(x.shape[1],keep)
                model=HistGradientBoostingRegressor(max_iter=85,max_leaf_nodes=15,min_samples_leaf=50,
                                                    l2_regularization=4,random_state=41)
                model.fit(x[:,take],ys);models[key,i]=(model,take)
    ys=[];xs=[]
    for r in cases('dev',m):
        h=np.asarray(r['history']);base=h[5:-1][:,[2,1]]
        scale=np.column_stack([design(h,m=='native',q)[1] for q in (False,True)])
        xs.extend(attr_features(h,base,scale,base,m=='native'));ys.extend(truth(r))
    weight=1/np.bincount(ys)[ys];weight/=weight.mean()
    attr=HistGradientBoostingClassifier(max_iter=65,max_leaf_nodes=10,min_samples_leaf=45,l2_regularization=4,random_state=42)
    attr.fit(xs,ys,sample_weight=weight)
    joblib.dump((models,attr),OUT/f'{m}-controls.joblib')


if __name__=='__main__':
    import sys
    fit(sys.argv[1])

