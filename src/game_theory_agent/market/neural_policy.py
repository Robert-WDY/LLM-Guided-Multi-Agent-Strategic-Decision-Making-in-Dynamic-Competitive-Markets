"""Small trained policy selector: public/own features -> feasible option."""
import json
import hashlib
from pathlib import Path
import numpy as np
from .actor_policies import options
from .protocols import sha256_hash

MODEL=Path(__file__).resolve().parents[3]/"artifacts/four-actor-policy-v1.json"
FEATURES=("remaining","demand","stockout","mean_price","cash","profit","sales","capacity",
          "resilience","orders","inventory","debt","reliability","hhi","outside","previous_spending","previous_refunds","price_anchor")


def artifact_hash(value):
    return "sha256:"+hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode("utf-8")).hexdigest()


def role(actor):
    return "company" if actor.startswith("company_") else actor if actor in ("consumers","government") else "supplier"


def features(obs):
    obs=dict(obs);p=obs.get("prices",{});obs["mean_price"]=sum(p.values())/max(1,len(p))
    if isinstance(obs.get("resilience"),dict):
        r=obs["resilience"];obs["resilience"]=sum(r.values())/max(1,len(r))
    return np.array([float(obs.get(k,0)) for k in FEATURES],dtype=np.float64)


def physics_hash(config):
    data=config.to_dict()
    for key in ("config_id","config_version","environment_version","schema_versions"):data.pop(key,None)
    return sha256_hash(data)


def fit(rows,seed=250101,epochs=400):
    rng=np.random.default_rng(seed);x=np.array([r["x"] for r in rows]);y=np.array([r["label"] for r in rows])
    count=len(rows[0]["options"]);center=x.mean(0);scale=x.std(0);scale[scale<1]=1;x=(x-center)/scale
    w1=rng.normal(0,0.12,(x.shape[1],16));b1=np.zeros(16);w2=rng.normal(0,0.12,(16,count));b2=np.zeros(count)
    losses=[]
    for epoch in range(epochs):
        h=np.tanh(x@w1+b1);logits=h@w2+b2;logits-=logits.max(1,keepdims=True)
        p=np.exp(logits);p/=p.sum(1,keepdims=True);loss=-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()
        if epoch in (0,epochs-1):losses.append(float(loss))
        g=p.copy();g[np.arange(len(y)),y]-=1;g/=len(y)
        gw2=h.T@g+0.0001*w2;gb2=g.sum(0);dh=(g@w2.T)*(1-h*h)
        gw1=x.T@dh+0.0001*w1;gb1=dh.sum(0)
        w1-=0.08*gw1;b1-=0.08*gb1;w2-=0.08*gw2;b2-=0.08*gb2
    return dict(options=rows[0]["options"],center=center.tolist(),scale=scale.tolist(),w1=w1.tolist(),b1=b1.tolist(),
                w2=w2.tolist(),b2=b2.tolist(),loss_initial=losses[0],loss_final=losses[-1],training_rows=len(rows))


def infer(model,x):
    z=(x-np.array(model["center"]))/np.array(model["scale"])
    logits=np.tanh(z@np.array(model["w1"])+np.array(model["b1"]))@np.array(model["w2"])+np.array(model["b2"])
    return model["options"][int(logits.argmax())]


def predict(actor,obs,*,path=MODEL,config=None):
    artifact=json.loads(Path(path).read_text(encoding="utf-8"))
    expected=artifact.pop("artifact_sha256")
    if artifact_hash(artifact)!=expected:raise ValueError("trained policy artifact hash mismatch")
    if config is not None and artifact["physics_hash"]!=physics_hash(config):raise ValueError("trained policy does not match market parameters")
    model=artifact["models"][role(actor)]
    if model["options"]!=list(options(actor)) or artifact["features"]!=list(FEATURES):raise ValueError("trained policy schema mismatch")
    return infer(model,features(obs))
