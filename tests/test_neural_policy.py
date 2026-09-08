import json
import numpy as np
import pytest
from game_theory_agent.market.neural_policy import fit,infer,predict,features,FEATURES,artifact_hash
from game_theory_agent.market.protocols import sha256_hash


def test_training_changes_weights_and_learns_nonlinear_boundary(tmp_path):
    rng=np.random.default_rng(25);x=rng.normal(0,1,(300,len(FEATURES)));x[:,2:]=0
    rows=[dict(x=v.tolist(),label=int(v[0]*v[1]>0),options=["balanced","value","margin","resilience"]) for v in x]
    model=fit(rows,epochs=800)
    assert model["loss_final"]<model["loss_initial"]*0.6
    holdout=rng.normal(0,1,(100,len(FEATURES)));holdout[:,2:]=0
    accuracy=sum(infer(model,v)==("value" if v[0]*v[1]>0 else "balanced") for v in holdout)/100
    assert accuracy>0.75
    artifact=dict(models={"company":model},features=list(FEATURES));artifact["artifact_sha256"]=artifact_hash(artifact)
    p=tmp_path/"model.json";p.write_text(json.dumps(artifact),encoding="utf-8")
    assert predict("company_A",{},path=p) in model["options"]
    artifact["models"]["company"]["w1"][0][0]+=1;p.write_text(json.dumps(artifact),encoding="utf-8")
    with pytest.raises(ValueError,match="hash"):predict("company_A",{},path=p)
