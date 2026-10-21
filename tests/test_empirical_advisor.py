from copy import deepcopy
import pytest
from game_theory_agent.game_theory.empirical_advisor import apply_calibration,VERSION
from test_calibrated_advisor import evidence

def test_empirical_gate_preserves_legacy_ranking_and_calibrates_lower_bound(evidence):
    c,s,r,cal=evidence;cal['policy']=VERSION;r['version']=VERSION
    out=apply_calibration(r,c.config_sha256,cal)
    assert out['disposition']=='recommend' and out['rankings']==r['rankings']
    assert out['empirical_calibration']['adjusted_margin']==pytest.approx(.3)
    cal['goals']['profit']['overprediction_budget']['5']=.6
    out=apply_calibration(r,c.config_sha256,cal)
    assert out['disposition']=='abstain' and out['action']==r['draft_action']

@pytest.mark.parametrize('kind',['missing','version','config','mode','goal','risk','scale','actor','budget','support','sample','horizon','unsafe'])
def test_empirical_scope_is_fail_closed(evidence,kind):
    c,s,r,cal=evidence;cal['policy']=VERSION
    if kind=='missing':cal=None
    elif kind=='version':cal['policy']='wrong'
    elif kind=='config':cal['config_sha256']=[]
    elif kind=='mode':r['mode']='interactive'
    elif kind=='goal':r['objective']['goal']='custom'
    elif kind=='risk':r['objective']['risk_aversion']=1.
    elif kind=='scale':r['company_count']=4
    elif kind=='actor':r['draft_action']['agent_id']='company_B'
    elif kind=='budget':r['search']['complete']=False
    elif kind=='support':r['public_features']['own_cash']+=1
    elif kind=='sample':cal['goals']['profit']['selected_cases']=1
    elif kind=='horizon':r['validation'].pop()
    elif kind=='unsafe':r['validation'][0]['safe']=False
    out=apply_calibration(r,c.config_sha256,cal)
    assert not out['empirical_calibration']['passed'] and out['action']==r['draft_action']
