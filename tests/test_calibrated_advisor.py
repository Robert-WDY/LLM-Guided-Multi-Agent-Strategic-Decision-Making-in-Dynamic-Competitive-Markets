from copy import deepcopy
from dataclasses import replace
import pytest
from game_theory_agent.game_theory.calibrated_advisor import apply_calibration, calibration_decision, plausible_states, public_features, POLICY_VERSION
from game_theory_agent.game_theory.reliable_advisor import ReliableRequest, advise
from game_theory_agent.game_theory.reliable_response import fit
from game_theory_agent.market import MarketEnv, load_market_config
from game_theory_agent.market.actor_policies import company_action
from game_theory_agent.market.protocols import state_hash
from game_theory_agent.game_theory.advisor_market import forecast

@pytest.fixture
def evidence():
    c=load_market_config('configs/market_v14_local.yaml');s=MarketEnv(c).reset(company_ids=['company_A','company_B'],episode_id='calibrated-test',episode_seed=123,max_rounds=20,cooperation_mode='combined_v1')
    draft=company_action(c,s,'company_A','balanced').to_dict()
    r=advise(c,s,ReliableRequest(draft_action=draft,goal='profit',mode='research',step_budget=1),model=fit([]))
    r.update(disposition='recommend',recommended_id='price_up',action={**draft,'price_cents':draft['price_cents']+1},public_features=public_features(c,s,'company_A'),uncertainty_validation=[])
    r['search']['complete']=True
    r['validation']=[dict(horizon=h,mean_gain=1.,lower_bound=.5,safe=True) for h in (3,5)]
    r['uncertainty_validation']=[dict(horizon=h,mean_gain=.8,lower_bound=.4,safe=True) for _ in range(2) for h in (3,5)]
    cal=dict(policy=POLICY_VERSION,config_sha256=[c.config_sha256],minimum_gain=0.,goals={'profit':dict(selected_cases=30,seed_clusters=20,overprediction_budget={'3':.2,'5':.2})},support={k:[v,v] for k,v in r['public_features'].items()})
    return c,s,r,cal

def test_overprediction_vetoes_model_positive_and_preserves_draft(evidence):
    c,s,r,cal=evidence;cal['goals']['profit']['overprediction_budget']['5']=1.1
    out=apply_calibration(r,c.config_sha256,cal)
    assert not out['empirical_calibration']['passed'] and out['action']==r['draft_action'] and out['proposed_action']==r['action']
    assert r['disposition']=='recommend'  # no mutation of persisted proposal

def test_supported_positive_margin_can_intervene(evidence):
    c,s,r,cal=evidence;out=apply_calibration(r,c.config_sha256,cal)
    assert out['disposition']=='recommend' and out['empirical_calibration']['adjusted_margin']>0

@pytest.mark.parametrize('kind',['missing','wrong_version','unknown_config','insufficient','out_of_support','incomplete','missing_horizon','unsafe','untrained_objective','untrained_mode','untrained_response','untrained_scale'])
def test_fail_closed_on_unsupported_cases(evidence,kind):
    c,s,r,cal=evidence
    if kind=='missing':cal=None
    elif kind=='wrong_version':cal['policy']='other'
    elif kind=='unknown_config':cal['config_sha256']=[]
    elif kind=='insufficient':cal['goals']['profit']['seed_clusters']=7
    elif kind=='out_of_support':r['public_features']['round']+=1
    elif kind=='incomplete':r['search']['complete']=False
    elif kind=='missing_horizon':r['uncertainty_validation'].pop()
    elif kind=='unsafe':r['uncertainty_validation'][0]['safe']=False
    elif kind=='untrained_objective':r['objective']['goal']='company'
    elif kind=='untrained_mode':r['mode']='interactive'
    elif kind=='untrained_response':r['response_model']='none'
    elif kind=='untrained_scale':r['company_count']=4
    assert not calibration_decision(r,c.config_sha256,cal)['passed']

def test_pressure_states_do_not_leak_private_peer_assets(evidence):
    c,s,_,_=evidence;p,_=forecast(c,s,'company_A');peer=s.companies[1]
    changed=replace(peer,brand=replace(peer.brand,brand_awareness_ppm=900000,service_quality_ppm=900000),financial=replace(peer.financial,cash_balance_cents=123456))
    altered=replace(s,companies=(s.companies[0],changed));altered=replace(altered,state_hash=state_hash(altered.to_dict()));q,_=forecast(c,altered,'company_A')
    assert public_features(c,s,'company_A')==public_features(c,altered,'company_A')
    assert [x.to_dict() for _,x in plausible_states(p,'company_A')]==[x.to_dict() for _,x in plausible_states(q,'company_A')]
    for _,x in plausible_states(p,'company_A'):
        env=MarketEnv(c);env.load_state(x)
        assert x.company('company_A')==s.company('company_A')


def test_extra_pressure_checks_share_the_original_compute_budget(evidence,monkeypatch):
    from game_theory_agent.game_theory import calibrated_advisor as module
    c,s,r,cal=evidence;r['search']['used_steps']=95
    monkeypatch.setattr(module.legacy,'advise',lambda *a,**k:deepcopy(r))
    req=ReliableRequest(draft_action=r['draft_action'],goal='profit',mode='research',step_budget=100)
    out=module.proposal(c,s,req,model=fit([]))
    assert not out['search']['complete'] and 95<=out['search']['used_steps']<=100
    assert apply_calibration(out,c.config_sha256,cal)['action']==r['draft_action']
