"""Empirically calibrated draft interventions; no evaluator states or policies.

Keeps v19 frozen. Unidentified public-state uncertainty is stress-tested, then
selected recommendations are calibrated against historical overprediction.
This is an empirical risk control, not a guarantee for an individual outcome.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import mean
import hashlib
import json
import math

from . import reliable_advisor as legacy
from .advisor_market import forecast, economic_key, BudgetExhausted
from .objectives import score
from .reliable_response import load_model
from game_theory_agent.market import CompanyAction
from game_theory_agent.market.protocols import state_hash

CALIBRATION_PATH=Path(__file__).resolve().parents[3]/'configs/reliable-calibration-v20.json'
POLICY_VERSION='calibrated-draft-v20.0.0'

def economic_change(a,b):
    return economic_key(CompanyAction.from_dict(a))!=economic_key(CompanyAction.from_dict(b))

def public_features(config,state,cid):
    # Only own private state and public market facts. No true peer cash/assets.
    c=state.company(cid)
    return dict(company_count=len(state.company_ids),round=state.round,remaining=state.rounds_remaining,
                own_cash=c.financial.cash_balance_cents,own_sales=c.commercial.sales_orders,
                own_capacity=c.operations.base_capacity_orders,own_price=c.commercial.price_cents,
                own_service=c.brand.service_quality_ppm,own_awareness=c.brand.brand_awareness_ppm,
                demand=state.market.realized_demand_orders)

def plausible_states(predicted,cid):
    """Stress two unidentified brand/service assets using own known benchmarks.

    The neutral reconstruction remains one scenario. These are assumptions,
    not estimates of competitors' true assets or inferred personalities.
    """
    own=predicted.company(cid);out=[('neutral',predicted)]
    for name,offset in [('peer_assets_like_own',0),('stronger_peer_assets',100000)]:
        peers=[]
        for peer in predicted.companies:
            if peer.company_id==cid:peers.append(peer);continue
            brand=replace(peer.brand,brand_awareness_ppm=max(peer.brand.brand_awareness_ppm,min(1000000,own.brand.brand_awareness_ppm+offset)),
                          service_quality_ppm=max(peer.brand.service_quality_ppm,min(1000000,own.brand.service_quality_ppm+offset)))
            peers.append(replace(peer,brand=brand))
        s=replace(predicted,companies=tuple(peers));s=replace(s,state_hash=state_hash(s.to_dict()));out.append((name,s))
    return out

def proposal(config,state,request,*,model=None):
    req=request if isinstance(request,legacy.ReliableRequest) else legacy.ReliableRequest.model_validate(request)
    model=load_model() if model is None else model
    r=legacy.advise(config,state,req,model=model)
    r['version']=POLICY_VERSION;r['legacy_recommended_id']=r['recommended_id']
    r['uncertainty_validation']=[];r['public_features']=public_features(config,state,req.company_id)
    if r['disposition']=='abstain':return r
    predicted,_=forecast(config,state,req.company_id)
    horizons=[1] if req.mode=='interactive' else [3,5];used=r['search']['used_steps']
    from time import perf_counter
    started=perf_counter()
    try:
        for name,s in plausible_states(predicted,req.company_id)[1:]:
            runner=legacy.DraftRollouts(config,s,req.model_copy(update={'step_budget':max(1,req.step_budget-used)}),model)
            for h in horizons:
                a=[runner.evaluate(r['action'],r['draft_action'],i,h) for i in range(3000,3004)]
                b=[runner.evaluate(r['draft_action'],r['draft_action'],i,h) for i in range(3000,3004)]
                gains=[score(x,r['objective'])['total']-score(y,r['objective'])['total'] for x,y in zip(a,b)]
                r['uncertainty_validation'].append(dict(state_assumption=name,horizon=min(h,state.rounds_remaining),mean_gain=mean(gains),
                    lower_bound=legacy.lower_bound(gains),safe=all(x['minimum_cash']>=req.cash_floor_cents and not x['exited'] for x in a),paired_gains=gains))
            used+=runner.used
    except BudgetExhausted:
        used+=runner.used;r['search']['complete']=False
    r['search']['used_steps']=used;r['search']['elapsed_seconds']+=perf_counter()-started
    r['limitations'].append('品牌和服务压力情景只表达未知资产的不确定性；未识别全部隐藏状态或证明因果修复。')
    return r

def load_calibration():
    if not CALIBRATION_PATH.exists():return None
    raw=CALIBRATION_PATH.read_bytes();value=json.loads(raw);value['artifact_sha256']=hashlib.sha256(raw).hexdigest();return value

def calibration_decision(r,config_sha,calibration):
    evidence=dict(policy=POLICY_VERSION,calibrated=False,passed=False,reasons=[],adjusted_margin=None)
    def reject(reason):evidence['reasons'].append(reason);return evidence
    if not calibration:return reject('尚无独立误差校准数据，保留原计划。')
    evidence['artifact_sha256']=calibration.get('artifact_sha256')
    if calibration.get('policy')!=POLICY_VERSION:return reject('校准版本不匹配。')
    if config_sha not in calibration.get('config_sha256',[]):return reject('当前市场配置不在校准范围。')
    if r['mode']!='research' or r['objective']['goal'] not in ('profit','welfare'):return reject('仅纯利润／纯福利的研究模式有误差校准；当前目标或期限未校准。')
    if r['objective']['discount']!=.95:return reject('当前折扣率未校准。')
    if r['response_model']!='conditional' or r['objective']['risk_aversion']!=.25 or r['objective']['cash_floor_cents']!=0:return reject('当前响应模型或风险参数未校准。')
    if r['company_count'] not in (2,5,10) or r['draft_action']['agent_id']!='company_A':return reject('当前主体或公司规模未校准。')
    if not r['search']['complete']:return reject('计算预算不足，校准检查不完整。')
    if r['disposition']=='abstain':return reject('模型验证未支持修改原计划。')
    goal=r['objective']['goal'];group=calibration.get('goals',{}).get(goal)
    if not group or group['selected_cases']<12 or group['seed_clusters']<8:return reject('该目标的实际干预校准样本不足。')
    support=calibration.get('support',{})
    for k,value in r['public_features'].items():
        bounds=support.get(k)
        if bounds is None or not bounds[0]<=value<=bounds[1]:return reject(f'公开观测超出校准支持范围：{k}。')
    checks=r['validation']+r['uncertainty_validation'];margins=[];details=[]
    for h in (3,5):
        rows=[x for x in checks if x['horizon']==h]
        if len(rows)!=3:return reject('缺少三种状态假设下的完整三／五轮检查。')
        # The empirical budget corrects selected-policy optimism; MC LCB alone is insufficient.
        predicted=min(x['mean_gain'] for x in rows);budget=group['overprediction_budget'][str(h)]
        margin=min(min(x['lower_bound'] for x in rows),predicted-budget)
        if not all(x['safe'] for x in rows):return reject('状态压力情景触及现金或退出约束。')
        margins.append(margin);details.append(dict(horizon=h,conservative_mean=predicted,empirical_error_budget=budget,adjusted_margin=margin))
    evidence.update(calibrated=True,adjusted_margin=min(margins),checks=details,selected_calibration_cases=group['selected_cases'],seed_clusters=group['seed_clusters'])
    evidence['passed']=min(margins)>calibration['minimum_gain']
    evidence['reasons'].append('经验误差调整后仍超过干预门槛。' if evidence['passed'] else '预测增益不足以覆盖历史高估误差或状态不确定性，保留原计划。')
    return evidence

def apply_calibration(result,config_sha,calibration):
    r=deepcopy(result);evidence=calibration_decision(r,config_sha,calibration);r['empirical_calibration']=evidence
    r['proposed_action']=r['action'];r['proposed_id']=r['recommended_id']
    if not evidence['passed']:
        r['action']=deepcopy(r['draft_action']);r['recommended_id']='agent_draft';r['recommended_label']='Agent原计划';r['disposition']='abstain'
    r['rationale']=' '.join(evidence['reasons'])
    r['limitations'].append('经验校准不保证单次收益为正；新市场、长期分布漂移及未校准目标不在验证范围。')
    return r

def advise(config,state,request,*,calibration=None,model=None):
    return apply_calibration(proposal(config,state,request,model=model),config.config_sha256,load_calibration() if calibration is None else calibration)
