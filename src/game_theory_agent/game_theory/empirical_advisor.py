"""v20.1: calibrate the existing ranking without an unvalidated asset reranking.

The v20 stress/reranking experiment failed equal-coverage evaluation. This
smaller correction retains v19's candidates/ranking and adjusts its lower
bounds with independently calibrated selected-policy overprediction errors.
"""
from copy import deepcopy
from pathlib import Path
import hashlib,json
from . import reliable_advisor as legacy
from .calibrated_advisor import public_features

VERSION='empirical-draft-v20.1.0'
PATH=Path(__file__).resolve().parents[3]/'configs/reliable-calibration-v201.json'

def load_calibration():
    if not PATH.exists():return None
    raw=PATH.read_bytes();v=json.loads(raw);v['artifact_sha256']=hashlib.sha256(raw).hexdigest();return v

def proposal(config,state,request):
    req=request if isinstance(request,legacy.ReliableRequest) else legacy.ReliableRequest.model_validate(request)
    r=legacy.advise(config,state,req);r['version']=VERSION;r['public_features']=public_features(config,state,req.company_id)
    return r

def apply_calibration(result,config_sha,cal):
    r=deepcopy(result);e=dict(policy=VERSION,passed=False,calibrated=False,adjusted_margin=None,reasons=[],checks=[])
    reason=None
    if not cal or cal.get('policy')!=VERSION:reason='缺少匹配的独立误差校准。'
    elif config_sha not in cal['config_sha256']:reason='市场配置未校准。'
    elif r['mode']!='research' or r['objective']['goal'] not in ('profit','welfare'):reason='当前目标或计算模式未校准。'
    elif r['objective']['discount']!=.95 or r['objective']['risk_aversion']!=.25 or r['objective']['cash_floor_cents']!=0 or r['response_model']!='conditional':reason='风险或响应参数未校准。'
    elif r['company_count'] not in (2,5,10) or r['draft_action']['agent_id']!='company_A':reason='主体或规模未校准。'
    elif not r['search']['complete']:reason='计算预算不足。'
    elif r['disposition']=='abstain':reason='模型验证未支持修改原计划。'
    else:
        goal=r['objective']['goal'];group=cal['goals'][goal]
        unsupported=[k for k,v in r['public_features'].items() if k not in cal['support'] or not cal['support'][k][0]<=v<=cal['support'][k][1]]
        if unsupported:reason='公开观测超出校准范围：'+','.join(unsupported)
        elif group['selected_cases']<12 or group['seed_clusters']<8:reason='实际干预校准样本不足。'
        else:
            for h in (3,5):
                check=next((x for x in r['validation'] if x['horizon']==h),None)
                if not check or not check['safe']:
                    reason='缺少完整期限验证或未满足现金约束。';break
                budget=group['overprediction_budget'][str(h)];margin=check['lower_bound']-budget
                e['checks'].append(dict(horizon=h,conservative_mean=check['lower_bound'],empirical_error_budget=budget,adjusted_margin=margin))
            if reason is None:
                e.update(calibrated=True,adjusted_margin=min(x['adjusted_margin'] for x in e['checks']),selected_calibration_cases=group['selected_cases'],seed_clusters=group['seed_clusters'])
                e['passed']=e['adjusted_margin']>0
                reason='原模型下界扣除历史高估误差后仍为正。' if e['passed'] else '模型内正下界不足以覆盖历史高估误差，保留原计划。'
    e['reasons']=[reason];e['artifact_sha256']=cal.get('artifact_sha256') if cal else None
    r['empirical_calibration']=e;r['proposed_action']=r['action'];r['proposed_id']=r['recommended_id']
    if not e['passed']:
        r.update(action=deepcopy(r['draft_action']),recommended_id='agent_draft',recommended_label='Agent原计划',disposition='abstain')
    r['rationale']=reason
    r['limitations'].append('v20.1是经验风险筛选，不声称修复全部隐藏状态、改进候选排序或保证单次盈利。')
    return r

def advise(config,state,request):return apply_calibration(proposal(config,state,request),config.config_sha256,load_calibration())
