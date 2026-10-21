"""Report v20.1 on its fresh confirmation, retaining v20's failed evidence."""
import report_reliable_v20 as reporting
from evaluate_reliable_v201 import *

if __name__=='__main__':
    reporting.DEST=DEST201;reporting.apply_calibration=empirical.apply_calibration;reporting.load_calibration=empirical.load_calibration
    result={}
    for split in ('calibration','holdout'):
        if (DEST201/(split+'.json')).exists():result[split]=reporting.outcomes(split)
    if (DEST201/'rolling.json').exists():
        rows=read_json(DEST201/'rolling.json')['results'];result['rolling']={}
        for g in ('profit','welfare'):
            rr=[r for r in rows if r['goal']==g];entry={}
            for mode in ('once','every'):entry[mode]=reporting.risk([{**r,'intervention':r[mode+'_interventions']>0,'gain':r[mode+'_gain']} for r in rr])
            entry['every_minus_once']=reporting.stats([{**r,'intervention':True,'gain':r['every_gain']-r['once_gain']} for r in rr])
            entry['positive_closed_loop_mean']=entry['every']['cluster_bootstrap_mean_ci95_cents'][0]>0;result['rolling'][g]=entry
    result['paid_phase']=dict(new_calls=0,status='blocked_original_preregistered_gate_and_expired_prices')
    write_json(DEST201/'analysis.json',result)
    print({g:{'passed':r['passed'],'interventions':r['v20']['interventions'],'losses':r['v20']['losses'],'mean_cents':r['v20']['all_case_mean_gain_cents']} for g,r in result.get('holdout',{}).get('goals',{}).items()},flush=True)
