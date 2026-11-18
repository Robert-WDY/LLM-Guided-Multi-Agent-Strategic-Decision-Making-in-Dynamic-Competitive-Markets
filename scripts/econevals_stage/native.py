"""Fresh paired test of the unmodified fresh-food advisor, independent of EconEvals."""
import sys
from concurrent.futures import ProcessPoolExecutor
from common import ROOT, OUT, RAW, save
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_reliable_v20 import snapshot
from reliable_v19_common import rollout
from game_theory_agent.game_theory.reliable_advisor import ReliableRequest,advise
from game_theory_agent.game_theory.calibrated_advisor import public_features
from game_theory_agent.game_theory.empirical_advisor import apply_calibration,load_calibration


def case(job):
    seed,regime=job
    c,s,mem,d,h=snapshot(regime,2,seed,4)
    draft=d.to_dict()
    def outcome(action):
        return rollout(c,s,mem,seed,action,draft)['metrics'][5]['profit']/100
    base=outcome(draft)
    row=dict(seed=seed,market=regime,A=base)
    for arm,mode in [('C','generic'),('D','conditional')]:
        req=ReliableRequest(draft_action=draft,goal='profit',mode='research',response_model=mode,seed=seed,public_history=h)
        r=advise(c,s,req)
        row[arm]=outcome(r['action'])
        row[arm+'_action']=r['recommended_id']
        row[arm+'_steps']=r['search']['used_steps']
        if arm=='D':
            r['public_features']=public_features(c,s,req.company_id)
            calibrated=apply_calibration(r,c.config_sha256,load_calibration())
            row['Dcal']=outcome(calibrated['action']);row['Dcal_action']=calibrated['recommended_id']
    return row


if __name__=='__main__':
    import numpy as np
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(case,[(s,r) for s in range(820001,820013) for r in ('normal','price_sensitive')]))
    stats={}
    for l,r in [('C','A'),('D','A'),('D','C'),('Dcal','A')]:
        vals=[np.mean([x[l]-x[r] for x in rows if x['seed']==s]) for s in range(820001,820013)]
        ci=np.quantile(np.random.default_rng(12).choice(vals,(5000,12)).mean(axis=1),[.025,.975])
        stats[l+'-'+r]=dict(mean=float(np.mean(vals)),ci=ci.tolist(),positive=sum(float(v)>0 for v in vals))
    save(OUT/'native.json',dict(cases=rows,comparisons=stats))
    print(stats)

