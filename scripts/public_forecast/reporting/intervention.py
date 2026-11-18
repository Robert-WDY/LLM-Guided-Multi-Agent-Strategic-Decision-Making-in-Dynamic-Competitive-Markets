"""Evaluate the marginal forecast for changing the price on exactly matched real-agent states."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import spearmanr
from collect import cases,OUT,save,exchange
from models import diagnosis,bundle
from gates import truth_delta
from analyze import cluster


def analyze(m):
    observations=[]
    for r in cases('rtest',m):
        d=r['opportunity'];t=d['t']
        h=r['history'][:t-1]
        a=np.array(h);b=np.asarray(r['history'][t-1]);alternative=np.asarray(r['alternative'][0])
        context='full' if m=='native' else 'basic'
        px=[(v,d['low'],d['high']) for v in (d['draft'],d['proposed'])]
        pred,*_=diagnosis(bundle(m),h,context,px)
        dq=alternative[2]-b[2];dqhat=pred[-1,0]-pred[-2,0]
        assert abs(alternative[1]-b[1])<1.e-9 and abs(pred[-1,1]-pred[-2,1])<1.e-9
        margin=np.median(a[-4:,3]/np.maximum(a[-4:,2],1.e-4))
        value=pred[-2:,0]*(margin+exchange(h)*(np.clip([d['draft'],d['proposed']],d['low'],d['high'])-a[-1,0]))
        observations.append(dict(seed=r['seed'],provider=r['provider'],y=dq,prediction=dqhat,profit_hat=value[1]-value[0],
                                 immediate=alternative[3]-b[3],terminal=truth_delta(r)))
    seeds=sorted({r['seed'] for r in observations})
    def avg(v):
        return np.mean([v(r) for r in observations])
    def err(r):return abs(r['prediction']-r['y'])-abs(r['y'])
    active=[r for r in observations if abs(r['y'])>1.e-8]
    def rank(a,b):return float(spearmanr([r[a] for r in observations],[r[b] for r in observations]).statistic)
    return dict(n=len(observations),forecast_mae=avg(lambda r:abs(r['prediction']-r['y'])),zero_effect_mae=avg(lambda r:abs(r['y'])),
                paired_mae_difference=cluster([np.mean([err(r) for r in observations if r['seed']==s]) for s in seeds]),
                nonzero_effect=len(active),sign_correct=float(np.mean([np.sign(r['prediction'])==np.sign(r['y']) for r in active])),
                rank_first_profit=rank('profit_hat','immediate'),rank_terminal_profit=rank('profit_hat','terminal'),
                rank_true_first_vs_terminal=rank('immediate','terminal'))


if __name__=='__main__':
    save(OUT/'marginal-effect.json',{m:analyze(m) for m in ('econ','native')})

