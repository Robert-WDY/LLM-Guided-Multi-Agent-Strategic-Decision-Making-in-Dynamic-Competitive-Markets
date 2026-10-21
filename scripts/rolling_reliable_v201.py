"""New v20.1 closed-loop confirmation; separate seeds from all previous runs."""
import evaluate_reliable_v20 as previous
from evaluate_reliable_v201 import *

def case_rolling(args):
    previous.DEST=DEST201
    previous.proposal=empirical.proposal
    previous.apply_calibration=empirical.apply_calibration
    previous.load_calibration=empirical.load_calibration
    return previous.rolling(args)

if __name__=='__main__':
    freeze201();p=DEST201/'rolling-implementation.json';digest=sha(Path(__file__))
    if p.exists():assert read_json(p)['sha256']==digest
    else:write_json(p,dict(sha256=digest,seeds=list(range(660001,660011)),rounds=6))
    jobs=[(REGIMES[(i+j)%6],n,660001+i,g) for i in range(10) for j,n in enumerate((2,5,10)) for g in ('profit','welfare')];rows=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for r in pool.map(case_rolling,jobs):
            rows.append(r)
            if len(rows)%6==0:print('v201 rolling',len(rows),'/60',flush=True)
    write_json(DEST201/'rolling.json',dict(results=rows,cases=len(rows)))
