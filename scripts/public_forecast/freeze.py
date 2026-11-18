"""Seal the estimators, thresholds, executable definitions and shared budget checkpoint."""
import hashlib
from pathlib import Path
import sqlite3
from collect import ROOT,OUT,SPLITS,RAW,save,read

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def freeze():
    sources=[*sorted(Path(__file__).parent.glob('*.py')),ROOT/'src/game_theory_agent/game_theory/public_forecast.py',
             *sorted((ROOT/'experiment-specs/public-forecast-v1').glob('*.md')),ROOT/'scripts/pricing_feedback/paid.py',
             ROOT/'scripts/pricing_feedback/study.py',ROOT/'scripts/pricing_feedback/units.py']
    sources+=[*sorted(OUT.glob('*.joblib')),*sorted(OUT.glob('*-thresholds.json'))]
    assert len(list(OUT.glob('*-gates.joblib')))==2
    sig={p.relative_to(ROOT).as_posix():digest(p) for p in sources}
    obj=dict(files=sig,seeds=SPLITS,baseline_id=12460,baseline_micro_yuan=23808940,total_cap_micro_yuan=50000000)
    if (OUT/'freeze.json').exists():assert read(OUT/'freeze.json')['files']==sig
    else:save(OUT/'freeze.json',obj)

if __name__=='__main__':freeze()

