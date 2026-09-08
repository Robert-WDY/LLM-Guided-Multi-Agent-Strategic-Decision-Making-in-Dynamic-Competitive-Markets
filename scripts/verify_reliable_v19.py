"""Independent engineering evidence, frozen-input verification and paid-stage admission."""
import xml.etree.ElementTree as ET
from reliable_v19_common import *

def verify_institution():
    files=sorted((OUT/'cooperation').glob('*.json.gz'));assert len(files)==1080
    checks=0
    for path in files:
        case=read(path);summary=case['summary'];n=summary['companies'];kind=summary['defection']
        # Use saved action order, not any assumed serialized state container shape.
        ids=list(case['transitions'][0]['actions'])
        defectors=set(ids[:0 if kind=='none' else 1 if kind=='one' else max(2,n//2)])
        for row in case['transitions']:
            for actor,promise in row['promises'].items():
                if actor not in defectors:
                    assert promise==row['realized'][actor], (path.name,actor,row['round'])
                    checks+=1
        if summary['threshold']=='unreachable':assert not summary['project_success']
    return dict(episodes=len(files),honest_promise_checks=checks,passed=True)

def main():
    freeze_market()
    for path,digest in read_json(OUT/'implementation-before-holdout.json').items():assert sha(ROOT/path)==digest
    xml=OUT/'backend-tests.xml'
    suites=ET.parse(xml).getroot();passed=sum(int(x.get('tests',0))-int(x.get('failures',0))-int(x.get('errors',0))-int(x.get('skipped',0)) for x in suites.iter('testsuite'))
    assert passed==481
    guard=ET.parse(OUT/'guard-tests.xml').getroot()
    guard_passed=sum(int(x.get('tests',0))-int(x.get('failures',0))-int(x.get('errors',0))-int(x.get('skipped',0)) for x in guard.iter('testsuite'))
    assert guard_passed==16
    log=(OUT/'frontend-final.log').read_text(encoding='utf-8-sig')
    assert '# pass 18' in log and '# fail 0' in log
    timing=read_json(OUT/'timing.json');ui=read_json(OUT/'ui-verification.json');analysis=read_json(OUT/'analysis.json')
    assert ui['passed'] and timing['interactive_under_5_seconds']
    institution=verify_institution()
    assert read_json(OUT/'sensitivity.json')['cases']==45
    assert read_json(OUT/'sequential-response.json')['passed']
    result=dict(version='reliable-strategic-advisor-v1',engineering_passed=True,backend_tests=passed,additional_paid_guard_tests=guard_passed,frontend_tests=18,ui=ui,institution=institution,latency=timing,
                frozen_market_unchanged=True,holdout_algorithm_unchanged=True,analysis_complete=True,
                advisor_effectiveness_passed=analysis['paid_economic_gate'],paid_gate_passed=analysis['paid_economic_gate'],
                paid_gate_reasons=analysis['paid_gate_reasons'],world_calibrated=False,
                scope='Engineering acceptance and research completion do not certify economic effectiveness. No held-out tuning.')
    write_json(OUT/'acceptance.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
