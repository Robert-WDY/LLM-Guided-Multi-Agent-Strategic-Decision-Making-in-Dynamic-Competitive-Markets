"""Freeze an evaluation bundle and the exact source changes, preserving old releases."""
import hashlib,json,shutil,zipfile
from xml.etree import ElementTree
from evaluate_market_v18 import ROOT,OUT,SPEC,status,read_json,write_json

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def accept():
 r=read_json(OUT/'analysis.json');stable=read_json(OUT/'stability.json');assert r['passed'] and r['final_replay'] and stable['passed']
 assert r['tournament_episodes']==2376 and r['tournament_rounds']==47520 and r['advisor_decisions']==864 and r['closed_loop_episodes']==54 and r['probes']['trajectories']==990
 assert all(x['advice']==x['complete'] for x in r['closed_loop'])
 contract=read_json(OUT/'contract-probes.json');ui=read_json(OUT/'ui-default-latency.json');regressions=read_json(OUT/'regression-reproductions.json')
 assert contract['passed'] and len(contract['results'])==66
 assert len(ui['results'])==3 and all(x['complete'] for x in ui['results'])
 assert len(read_json(OUT/'latency.json')['results'])==27
 assert regressions['passed'] and len(regressions['records'])==3
 assert stable['cases']==18 and stable['advisor_calls']==108
 test=ROOT.parent.parent/'migration_2026-09-07/market-v18-final2.xml';s=ElementTree.parse(test).getroot().find('testsuite');assert int(s.get('tests'))==474 and all(int(s.get(k,'0'))==0 for k in ('errors','failures','skipped'))
 for path,expected in read_json(OUT/'implementation-final.json').items():assert sha(ROOT/path)==expected
 old=ROOT/'releases/local-market-v17.0.0/manifest.json';m=read_json(old);assert sha(old.parent/'local-market-v17.0.0-source.zip')==m['archive_sha256']
 for release,expected in m['old_release_hashes'].items():assert sha(ROOT/'releases'/release/(release+'-source.zip'))==expected
 budget=status();assert budget['new_calls']==186 and budget['reserved_cny']==9.993565
 report=dict(engineering_passed=True,backend_tests=474,tournament_episodes=2376,tournament_rounds=47520,advice_cases=288,advice_decisions=864,closed_loop_episodes=54,closed_loop_rounds=540,closed_loop_decisions=r['closed_loop_advice'],probes=990,probe_rounds=4950,contract_probe_trajectories=132,contract_probe_rounds=660,stability_cases=18,stability_calls=108,latency_calls=27,ui_default_latency_calls=3,arithmetic_checks=r['arithmetic_checks'],old_releases_unchanged=True,world_calibrated=False,new_paid_calls=0,budget_reserved_cny=9.993565,summary='Engineering pass is independent from economic effectiveness and latency gates, recorded per group in analysis.json.')
 report.update(v17_effectiveness_groups_passed=sum(x['effectiveness_pass'] for x in r['advisor_groups'] if x['variant']=='v17'),v17_effectiveness_groups=36,serial_latency_groups_passed=sum(x['passed'] for x in r['latency']),serial_latency_groups=9,default_ui_seconds={str(x['companies']):x['seconds'] for x in ui['results']},exact_regressions_reproduced=3)
 write_json(OUT/'acceptance.json',report);return report,test

def freeze():
 report,test=accept();dest=ROOT/'releases/market-evaluation-v18.0.0';assert not dest.exists(),'never overwrite frozen releases';dest.mkdir()
 files={p.relative_to(OUT).as_posix():p for p in OUT.rglob('*') if p.is_file() and (p.suffix in ('.json','.gz','.md') or p.name.endswith('.png'))}
 files['backend-tests.xml']=test
 with zipfile.ZipFile(dest/'evaluation-evidence.zip','x',zipfile.ZIP_DEFLATED) as z:
  for name,p in files.items():z.write(p,name)
 with zipfile.ZipFile(dest/'evaluation-evidence.zip') as z:
  assert z.testzip() is None
  for name,p in files.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha(p)
 source=[ROOT/'README.md',ROOT/'src/game_theory_agent/game_theory/advisor_audit.py',ROOT/'src/game_theory_agent/market/actor_policies.py',ROOT/'tests/test_market_evaluation_v18.py',ROOT/'docs/market-evaluation-v18-results.md',ROOT/'scripts/time_market_v18_ui.py',*ROOT.glob('scripts/*market_v18.py')]
 with zipfile.ZipFile(dest/'evaluation-source-changes.zip','x',zipfile.ZIP_DEFLATED) as z:
  for p in source:z.write(p,p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(dest/'evaluation-source-changes.zip') as z:
  for p in source:assert hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha(p)
 manifest=dict(release='market-evaluation-v18.0.0',base_release='local-market-v17.0.0',scope='Evaluation bundle plus audited bug fixes, not a claim that every strategy passes or the economy is calibrated',acceptance=report,evidence_files={n:sha(p) for n,p in files.items()},source_files={p.relative_to(ROOT).as_posix():sha(p) for p in source},evidence_sha256=sha(dest/'evaluation-evidence.zip'),source_sha256=sha(dest/'evaluation-source-changes.zip'))
 write_json(dest/'manifest.json',manifest);shutil.copy2(ROOT/'docs/market-evaluation-v18-results.md',dest/'results.md');print(json.dumps(dict(files=len(files),evidence_sha256=manifest['evidence_sha256'],source_sha256=manifest['source_sha256'])))

if __name__=='__main__':
 import sys
 if '--accept-only' in sys.argv:print(json.dumps(accept()[0]))
 else:freeze()
