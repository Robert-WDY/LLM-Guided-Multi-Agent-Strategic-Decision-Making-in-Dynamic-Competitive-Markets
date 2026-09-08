"""Freeze accepted v16 source/evidence and verify every archived byte."""
import hashlib,json,shutil,sys,zipfile
from pathlib import Path
from xml.etree import ElementTree
from game_theory_agent.local_budget import status
ROOT=Path(__file__).resolve().parents[1];OUTER=ROOT.parent.parent;DEST=ROOT/'releases/local-market-v16.0.0';RUNS=ROOT/'runs/advisor-v16'
OLD={
 'local-market-v15.0.1':'5124d670dc75f8b25f93c59638dceabcb959c4a371d93065bb1a9dbb75537982',
 'local-market-v15.0.0':'a8296155943e5f190f95b57dd80f8c3fa0ed10b80cc5f581c02a05de6a50dbca',
 'local-market-v14.0.0':'fa4c93f6014951bac9b317650e67125acae19f7f65da670291bdcc470ff02555',
 'local-market-v13.1.0':'2861d5b9c877a5a2370995fb3c0677f46e60057f6261a2ec12a9670b7e1a3e60',
 'v10-internal-beta-1':'76d2b6954e466a43569e163aa1548abcd6ae87b8fcffb30637ea8b1020f582fa',
}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
 assert not DEST.exists(),'never overwrite frozen releases'
 tests=OUTER/'migration_2026-09-07/advisor-v16-final.xml';suite=ElementTree.parse(tests).getroot().find('testsuite');assert int(suite.get('tests'))==461 and all(int(suite.get(k,'0'))==0 for k in ('errors','failures','skipped'))
 qa=read(RUNS/'acceptance.json');assert qa['passed'] and qa['browser_qa'] and qa['backup_restore'];assert qa['frontend']==dict(build=True,typecheck=True,lint=True,tests=14)
 stages=[read(RUNS/f'stage{i}.json') for i in range(1,7)];assert all(s['passed'] for s in stages)
 execution=read(RUNS/'execution-acceptance.json');assert execution['passed'] and execution['episodes']==8 and len(execution['independent_deviation_checks'])==6
 assert read(RUNS/'backup-restore.json')['all_api_and_hashes_equal'];budget=status();assert budget['reserved_cny']==9.993565 and budget['new_calls']==186
 for name,expected in OLD.items():assert digest(ROOT/'releases'/name/(name+'-source.zip'))==expected
 files=[]
 for folder in ('src','configs','artifacts','tests','scripts','docs','experiment-specs','frontend/app','frontend/public','frontend/worker','frontend/db','frontend/build','frontend/tests','frontend/drizzle'):files.extend((ROOT/folder).rglob('*'))
 files.extend((ROOT/'frontend').glob('*'));files.extend(ROOT/n for n in ('README.md','AGENTS.md','pyproject.toml','requirements-local.lock.txt','.gitignore'))
 allowed={'.py','.yaml','.yml','.json','.jsonc','.toml','.md','.ts','.tsx','.mjs','.css','.svg','.png','.sql','.txt','.ps1','.cmd'}
 files=sorted(set(p for p in files if p.is_file() and p.suffix in allowed and not p.name.startswith('.env') and not p.name.startswith('.advisor-render') and not any(v in {'__pycache__','.tmp','node_modules','dist','.next','.local-state','.git'} for v in p.relative_to(ROOT).parts)))
 content={p.relative_to(ROOT).as_posix():p for p in files};assert all(k in content for k in ('src/game_theory_agent/game_theory/advisor_search.py','frontend/app/advisor-v16.tsx','docs/advisor-v16-results.md'))
 outer=('START_MARKET.ps1','START_MARKET.cmd','STOP_MARKET.cmd','STATUS_MARKET.cmd','BACKUP_MARKET.cmd','START_BACKEND.ps1','START_FRONTEND.ps1','LOCAL_MARKET_GUIDE.md','START_HERE.md','PENDING_FEATURES_2026-09-07.md')
 hashes={n:digest(p) for n,p in content.items()};DEST.mkdir(parents=True);archive=DEST/'local-market-v16.0.0-source.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
  for name,path in content.items():bundle.write(path,'game-theory-agent/'+ROOT.name+'/'+name)
  for name in outer:bundle.write(OUTER/name,name)
 with zipfile.ZipFile(archive) as bundle:
  assert bundle.testzip() is None
  for name,expected in hashes.items():assert hashlib.sha256(bundle.read('game-theory-agent/'+ROOT.name+'/'+name)).hexdigest()==expected
 evidence={p.relative_to(RUNS).as_posix():p for p in RUNS.rglob('*.json') if not any(v in ('local-backup','api-records') for v in p.relative_to(RUNS).parts) and p.name!='restore-before.json'}
 with zipfile.ZipFile(DEST/'advisor-v16-evidence.zip','x',zipfile.ZIP_DEFLATED) as bundle:
  for name,path in evidence.items():bundle.write(path,name)
 with zipfile.ZipFile(DEST/'advisor-v16-evidence.zip') as bundle:assert bundle.testzip() is None
 for name in ('acceptance.json','backup-restore.json','execution-acceptance.json',*(f'stage{i}.json' for i in range(1,7))):shutil.copy2(RUNS/name,DEST/name)
 shutil.copy2(tests,DEST/'backend-tests.xml')
 manifest=dict(release='local-market-v16.0.0',scope='single-user local synthetic market; objective advisor with finite search and restricted multiplayer diagnostics',config='configs/market_v14_local.yaml',python=sys.version,source_files=hashes,outer_files={n:digest(OUTER/n) for n in outer},archive_sha256=digest(archive),evidence_sha256=digest(DEST/'advisor-v16-evidence.zip'),evidence_files={n:digest(p) for n,p in evidence.items()},backend_tests=461,frontend_tests=14,stages=[s.get('cases') for s in stages],holdout_cases=18,holdout_positive=17,holdout_baseline=1,closed_loop_episodes=8,closed_loop_rounds=40,independent_deviation_values=58,new_real_calls=0,budget_reserved_cny=budget['reserved_cny'],global_optimality_proven=False,old_release_hashes=OLD,excluded=['credentials','runtime databases','local research log','venv','node_modules','generated frontend; rebuild with build_local.ps1'])
 (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(files=len(hashes),evidence_files=len(evidence),archive_sha256=manifest['archive_sha256'])))
if __name__=='__main__':main()
