"""Freeze the tested personal local release; never overwrite prior versions."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import zipfile
from xml.etree import ElementTree

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'releases/local-market-v13.1.0'
OUTER=ROOT.parent.parent

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if DEST.exists(): raise ValueError('Release already frozen; use a new version')
    tests=OUTER/'migration_2026-09-07/stage19-final-tests.xml'
    suite=ElementTree.parse(tests).getroot().find('testsuite')
    assert int(suite.get('tests'))==418 and all(int(suite.get(k,'0'))==0 for k in ('errors','failures'))
    evidence={}
    for name,path in {'http':'runs/local-market-stage19/summary.json','deployment':'runs/local-release-acceptance-r2/summary.json','engine':'runs/supplier-payback-v16/summary.json','frontend':'.tmp/local-frontend-validation.json'}.items():
        evidence[name]=json.loads((ROOT/path).read_text(encoding='utf-8'))
    assert evidence['http']['passed'] and evidence['deployment']['passed']
    assert evidence['engine']['passed'] and evidence['engine']['promotion_passed']
    assert evidence['frontend']=={'build':True,'typecheck':True,'lint':True,'tests':11,'browser_qa':False}
    files=[]
    for folder in ('src','configs','tests','scripts','docs','experiment-specs','frontend/app','frontend/public','frontend/worker','frontend/db','frontend/build','frontend/tests','frontend/drizzle'):
        files.extend((ROOT/folder).rglob('*'))
    files.extend((ROOT/'frontend').glob('*'))
    files.extend(ROOT/name for name in ('README.md','AGENTS.md','pyproject.toml','requirements-local.lock.txt','.gitignore'))
    allowed={'.py','.yaml','.yml','.json','.jsonc','.toml','.md','.ts','.tsx','.mjs','.css','.svg','.png','.sql','.txt','.ps1','.cmd'}
    files=sorted(set(p for p in files if p.is_file() and p.suffix in allowed and not p.name.startswith('.env') and not any(v in {'__pycache__','.tmp','node_modules','dist','.next','.local-state','.git'} for v in p.relative_to(ROOT).parts)))
    content={p.relative_to(ROOT).as_posix():p for p in files}
    # Outer convenience launchers retain their original relative directory layout.
    outer_names=('START_MARKET.ps1','START_MARKET.cmd','STOP_MARKET.cmd','STATUS_MARKET.cmd','BACKUP_MARKET.cmd','START_BACKEND.ps1','START_FRONTEND.ps1','LOCAL_MARKET_GUIDE.md','START_HERE.md')
    file_hashes={name:digest(p) for name,p in content.items()}
    DEST.mkdir(parents=True)
    archive=DEST/'local-market-v13.1.0-source.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as bundle:
        prefix='game-theory-agent/'+ROOT.name+'/'
        for name,path in content.items(): bundle.write(path,prefix+name)
        for name in outer_names: bundle.write(OUTER/name,name)
    shutil.copy2(tests,DEST/'backend-tests.xml')
    for name,report in evidence.items(): (DEST/f'{name}-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=dict(release='local-market-v13.1.0',scope='single user, local loopback, synthetic rule market',config='configs/market_v13_1_local.yaml',python=sys.version,source_files=file_hashes,outer_files={n:digest(OUTER/n) for n in outer_names},source_sha256=hashlib.sha256(json.dumps(file_hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest(),archive_sha256=digest(archive),backend_tests=418,frontend_tests=11,new_real_model_calls=0,browser_qa=False,excluded=['credentials','runtime databases','local research log','venv','node_modules','generated frontend; rebuild with build_local.ps1'])
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'archive':str(archive),'files':len(files),'archive_sha256':manifest['archive_sha256'],'source_sha256':manifest['source_sha256']}))

if __name__=='__main__':main()
