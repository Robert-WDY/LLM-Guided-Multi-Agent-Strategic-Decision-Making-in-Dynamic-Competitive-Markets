"""Immutable research deliverable; economic failure remains visible in the manifest."""
import subprocess,zipfile,shutil
from reliable_v19_common import *

def bundle(path,paths,root):
    assert not path.exists(),'Never overwrite a frozen archive'
    hashes={p.relative_to(root).as_posix():sha(p) for p in sorted(paths)}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in hashes:z.write(root/name,name)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for name,digest in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    return dict(file=path.name,sha256=sha(path),files=hashes)

def main():
    freeze_market();dest=ROOT/'releases/reliable-advisor-v19.0.0';dest.mkdir(exist_ok=True)
    assert not (dest/'manifest.json').exists(),'Release is already frozen'
    acceptance=read_json(OUT/'acceptance.json');real=read_json(OUT/'real/status.json')
    old=ROOT/'releases/market-evaluation-v18.0.0';manifest=read_json(old/'manifest.json')
    assert sha(old/'evaluation-evidence.zip')==manifest['evidence_sha256']
    assert sha(old/'evaluation-source-changes.zip')==manifest['source_sha256']
    prior={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'releases').rglob('*.zip') if dest not in p.parents}
    source_names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')+subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    source=[]
    for name in set(source_names):
        p=ROOT/name
        if not name or not p.is_file() or name.startswith('releases/') or p.suffix=='.zip':continue
        assert p.name!='PROJECT_EVOLUTION_RECORD.md' and p.name!='.env'
        source.append(p)
    source_bundle=bundle(dest/'source.zip',source,ROOT)
    evidence_bundle=bundle(dest/'evidence.zip',[p for p in OUT.rglob('*') if p.is_file() and p.suffix!='.tmp'],OUT)
    for name,digest in prior.items():assert sha(ROOT/name)==digest
    shutil.copyfile(ROOT/'docs/reliable-advisor-results.md',dest/'results.md')
    base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    write_json(dest/'manifest.json',dict(release='reliable-advisor-v19.0.0',base_commit=base,scope='Local research snapshot: implementation and experiments, not universal economic effectiveness or real-world calibration',
        acceptance=acceptance,real_experiment=real,source=source_bundle,evidence=evidence_bundle,prior_archives_unchanged=prior,report_sha256=sha(dest/'results.md')))
    print(json.dumps(dict(source_sha256=source_bundle['sha256'],evidence_sha256=evidence_bundle['sha256'],source_files=len(source),evidence_files=len(evidence_bundle['files'])),ensure_ascii=False))
if __name__=='__main__':main()
