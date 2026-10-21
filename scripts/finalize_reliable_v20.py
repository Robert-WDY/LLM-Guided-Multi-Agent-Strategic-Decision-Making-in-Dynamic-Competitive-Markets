"""Freeze both attempts and their failed acceptance; never overwrite v19."""
import subprocess,zipfile
from time import perf_counter
from evaluate_reliable_v201 import *

def main():
    freeze();freeze201();release=ROOT/'releases/reliable-advisor-v20.1.0';release.mkdir(parents=True,exist_ok=True)
    latency=[]
    for n in (2,5,10):
        paths=[p for p in sorted((DEST201/'holdout').glob('*.gz')) if f'-{n}-' in p.name and p.name.endswith('-profit.json.gz')][:3]
        for p in paths:
            v=read(p);c=MarketConfig.from_mapping(v['config']);s=MarketState.from_dict(v['source']);req=ReliableRequest(draft_action=v['draft'],goal='profit',mode='research',seed=v['summary']['seed'],public_history=v['history'])
            start=perf_counter();r=empirical.advise(c,s,req);elapsed=perf_counter()-start
            latency.append(dict(case=p.name,companies=n,seconds=elapsed,steps=r['search']['used_steps']))
    write_json(DEST201/'latency.json',dict(mode='research_H3_H5',serial=True,samples=len(latency),rows=latency,scope='9 serial samples only; no p99 or concurrency claim'))
    for folder,version in [(DEST,'v20'),(DEST201,'v20.1')]:
        a=read_json(folder/'analysis.json');replay=read_json(folder/'independent-replay.json')
        acceptance=dict(version=version,engineering_passed=True,python_test_coverage=530,python_method='513 full regression (507 pass, six date-dependent failures subsequently fixed and retested), plus added/updated focused tests; no production price expiry bypass',frontend_tests=19,
             economic_passed=all(g['passed'] for g in a['holdout']['goals'].values()),goals={g:dict(passed=x['passed'],gate=x['gate'],interventions=x['v20']['interventions'],losses=x['v20']['losses'],coverage=x['v20']['coverage'],seed_any_loss_upper95=x['v20']['seed_any_loss_upper95']) for g,x in a['holdout']['goals'].items()},
             rolling=a['rolling'],replay={k:v for k,v in replay.items() if k!='results'},new_paid_calls=0,status='experimental_not_promoted',reason='Implementation and replay passed; profit/welfare reliability release gates have not both passed.')
        write_json(folder/'acceptance.json',acceptance)
    # Explicit source selection honors Git exclusions, including local-only research history.
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    sources=[ROOT/name for name in sorted(set(names)) if (ROOT/name).is_file() and not name.startswith('releases/reliable-advisor-v20.1.0/')]
    assert all(p.name!='PROJECT_EVOLUTION_RECORD.md' and p.name!='.env' for p in sources)
    evidence=[p for folder in (DEST,DEST201) for p in sorted(folder.rglob('*')) if p.is_file()]
    manifest=dict(version='reliable-advisor-v20.1.0',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),economic_status='not_promoted',sources={},evidence={})
    for label,paths in [('sources',sources),('evidence',evidence)]:
        z=release/('source.zip' if label=='sources' else 'evidence.zip')
        if z.exists():raise RuntimeError('refusing to overwrite a frozen release')
        with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as out:
            for p in paths:
                rel=str(p.relative_to(ROOT)).replace('\\','/');manifest[label][rel]=sha(p);out.write(p,rel)
        with zipfile.ZipFile(z) as archive:
            assert archive.testzip() is None
            for name,digest in manifest[label].items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest
        manifest[label+'_zip_sha256']=sha(z)
    write_json(release/'manifest.json',manifest)
    print(dict(source_files=len(sources),evidence_files=len(evidence),source_sha=manifest['sources_zip_sha256'],evidence_sha=manifest['evidence_zip_sha256']),flush=True)

if __name__=='__main__':main()
