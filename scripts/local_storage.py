"""Consistent local session backup and verified restore, excluding budget rollback."""
import argparse
from contextlib import closing
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

ROOT=Path(__file__).resolve().parents[1]

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify(path):
    with closing(sqlite3.connect(Path(path).as_uri()+'?mode=ro',uri=True)) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok': raise ValueError('SQLite integrity check failed')
        rows=db.execute('SELECT document,sha256 FROM checkpoints').fetchall()
        for document,checksum in rows:
            if hashlib.sha256(document.encode()).hexdigest()!=checksum: raise ValueError('Checkpoint checksum mismatch')
            json.loads(document)
        return len(rows)

def backup(data_dir, destination, config_path=None):
    data_dir, destination=Path(data_dir).resolve(),Path(destination).resolve()
    source=data_dir/'sessions.sqlite3'
    if not source.is_file(): raise ValueError('No saved experiments yet')
    destination.mkdir(parents=True,exist_ok=False)
    target=destination/'sessions.sqlite3'
    with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)) as src, closing(sqlite3.connect(target)) as dst: src.backup(dst)
    count=verify(target)
    manifest=dict(schema='local-market-backup-v1',created_at=datetime.now(UTC).isoformat(),sha256=digest(target),episodes=count,config_file_sha256=digest(ROOT/'configs/market_v13_1_local.yaml'),budget_restored=False)
    if config_path:
        from game_theory_agent.market import load_market_config
        manifest.update(schema="local-market-backup-v2",config_sha256=load_market_config(config_path).config_sha256,workbench_files={})
        queue=data_dir/"workbench"/"queue.sqlite3"
        if queue.exists():
            if (data_dir/"workbench").is_symlink():raise ValueError("workbench must be inside the data directory")
            with closing(sqlite3.connect(queue)) as db:
                db.execute("BEGIN IMMEDIATE")
                if db.execute("SELECT count(*) FROM jobs WHERE status IN ('running','queued')").fetchone()[0]:
                    raise ValueError("Pause active research jobs before backup")
                wb=destination/"workbench";wb.mkdir()
                with closing(sqlite3.connect(queue.as_uri()+"?mode=ro",uri=True)) as src,closing(sqlite3.connect(wb/"queue.sqlite3")) as dst:src.backup(dst)
                for source_file in (data_dir/"workbench").rglob("*.json"):
                    if source_file.is_symlink() or not source_file.resolve().is_relative_to((data_dir/"workbench").resolve()):raise ValueError("invalid workbench backup path")
                    output=wb/source_file.relative_to(data_dir/"workbench");output.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source_file,output)
                manifest["workbench_files"]={p.relative_to(destination).as_posix():digest(p) for p in wb.rglob("*") if p.is_file()}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

def restore(data_dir, source, config_path=None):
    # Caller must have stopped every backend that owns this directory.
    data_dir,source=Path(data_dir).resolve(),Path(source).resolve()
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') not in {'local-market-backup-v1','local-market-backup-v2'} or digest(source/'sessions.sqlite3')!=manifest.get('sha256'): raise ValueError('Backup checksum/schema mismatch')
    if manifest["schema"]=="local-market-backup-v2":
        from game_theory_agent.market import load_market_config
        if not config_path or manifest["config_sha256"]!=load_market_config(config_path).config_sha256:raise ValueError("Backup configuration differs from this release")
        for name,checksum in manifest["workbench_files"].items():
            p=(source/name).resolve()
            if not p.is_relative_to(source/"workbench") or digest(p)!=checksum:raise ValueError("Workbench backup checksum/path mismatch")
        actual_files={p.relative_to(source).as_posix() for p in (source/"workbench").rglob("*") if p.is_file()}
        if actual_files!=set(manifest["workbench_files"]):raise ValueError("Workbench backup contains unlisted files")
        current=(data_dir/"workbench").resolve()
        if not current.is_relative_to(data_dir) or current==data_dir:raise ValueError("invalid workbench restore target")
        if (source/"workbench"/"queue.sqlite3").exists():
            with closing(sqlite3.connect(source/"workbench"/"queue.sqlite3")) as db:
                if db.execute("PRAGMA integrity_check").fetchone()[0]!="ok":raise ValueError("Workbench SQLite integrity failed")
    elif manifest.get('config_file_sha256')!=digest(ROOT/'configs/market_v13_1_local.yaml'): raise ValueError('Backup configuration differs from this release')
    verify(source/'sessions.sqlite3')
    data_dir.mkdir(parents=True,exist_ok=True)
    target=data_dir/'sessions.sqlite3'
    # Never overwrite the only prior copy, even on a failed restore.
    prior=data_dir/'backups'/('before-restore-'+datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ'))
    prior.mkdir(parents=True,exist_ok=False)
    for name in ('sessions.sqlite3','sessions.sqlite3-wal','sessions.sqlite3-shm'):
        path=data_dir/name
        if path.exists(): shutil.copy2(path,prior/name)
    stage=data_dir/'sessions.restore.pending'
    if stage.exists(): raise ValueError('An unfinished restore exists; inspect it first')
    shutil.copy2(source/'sessions.sqlite3',stage)
    verify(stage)
    # A stopped SQLite reader can still have WAL from its previous database.
    # Preserve these files under the recovery directory before replacing main DB.
    for suffix in ('-wal','-shm'):
        path=Path(str(target)+suffix)
        if path.exists(): path.rename(prior/(path.name+'.original'))
    stage.replace(target)
    if manifest["schema"]=="local-market-backup-v2":
        current=(data_dir/"workbench").resolve()
        if not current.is_relative_to(data_dir) or current==data_dir:raise ValueError("invalid workbench restore target")
        if current.exists():current.rename(prior/"workbench")
        if (source/"workbench").exists():shutil.copytree(source/"workbench",current)
    return dict(restored=True,episodes=manifest['episodes'],previous_copy=str(prior),budget_restored=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['backup','restore'])
    parser.add_argument('data_dir',type=Path)
    parser.add_argument('path',type=Path)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    print(json.dumps(globals()[args.action](args.data_dir,args.path,config_path=args.config),ensure_ascii=False))
