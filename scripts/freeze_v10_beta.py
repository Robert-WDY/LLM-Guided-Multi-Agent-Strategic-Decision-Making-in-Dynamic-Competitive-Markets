"""Create an immutable, credential-free local beta source snapshot."""
import hashlib
import importlib.metadata
import json
import shutil
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"releases/v10-internal-beta-1"
TESTS=ROOT.parent.parent/"migration_2026-09-07/v10-beta-tests.xml"


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists(): raise RuntimeError("frozen release already exists; use a new version")
    suite=ElementTree.parse(TESTS).getroot().find("testsuite")
    assert suite is not None and int(suite.get("tests","0"))>=348
    assert all(int(suite.get(k,"0"))==0 for k in ("errors","failures","skipped"))
    gates=json.loads((ROOT/"runs/v10-beta-validation/summary.json").read_text(encoding="utf-8"))
    assert gates["engineering_gate_passed"]
    assert json.loads((ROOT/".tmp/v10-restart-smoke/result.json").read_text())["passed"]
    front=json.loads((ROOT/".tmp/v10-frontend-validation.json").read_text())
    assert front["build"] and front["typecheck"] and front["lint"] and front["tests"]==8
    candidates=[]
    for directory in ("src","tests","configs","scripts","docs","experiment-specs","frontend/app","frontend/public","frontend/worker","frontend/db","frontend/build","frontend/tests","frontend/drizzle"):
        candidates.extend((ROOT/directory).rglob("*"))
    candidates.extend(ROOT.glob("*.toml"))
    candidates.extend([ROOT/"README.md",ROOT/"AGENTS.md",ROOT/".gitignore"])
    candidates.extend(p for p in (ROOT/"frontend").iterdir() if p.is_file())
    allowed={".py",".yaml",".yml",".json",".jsonc",".toml",".md",".ts",".tsx",".mjs",".css",".svg",".sql",".txt"}
    files=sorted({p for p in candidates if p.is_file() and p.suffix in allowed and not any(part in {"__pycache__","node_modules","dist",".next",".git",".local-state",".tmp"} for part in p.relative_to(ROOT).parts) and not p.name.startswith(".env") and not p.name.startswith(".market-v10-render-")})
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    source=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    DEST.mkdir(parents=True)
    archive=DEST/"v10-internal-beta-1-source.zip"
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as bundle:
        for p in files: bundle.write(p,p.relative_to(ROOT).as_posix())
    dependencies={dist.metadata["Name"]:dist.version for dist in importlib.metadata.distributions() if dist.metadata.get("Name")}
    (DEST/"python-dependencies.json").write_text(json.dumps(dependencies,sort_keys=True,indent=2),encoding="utf-8")
    shutil.copyfile(TESTS,DEST/"backend-tests.xml")
    shutil.copyfile(ROOT/".tmp/v10-restart-smoke/result.json",DEST/"restart-smoke.json")
    shutil.copyfile(ROOT/".tmp/v10-frontend-validation.json",DEST/"frontend-validation.json")
    shutil.copyfile(ROOT/"runs/v10-beta-validation/summary.json",DEST/"synthetic-summary.json")
    manifest={"release":"v10-internal-beta-1","created_at":datetime.now(UTC).isoformat(),"scope":"local single-process internal beta; no public deployment certification","source_sha256":source,"archive_sha256":digest(archive),"files":hashes,"python_version":sys.version,"backend_tests":int(suite.get("tests")),"frontend_tests":8,"synthetic_episodes":372,"synthetic_rounds":3552,"real_experiments":"run after freeze, bounded by 10 CNY; results stored separately","excluded":["credentials","environment files","runtime databases","provider raw runs","git metadata","local research log","venv","node_modules"]}
    (DEST/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2),encoding="utf-8")
    print(json.dumps({"source_sha256":source,"files":len(files),"archive_bytes":archive.stat().st_size,"release":str(DEST)},ensure_ascii=False))


if __name__=="__main__": main()
