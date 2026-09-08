"""Freeze the accepted local v14 source and evidence without replacing older releases."""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
OUTER = ROOT.parent.parent
DEST = ROOT / "releases/local-market-v14.0.0"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise ValueError("Release already exists; never overwrite a frozen release")
    tests = OUTER / "migration_2026-09-07/stage28-release.xml"
    suite = ElementTree.parse(tests).getroot().find("testsuite")
    assert int(suite.get("tests")) == 444
    assert all(int(suite.get(k, "0")) == 0 for k in ("errors", "failures", "skipped"))
    qa = json.loads((ROOT / "runs/stage28-release/acceptance.json").read_text(encoding="utf-8"))
    assert qa["passed"] and qa["browser_qa"] and qa["backup_restore"]
    assert qa["frontend"] == dict(build=True, typecheck=True, lint=True, tests=11)
    real = json.loads((ROOT / "runs/stage27-long-real/summary.json").read_text(encoding="utf-8"))
    assert real["passed"] and real["real_model_calls"] == 160 and real["real_attempts"] == 161
    assert real["budget_after"]["reserved_cny"] <= 10
    old = ROOT / "releases/local-market-v13.1.0/local-market-v13.1.0-source.zip"
    assert digest(old) == "2861d5b9c877a5a2370995fb3c0677f46e60057f6261a2ec12a9670b7e1a3e60"
    files = []
    for folder in ("src", "configs", "artifacts", "tests", "scripts", "docs", "experiment-specs", "frontend/app", "frontend/public", "frontend/worker", "frontend/db", "frontend/build", "frontend/tests", "frontend/drizzle"):
        files.extend((ROOT / folder).rglob("*"))
    files.extend((ROOT / "frontend").glob("*"))
    files.extend(ROOT / name for name in ("README.md", "AGENTS.md", "pyproject.toml", "requirements-local.lock.txt", ".gitignore"))
    allowed = {".py", ".yaml", ".yml", ".json", ".jsonc", ".toml", ".md", ".ts", ".tsx", ".mjs", ".css", ".svg", ".png", ".sql", ".txt", ".ps1", ".cmd"}
    files = sorted(set(p for p in files if p.is_file() and p.suffix in allowed and not p.name.startswith(".env") and not any(v in {"__pycache__", ".tmp", "node_modules", "dist", ".next", ".local-state", ".git"} for v in p.relative_to(ROOT).parts)))
    content = {p.relative_to(ROOT).as_posix(): p for p in files}
    assert "artifacts/four-actor-policy-v1.json" in content
    outer = ("START_MARKET.ps1", "START_MARKET.cmd", "STOP_MARKET.cmd", "STATUS_MARKET.cmd", "BACKUP_MARKET.cmd", "START_BACKEND.ps1", "START_FRONTEND.ps1", "LOCAL_MARKET_GUIDE.md", "START_HERE.md", "PENDING_FEATURES_2026-09-07.md")
    hashes = {n: digest(p) for n, p in content.items()}
    DEST.mkdir(parents=True)
    archive = DEST / "local-market-v14.0.0-source.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        for name, path in content.items():
            bundle.write(path, "game-theory-agent/" + ROOT.name + "/" + name)
        for name in outer:
            bundle.write(OUTER / name, name)
    shutil.copy2(tests, DEST / "backend-tests.xml")
    for name in ("acceptance.json", "backup-restore.json"):
        shutil.copy2(ROOT / "runs/stage28-release" / name, DEST / name)
    for stage in ("stage21-personas-r2", "stage22-suppliers-r3", "stage23-government", "stage24-selfplay", "stage25-training", "stage27-long-real"):
        source = ROOT / "runs" / stage / "summary.json"
        shutil.copy2(source, DEST / (stage + ".json"))
    manifest = dict(release="local-market-v14.0.0", scope="single user local synthetic four-actor research market", config="configs/market_v14_local.yaml", python=sys.version, source_files=hashes, outer_files={n: digest(OUTER / n) for n in outer}, archive_sha256=digest(archive), source_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(",", ":")).encode()).hexdigest(), backend_tests=444, frontend_tests=11, browser_qa=True, main_episodes=254, main_rounds=5020, new_real_attempts=180, long_real_valid_choices=160, long_real_attempts=161, budget_reserved_cny=9.938022, excluded=["credentials", "runtime databases", "local research log", "venv", "node_modules", "generated frontend; rebuild with build_local.ps1"])
    (DEST / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(archive=str(archive), files=len(files), archive_sha256=manifest["archive_sha256"])))


if __name__ == "__main__":
    main()
