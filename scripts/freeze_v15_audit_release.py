"""Freeze the accepted local v15 source and evidence without replacing older releases."""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
OUTER = ROOT.parent.parent
DEST = ROOT / "releases/local-market-v15.0.1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise ValueError("Release already exists; never overwrite a frozen release")
    tests = OUTER / "migration_2026-09-07/theory-v15-audit.xml"
    suite = ElementTree.parse(tests).getroot().find("testsuite")
    assert int(suite.get("tests")) == 455
    assert all(int(suite.get(k, "0")) == 0 for k in ("errors", "failures", "skipped"))
    qa = json.loads((ROOT / "runs/theory-v15-audit/acceptance.json").read_text(encoding="utf-8"))
    assert qa["passed"] and qa["browser_qa"] and qa["backup_restore"]
    assert qa["frontend"] == dict(build=True, typecheck=True, lint=True, tests=12)
    real = json.loads((ROOT / "runs/theory-stage34/summary.json").read_text(encoding="utf-8"))
    assert real["passed"] and real["valid_choices"] == 6 and real["attempts"] == 6
    assert real["budget_after"]["reserved_cny"] <= 10
    old = ROOT / "releases/local-market-v15.0.0/local-market-v15.0.0-source.zip"
    assert digest(old) == "a8296155943e5f190f95b57dd80f8c3fa0ed10b80cc5f581c02a05de6a50dbca"
    for stage in range(29,35):
        evidence=json.loads((ROOT / f"runs/theory-stage{stage}/summary.json").read_text(encoding="utf-8"))
        assert evidence["passed"]
    audit = json.loads((ROOT / "runs/theory-v15-audit/summary.json").read_text(encoding="utf-8"))
    assert audit["passed"] and audit["new_real_calls"] == 0
    assert audit["learning_runs"] == 48 and audit["market_episodes"] == 6
    files = []
    for folder in ("src", "configs", "artifacts", "tests", "scripts", "docs", "experiment-specs", "frontend/app", "frontend/public", "frontend/worker", "frontend/db", "frontend/build", "frontend/tests", "frontend/drizzle"):
        files.extend((ROOT / folder).rglob("*"))
    files.extend((ROOT / "frontend").glob("*"))
    files.extend(ROOT / name for name in ("README.md", "AGENTS.md", "pyproject.toml", "requirements-local.lock.txt", ".gitignore"))
    allowed = {".py", ".yaml", ".yml", ".json", ".jsonc", ".toml", ".md", ".ts", ".tsx", ".mjs", ".css", ".svg", ".png", ".sql", ".txt", ".ps1", ".cmd"}
    files = sorted(set(p for p in files if p.is_file() and p.suffix in allowed and not p.name.startswith(".env") and not any(v in {"__pycache__", ".tmp", "node_modules", "dist", ".next", ".local-state", ".git"} for v in p.relative_to(ROOT).parts)))
    content = {p.relative_to(ROOT).as_posix(): p for p in files}
    assert "artifacts/four-actor-policy-v1.json" in content
    assert all(n in content for n in ("src/game_theory_agent/game_theory/lab.py", "src/game_theory_agent/game_theory/service.py", "src/game_theory_agent/game_theory/market_strategies.py", "frontend/app/theory-lab.tsx", "frontend/app/theory-lab.css"))
    outer = ("START_MARKET.ps1", "START_MARKET.cmd", "STOP_MARKET.cmd", "STATUS_MARKET.cmd", "BACKUP_MARKET.cmd", "START_BACKEND.ps1", "START_FRONTEND.ps1", "LOCAL_MARKET_GUIDE.md", "START_HERE.md", "PENDING_FEATURES_2026-09-07.md")
    hashes = {n: digest(p) for n, p in content.items()}
    DEST.mkdir(parents=True)
    archive = DEST / "local-market-v15.0.1-source.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        for name, path in content.items():
            bundle.write(path, "game-theory-agent/" + ROOT.name + "/" + name)
        for name in outer:
            bundle.write(OUTER / name, name)
    shutil.copy2(tests, DEST / "backend-tests.xml")
    for name in ("acceptance.json", "backup-restore.json"):
        shutil.copy2(ROOT / "runs/theory-v15-audit" / name, DEST / name)
    for stage in tuple(f"theory-stage{i}" for i in range(29,35)):
        source = ROOT / "runs" / stage / "summary.json"
        shutil.copy2(source, DEST / (stage + ".json"))
    shutil.copy2(ROOT / "runs/theory-stage34/market-execution-replay.json", DEST / "real-market-replay.json")
    shutil.copy2(ROOT / "runs/theory-v15-audit/summary.json", DEST / "audit-experiments.json")
    manifest = dict(release="local-market-v15.0.1", scope="single user local four-actor synthetic market with eight game-theory topics and five company policies", config="configs/market_v14_local.yaml", python=sys.version, source_files=hashes, outer_files={n: digest(OUTER / n) for n in outer}, archive_sha256=digest(archive), source_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(",", ":")).encode()).hexdigest(), backend_tests=455, frontend_tests=12, browser_qa=True, repeated_episodes=576, repeated_rounds=34560, learning_runs=64, learning_rounds=192000, analytic_conditions=81, market_episodes=48, market_rounds=960, new_real_attempts=0, historical_real_valid_choices=6, audit_learning_runs=48, audit_learning_rounds=99216, audit_market_episodes=6, audit_market_rounds=120, budget_reserved_cny=9.993565, excluded=["credentials", "runtime databases", "local research log", "venv", "node_modules", "generated frontend; rebuild with build_local.ps1"])
    (DEST / "manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(archive=str(archive), files=len(files), archive_sha256=manifest["archive_sha256"])))


if __name__ == "__main__":
    main()
