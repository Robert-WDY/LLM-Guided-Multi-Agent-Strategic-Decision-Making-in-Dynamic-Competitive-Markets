"""Windows 一键启动器的进程边界与无副作用验证。"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start.ps1"
START_BAT = ROOT / "start.bat"
VENV = ROOT / ".venv"
STARTUP_LOGS = ROOT / "~temp" / "logs" / "startup"


def test_validate_only_reports_configuration_without_local_side_effects():
    """防止验证模式创建环境、日志或报告错误端口。"""

    powershell = shutil.which("powershell.exe")
    assert powershell is not None, "Windows PowerShell is required for this test"

    before = (VENV.exists(), STARTUP_LOGS.exists())
    completed = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-ValidateOnly",
            "-NoBrowser",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])

    assert payload == {
        "mode": "validate",
        "project_root": str(ROOT),
        "ports": {"frontend": 3210, "api": 8010, "agent": 8011},
        "urls": {
            "frontend": "http://127.0.0.1:3210/",
            "api_health": "http://127.0.0.1:8010/api/health",
            "agent_health": "http://127.0.0.1:8011/health",
        },
        "requirements": str(ROOT / "requirements.txt"),
        "package_lock": str(ROOT / "frontend" / "package-lock.json"),
    }
    assert (VENV.exists(), STARTUP_LOGS.exists()) == before


def test_cmd_bat_chain_preserves_utf8_and_exits():
    """防止双击等价链路乱码或在自动化验证后保留窗口。"""

    completed = subprocess.run(
        ["cmd.exe", "/d", "/s", "/c", "call start.bat -ValidateOnly -NoBrowser"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
    )
    assert "Encoding probe: 中文" in completed.stdout
    assert "�" not in completed.stdout
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert not lines[-1].startswith("PS "), completed.stdout
    assert json.loads(lines[-1])["mode"] == "validate"


def test_launcher_stops_target_ports_before_frontend_dependency_sync():
    """防止运行中的前端锁住 npm ci 需要替换的 Windows 原生模块。"""

    script = SCRIPT.read_text(encoding="utf-8-sig")
    execution = script[script.index("$PreviousControllerToken") :]

    assert execution.index("Stop-TargetPortListeners") < execution.index(
        "Sync-FrontendEnvironment"
    )
