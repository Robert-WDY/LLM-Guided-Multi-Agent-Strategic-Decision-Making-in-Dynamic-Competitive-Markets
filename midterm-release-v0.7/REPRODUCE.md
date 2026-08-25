# 复现说明

## 环境

- Windows PowerShell 7
- Python 3.13
- Node.js 22.13 或更高
- 项目依赖按根目录 `pyproject.toml` 与 `frontend/package-lock.json` 安装

## 一键验证发布证据

```powershell
.\midterm-release-v0.7\scripts\verify_release.ps1
```

验证内容包括文件 SHA-256、实验目录白名单、三个选中 RoundEvent 的 Economic/Interaction/Information/Belief/Advisor/Adoption Replay、隐藏状态泄漏和 Forecast Report Hash。Forecast 预测门禁失败是冻结结果，验证脚本要求它继续保持“真实 LLM 门关闭”。

## 单独重建选中实验

```powershell
.\midterm-release-v0.7\scripts\replay_selected_runs.ps1
```

## 运行测试

```powershell
$env:PYTHONPATH = "src"
pytest -q
ruff check src\game_theory_agent\api.py src\game_theory_agent\strategic_reliability\forecast_reliability.py src\game_theory_agent\experiments\midterm_forecast_calibration.py src\game_theory_agent\experiments\midterm_release_verification.py tests\test_api.py tests\test_forecast_reliability.py
python -m compileall -q src tests

Set-Location frontend
npm run lint
npm test
```

全仓 Ruff 仍会报告若干早期实验入口在 `sys.path` 引导后的 E402 和一个历史 `Any` 导入问题；这些文件不属于本次发布修改范围，253 项后端测试均通过。本发布以严格的变更/发布范围 Ruff 作为门禁，并在 `TEST_RESULTS.json` 记录该遗留项。

## 零 Token 校准

```powershell
$env:PYTHONPATH = "src"
python -m game_theory_agent.experiments.midterm_forecast_calibration
```

该命令计算量较大，会完整重建两遍 435 条样本。它应以退出码 2 结束，因为冻结 Holdout 门禁未通过；这不是脚本错误。正式中期证据已经保存在 `selected-runs/forecast-calibration`，答辩当天无需重新运行。
