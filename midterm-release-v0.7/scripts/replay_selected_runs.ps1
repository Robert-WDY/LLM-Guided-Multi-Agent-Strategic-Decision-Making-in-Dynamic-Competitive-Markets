$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$releaseRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $projectRoot "src"
Set-Location $projectRoot
python -m game_theory_agent.experiments.midterm_release_verification --release-root $releaseRoot
if ($LASTEXITCODE -ne 0) { throw "中期发布回放失败。" }
