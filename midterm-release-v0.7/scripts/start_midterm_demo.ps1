$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$releaseRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimeLogRoot = Join-Path $projectRoot "logs\midterm-v0.7"
New-Item -ItemType Directory -Path $runtimeLogRoot -Force | Out-Null

if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "未找到 Python。" }
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw "未找到 Node.js。" }
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw "未找到 npm。" }
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot "frontend\node_modules"))) { throw "前端依赖未安装，请先在 frontend 执行 npm install。" }
if (-not (Test-Path -LiteralPath (Join-Path $releaseRoot "configs\market_v4.yaml"))) { throw "冻结市场配置缺失。" }

$env:PYTHONPATH = Join-Path $projectRoot "src"
$env:MARKET_CONFIG_PATH = Join-Path $releaseRoot "configs\market_v4.yaml"
$backend = Start-Process -FilePath "python" -ArgumentList @("-m", "uvicorn", "game_theory_agent.api:app", "--host", "127.0.0.1", "--port", "8010") -WorkingDirectory $projectRoot -RedirectStandardOutput (Join-Path $runtimeLogRoot "backend.out.log") -RedirectStandardError (Join-Path $runtimeLogRoot "backend.err.log") -WindowStyle Hidden -PassThru
$frontend = Start-Process -FilePath "npm.cmd" -ArgumentList @("run", "dev") -WorkingDirectory (Join-Path $projectRoot "frontend") -RedirectStandardOutput (Join-Path $runtimeLogRoot "frontend.out.log") -RedirectStandardError (Join-Path $runtimeLogRoot "frontend.err.log") -WindowStyle Hidden -PassThru

$healthy = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:8010/api/health" -TimeoutSec 2
        if ($response.status -eq "ok") { $healthy = $true; break }
    } catch { Start-Sleep -Seconds 1 }
}
if (-not $healthy) {
    Stop-Process -Id $backend.Id, $frontend.Id -Force -ErrorAction SilentlyContinue
    throw "后端健康检查失败，请查看 logs\midterm-v0.7。"
}

Write-Host "后端 PID: $($backend.Id)，前端 PID: $($frontend.Id)"
Write-Host "本演示使用冻结档案，不调用真实 LLM。"
Start-Process "http://localhost:3210"
