$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $projectRoot "configs\market_v10_multi_objective.yaml"
$sourcePath = Join-Path $projectRoot "src"
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Final market config not found: $configPath"
}

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project Python is missing: $pythonPath. Create .venv with Python 3.11+ and install .[test]."
}

$env:PYTHONPATH = $sourcePath
$env:MARKET_CONFIG_PATH = $configPath

if (-not $env:MARKET_CONTROLLER_TOKEN) {
    $env:MARKET_CONTROLLER_TOKEN = "local-final-market-$([guid]::NewGuid().ToString('N'))"
}

# This bypass is accepted only while MARKET_API_HOST is loopback. It lets the
# local browser create protected experiments without exposing the token.
$env:MARKET_API_HOST = "127.0.0.1"
$env:MARKET_LOCAL_DEV_UNAUTHENTICATED_CONTROLLER = "1"

Write-Host "Starting final strategic market: market-v10-multi-objective"
Write-Host "Private API: http://127.0.0.1:8010"
Write-Host "Agent gateway: http://127.0.0.1:8011"
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -m game_theory_agent.api
    if ($LASTEXITCODE -ne 0) { throw "Backend exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
