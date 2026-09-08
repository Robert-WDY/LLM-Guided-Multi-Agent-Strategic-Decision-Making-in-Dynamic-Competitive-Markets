$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $projectRoot "configs\market_v11_supplier.yaml"
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw "Project Python missing: $pythonPath" }
$env:PYTHONPATH = Join-Path $projectRoot "src"
$env:MARKET_CONFIG_PATH = $configPath
if (-not $env:MARKET_CONTROLLER_TOKEN) {
    $env:MARKET_CONTROLLER_TOKEN = "local-supplier-$([guid]::NewGuid().ToString('N'))"
}
$env:MARKET_API_HOST = "127.0.0.1"
$env:MARKET_LOCAL_DEV_UNAUTHENTICATED_CONTROLLER = "1"
Write-Host "Starting v11 autonomous supplier pricing on http://127.0.0.1:8010"
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -m game_theory_agent.api
    if ($LASTEXITCODE -ne 0) { throw "Backend exited with code $LASTEXITCODE" }
} finally { Pop-Location }
