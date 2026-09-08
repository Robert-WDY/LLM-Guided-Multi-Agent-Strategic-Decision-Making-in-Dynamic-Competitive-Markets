$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$frontendPath = Join-Path $projectRoot "frontend"

if (-not (Test-Path -LiteralPath (Join-Path $frontendPath "package.json") -PathType Leaf)) {
    throw "Frontend package not found: $frontendPath"
}

$env:NEXT_PUBLIC_MARKET_API_URL = "http://127.0.0.1:8010/api"
$env:NEXT_PUBLIC_LOCAL_CONTROLLER_BYPASS = "1"

Write-Host "Starting final market research console: http://127.0.0.1:3210"
$nodePath = (Get-Command node -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$npmCliPath = Join-Path (Split-Path -Parent $nodePath) "node_modules\npm\bin\npm-cli.js"
if (Test-Path -LiteralPath $npmCliPath -PathType Leaf) {
    # Use the npm installed beside Node; avoids a stale global npm shim.
    & $nodePath $npmCliPath --prefix $frontendPath run dev
}
else {
    npm --prefix $frontendPath run dev
}
if ($LASTEXITCODE -ne 0) { throw "Frontend exited with code $LASTEXITCODE" }
