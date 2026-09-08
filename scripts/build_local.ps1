$ErrorActionPreference='Stop'
$repository=Split-Path -Parent $PSScriptRoot
$node=(Get-Command node.exe -ErrorAction SilentlyContinue).Source
if (-not $node -and (Test-Path -LiteralPath 'D:\node\node.exe')) { $node='D:\node\node.exe' }
if (-not $node) { throw 'Node.js missing.' }
$npm=Join-Path (Split-Path -Parent $node) 'node_modules\npm\bin\npm-cli.js'
$env:PATH=(Split-Path -Parent $node)+';'+$env:PATH
$env:NEXT_PUBLIC_MARKET_API_URL='http://127.0.0.1:8010/api'
$env:NEXT_PUBLIC_LOCAL_CONTROLLER_BYPASS='1'
Push-Location -LiteralPath (Join-Path $repository 'frontend')
try { & $node $npm run test; if ($LASTEXITCODE -ne 0) { throw 'Production build/tests failed.' } } finally { Pop-Location }
