$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$releaseRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$checksumFile = Join-Path $releaseRoot "ARTIFACT_CHECKSUMS.sha256"

if (-not (Test-Path -LiteralPath $checksumFile)) { throw "缺少 ARTIFACT_CHECKSUMS.sha256。" }
foreach ($line in Get-Content -LiteralPath $checksumFile) {
    if (-not $line.Trim()) { continue }
    $parts = $line -split "  ", 2
    if ($parts.Count -ne 2) { throw "无效校验行：$line" }
    $path = Join-Path $releaseRoot $parts[1]
    if (-not (Test-Path -LiteralPath $path)) { throw "缺少发布文件：$($parts[1])" }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $parts[0].ToLowerInvariant()) { throw "SHA-256 不一致：$($parts[1])" }
}

$env:PYTHONPATH = Join-Path $projectRoot "src"
Set-Location $projectRoot
python -m game_theory_agent.experiments.midterm_release_verification --release-root $releaseRoot
if ($LASTEXITCODE -ne 0) { throw "发布回放或研究门禁验证失败。" }
Write-Host "Strategic Research MVP v0.7 发布验证通过。"
