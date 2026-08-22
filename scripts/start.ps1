[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$ForceInstall,
    [switch]$ValidateOnly,
    [switch]$SmokeTest
)

# Windows 一键启动：准备依赖、启动前后端、验证健康状态并统一清理进程。
$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0
Import-Module Microsoft.PowerShell.Utility -ErrorAction SilentlyContinue
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$FrontendPath = Join-Path $ProjectRoot "frontend"
$PackageLockPath = Join-Path $FrontendPath "package-lock.json"
$VinextCliPath = Join-Path $FrontendPath "node_modules\vinext\dist\cli.js"
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$FrontendPort = 3210
$ApiPort = 8010
$AgentPort = 8011
$FrontendUrl = "http://127.0.0.1:$FrontendPort/"
$ApiHealthUrl = "http://127.0.0.1:$ApiPort/api/health"
$AgentHealthUrl = "http://127.0.0.1:$AgentPort/health"

function Write-Step {
    param([string]$Message)
    Write-Host "[market-agents] $Message" -ForegroundColor Cyan
}

function Assert-RequiredLayout {
    foreach ($Path in @($RequirementsPath, $PackageLockPath, (Join-Path $ProjectRoot "pyproject.toml"))) {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "Required file is missing: $Path"
        }
    }
}

function Get-WorkingPython {
    $Candidates = New-Object System.Collections.Generic.List[string]
    if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
        $Candidates.Add($VenvPython)
    }
    foreach ($Name in @("python.exe", "python3.exe")) {
        foreach ($Command in @(Get-Command $Name -All -ErrorAction SilentlyContinue)) {
            if ($Command.Source) {
                $Candidates.Add($Command.Source)
            }
        }
    }
    $LocalPrograms = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (Test-Path -LiteralPath $LocalPrograms -PathType Container) {
        foreach ($Candidate in @(Get-ChildItem -LiteralPath $LocalPrograms -Filter python.exe -File -Recurse -ErrorAction SilentlyContinue)) {
            $Candidates.Add($Candidate.FullName)
        }
    }
    foreach ($Candidate in @($Candidates | Select-Object -Unique)) {
        try {
            $VersionText = & $Candidate -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
            if ($LASTEXITCODE -ne 0 -or -not $VersionText) { continue }
            if ([version]($VersionText | Select-Object -Last 1) -ge [version]"3.11") {
                return $Candidate
            }
        }
        catch { continue }
    }
    throw "Python 3.11 or newer was not found. Install it and ensure python.exe is available on PATH."
}

function Get-Sha256FileHash {
    param([string]$Path)
    $Hasher = Get-Command Get-FileHash -ErrorAction SilentlyContinue
    if ($Hasher) {
        return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    }
    $Sha = [System.Security.Cryptography.SHA256]::Create()
    $Stream = [System.IO.File]::OpenRead($Path)
    try {
        return ([System.BitConverter]::ToString($Sha.ComputeHash($Stream))).Replace("-", "").ToUpperInvariant()
    }
    finally {
        $Stream.Dispose()
        $Sha.Dispose()
    }
}

function Get-CombinedHash {
    param([string[]]$Paths)
    $Text = ($Paths | ForEach-Object {
        if (-not (Test-Path -LiteralPath $_ -PathType Leaf)) {
            throw "Cannot hash missing file: $_"
        }
        Get-Sha256FileHash $_
    }) -join "`n"
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
    $Sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($Sha.ComputeHash($Bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally { $Sha.Dispose() }
}

function Sync-PythonEnvironment {
    $Python = Get-WorkingPython
    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        Write-Step "Creating .venv with $Python"
        & $Python -m venv $VenvPath
        if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv." }
    }
    $Marker = Join-Path $VenvPath ".requirements.sha256"
    $ExpectedHash = Get-CombinedHash @($RequirementsPath, (Join-Path $ProjectRoot "pyproject.toml"))
    $InstalledHash = if (Test-Path -LiteralPath $Marker -PathType Leaf) {
        (Get-Content -LiteralPath $Marker -Raw -Encoding UTF8).Trim()
    } else { "" }
    if ($ForceInstall -or $InstalledHash -ne $ExpectedHash) {
        Write-Step "Installing Python dependencies"
        & $VenvPython -m pip install --disable-pip-version-check -r $RequirementsPath
        if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }
        Set-Content -LiteralPath $Marker -Value $ExpectedHash -Encoding ASCII
    }
    else { Write-Step "Python dependencies are current" }
}

function Sync-FrontendEnvironment {
    $Npm = Get-Command "npm.cmd" -ErrorAction SilentlyContinue
    if (-not $Npm) { throw "npm.cmd was not found. Install Node.js 22.13 or newer." }
    $NodeVersionText = & node.exe --version
    if ($LASTEXITCODE -ne 0 -or -not $NodeVersionText) { throw "node.exe could not be executed." }
    $NodeVersion = [version]($NodeVersionText.TrimStart("v"))
    if ($NodeVersion -lt [version]"22.13") {
        throw "Node.js 22.13 or newer is required; found $NodeVersion."
    }
    $NodeModules = Join-Path $FrontendPath "node_modules"
    $Marker = Join-Path $NodeModules ".package-lock.sha256"
    $ExpectedHash = (Get-Sha256FileHash $PackageLockPath).ToLowerInvariant()
    $InstalledHash = if (Test-Path -LiteralPath $Marker -PathType Leaf) {
        (Get-Content -LiteralPath $Marker -Raw -Encoding UTF8).Trim()
    } else { "" }
    if ($ForceInstall -or -not (Test-Path -LiteralPath $NodeModules -PathType Container) -or $InstalledHash -ne $ExpectedHash) {
        Write-Step "Installing frontend dependencies"
        Push-Location $FrontendPath
        try {
            & $Npm.Source ci | Out-Host
            if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
        }
        finally { Pop-Location }
        Set-Content -LiteralPath $Marker -Value $ExpectedHash -Encoding ASCII
    }
    else { Write-Step "Frontend dependencies are current" }
    return $Npm.Source
}

function Build-Frontend {
    param([string]$NpmCommand)
    Write-Step "Building frontend"
    Push-Location $FrontendPath
    try {
        & $NpmCommand run build | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
    }
    finally { Pop-Location }
}

function Stop-TargetPortListeners {
    param([int[]]$Ports)
    $ProcessIds = New-Object System.Collections.Generic.List[int]
    foreach ($Port in $Ports) {
        foreach ($Connection in @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)) {
            if ($Connection.OwningProcess -gt 0) { $ProcessIds.Add([int]$Connection.OwningProcess) }
        }
        foreach ($Line in @(& netstat.exe -ano -p TCP)) {
            if ($Line -match "^\s*TCP\s+\S+:$Port\s+\S+\s+LISTENING\s+(\d+)\s*$") {
                $ProcessIds.Add([int]$Matches[1])
            }
        }
    }
    foreach ($ProcessId in @($ProcessIds | Where-Object { $_ -gt 0 } | Select-Object -Unique)) {
        Write-Step "Stopping process $ProcessId on a target port"
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
    }
}

function Wait-ForUrl {
    param(
        [string]$Url,
        [int]$TimeoutSeconds,
        [System.Diagnostics.Process]$ProcessToWatch,
        [string]$ServiceName,
        [string]$LogDirectory
    )
    $Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $Deadline) {
        if ($ProcessToWatch -and $ProcessToWatch.HasExited) {
            $ProcessToWatch.WaitForExit()
            throw "$ServiceName exited during startup (PID $($ProcessToWatch.Id)). Check $LogDirectory."
        }
        try {
            $Response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
            if ($Response.StatusCode -ge 200 -and $Response.StatusCode -lt 500) { return }
        }
        catch { Start-Sleep -Milliseconds 500 }
    }
    throw "Timed out waiting for $Url"
}

function Test-UrlAvailable {
    param([string]$Url)
    try {
        $Response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
        return ($Response.StatusCode -ge 200 -and $Response.StatusCode -lt 500)
    }
    catch { return $false }
}

function Assert-LocalCommands {
    [void](Get-WorkingPython)
    if (-not (Get-Command "npm.cmd" -ErrorAction SilentlyContinue)) {
        throw "npm.cmd was not found. Install Node.js 22.13 or newer."
    }
    $NodeVersionText = & node.exe --version
    if ($LASTEXITCODE -ne 0 -or -not $NodeVersionText) { throw "node.exe could not be executed." }
    $NodeVersion = [version]($NodeVersionText.TrimStart("v"))
    if ($NodeVersion -lt [version]"22.13") {
        throw "Node.js 22.13 or newer is required; found $NodeVersion."
    }
}

function Normalize-ProcessPathVariable {
    $CurrentPath = [Environment]::GetEnvironmentVariable("Path", "Process")
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $CurrentPath, "Process")
}

function Stop-LaunchedProcessTree {
    param([System.Diagnostics.Process]$Process)
    if (-not $Process.HasExited) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    }
}

Assert-RequiredLayout

if ($ValidateOnly) {
    Assert-LocalCommands
    $EncodingProbe = -join ([char[]](0x4E2D, 0x6587))
    Write-Step "Encoding probe: $EncodingProbe"
    [ordered]@{
        mode = "validate"
        project_root = $ProjectRoot
        ports = [ordered]@{ frontend = $FrontendPort; api = $ApiPort; agent = $AgentPort }
        urls = [ordered]@{
            frontend = $FrontendUrl
            api_health = $ApiHealthUrl
            agent_health = $AgentHealthUrl
        }
        requirements = $RequirementsPath
        package_lock = $PackageLockPath
    } | ConvertTo-Json -Compress
    exit 0
}

$StartedProcesses = New-Object System.Collections.Generic.List[System.Diagnostics.Process]
$LaunchEnvironment = [ordered]@{
    PYTHONPATH = (Join-Path $ProjectRoot "src")
    MARKET_API_HOST = "127.0.0.1"
    MARKET_API_PORT = "$ApiPort"
    MARKET_AGENT_HOST = "127.0.0.1"
    MARKET_AGENT_PORT = "$AgentPort"
    MARKET_AGENT_GATEWAY_ENABLED = "1"
    NEXT_PUBLIC_MARKET_API_URL = "http://127.0.0.1:$ApiPort/api"
    PYTHONUTF8 = "1"
    PYTHONIOENCODING = "utf-8"
}
$PreviousEnvironment = @{}
foreach ($Name in $LaunchEnvironment.Keys) {
    $PreviousEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, "Process")
}
$PreviousControllerToken = [Environment]::GetEnvironmentVariable("MARKET_CONTROLLER_TOKEN", "Process")
$HadControllerToken = -not [string]::IsNullOrWhiteSpace($PreviousControllerToken)

try {
    Stop-TargetPortListeners @($FrontendPort, $ApiPort, $AgentPort)
    Sync-PythonEnvironment
    $NpmCommand = Sync-FrontendEnvironment
    Build-Frontend $NpmCommand
    $FrontendNode = Get-Command "node.exe" -ErrorAction SilentlyContinue
    if (-not $FrontendNode) { throw "node.exe was not found. Install Node.js 22.13 or newer." }
    if (-not (Test-Path -LiteralPath $VinextCliPath -PathType Leaf)) {
        throw "Vinext CLI was not found: $VinextCliPath"
    }
    $LogDirectory = Join-Path $ProjectRoot "~temp\logs\startup"
    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    $Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    foreach ($Name in $LaunchEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($Name, $LaunchEnvironment[$Name], "Process")
    }
    if (-not $HadControllerToken) {
        [Environment]::SetEnvironmentVariable(
            "MARKET_CONTROLLER_TOKEN",
            ([guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N")),
            "Process"
        )
    }
    Normalize-ProcessPathVariable

    Write-Step "Starting API and Agent Gateway"
    $Backend = Start-Process -FilePath $VenvPython `
        -ArgumentList @("-m", "game_theory_agent.api") `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogDirectory "$Timestamp-backend.stdout.log") `
        -RedirectStandardError (Join-Path $LogDirectory "$Timestamp-backend.stderr.log") `
        -PassThru
    $StartedProcesses.Add($Backend)

    Write-Step "Starting frontend"
    $Frontend = Start-Process -FilePath $FrontendNode.Source `
        -ArgumentList @($VinextCliPath, "start", "--port", "$FrontendPort", "--hostname", "127.0.0.1") `
        -WorkingDirectory $FrontendPath `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogDirectory "$Timestamp-frontend.stdout.log") `
        -RedirectStandardError (Join-Path $LogDirectory "$Timestamp-frontend.stderr.log") `
        -PassThru
    $StartedProcesses.Add($Frontend)

    Wait-ForUrl $ApiHealthUrl 60 $Backend "API" $LogDirectory
    Wait-ForUrl $AgentHealthUrl 60 $Backend "Agent Gateway" $LogDirectory
    Wait-ForUrl $FrontendUrl 60 $Frontend "Frontend" $LogDirectory
    Write-Step "Ready: $FrontendUrl"
    if (-not $NoBrowser) { Start-Process $FrontendUrl | Out-Null }
    if ($SmokeTest) {
        Write-Step "Smoke test passed; stopping launched services"
        exit 0
    }

    Write-Step "Press Ctrl+C to stop all launched services"
    while ($true) {
        if ($Backend.HasExited -and (-not (Test-UrlAvailable $ApiHealthUrl) -or -not (Test-UrlAvailable $AgentHealthUrl))) {
            throw "A launched service exited unexpectedly (PID $($Backend.Id), code $($Backend.ExitCode)). Check $LogDirectory."
        }
        if ($Frontend.HasExited -and -not (Test-UrlAvailable $FrontendUrl)) {
            throw "A launched service exited unexpectedly (PID $($Frontend.Id), code $($Frontend.ExitCode)). Check $LogDirectory."
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    foreach ($Process in $StartedProcesses) {
        Stop-LaunchedProcessTree $Process
        $Process.Dispose()
    }
    try { Stop-TargetPortListeners @($FrontendPort, $ApiPort, $AgentPort) }
    catch { Write-Warning "Some launched port listeners could not be stopped automatically: $($_.Exception.Message)" }
    if ($HadControllerToken) {
        [Environment]::SetEnvironmentVariable("MARKET_CONTROLLER_TOKEN", $PreviousControllerToken, "Process")
    }
    else {
        [Environment]::SetEnvironmentVariable("MARKET_CONTROLLER_TOKEN", $null, "Process")
    }
    foreach ($Name in $LaunchEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($Name, $PreviousEnvironment[$Name], "Process")
    }
}
