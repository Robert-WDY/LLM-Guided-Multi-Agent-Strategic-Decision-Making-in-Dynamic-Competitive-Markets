param(
    [ValidateSet('Start','Stop','Status','Backup','Restore')][string]$Action='Start',
    [string]$BackupPath,
    [string]$DataDirectory,
    [string]$ConfigPath,
    [int]$ApiPort=8010,
    [int]$FrontendPort=3210
)
$ErrorActionPreference='Stop'
$repository=Split-Path -Parent $PSScriptRoot
if (-not $ConfigPath) { $ConfigPath=Join-Path $repository 'configs\market_v14_local.yaml' }
if (-not $DataDirectory) { $DataDirectory=Join-Path $repository '.local-state-v14-release' }
$DataDirectory=[IO.Path]::GetFullPath($DataDirectory)
New-Item -ItemType Directory -Path $DataDirectory -Force | Out-Null
$registryPath=Join-Path $DataDirectory 'processes.json'
$pythonVenv=Join-Path $repository '.venv\Scripts\python.exe'
$env:PYTHONPATH=(Join-Path $repository 'src')+';'+(Join-Path $repository '.venv\Lib\site-packages')
$services=@()
if (Test-Path -LiteralPath $registryPath) { $services=Get-Content -LiteralPath $registryPath -Raw | ConvertFrom-Json; $services=@($services | Where-Object { $_ }) }
function Find-Owned($service) {
    $process=Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$service.pid)"
    if (-not $process) { return $null }
    if ($process.CreationDate.ToUniversalTime().Ticks -ne ([datetime]$service.created).ToUniversalTime().Ticks -or $process.CommandLine -ne $service.commandLine) { throw "Process ID reused or ownership changed: $($service.pid). Refusing to stop it." }
    return $process
}
function Write-Registry { ConvertTo-Json -InputObject @($script:services) -Depth 5 | Set-Content -LiteralPath $registryPath -Encoding utf8 }
function Assert-Free([int]$port) {
    $listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,$port)
    try { $listener.Start() } catch { throw "Port $port is already occupied. Stop the existing service first; no process was killed." } finally { $listener.Stop() }
}
function Wait-Ready([string]$url) {
    for ($attempt=0;$attempt -lt 80;$attempt++) {
        try { $reply=Invoke-WebRequest -Uri $url -TimeoutSec 2 -UseBasicParsing; if ($reply.StatusCode -eq 200) { return } } catch {}
        Start-Sleep -Milliseconds 250
    }
    throw "Service did not become ready: $url. See $DataDirectory logs."
}
function Spawn-Service($name,$executable,$arguments,$directory) {
    $process=Start-Process -FilePath $executable -ArgumentList $arguments -WorkingDirectory $directory -WindowStyle Hidden -RedirectStandardOutput (Join-Path $DataDirectory "$name.stdout.log") -RedirectStandardError (Join-Path $DataDirectory "$name.stderr.log") -PassThru
    Start-Sleep -Milliseconds 300
    $identity=Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)"
    if (-not $identity) { throw "$name exited; inspect its logs." }
    $script:services+=@{ name=$name;pid=$process.Id;created=$identity.CreationDate.ToUniversalTime().ToString('o');commandLine=$identity.CommandLine;apiPort=$ApiPort;frontendPort=$FrontendPort }
    Write-Registry
}
switch ($Action) {
    'Status' {
        foreach ($service in $services) { $running=[bool](Find-Owned $service); Write-Output "$($service.name): running=$running pid=$($service.pid)" }
        if (-not $services.Count) { Write-Output 'No managed services running.' }
    }
    'Stop' {
        $backend=$services | Where-Object name -eq 'backend' | Select-Object -First 1
        if ($backend -and (Find-Owned $backend)) {
            $runState=Invoke-RestMethod -Uri "http://127.0.0.1:$($backend.apiPort)/api/v1/controller/runtime-status" -TimeoutSec 5
            if ($runState.active_runs -gt 0) { throw 'A round is running. Pause in the UI and wait for settlement before stopping.' }
        }
        foreach ($service in $services) {
            if (Find-Owned $service) { Stop-Process -Id $service.pid; Wait-Process -Id $service.pid -Timeout 10 -ErrorAction SilentlyContinue }
        }
        $script:services=@(); Write-Registry
        Write-Output 'Managed services stopped. Experiments and budget retained.'
    }
    'Backup' {
        if (-not $BackupPath) { $BackupPath=Join-Path $DataDirectory ('backups\'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')) }
        & $pythonVenv (Join-Path $PSScriptRoot 'local_storage.py') backup $DataDirectory $BackupPath --config $ConfigPath
        if ($LASTEXITCODE -ne 0) { throw 'Backup failed.' }
        Write-Output "Backup: $BackupPath"
    }
    'Restore' {
        if (-not $BackupPath) { throw 'Supply -BackupPath with the backup directory.' }
        foreach ($service in $services) { if (Find-Owned $service) { throw 'Stop managed services before restoring.' } }
        Assert-Free $ApiPort
        & $pythonVenv (Join-Path $PSScriptRoot 'local_storage.py') restore $DataDirectory $BackupPath --config $ConfigPath
        if ($LASTEXITCODE -ne 0) { throw 'Restore failed; previous copies preserved.' }
    }
    'Start' {
        $live=@($services | Where-Object { Find-Owned $_ })
        if ($live.Count) {
            if ($live.Count -ne 2) { throw 'Partial startup detected. Run Stop, inspect logs, then Start.' }
            Wait-Ready "http://127.0.0.1:$($live[0].apiPort)/api/health"
            Wait-Ready "http://127.0.0.1:$($live[0].frontendPort)/"
            Write-Output "Already running: http://localhost:$($live[0].frontendPort)/"; break
        }
        Assert-Free $ApiPort; Assert-Free $FrontendPort
        $pythonBase=(& $pythonVenv -c 'import sys; print(sys._base_executable)').Trim()
        if ($LASTEXITCODE -ne 0) { throw 'Python environment is unavailable.' }
        $node=(Get-Command node.exe -ErrorAction SilentlyContinue).Source
        if (-not $node -and (Test-Path -LiteralPath 'D:\node\node.exe')) { $node='D:\node\node.exe' }
        if (-not $node) { throw 'Node.js was not found.' }
        $frontend=Join-Path $repository 'frontend'
        if (-not (Test-Path -LiteralPath (Join-Path $frontend 'dist\server\index.js'))) { throw 'Production frontend is missing. Run scripts/build_local.ps1 first.' }
        & $pythonVenv -c 'from game_theory_agent.local_budget import initialize; import json; print(json.dumps(initialize()))'
        if ($LASTEXITCODE -ne 0) { throw 'Budget initialization failed; existing data was not reset.' }
        $env:MARKET_CONFIG_PATH=(Resolve-Path -LiteralPath $ConfigPath).Path
        $env:MARKET_SESSION_DB=Join-Path $DataDirectory 'sessions.sqlite3'
        $env:MARKET_API_HOST='127.0.0.1'; $env:MARKET_API_PORT=[string]$ApiPort
        $env:MARKET_AGENT_GATEWAY_ENABLED='0'; $env:MARKET_PERSISTENCE='1'
        $env:MARKET_LOCAL_DEV_UNAUTHENTICATED_CONTROLLER='1'
        $script:services=@()
        Spawn-Service 'backend' $pythonBase '-m game_theory_agent.api' $repository
        Wait-Ready "http://127.0.0.1:$ApiPort/api/health"
        Spawn-Service 'frontend' $node ('"'+(Join-Path $frontend 'node_modules\vinext\dist\cli.js')+'" start --port '+$FrontendPort+' --hostname 127.0.0.1') $frontend
        Wait-Ready "http://127.0.0.1:$FrontendPort/"
        Write-Output "Local market ready: http://localhost:$FrontendPort/"
    }
}
