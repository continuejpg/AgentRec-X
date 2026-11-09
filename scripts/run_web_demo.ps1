# AgentRec-X Web Demo - one-command launcher (Windows host + WSL Ubuntu-22.04)
#
#   powershell -ExecutionPolicy Bypass -File scripts/run_web_demo.ps1
#
# What it starts:
#   1) FastAPI BFF inside WSL        http://127.0.0.1:8787  (P17 MCP servers, BGE, CPU only)
#   2) Next.js frontend on Windows   http://127.0.0.1:3000
#
# It waits for the real /api/health readiness (MCP cold start is ~35-45 s), prints the
# per-server state, and only then opens the browser. Use -NoBrowser to skip that.
#
# Logs: runs/web_demo/logs/{bff.console.log,bff.console.err.log,frontend.console.log,frontend.console.err.log}
# PIDs: runs/web_demo/launcher_pids.json + runs/web_demo/bff.pid (Linux pid)
# Stop: powershell -ExecutionPolicy Bypass -File scripts/stop_web_demo.ps1

[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$SkipInstall,
    [int]$BffPort = 8787,
    [int]$FrontendPort = 3000,
    [int]$ReadyTimeoutSec = 420
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Runs = Join-Path $Root "runs\web_demo"
$Logs = Join-Path $Runs "logs"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

function Convert-ToWslPath([string]$WindowsPath) {
    $drive = $WindowsPath.Substring(0, 1).ToLowerInvariant()
    $rest = $WindowsPath.Substring(2).Replace('\', '/')
    return "/mnt/$drive$rest"
}

$WslRoot = Convert-ToWslPath $Root
$BffConsole = Join-Path $Logs "bff.console.log"
$BffConsoleErr = Join-Path $Logs "bff.console.err.log"
$FrontendConsole = Join-Path $Logs "frontend.console.log"
$FrontendConsoleErr = Join-Path $Logs "frontend.console.err.log"

function Start-DemoProcess([string]$FilePath, [string[]]$Arguments, [string]$StdOut, [string]$StdErr, [string]$WorkingDirectory = "") {
    $params = @{
        FilePath               = $FilePath
        ArgumentList           = $Arguments
        PassThru               = $true
        WindowStyle            = "Hidden"
        RedirectStandardOutput = $StdOut
        RedirectStandardError  = $StdErr
    }
    if ($WorkingDirectory) { $params["WorkingDirectory"] = $WorkingDirectory }
    return Start-Process @params
}

function Stop-ProcessTree([int]$ProcessId) {
    if ($ProcessId -le 0) { return }
    try { & taskkill.exe /PID $ProcessId /T /F 2>$null | Out-Null } catch { }
}

Write-Host "AgentRec-X Web Demo launcher" -ForegroundColor Cyan
Write-Host "  repo (Windows): $Root"
Write-Host "  repo (WSL):     $WslRoot"
Write-Host ""

# ---------------------------------------------------------------------------
# 0) prerequisites
# ---------------------------------------------------------------------------
$webDir = Join-Path $Root "web"
if (-not (Test-Path (Join-Path $webDir "package.json"))) {
    throw "web/package.json not found; the Next.js frontend is not built yet."
}
if (-not (Test-Path (Join-Path $webDir "node_modules"))) {
    if ($SkipInstall) {
        throw "web/node_modules missing and -SkipInstall was given."
    }
    Write-Host "[1/4] Installing frontend dependencies (pnpm --dir web install)..." -ForegroundColor Yellow
    & cmd.exe /c "cd /d `"$Root`" && pnpm --dir web install" 2>&1 | Tee-Object -FilePath (Join-Path $Logs "pnpm-install.log") | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "pnpm install failed (see runs/web_demo/logs/pnpm-install.log)" }
}

# ---------------------------------------------------------------------------
# 1) BFF in WSL
# ---------------------------------------------------------------------------
Write-Host "[2/4] Starting FastAPI BFF in WSL (cold start ~35-45 s)..." -ForegroundColor Yellow
$bffArgs = @(
    "-d", "Ubuntu-22.04",
    "--cd", $WslRoot,
    "--", "env",
    "AGENTRECX_WEB_DEMO_PORT=$BffPort",
    "AGENTRECX_WEB_DEMO_SEARCH_MODE=bge",
    "AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES=1",
    "bash", "scripts/run_web_demo_bff.sh"
)
$bff = Start-DemoProcess -FilePath "wsl.exe" -Arguments $bffArgs -StdOut $BffConsole -StdErr $BffConsoleErr

# ---------------------------------------------------------------------------
# 2) Next.js frontend on Windows
# ---------------------------------------------------------------------------
Write-Host "[3/4] Starting Next.js frontend on port $FrontendPort..." -ForegroundColor Yellow
$frontend = Start-DemoProcess -FilePath "cmd.exe" `
    -Arguments @("/c", "pnpm exec next dev -p $FrontendPort") `
    -WorkingDirectory $webDir -StdOut $FrontendConsole -StdErr $FrontendConsoleErr

$Pids = @{
    started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    bff_wsl_wrapper_pid = $bff.Id
    frontend_pid = $frontend.Id
    bff_port = $BffPort
    frontend_port = $FrontendPort
}
$Pids | ConvertTo-Json | Set-Content -Path (Join-Path $Runs "launcher_pids.json") -Encoding UTF8

# ---------------------------------------------------------------------------
# 3) wait for readiness
# ---------------------------------------------------------------------------
$healthUrl = "http://127.0.0.1:$BffPort/api/health"
$frontendUrl = "http://127.0.0.1:$FrontendPort/"
$deadline = (Get-Date).AddSeconds($ReadyTimeoutSec)
$lastStatus = ""
$health = $null

try {
    while ((Get-Date) -lt $deadline) {
        if ($bff.HasExited) {
            throw "BFF process exited early (exit=$($bff.ExitCode)). See runs/web_demo/logs/bff.console.err.log"
        }
        try {
            $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 5
            $serverStates = ($health.servers | ForEach-Object { "$($_.profile_id)=$($_.status)" }) -join " "
            $line = "  BFF status=$($health.status) ready=$($health.ready) | $serverStates"
            if ($line -ne $lastStatus) { Write-Host $line; $lastStatus = $line }
            if ($health.ready) { break }
        } catch {
            if ($lastStatus -ne "  BFF not reachable yet (uvicorn starting)") {
                Write-Host "  BFF not reachable yet (uvicorn starting)"; $lastStatus = "  BFF not reachable yet (uvicorn starting)"
            }
        }
        Start-Sleep -Seconds 2
    }
    if (-not $health -or -not $health.ready) {
        throw "BFF did not become ready within $ReadyTimeoutSec s. See runs/web_demo/logs/."
    }

    $frontendDeadline = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $frontendDeadline) {
        if ($frontend.HasExited) {
            throw "Frontend process exited early (exit=$($frontend.ExitCode)). See runs/web_demo/logs/frontend.console.err.log"
        }
        try {
            $resp = Invoke-WebRequest -Uri $frontendUrl -UseBasicParsing -TimeoutSec 5
            if ($resp.StatusCode -ge 200) { break }
        } catch { }
        Start-Sleep -Seconds 1
    }
} catch {
    Write-Host "Launch failed: $($_.Exception.Message)" -ForegroundColor Red
    Stop-ProcessTree $frontend.Id
    Stop-ProcessTree $bff.Id
    # The Windows wsl.exe wrapper may not own the Linux child; ask WSL to stop the recorded PID.
    $bffPidPath = Join-Path $Runs "bff.pid"
    if (Test-Path $bffPidPath) {
        $linuxPid = (Get-Content $bffPidPath -Raw).Trim()
        if ($linuxPid -match '^\d+$') {
            & wsl.exe -d Ubuntu-22.04 -- bash -lc "kill -TERM $linuxPid 2>/dev/null || true"
        }
    }
    throw
}

# ---------------------------------------------------------------------------
# 4) ready
# ---------------------------------------------------------------------------
$serverLines = $health.servers | ForEach-Object {
    $startup = if ($null -ne $_.startup_ms) { "{0:N0} ms" -f $_.startup_ms } else { "-" }
    "    {0,-14} {1,-8} search={2} writes={3} startup={4}" -f $_.profile_id, $_.status, $_.search_mode, $_.writes_enabled, $startup
}
Write-Host ""
Write-Host "[4/4] Demo is ready." -ForegroundColor Green
Write-Host "  Frontend : http://127.0.0.1:$FrontendPort"
Write-Host "  BFF      : http://127.0.0.1:$BffPort/api/health"
Write-Host "  Profiles :"
$serverLines | ForEach-Object { Write-Host $_ }
Write-Host "  Logs     : runs/web_demo/logs/"
Write-Host "  Stop     : powershell -ExecutionPolicy Bypass -File scripts/stop_web_demo.ps1"
Write-Host ""

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$FrontendPort/"
}
