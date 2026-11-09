# AgentRec-X Web Demo - one-command E2E acceptance run.
#
#   powershell -ExecutionPolicy Bypass -File scripts/run_web_demo_e2e.ps1
#
# What it does:
#   1) preflight: refuse to start if ports 3000/8787 are already used by another process
#      (never kills anything it did not start)
#   2) install e2e/ dependencies on first use with PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
#      (Playwright drives the system Chrome via channel="chrome")
#   3) start the real demo with scripts/run_web_demo.ps1 -NoBrowser (BFF in WSL + Next.js)
#   4) snapshot the real /api/health readiness + startup timings into runs/web_demo/e2e/launch.json
#   5) run the Playwright suite against the real stack
#   6) always stop the demo (scripts/stop_web_demo.ps1) and verify ports were released, then write
#      runs/web_demo/e2e/summary.json + REPORT.md and print the artifact paths.
#
# Options:
#   -BffPort / -FrontendPort / -ReadyTimeoutSec   ports/timeout for the demo launcher
#   -SkipInstall                                  fail instead of running pnpm install
#   -KeepDemoRunning                              do not stop the demo (debugging only)
#
# Exit codes: 0 = all tests passed, 1 = launcher/test failure, 2 = ports busy (nothing started),
#             3 = tests passed but the demo stop left a listener.
[CmdletBinding()]
param(
    [int]$BffPort = 8787,
    [int]$FrontendPort = 3000,
    [int]$ReadyTimeoutSec = 480,
    [switch]$SkipInstall,
    [switch]$KeepDemoRunning
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$E2EDir = Join-Path $Root "e2e"
$Runs = Join-Path $Root "runs\web_demo"
$E2EOut = Join-Path $Runs "e2e"
$Shots = Join-Path $Runs "screenshots"
$Traces = Join-Path $Runs "traces"
$Logs = Join-Path $Runs "logs"
$SummaryPath = Join-Path $E2EOut "summary.json"
$ReportPath = Join-Path $E2EOut "REPORT.md"
$LaunchPath = Join-Path $E2EOut "launch.json"
$CleanupPath = Join-Path $E2EOut "cleanup.json"

function Get-PortListener([int]$Port) {
    return Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
}

function Wait-PortFree([int]$Port, [int]$TimeoutSec) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    do {
        $listener = Get-PortListener -Port $Port
        if (-not $listener) { return $true }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    return $false
}

Write-Host "AgentRec-X Web Demo - E2E acceptance" -ForegroundColor Cyan
Write-Host "  repo      : $Root"
Write-Host "  e2e       : $E2EDir"
Write-Host "  artifacts : $Runs"
Write-Host ""

# ---------------------------------------------------------------------------
# 0) artifact dirs + preflight
# ---------------------------------------------------------------------------
foreach ($dir in @($E2EOut, $Shots, $Traces, $Logs)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
$resolvedRuns = (Resolve-Path -LiteralPath $Runs).Path
if ($resolvedRuns -notlike "$Root*") {
    throw "Refusing to clean artifacts outside the repository: $resolvedRuns"
}

$busy = @()
foreach ($port in @($BffPort, $FrontendPort)) {
    $listener = Get-PortListener -Port $port
    if ($listener) { $busy += ("port {0} (pid {1})" -f $port, $listener.OwningProcess) }
}
if ($busy.Count -gt 0) {
    Write-Host "Refusing to start: $($busy -join ', ') already in use." -ForegroundColor Red
    Write-Host "This acceptance launcher never kills other processes. Stop the existing demo/frontend first:" -ForegroundColor Red
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/stop_web_demo.ps1" -ForegroundColor Red
    exit 2
}

# Fresh acceptance run: remove only artifacts this launcher owns under runs/web_demo/.
$artifactFiles = @($SummaryPath, $ReportPath, $LaunchPath, $CleanupPath,
    (Join-Path $E2EOut "preflight.json"), (Join-Path $E2EOut "evidence.jsonl"))
foreach ($file in $artifactFiles) {
    if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file -Force }
}
$payloadDir = Join-Path $E2EOut "payloads"
if (Test-Path -LiteralPath $payloadDir) { Remove-Item -LiteralPath $payloadDir -Recurse -Force }
Get-ChildItem -LiteralPath $Shots -Filter *.png -File -ErrorAction SilentlyContinue | Remove-Item -Force
if (Test-Path -LiteralPath $Traces) { Get-ChildItem -LiteralPath $Traces -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force }

# ---------------------------------------------------------------------------
# 1) e2e dependencies (system Chrome; never download Playwright browsers)
# ---------------------------------------------------------------------------
$playwrightPkg = Join-Path $E2EDir "node_modules\@playwright\test\package.json"
if (-not (Test-Path -LiteralPath $playwrightPkg)) {
    if ($SkipInstall) {
        throw "e2e dependencies are missing and -SkipInstall was given. Run: pnpm --dir e2e install"
    }
    Write-Host "[1/6] Installing E2E dependencies (@playwright/test; browser download skipped)..." -ForegroundColor Yellow
    $env:PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD = "1"
    & cmd.exe /c "cd /d `"$E2EDir`" && pnpm install"
    if ($LASTEXITCODE -ne 0) { throw "pnpm install failed in e2e/ (see the output above)" }
} else {
    Write-Host "[1/6] E2E dependencies already installed."
}

$chrome = $null
foreach ($candidate in @(
        "C:\Program Files\Google\Chrome\Application\chrome.exe",
        "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe")) {
    if (Test-Path -LiteralPath $candidate) { $chrome = $candidate; break }
}
if (-not $chrome) {
    throw "System Google Chrome was not found; Playwright channel='chrome' requires it (no browser download is used)."
}
$chromeVersion = (Get-Item -LiteralPath $chrome).VersionInfo.ProductVersion

# ---------------------------------------------------------------------------
# 2) start the real demo, 3) snapshot readiness
# ---------------------------------------------------------------------------
$testsExit = 1
$launcherFailure = $null
$demoStarted = $false
$testsStartedAt = $null
$testsFinishedAt = $null
$cleanup = $null
$startedAt = Get-Date
$readyAt = $null

try {
    Write-Host "[2/6] Starting the real demo (scripts/run_web_demo.ps1 -NoBrowser; BFF cold start ~35-48 s)..." -ForegroundColor Yellow
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\run_web_demo.ps1") `
        -NoBrowser -BffPort $BffPort -FrontendPort $FrontendPort -ReadyTimeoutSec $ReadyTimeoutSec
    if ($LASTEXITCODE -ne 0) {
        throw "scripts/run_web_demo.ps1 failed (exit $LASTEXITCODE); see runs/web_demo/logs/"
    }
    $demoStarted = $true
    $readyAt = Get-Date

    Write-Host "[3/6] Snapshotting real /api/health readiness..." -ForegroundColor Yellow
    $healthUrl = "http://127.0.0.1:$BffPort/api/health"
    $health = $null
    $healthDeadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $healthDeadline) {
        try {
            $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 10
            if ($health.ready) { break }
        } catch { }
        Start-Sleep -Seconds 1
    }
    if (-not $health -or -not $health.ready) {
        throw "BFF /api/health did not report ready=true after the launcher reported ready."
    }

    $startupMs = [int]((($readyAt - $startedAt).TotalSeconds) * 1000)
    $serverInfo = @()
    foreach ($server in @($health.servers)) {
        $toolCount = 0
        if ($server.tools) { $toolCount = @($server.tools).Count }
        $serverInfo += [ordered]@{
            profile_id     = $server.profile_id
            status         = $server.status
            search_mode    = $server.search_mode
            writes_enabled = $server.writes_enabled
            startup_ms     = $server.startup_ms
            tools_count    = $toolCount
            error          = $server.error
        }
    }
    $playwrightVersion = "unknown"
    try {
        $playwrightPkgJson = Get-Content -LiteralPath $playwrightPkg -Raw | ConvertFrom-Json
        $playwrightVersion = $playwrightPkgJson.version
    } catch { }
    $nodeVersion = (cmd.exe /c "node --version") -replace "\s+$", ""
    $launch = [ordered]@{
        schema                = "agentrecx.web_demo.e2e.launch.v1"
        started_at            = $startedAt.ToUniversalTime().ToString("o")
        ready_at              = $readyAt.ToUniversalTime().ToString("o")
        startup_ms            = $startupMs
        bff_port              = $BffPort
        frontend_port         = $FrontendPort
        catalogue_size        = $health.catalogue_size
        memory_writes_enabled = $health.memory_writes_enabled
        servers               = $serverInfo
        playwright_version    = $playwrightVersion
        chrome_version        = $chromeVersion
        chrome_path           = $chrome
        node_version          = $nodeVersion
    }
    $launch | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $LaunchPath -Encoding UTF8
    Write-Host ("  real readiness: ready={0} catalogue={1} startup={2:N1} s" -f $health.ready, $health.catalogue_size, ($startupMs / 1000.0)) -ForegroundColor Green
    foreach ($server in $serverInfo) {
        Write-Host ("  {0}: status={1} search_mode={2} writes={3} tools={4} startup_ms={5}" -f `
                $server.profile_id, $server.status, $server.search_mode, $server.writes_enabled, $server.tools_count, $server.startup_ms)
    }

    # -----------------------------------------------------------------------
    # 4) Playwright against the real stack
    # -----------------------------------------------------------------------
    $env:E2E_BASE_URL = "http://127.0.0.1:$FrontendPort"
    $env:E2E_BFF_URL = "http://127.0.0.1:$BffPort"
    $env:E2E_SUMMARY_PATH = $SummaryPath
    $env:E2E_ARTIFACT_ROOT = $Runs
    $env:PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD = "1"
    Write-Host "[4/6] Running Playwright E2E (real stack, no mocks)..." -ForegroundColor Yellow
    $testsStartedAt = Get-Date
    & cmd.exe /c "cd /d `"$E2EDir`" && pnpm exec playwright test"
    $testsExit = $LASTEXITCODE
    $testsFinishedAt = Get-Date
    Write-Host "  Playwright exit code: $testsExit" -ForegroundColor $(if ($testsExit -eq 0) { "Green" } else { "Red" })
} catch {
    $launcherFailure = $_
    Write-Host "E2E acceptance failed before/during the run: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    # -----------------------------------------------------------------------
    # 5) always stop what we started, verify no orphan listeners remain
    # -----------------------------------------------------------------------
    $remaining = @()
    $stopped = $true
    if ($KeepDemoRunning -and $demoStarted) {
        Write-Host "[5/6] -KeepDemoRunning: leaving the demo running." -ForegroundColor Yellow
        $stopped = $false
        foreach ($port in @($BffPort, $FrontendPort)) {
            $listener = Get-PortListener -Port $port
            if ($listener) { $remaining += ("{0}(pid {1})" -f $port, $listener.OwningProcess) }
        }
    } else {
        Write-Host "[5/6] Stopping the demo (scripts/stop_web_demo.ps1)..." -ForegroundColor Yellow
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\stop_web_demo.ps1") | Out-Host
        $bffFree = Wait-PortFree -Port $BffPort -TimeoutSec 30
        $frontendFree = Wait-PortFree -Port $FrontendPort -TimeoutSec 45
        if (-not ($bffFree -and $frontendFree)) {
            Write-Host "  Ports still busy; retrying stop once..." -ForegroundColor Yellow
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\stop_web_demo.ps1") | Out-Host
            $bffFree = Wait-PortFree -Port $BffPort -TimeoutSec 15
            $frontendFree = Wait-PortFree -Port $FrontendPort -TimeoutSec 20
        }
        foreach ($port in @($BffPort, $FrontendPort)) {
            $listener = Get-PortListener -Port $port
            if ($listener) { $remaining += ("{0}(pid {1})" -f $port, $listener.OwningProcess) }
        }
        $stopped = ($remaining.Count -eq 0)
        if ($stopped) {
            Write-Host "  Demo stopped; ports $BffPort and $FrontendPort released." -ForegroundColor Green
        } else {
            Write-Host "  Warning: still listening after stop: $($remaining -join ', '). No unrelated process was killed." -ForegroundColor Red
        }
    }
    $testsStartedIso = $null
    $testsFinishedIso = $null
    if ($testsStartedAt) { $testsStartedIso = $testsStartedAt.ToUniversalTime().ToString("o") }
    if ($testsFinishedAt) { $testsFinishedIso = $testsFinishedAt.ToUniversalTime().ToString("o") }
    $cleanup = [ordered]@{
        schema             = "agentrecx.web_demo.e2e.cleanup.v1"
        stopped            = $stopped
        remaining_ports    = $remaining
        keep_demo_running  = [bool]$KeepDemoRunning
        stopped_at         = (Get-Date).ToUniversalTime().ToString("o")
        tests_started_at   = $testsStartedIso
        tests_finished_at  = $testsFinishedIso
        tests_exit_code    = $testsExit
    }
    $cleanup | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $CleanupPath -Encoding UTF8
}

# ---------------------------------------------------------------------------
# 6) report + console summary
# ---------------------------------------------------------------------------
Write-Host "[6/6] Generating runs/web_demo/e2e/summary.json + REPORT.md..." -ForegroundColor Yellow
& cmd.exe /c "cd /d `"$Root`" && node e2e\generate-report.mjs"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Warning: report generator returned exit code $LASTEXITCODE; run 'node e2e/generate-report.mjs' manually." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== E2E acceptance summary ===" -ForegroundColor Cyan
if (Test-Path -LiteralPath $SummaryPath) {
    try {
        $summary = Get-Content -LiteralPath $SummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $verdict = "UNKNOWN"
        if ($summary.verdict) { $verdict = $summary.verdict.result }
        Write-Host ("  verdict : {0}" -f $verdict)
        Write-Host ("  tests   : {0} passed / {1} failed / {2} skipped (of {3}); playwright {4:N1} s" -f `
                $summary.totals.passed, $summary.totals.failed, $summary.totals.skipped, $summary.totals.expected_tests, ($summary.totals.duration_ms / 1000.0))
        if ($summary.launch) {
            Write-Host ("  startup : {0:N1} s (real /api/health ready)" -f ($summary.launch.startup_ms / 1000.0))
        }
        if ($summary.artifacts) {
            Write-Host ("  shots   : {0}" -f (($summary.artifacts.screenshots | ForEach-Object { $_.name }) -join ", "))
            $traceCount = 0
            if ($summary.artifacts.traces) { $traceCount = @($summary.artifacts.traces).Count }
            Write-Host ("  traces  : {0} file(s) (retain-on-failure)" -f $traceCount)
        }
    } catch {
        Write-Host "  Could not parse summary.json: $($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  summary.json was not produced." -ForegroundColor Red
}
Write-Host "  screenshots : $Shots"
Write-Host "  traces      : $Traces"
Write-Host "  summary     : $SummaryPath"
Write-Host "  report      : $ReportPath"
Write-Host "  launch      : $LaunchPath"
Write-Host "  cleanup     : $CleanupPath"
Write-Host "  logs        : $Logs"
Write-Host ""

if ($launcherFailure) {
    Write-Host "FAILED: $($launcherFailure.Exception.Message)" -ForegroundColor Red
    exit 1
}
if (-not $KeepDemoRunning -and $cleanup -ne $null -and $cleanup.stopped -ne $true) {
    Write-Host "FAILED: demo stop left listeners: $($cleanup.remaining_ports -join ', ')" -ForegroundColor Red
    exit 3
}
if ($testsExit -ne 0) {
    Write-Host "FAILED: playwright exit code $testsExit (see $ReportPath)" -ForegroundColor Red
    exit $testsExit
}
Write-Host "PASSED: all E2E acceptance checks passed." -ForegroundColor Green
exit 0
