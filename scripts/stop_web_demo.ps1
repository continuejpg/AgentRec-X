# Stop the AgentRec-X Web Demo started by scripts/run_web_demo.ps1.
#   powershell -ExecutionPolicy Bypass -File scripts/stop_web_demo.ps1
$ErrorActionPreference = "Continue"

$Root = Split-Path -Parent $PSScriptRoot
$Runs = Join-Path $Root "runs\web_demo"
$pidsPath = Join-Path $Runs "launcher_pids.json"
$bffPidPath = Join-Path $Runs "bff.pid"

if (Test-Path $pidsPath) {
    $pids = Get-Content $pidsPath -Raw | ConvertFrom-Json
    Write-Host "Stopping frontend process tree (pid $($pids.frontend_pid))..."
    & taskkill.exe /PID $pids.frontend_pid /T /F 2>$null | Out-Null
    Write-Host "Stopping WSL launcher wrapper (pid $($pids.bff_wsl_wrapper_pid))..."
    & taskkill.exe /PID $pids.bff_wsl_wrapper_pid /T /F 2>$null | Out-Null
    Remove-Item $pidsPath -Force -ErrorAction SilentlyContinue
}

# The BFF writes its Linux PID; kill it inside WSL as a fallback (killing the Windows wsl.exe
# wrapper is not guaranteed to stop the Linux child).
if (Test-Path $bffPidPath) {
    $linuxPid = (Get-Content $bffPidPath -Raw).Trim()
    if ($linuxPid -match '^\d+$') {
        Write-Host "Stopping BFF in WSL (pid $linuxPid)..."
        & wsl.exe -d Ubuntu-22.04 -- bash -lc "kill -TERM $linuxPid 2>/dev/null || true; sleep 1; kill -KILL $linuxPid 2>/dev/null || true"
    }
    Remove-Item $bffPidPath -Force -ErrorAction SilentlyContinue
}

# Any orphaned Next.js dev server bound to 3000/8787 is reported, not silently killed.
foreach ($port in 3000, 8787) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        Write-Host "Warning: port $port still has a listener (pid $($conn[0].OwningProcess)); inspect with Get-Process -Id $($conn[0].OwningProcess)."
    }
}
Write-Host "Stop requested. Remove runs/web_demo/launcher_pids.json manually if it becomes stale."
