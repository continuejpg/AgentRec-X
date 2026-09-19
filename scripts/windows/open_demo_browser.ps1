# AgentRec-X - detached browser watcher for the one-click Windows launcher.
#
# Started by start-agentrecx.cmd as a background convenience helper. It waits for
# the AgentRec-X demo to answer on localhost and then opens the Windows default
# browser at the demo page.
#
# Trust boundaries - this script:
#   * is CONVENIENCE ONLY. It never starts, stops, signals or supervises the demo,
#     and its success or failure has no bearing on whether the service started;
#   * only ever talks to 127.0.0.1;
#   * gives up silently after the timeout instead of retrying forever.
#
# It is not invoked by scripts/start_demo.sh (the authoritative Linux launcher);
# it is a Windows-side presentation concern only.

[CmdletBinding()]
param(
    [int]$Port = 8000,
    # Named BindHost, not Host: $Host is a PowerShell automatic variable.
    [string]$BindHost = "127.0.0.1",
    [int]$TimeoutSeconds = 180,
    # Open the API docs page instead of the demo page.
    [switch]$Docs,
    # Poll only, do not launch a browser (used for diagnosis).
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"

$path = if ($Docs) { "/docs" } else { "/demo/" }
$url = "http://{0}:{1}{2}" -f $BindHost, $Port, $path
$healthUrl = "http://{0}:{1}/health" -f $BindHost, $Port

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$ready = $false

while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 3
        if ($response.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        # The server is almost certainly not listening yet; keep waiting.
    }
    Start-Sleep -Milliseconds 750
}

if (-not $ready) {
    # Silent by design: the console running the demo is the source of truth.
    exit 1
}

if ($NoLaunch) {
    Write-Output "ready: $healthUrl"
    exit 0
}

try {
    Start-Process $url
} catch {
    # A browser that cannot be opened must not look like a demo failure.
    exit 1
}

exit 0
