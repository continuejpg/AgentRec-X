<#
.SYNOPSIS
    Windows-side discovery and verification for the AgentRec-X M11.5 demo.

.DESCRIPTION
    Invoked by start-agentrecx.cmd. This script ONLY resolves and verifies:

      1. enumerates the distributions WSL actually has registered
         (wsl.exe --list --quiet) - a -d value is never guessed;
      2. derives a candidate distro and Linux path from the launcher's own
         location (passed in by the shim), never from the current directory;
      3. asks the candidate distro to verify that <repo>/scripts/start_demo.sh
         exists and is executable;
      4. falls back to verifying the same path in the other installed distros;
      5. writes the verified (distro, repo) pair to an output file, or a
         diagnostic block, and always exits with a clear status.

    It does NOT launch anything. The .cmd shim performs the final
    `wsl.exe -d <distro> -- bash -lc ...` call, because batch forwards a quoted
    command string to wsl.exe more predictably than PowerShell 5.1 marshals
    native arguments.

    Windows PowerShell 5.1 compatibility (deliberate):
      * .NET ProcessStartInfo.ArgumentList does NOT exist on .NET Framework, so
        discovery uses Start-Process -ArgumentList (the 5.1-safe form);
      * no ternary, no null-coalescing, no PS7-only operators;
      * wsl.exe output is forced to UTF-8 via WSL_UTF8 to avoid UTF-16LE mangling.

    Trust boundaries: no background process, no PID file, no kill, no install, no
    environment/artifact/preflight/application logic (all of that lives in
    scripts/start_demo.sh).
#>

[CmdletBinding()]
param(
    # The wrapper's own directory in Windows form (from %~dp0), either
    # \\wsl.localhost\<Distro>\<linux path> or \\wsl$\<Distro>\<linux path>.
    [string]$WindowsPath = '',

    # Linux path corresponding to $WindowsPath, derived lexically by the shim.
    [string]$LinuxFolder = '',

    # Candidate distro (from the UNC share name or AGENTRECX_DISTRO). Verified.
    [string]$Distro = '',

    # Where to write the response. The .cmd shim reads this file; it is used
    # instead of stdout so the shim never has to parse command output.
    [string]$OutFile = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$env:WSL_UTF8 = '1'
$WrapperVersion = 'M11.5-windows-3'

if (-not $OutFile) {
    Write-Host 'ERROR: -OutFile is required.' -ForegroundColor Red
    exit 8
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

function Get-WslInstalledDistro {
    <#
      Return the distro names WSL actually has registered.

      Start-Process -ArgumentList is used (not ProcessStartInfo.ArgumentList)
      because Windows PowerShell 5.1 runs on .NET Framework, where ArgumentList
      does not exist.
    #>
    [CmdletBinding()]
    param()

    $exe = Join-Path $env:SystemRoot 'System32\wsl.exe'
    if (-not (Test-Path -LiteralPath $exe)) { $exe = 'wsl.exe' }

    $stdoutFile = [System.IO.Path]::GetTempFileName()
    $stderrFile = [System.IO.Path]::GetTempFileName()
    try {
        # Splatting (no backtick line-continuations) keeps this readable and avoids
        # the fragile trailing-whitespace pitfalls of ` continuation.
        $startArguments = @{
            FilePath               = $exe
            ArgumentList           = @('--list', '--quiet')
            NoNewWindow            = $true
            Wait                   = $true
            PassThru               = $true
            RedirectStandardOutput = $stdoutFile
            RedirectStandardError  = $stderrFile
        }
        $process = Start-Process @startArguments

        $stdout = ''
        $stderr = ''
        try { $stdout = [System.IO.File]::ReadAllText($stdoutFile) } catch { }
        try { $stderr = [System.IO.File]::ReadAllText($stderrFile) } catch { }

        if ($process.ExitCode -ne 0) {
            throw ("wsl.exe --list --quiet failed (exit {0}). {1}" -f `
                $process.ExitCode, $stderr.Trim())
        }

        $names = @()
        foreach ($line in ($stdout -split "`r?`n")) {
            # Defensive: strip NUL/BOM artefacts, then surrounding whitespace.
            $name = ($line -replace "`0", '').Trim([char]0xFEFF).Trim()
            if ($name -and $name -ne '*') { $names += $name }
        }
        return , $names
    }
    finally {
        Remove-Item -LiteralPath $stdoutFile, $stderrFile -ErrorAction SilentlyContinue
    }
}

function Test-RepoRoot {
    <#
      Ask one distro whether $LinuxPath is the AgentRec-X repository root with an
      executable scripts/start_demo.sh. Returns $false when the distro does not
      exist or the path is wrong, which is how a bad candidate is rejected.

      The single-quoted path keeps it opaque to the login shell: no variable
      interpolation can occur, so a verified path cannot be corrupted here.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$DistroName,
        [Parameter(Mandatory = $true)][string]$LinuxPath
    )

    $repo = $LinuxPath.TrimEnd('/')
    if (-not $repo) { return $false }

    $command = "set -e; cd '$repo'; test -f recommendation/config.py; test -f AGENTS.md; test -x scripts/start_demo.sh"

    $exe = Join-Path $env:SystemRoot 'System32\wsl.exe'
    if (-not (Test-Path -LiteralPath $exe)) { $exe = 'wsl.exe' }

    try {
        $verifyArguments = @{
            FilePath     = $exe
            ArgumentList = @('-d', $DistroName, '--', 'bash', '-c', $command)
            NoNewWindow  = $true
            Wait         = $true
            PassThru     = $true
        }
        $process = Start-Process @verifyArguments
        return ($process.ExitCode -eq 0)
    }
    catch {
        return $false
    }
}


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #

function Write-Response {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Status,
        [string]$Message = '',
        [string]$VerifiedDistro = '',
        [string]$VerifiedRepo = '',
        [string[]]$Installed = @(),
        [string]$Candidate = ''
    )

    $lines = @(
        "status=$Status",
        "version=$WrapperVersion",
        "windowspath=$WindowsPath",
        "linuxfolder=$LinuxFolder",
        "candidate=$Candidate",
        "distro=$VerifiedDistro",
        "repo=$VerifiedRepo",
        "installed=$($Installed -join ',')",
        "message=$Message"
    )
    [System.IO.File]::WriteAllLines($OutFile, $lines)
    Write-Host ''
    Write-Host "start-agentrecx: $Status - $Message"
    if ($VerifiedDistro) {
        Write-Host "  distro     : $VerifiedDistro"
        Write-Host "  repository : $VerifiedRepo"
        Write-Host "  launcher   : $VerifiedRepo/scripts/start_demo.sh"
    }
    if ($Status -ne 'ok') {
        if ($Installed.Count -gt 0) {
            Write-Host '  installed WSL distributions:'
            foreach ($name in $Installed) { Write-Host "    - $name" }
        }
        else {
            Write-Host '  installed WSL distributions: (none reported)'
        }
    }
}

try {
    Write-Host '======================================================================'
    Write-Host ' AgentRec-X - Windows launcher: resolving WSL distro and repository'
    Write-Host '======================================================================'
    Write-Host "  launcher    : $WrapperVersion"
    Write-Host "  windows path: $WindowsPath"
    Write-Host "  linux path  : $LinuxFolder"
    Write-Host '  current dir : (never used for resolution)'
    Write-Host ''
    Write-Host '  [1/2] wsl.exe --list --quiet'

    $installed = Get-WslInstalledDistro

    $candidate = $Distro
    if (-not $candidate -and $env:AGENTRECX_DISTRO) { $candidate = $env:AGENTRECX_DISTRO }

    $linuxPath = $LinuxFolder
    if ($env:AGENTRECX_REPO) { $linuxPath = $env:AGENTRECX_REPO }

    # A checkout can also live on a Windows drive, which WSL mounts at
    # /mnt/<drive>. A drive path has no UNC share to parse, so the batch shim
    # hands us the raw Windows path and the translation happens here, where the
    # drive letter can be lower-cased (/mnt/D does not exist; the mount point is
    # lower case). This is only a candidate: Test-RepoRoot below still has to
    # prove the path inside a distro, so a wrong guess fails verification just
    # like an unverifiable path always did and can never launch the wrong tree.
    if ($linuxPath -match '^([A-Za-z]):[\\/](.*)$') {
        $linuxPath = '/mnt/' + $Matches[1].ToLower() + '/' + ($Matches[2] -replace '\\', '/')
    }

    if (-not $linuxPath) {
        Write-Response -Status 'error' -Installed $installed -Candidate $candidate `
            -Message 'no Linux path could be derived from the launcher location'
        exit 1
    }

    Write-Host '  [2/2] verifying the repository inside WSL'

    if ($installed.Count -eq 0) {
        Write-Response -Status 'error' -Installed $installed -Candidate $candidate `
            -Message 'WSL reported no installed distributions; install one with: wsl --install -d Ubuntu-22.04'
        exit 3
    }

    # Candidate order: the preferred distro first (only if it is really
    # installed), then every other installed distro.
    $order = @()
    if ($candidate -and ($installed -contains $candidate)) { $order += $candidate }
    foreach ($name in $installed) {
        if ($order -notcontains $name) { $order += $name }
    }

    $foundDistro = ''
    foreach ($name in $order) {
        Write-Host "        trying $name ..."
        if (Test-RepoRoot -DistroName $name -LinuxPath $linuxPath) {
            $foundDistro = $name
            break
        }
    }

    if (-not $foundDistro) {
        Write-Response -Status 'error' -Installed $installed -Candidate $candidate `
            -Message "the AgentRec-X repository was not found at '$linuxPath' in any installed WSL distro"
        exit 2
    }

    $note = ''
    if ($candidate -and $foundDistro -ne $candidate) {
        $note = "the UNC share name '$candidate' is not a registered distro; '$foundDistro' was verified instead"
    }
    elseif (-not $candidate) {
        $note = "'$foundDistro' was verified by probing the path"
    }
    else {
        $note = "'$foundDistro' verified"
    }

    Write-Response -Status 'ok' -Message $note -Installed $installed `
        -Candidate $candidate -VerifiedDistro $foundDistro -VerifiedRepo $linuxPath.TrimEnd('/')
    exit 0
}
catch {
    Write-Host "  resolution error: $($_.Exception.Message)" -ForegroundColor Red
    $lines = @(
        'status=error',
        "version=$WrapperVersion",
        "windowspath=$WindowsPath",
        "linuxfolder=$LinuxFolder",
        "candidate=$Distro",
        'distro=',
        'repo=',
        'installed=',
        "message=$($_.Exception.Message)"
    )
    try { [System.IO.File]::WriteAllLines($OutFile, $lines) } catch { }
    exit 4
}
