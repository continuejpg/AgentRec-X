@echo off
REM ===========================================================================
REM  AgentRec-X - one-click Windows launcher (M11.5) - double-click shim ONLY.
REM
REM  Flow:
REM
REM      start-agentrecx.cmd
REM        [1] derive two strings from %~dp0 (never %CD%)
REM        [2] powershell -File scripts\windows\start_agentrecx.ps1
REM              -> wsl.exe --list --quiet        (real registered distros)
REM              -> verify <repo>/scripts/start_demo.sh inside that distro
REM            (the helper writes its verified result to a temp file)
REM        [3] wsl.exe -d <verified> -- bash -lc "cd <repo> && exec .../start_demo.sh"
REM
REM  Why this shim is small: the first version parsed UNC paths AND called
REM  wsl.exe inside for /f in batch. That failed on real Windows with
REM  "UNC paths are not supported" and a garbled repository path. All discovery
REM  now lives in PowerShell. The final launch stays here because batch forwards
REM  one quoted command string to wsl.exe more predictably than Windows
REM  PowerShell 5.1 marshals native arguments.
REM
REM  This shim never:
REM    * uses the current directory (a UNC CWD is illegal in cmd.exe);
REM    * invokes wsl.exe -d with an unverified name;
REM    * prints a repository path that has not been verified;
REM    * starts a background service, writes a PID file or kills anything.
REM
REM  Optional overrides (environment variables):
REM    AGENTRECX_DISTRO   force a distro (still verified against wsl --list)
REM    AGENTRECX_REPO     force the Linux repository path (still verified)
REM    AGENTRECX_PORT     demo port (default: the Linux launcher's own default)
REM    AGENTRECX_NO_PAUSE set to 1 to skip the final pause
REM  Flags: --self-test (resolve + verify, start nothing)   --no-browser
REM ===========================================================================

setlocal EnableExtensions

REM Keep wsl.exe output clean (UTF-8 instead of UTF-16LE).
set "WSL_UTF8=1"

REM --- [1] derive the two paths from this file's own location ---------------
REM A \\wsl.localhost\ UNC path is <prefix>\<Distro>\<linux path>. Dropping only
REM the prefix would leave the distro component glued to the front and shift the
REM whole path up one level, so BOTH the prefix and the share component are
REM dropped here.
set "AGENTRECX_WIN=%~dp0"
if "%AGENTRECX_WIN:~-1%"=="\" set "AGENTRECX_WIN=%AGENTRECX_WIN:~0,-1%"

set "AGENTRECX_REST="
if /i "%AGENTRECX_WIN:~0,16%"=="\\wsl.localhost\" set "AGENTRECX_REST=%AGENTRECX_WIN:~16%"
if /i "%AGENTRECX_WIN:~0,7%"=="\\wsl$\" set "AGENTRECX_REST=%AGENTRECX_WIN:~7%"

REM "tokens=1,* delims=\" splits <Distro>\<linux path> into the share component
REM and the remainder. A for /f over a plain string (no backquoted command) is
REM safe: nothing is executed and nothing can garble the value.
set "AGENTRECX_GUESS="
set "AGENTRECX_AFTER="
if defined AGENTRECX_REST for /f "tokens=1,* delims=\" %%d in ("%AGENTRECX_REST%") do (
    set "AGENTRECX_GUESS=%%d"
    set "AGENTRECX_AFTER=%%e"
)

if not "%AGENTRECX_AFTER%"=="" (
    set "AGENTRECX_LINUX=/%AGENTRECX_AFTER%"
) else (
    REM Not a WSL UNC path (for example a copied launcher). The helper will need
    REM either AGENTRECX_REPO or a location it can verify.
    set "AGENTRECX_LINUX=%AGENTRECX_WIN%"
)
set "AGENTRECX_LINUX=%AGENTRECX_LINUX:\=/%"

REM AGENTRECX_DISTRO, when set, overrides the share-derived candidate.
if not "%AGENTRECX_DISTRO%"=="" set "AGENTRECX_GUESS=%AGENTRECX_DISTRO%"

set "AGENTRECX_PS1=%~dp0scripts\windows\start_agentrecx.ps1"
if not exist "%AGENTRECX_PS1%" (
    echo.
    echo  ERROR: PowerShell helper not found:
    echo    %AGENTRECX_PS1%
    echo.
    echo  This launcher expects the full AgentRec-X repository. Run it from inside
    echo  the repository over the WSL share, for example:
    echo    \\wsl.localhost\Ubuntu-22.04\home\^<user^>\AgentRec-X\
    echo.
    set "AGENTRECX_EXIT=1"
    goto :done
)

REM --- [2] let PowerShell resolve and verify (writes a response file) -------
set "AGENTRECX_OUT=%TEMP%\agentrecx-launch-%RANDOM%%RANDOM%.txt"
if exist "%AGENTRECX_OUT%" del "%AGENTRECX_OUT%" >nul 2>&1

set "AGENTRECX_PSARGS=-NoProfile -ExecutionPolicy Bypass -File "%AGENTRECX_PS1%" -OutFile "%AGENTRECX_OUT%" -WindowsPath "%AGENTRECX_WIN%" -LinuxFolder "%AGENTRECX_LINUX%""
if not "%AGENTRECX_GUESS%"=="" set "AGENTRECX_PSARGS=%AGENTRECX_PSARGS% -Distro "%AGENTRECX_GUESS%""

powershell.exe %AGENTRECX_PSARGS%

if not exist "%AGENTRECX_OUT%" (
    echo.
    echo  ERROR: the Windows helper produced no result file.
    echo         Expected: %AGENTRECX_OUT%
    echo.
    set "AGENTRECX_EXIT=1"
    goto :done
)

set "AGENTRECX_STATUS="
set "AGENTRECX_DISTRO_OK="
set "AGENTRECX_REPO_OK="
set "AGENTRECX_MESSAGE="
for /f "usebackq tokens=1* delims==" %%a in ("%AGENTRECX_OUT%") do (
    if /i "%%a"=="status"  set "AGENTRECX_STATUS=%%b"
    if /i "%%a"=="distro"  set "AGENTRECX_DISTRO_OK=%%b"
    if /i "%%a"=="repo"    set "AGENTRECX_REPO_OK=%%b"
    if /i "%%a"=="message" set "AGENTRECX_MESSAGE=%%b"
)

if not "%AGENTRECX_STATUS%"=="ok" (
    echo.
    echo  ERROR: could not start AgentRec-X.
    echo    %AGENTRECX_MESSAGE%
    echo.
    echo  Set the values explicitly and try again:
    echo    set AGENTRECX_REPO=/home/^<user^>/AgentRec-X
    echo    set AGENTRECX_DISTRO=^<distro^>
    echo.
    set "AGENTRECX_EXIT=1"
    goto :done
)

REM Defensive: never launch with an empty or unverified value.
if "%AGENTRECX_DISTRO_OK%"=="" (
    echo  ERROR: helper reported success without a verified distro.
    set "AGENTRECX_EXIT=1"
    goto :done
)
if "%AGENTRECX_REPO_OK%"=="" (
    echo  ERROR: helper reported success without a verified repository path.
    set "AGENTRECX_EXIT=1"
    goto :done
)

REM --- [3] self-test, or launch ---------------------------------------------
REM The launch command is written INLINE on the wsl.exe line, never stored in a
REM variable: cmd would otherwise treat the "&&" as a command separator while
REM parsing the `set` line. Writing it inline removes that whole class of
REM quoting/escaping ambiguity. The port suffix is appended only when set.
set "AGENTRECX_PORTARG="
if not "%AGENTRECX_PORT%"=="" set "AGENTRECX_PORTARG=--port %AGENTRECX_PORT%"

echo.
echo ======================================================================
echo  AgentRec-X - starting the M11 demo
echo ======================================================================
echo  distro     : %AGENTRECX_DISTRO_OK%   (verified against wsl.exe --list)
echo  repository : %AGENTRECX_REPO_OK%
echo  launcher   : scripts/start_demo.sh   (verified present + executable)
echo.

if /i "%~1"=="--self-test" (
    echo  SELF-TEST ONLY - nothing was started.
    echo  A real run would execute:
    echo    wsl.exe -d %AGENTRECX_DISTRO_OK% -- bash -lc "cd '%AGENTRECX_REPO_OK%' && exec ./scripts/start_demo.sh %AGENTRECX_PORTARG%"
    echo.
    set "AGENTRECX_EXIT=0"
    goto :done
)

echo  The demo runs in this console. Press Ctrl+C to stop it.
echo.

if /i "%~1"=="--no-browser" goto :launch
if not exist "%~dp0scripts\windows\open_demo_browser.ps1" goto :launch

set "AGENTRECX_BROWSER_PORT=8000"
if not "%AGENTRECX_PORT%"=="" set "AGENTRECX_BROWSER_PORT=%AGENTRECX_PORT%"
start "AgentRec-X browser" /min cmd /c powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0scripts\windows\open_demo_browser.ps1" -Port %AGENTRECX_BROWSER_PORT%

:launch
echo  command    : wsl.exe -d %AGENTRECX_DISTRO_OK% -- bash -lc "cd '%AGENTRECX_REPO_OK%' && exec ./scripts/start_demo.sh %AGENTRECX_PORTARG%"
echo.
wsl.exe -d %AGENTRECX_DISTRO_OK% -- bash -lc "cd '%AGENTRECX_REPO_OK%' && exec ./scripts/start_demo.sh %AGENTRECX_PORTARG%"
set "AGENTRECX_EXIT=%ERRORLEVEL%"
echo.
echo  AgentRec-X demo stopped (exit code %AGENTRECX_EXIT%).
goto :done

:done
if "%AGENTRECX_EXIT%"=="" set "AGENTRECX_EXIT=1"
if exist "%AGENTRECX_OUT%" del "%AGENTRECX_OUT%" >nul 2>&1
echo.
if not "%AGENTRECX_NO_PAUSE%"=="1" pause
endlocal & exit /b %AGENTRECX_EXIT%
