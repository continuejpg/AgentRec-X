#!/usr/bin/env bash
#
# AgentRec-X local demo launcher (Milestone 11.5).
#
#   ./scripts/start_demo.sh [--host HOST] [--port PORT] [--device DEVICE]
#                           [--verify] [--doctor] [--json] [--help]
#
# Starts the accepted M11 multi-turn web demo in the FOREGROUND using the existing
# FastAPI/Uvicorn entry point (`recommendation.api.app`).
#
# Trust boundaries -- this script:
#
#   * performs NO installation and mutates NO environment.  It never runs pip.  If
#     the virtualenv is missing it stops and tells you to run ./scripts/setup_demo.sh;
#   * never starts a background process, never writes a PID file, never rotates logs;
#   * never kills, signals or stops any process.  If the port is busy it classifies
#     the listener and refuses to start -- it does not reclaim the port;
#   * works from any current working directory (the repository root is derived from
#     this script's own location).
#
# Stop the demo with Ctrl+C.  Fresh environment preparation is a separate command:
# ./scripts/setup_demo.sh
#
set -Eeuo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

# Make the repository importable no matter which directory this script is invoked
# from. Without this, `python -m recommendation.api.app` would only work when the
# CWD happened to be the repository root, defeating location independence.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

HOST="127.0.0.1"
PORT="8000"
DEVICE="cpu"
VERIFY=0
DOCTOR_ONLY=0
JSON=0

# Readable preflight for a missing interpreter: do not fall back to the system
# python, because the accepted dependency closure lives in the virtualenv.
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "ERROR: virtualenv interpreter not found: ${VENV_PYTHON}" >&2
    echo "       Run ./scripts/setup_demo.sh first (setup and start are separate)." >&2
    exit 1
fi

usage() {
    cat <<'EOF'
AgentRec-X local demo launcher

Usage:
  ./scripts/start_demo.sh [options]

Options:
  --host HOST     bind host (default 127.0.0.1)
  --port PORT     bind port (default 8000)
  --device DEVICE inference device (default cpu; CUDA is never required)
  --verify        full SHA-256 artifact verification during preflight
  --doctor        run the environment/artifact doctor and exit (no server)
  --json          machine-readable doctor/preflight output
  -h, --help      show this help

The demo runs in the foreground. Press Ctrl+C to stop it.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)   HOST="${2:?--host needs a value}"; shift 2 ;;
        --port)   PORT="${2:?--port needs a value}"; shift 2 ;;
        --device) DEVICE="${2:?--device needs a value}"; shift 2 ;;
        --verify) VERIFY=1; shift ;;
        --doctor) DOCTOR_ONLY=1; shift ;;
        --json)   JSON=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "ERROR: --port must be an integer in 1..65535, got '${PORT}'" >&2
    exit 2
fi

# libgomp rejects an empty or non-positive OMP_NUM_THREADS on every torch import,
# which would otherwise kill the server with a confusing message.  The sandbox has
# been observed exporting 0; normalise it here (pytest does the same in conftest.py).
NORMALISED_OMP="$(
    "${VENV_PYTHON}" -c \
        'import os, sys; sys.path.insert(0, sys.argv[1]); from recommendation.local_demo import normalise_omp_num_threads as n; print(n(os.environ.get("OMP_NUM_THREADS")) or "")' \
        "${REPO_ROOT}" 2>/dev/null || true
)"
if [[ -n "${NORMALISED_OMP}" ]]; then
    export OMP_NUM_THREADS="${NORMALISED_OMP}"
fi

run_support() {
    "${VENV_PYTHON}" -m recommendation.local_demo "$@"
}

if (( DOCTOR_ONLY == 1 )); then
    echo "======================================================================"
    echo " AgentRec-X local demo doctor"
    echo "======================================================================"
    echo "Repository root : ${REPO_ROOT}"
    echo "Current dir     : $(pwd -P)"
    echo
    ARGS=(doctor)
    (( VERIFY == 1 )) && ARGS+=(--verify)
    (( JSON == 1 )) && ARGS+=(--json)
    run_support "${ARGS[@]}"
    exit $?
fi

echo "======================================================================"
echo " AgentRec-X local demo"
echo "======================================================================"
echo "[1/4] Repository"
echo "      repo root : ${REPO_ROOT}"
echo "      cwd       : $(pwd -P)"
echo "      python    : ${VENV_PYTHON}"

echo
echo "[2/4] Preflight (read-only: no installation, no environment mutation)"
PREFLIGHT_ARGS=(preflight --host "${HOST}" --port "${PORT}")
(( VERIFY == 1 )) && PREFLIGHT_ARGS+=(--verify)
(( JSON == 1 )) && PREFLIGHT_ARGS+=(--json)
if ! run_support "${PREFLIGHT_ARGS[@]}"; then
    echo >&2
    echo "ERROR: preflight failed; refusing to start." >&2
    echo "       Fix the reported problems, or run ./scripts/setup_demo.sh." >&2
    exit 1
fi

# Classify the port explicitly so the three outcomes get three distinct messages.
# Classify the port explicitly so the three outcomes get three distinct messages.
# A single -c invocation keeps stdin untouched (the entry point below may read it).
PORT_REPORT="$(
    { "${VENV_PYTHON}" -c '
import sys
sys.path.insert(0, sys.argv[3])
from recommendation.local_demo import classify_port
status = classify_port(sys.argv[1], int(sys.argv[2]))
print("%s\t%s" % (status.state, status.detail))
' "${HOST}" "${PORT}" "${REPO_ROOT}" 2>/dev/null; } || echo $'unknown\tport probe failed'
)"
PORT_STATE="${PORT_REPORT%%$'\t'*}"
PORT_DETAIL="${PORT_REPORT#*$'\t'}"
[[ -z "${PORT_STATE}" ]] && PORT_STATE="unknown"
[[ "${PORT_STATE}" == "${PORT_DETAIL}" ]] && PORT_DETAIL=""

echo
echo "[3/4] Port ${HOST}:${PORT} -> ${PORT_STATE}"
[[ -n "${PORT_DETAIL}" ]] && echo "      ${PORT_DETAIL}"

case "${PORT_STATE}" in
    free)
        echo "      port is free; starting the server."
        ;;
    agentrecx_running)
        echo
        echo "An AgentRec-X demo server is ALREADY running on ${HOST}:${PORT}."
        echo "Not starting a second instance (two servers would share one memory database)."
        echo
        echo "  Demo : http://${HOST}:${PORT}/demo/"
        echo "  Docs : http://${HOST}:${PORT}/docs"
        echo
        echo "Stop the running instance with Ctrl+C in its own terminal, or start this"
        echo "one on another port:  ./scripts/start_demo.sh --port 8011"
        exit 0
        ;;
    foreign)
        echo
        echo "ERROR: ${HOST}:${PORT} is occupied by a process that is not an AgentRec-X" >&2
        echo "       demo server. This launcher never stops or kills another process." >&2
        echo "       Choose a different port:  ./scripts/start_demo.sh --port 8011" >&2
        exit 3
        ;;
    *)
        echo >&2
        echo "ERROR: could not classify port ${HOST}:${PORT}; refusing to start." >&2
        exit 1
        ;;
esac

echo
echo "[4/4] Starting Uvicorn (foreground). Press Ctrl+C to stop."
echo
echo "  Demo      : http://${HOST}:${PORT}/demo/"
echo "  API docs  : http://${HOST}:${PORT}/docs"
echo "  Health    : http://${HOST}:${PORT}/health"
echo "  Demo ready: http://${HOST}:${PORT}/v1/demo/health"
echo "  Model info: http://${HOST}:${PORT}/v1/model"
echo

# Foreground, no daemon. Uvicorn owns its own graceful SIGINT/SIGTERM shutdown; the
# trap exists so a terminated wrapper still waits for and reports the child's exit
# status instead of leaving an orphan.
SERVER_PID=""
on_signal() {
    local signal="$1"
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -"${signal}" "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

# The existing FastAPI entry point remains the actual service entrypoint. The device
# is passed through the documented AGENTRECX_* environment path; host and port are
# passed as CLI flags (which are passed straight to uvicorn.run).
AGENTRECX_DEVICE="${DEVICE}" \
    "${VENV_PYTHON}" -m recommendation.api.app \
        --host "${HOST}" \
        --port "${PORT}" \
        --device "${DEVICE}" &
SERVER_PID=$!

set +e
wait "${SERVER_PID}"
STATUS=$?
set -e

trap - INT TERM
echo
echo "AgentRec-X demo stopped (exit status ${STATUS})."
exit "${STATUS}"
