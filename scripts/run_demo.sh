#!/usr/bin/env bash
#
# AgentRec-X - one-command offline demo (packaging phase).
#
#   ./scripts/run_demo.sh [--host HOST] [--port PORT] [--no-verify]
#
# What this does, in order:
#
#   1. creates .venv when missing and installs the runtime + CPU-only PyTorch
#      (requirements.txt then requirements-cpu.txt - never a bare `pip install torch`);
#   2. builds the small synthetic demo catalogue and its checkpoint/mappings
#      (experiments.build_demo_catalog) - no download, no network, no credentials;
#   3. runs the canonical scenario once and prints the trajectory, so the run is
#      inspectable before the server starts;
#   4. starts the FastAPI server on the chosen port, offline.
#
# This is the native path.  `docker compose up` performs the same steps inside a container;
# see DOCKER.md.
#
# It never writes a credential, never enables a live model provider and never edits the
# environment outside .venv and runs/demo_catalog.
#
# Real-provider mode is a separate, explicit opt-in documented in README.md; it needs
# AGENTRECX_LLM_* plus AGENTRECX_AGENT_POLICY=llm.
#
set -Eeuo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

HOST="127.0.0.1"
PORT="8000"
VERIFY=1
SKIP_INSTALL=0
DEMO_ARTIFACTS="${REPO_ROOT}/runs/demo_catalog"

usage() {
    cat <<'EOF'
AgentRec-X one-command offline demo

Usage:
  ./scripts/run_demo.sh [options]

Options:
  --host HOST        bind host (default 127.0.0.1)
  --port PORT        bind port (default 8000)
  --no-verify        skip the dependency install check (faster restart)
  --skip-install     assume .venv is already prepared
  -h, --help         show this help

Everything runs offline. No API key is read. Ctrl+C stops the server.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="${2:?--host needs a value}"; shift 2 ;;
        --port) PORT="${2:?--port needs a value}"; shift 2 ;;
        --no-verify) VERIFY=0; shift ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! [[ "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "ERROR: --port must be an integer in 1..65535, got '${PORT}'" >&2
    exit 2
fi

echo "======================================================================"
echo " AgentRec-X one-command demo (offline)"
echo "======================================================================"
echo " repository : ${REPO_ROOT}"

# ---- [1/4] environment --------------------------------------------------- #
if [[ ! -x "${VENV_PYTHON}" ]]; then
    if (( SKIP_INSTALL == 1 )); then
        echo "ERROR: --skip-install given but ${VENV_PYTHON} does not exist." >&2
        exit 1
    fi
    echo "[1/4] Creating .venv and installing dependencies (first run only)"
    python3 -m venv "${REPO_ROOT}/.venv"
    "${VENV_PYTHON}" -m pip install --quiet --upgrade pip
    "${VENV_PYTHON}" -m pip install --quiet -r "${REPO_ROOT}/requirements.txt"
    # The CPU wheel index is what keeps this from pulling multi-GB CUDA userspace.
    "${VENV_PYTHON}" -m pip install --quiet -r "${REPO_ROOT}/requirements-cpu.txt"
else
    echo "[1/4] Environment ready: ${VENV_PYTHON}"
fi

if (( VERIFY == 1 )); then
    echo "      verifying the dependency closure (pip check)"
    "${VENV_PYTHON}" -m pip check >/dev/null 2>&1 || {
        echo "ERROR: 'pip check' failed; the environment is inconsistent." >&2
        echo "       Re-run without --skip-install, or rebuild .venv." >&2
        exit 1
    }
fi

# libgomp rejects an empty/non-positive OMP_NUM_THREADS on torch import.
if [[ -z "${OMP_NUM_THREADS:-}" || "${OMP_NUM_THREADS}" == "0" ]]; then
    export OMP_NUM_THREADS=1
fi

# ---- [2/4] demo artifacts ------------------------------------------------ #
echo
echo "[2/4] Demo catalogue and checkpoint (synthetic, untrained - no quality claim)"
"${VENV_PYTHON}" -m experiments.build_demo_catalog --out "${DEMO_ARTIFACTS}" --quiet
echo "      artifacts : ${DEMO_ARTIFACTS}"

# ---- [3/4] canonical scenario -------------------------------------------- #
echo
echo "[3/4] Canonical scenario (one full agent turn, offline)"
echo "----------------------------------------------------------------------"
"${VENV_PYTHON}" -m experiments.demo_scenario
echo "----------------------------------------------------------------------"

# ---- [4/4] server ------------------------------------------------------- #
echo
echo "[4/4] Starting the API. Ctrl+C to stop."
echo
echo "  Demo page  : http://${HOST}:${PORT}/demo/"
echo "  API docs   : http://${HOST}:${PORT}/docs"
echo "  Health     : http://${HOST}:${PORT}/health"
echo "  Agent      : POST http://${HOST}:${PORT}/v1/demo/agent/recommend"
echo

AGENTRECX_CHECKPOINT_PATH="${DEMO_ARTIFACTS}/checkpoint.pt" \
AGENTRECX_MAPPINGS_PATH="${DEMO_ARTIFACTS}/mappings.json" \
AGENTRECX_CATALOG_METADATA_PATH="${REPO_ROOT}/recommendation/demo/artifacts/demo_products.jsonl" \
AGENTRECX_MEMORY_DB="${DEMO_ARTIFACTS}/preference_memory.sqlite3" \
AGENTRECX_VERIFY_CHECKPOINT=0 \
AGENTRECX_MANIFEST_PATH=none \
AGENTRECX_DEVICE=cpu \
    exec "${VENV_PYTHON}" -m recommendation.api.app --host "${HOST}" --port "${PORT}" --device cpu
