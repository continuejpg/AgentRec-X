#!/usr/bin/env bash
#
# AgentRec-X local demo environment preparation (Milestone 11.5).
#
#   ./scripts/setup_demo.sh [--skip-install] [--help]
#
# This is the ONLY script that mutates the environment. It is deliberately separate
# from ./scripts/start_demo.sh: a normal start never installs anything.
#
# What it does, in order:
#
#   1. validate a supported Python (3.10.x);
#   2. create <repo>/.venv when it is missing;
#   3. install the generic runtime dependencies (requirements.txt);
#   4. install the accepted CPU-only PyTorch from the official CPU wheel index
#      (requirements-cpu.txt) -- never an unqualified `pip install torch`;
#   5. assert the result really is CPU-only (torch.version.cuda is None, no `nvidia`
#      package directory) and that `pip check` is clean;
#   6. print the demo readiness report and exit with its verdict.
#
# Step 6 owns no artifact truth of its own: the runtime artifact list, paths, sizes and
# digests come from config/demo_runtime_artifacts.json through scripts/fetch_artifacts.py,
# and are verified by that module's own verifier.  The report ends in
#   DEMO READINESS: READY      (start_demo.sh can serve the product)
#   DEMO READINESS: BLOCKED    (a required item is missing, with the recovery command)
# The LLM provider and the artifact-recovery credential are OPTIONAL and never block.
#
# It never downloads, regenerates or modifies an artifact, and it never touches a
# running demo server.
#
set -Eeuo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
VENV_DIR="${REPO_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

# Make the repository importable regardless of the caller's current directory, so the
# final `python -m recommendation.local_demo doctor` works from anywhere.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SUPPORTED_PYTHON_MAJOR=3
SUPPORTED_PYTHON_MINOR=10
SKIP_INSTALL=0

usage() {
    cat <<'EOF'
AgentRec-X local demo environment setup

Usage:
  ./scripts/setup_demo.sh [options]

Options:
  --skip-install   create the venv if needed, but install nothing
                   (use when the environment is already prepared)
  -h, --help       show this help

After a successful setup, start the demo with:  ./scripts/start_demo.sh
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-install) SKIP_INSTALL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

echo "======================================================================"
echo " AgentRec-X local demo setup"
echo "======================================================================"
echo "Repository root : ${REPO_ROOT}"
echo "Virtualenv      : ${VENV_DIR}"

# --------------------------------------------------------------------------- #
# [1/6] Python
# --------------------------------------------------------------------------- #
echo
echo "[1/6] Python interpreter"

BOOTSTRAP_PYTHON=""
for candidate in "python${SUPPORTED_PYTHON_MAJOR}.${SUPPORTED_PYTHON_MINOR}" python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
        version="$("${candidate}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
        if [[ "${version}" == "${SUPPORTED_PYTHON_MAJOR}.${SUPPORTED_PYTHON_MINOR}" ]]; then
            BOOTSTRAP_PYTHON="$(command -v "${candidate}")"
            break
        fi
    fi
done

if [[ -z "${BOOTSTRAP_PYTHON}" ]]; then
    echo "ERROR: no Python ${SUPPORTED_PYTHON_MAJOR}.${SUPPORTED_PYTHON_MINOR}.x interpreter found on PATH." >&2
    echo "       On WSL2 Ubuntu 22.04: sudo apt install python${SUPPORTED_PYTHON_MAJOR}.${SUPPORTED_PYTHON_MINOR}-venv" >&2
    exit 1
fi

echo "      found : ${BOOTSTRAP_PYTHON}"
echo "      version: $("${BOOTSTRAP_PYTHON}" -c 'import sys; print(sys.version.split()[0])')"

# If the venv already exists, it must be the supported series too.
if [[ -x "${VENV_PYTHON}" ]]; then
    existing="$("${VENV_PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    if [[ "${existing}" != "${SUPPORTED_PYTHON_MAJOR}.${SUPPORTED_PYTHON_MINOR}" ]]; then
        echo "ERROR: ${VENV_PYTHON} is Python ${existing}, expected ${SUPPORTED_PYTHON_MAJOR}.${SUPPORTED_PYTHON_MINOR}.x" >&2
        echo "       Remove ${VENV_DIR} and re-run this script to rebuild it." >&2
        exit 1
    fi
fi

# --------------------------------------------------------------------------- #
# [2/6] Virtualenv
# --------------------------------------------------------------------------- #
echo
echo "[2/6] Virtualenv"
if [[ -x "${VENV_PYTHON}" ]]; then
    echo "      already present: ${VENV_PYTHON}"
else
    echo "      creating ${VENV_DIR} ..."
    "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"
    echo "      created"
fi

# --------------------------------------------------------------------------- #
# [3/6] / [4/6] Dependencies
# --------------------------------------------------------------------------- #
# Preparation runs only when something is actually missing: a complete environment is
# left untouched rather than reinstalled on every onboarding check.
ENV_COMPLETE=0
if [[ -x "${VENV_PYTHON}" ]] \
   && "${VENV_PYTHON}" -m pip check >/dev/null 2>&1 \
   && "${VENV_PYTHON}" -c 'import torch, numpy' >/dev/null 2>&1; then
    ENV_COMPLETE=1
fi

if (( SKIP_INSTALL == 1 )); then
    echo
    echo "[3/6] Runtime dependencies  : SKIPPED (--skip-install)"
    echo "[4/6] CPU-only PyTorch       : SKIPPED (--skip-install)"
elif (( ENV_COMPLETE == 1 )); then
    echo
    echo "[3/6] Runtime dependencies  : already satisfied (pip check clean, torch importable)"
    echo "[4/6] CPU-only PyTorch       : already satisfied"
else
    echo
    echo "[3/6] Runtime dependencies (requirements.txt)"
    "${VENV_PYTHON}" -m pip install --upgrade --quiet pip
    "${VENV_PYTHON}" -m pip install -r "${REPO_ROOT}/requirements.txt"

    echo
    echo "[4/6] CPU-only PyTorch (requirements-cpu.txt, official CPU wheel index)"
    echo "      NOTE: a plain 'pip install torch' would pull CUDA/nvidia packages; this"
    echo "            file pins torch==2.1.2+cpu from download.pytorch.org/whl/cpu."
    "${VENV_PYTHON}" -m pip install -r "${REPO_ROOT}/requirements-cpu.txt"
fi

# --------------------------------------------------------------------------- #
# [5/6] Dependency contract
# --------------------------------------------------------------------------- #
echo
echo "[5/6] Dependency contract"
"${VENV_PYTHON}" -m pip check

"${VENV_PYTHON}" - <<'PY'
import sys

try:
    import numpy
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"ERROR: NumPy is not importable: {exc!r}")
print(f"      numpy : {numpy.__version__}")
if numpy.__version__ != "1.26.4":
    raise SystemExit(f"ERROR: expected the pinned numpy==1.26.4, found {numpy.__version__}")

try:
    import torch
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"ERROR: PyTorch is not importable: {exc!r}")
print(f"      torch : {torch.__version__}")

if torch.version.cuda is not None:
    raise SystemExit(
        f"ERROR: this is a CUDA build of PyTorch (torch.version.cuda={torch.version.cuda}). "
        "The demo must stay CPU-only; reinstall from requirements-cpu.txt."
    )
if torch.cuda.is_available():
    raise SystemExit("ERROR: a CUDA device is reported available; expected CPU-only.")
print("      torch.version.cuda : None")
print("      cuda available     : False")
print("      device             : cpu")
PY

# --------------------------------------------------------------------------- #
# [6/6] Demo readiness
# --------------------------------------------------------------------------- #
# The provider configuration is read from <repo>/.env, exactly as ./scripts/start_demo.sh
# reads it, so this report describes the environment the server will actually get.  The
# loader is not re-implemented here: sourcing the launcher defines it and returns before
# the launcher can do anything else.
# shellcheck source=./start_demo.sh
source "${SCRIPT_DIR}/start_demo.sh"
load_dotenv_into_environment "${REPO_ROOT}/.env"

echo
echo "[6/6] Demo readiness (runtime artifacts verified by scripts/fetch_artifacts.py)"
"${VENV_PYTHON}" -m recommendation.demo.readiness --repo-root "${REPO_ROOT}"
READINESS_STATUS=$?

if (( READINESS_STATUS != 0 )); then
    echo >&2
    echo "Setup finished, but the demo cannot start yet: a required item is missing above." >&2
fi
exit "${READINESS_STATUS}"
