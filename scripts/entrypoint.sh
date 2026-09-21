#!/usr/bin/env bash
#
# Container entrypoint for the AgentRec-X demo image.
#
# Responsibilities, deliberately kept small:
#
#   1. refuse to start a live model provider unless it was explicitly requested, so a
#      container that happens to inherit AGENTRECX_LLM_* cannot start billing by accident;
#   2. generate the demo artifacts when the mounted volume does not have them (a fresh
#      volume), reusing the image's copies when it does;
#   3. exec the accepted server entry point, so signals reach uvicorn directly and the
#      container's exit status is the server's.
#
# It never prints an environment variable's value, so a credential passed at run time cannot
# appear in the container logs.
#
set -Eeuo pipefail

DEMO_DIR="${AGENTRECX_DEMO_ARTIFACTS_DIR:-/app/runs/demo_catalog}"
POLICY="${AGENTRECX_AGENT_POLICY:-deterministic}"
# The interpreter that serves the app.  Overridable so a test can point the entrypoint at a
# specific virtualenv instead of whatever `python` happens to resolve to on PATH.
PYTHON_BIN="${AGENTRECX_PYTHON:-python}"

if [[ "${POLICY}" == "llm" ]]; then
    # Real-provider mode is an explicit opt-in that needs a base URL and a model name; the key
    # may legitimately be absent for a local server that needs none.
    for required in AGENTRECX_LLM_BASE_URL AGENTRECX_LLM_MODEL; do
        if [[ -z "${!required:-}" ]]; then
            echo "ERROR: AGENTRECX_AGENT_POLICY=llm requires ${required}." >&2
            echo "       See DOCKER.md for the real-provider configuration." >&2
            exit 2
        fi
    done
    echo "agent policy : llm (explicit opt-in; provider calls will be billed to your account)"
else
    # Make the offline guarantee explicit rather than assumed, and unset the provider
    # variables so nothing downstream can read them.
    unset AGENTRECX_LLM_BASE_URL AGENTRECX_LLM_MODEL AGENTRECX_LLM_API_KEY \
          AGENTRECX_LLM_PROFILE AGENTRECX_LLM_TIMEOUT AGENTRECX_LLM_JSON_MODE \
          AGENTRECX_LLM_THINKING AGENTRECX_LLM_INPUT_PRICE AGENTRECX_LLM_OUTPUT_PRICE || true
    echo "agent policy : deterministic (offline; no provider configured)"
fi

export AGENTRECX_DEMO_ARTIFACTS_DIR="${DEMO_DIR}"
if [[ ! -s "${DEMO_DIR}/checkpoint.pt" || ! -s "${DEMO_DIR}/mappings.json" ]]; then
    echo "demo artifacts missing in ${DEMO_DIR}; generating them now"
    "${PYTHON_BIN}" -m experiments.build_demo_catalog --out "${DEMO_DIR}" --quiet
fi

export AGENTRECX_CHECKPOINT_PATH="${AGENTRECX_CHECKPOINT_PATH:-${DEMO_DIR}/checkpoint.pt}"
export AGENTRECX_MAPPINGS_PATH="${AGENTRECX_MAPPINGS_PATH:-${DEMO_DIR}/mappings.json}"
export AGENTRECX_CATALOG_METADATA_PATH="${AGENTRECX_CATALOG_METADATA_PATH:-/app/recommendation/demo/artifacts/demo_products.jsonl}"
export AGENTRECX_MEMORY_DB="${AGENTRECX_MEMORY_DB:-${DEMO_DIR}/preference_memory.sqlite3}"

HOST="${AGENTRECX_HOST:-0.0.0.0}"
PORT="${AGENTRECX_PORT:-8000}"
echo "starting AgentRec-X demo on http://${HOST}:${PORT}/demo/"

exec "${PYTHON_BIN}" -m recommendation.api.app --host "${HOST}" --port "${PORT}" --device "${AGENTRECX_DEVICE:-cpu}"
