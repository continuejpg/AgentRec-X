#!/usr/bin/env bash
# One-command launcher for the AgentRec-X Web Demo BFF (WSL Ubuntu-22.04 / Linux).
#
#   bash scripts/run_web_demo_bff.sh
#
# Defaults for this launcher (chosen for the demo):
#   AGENTRECX_WEB_DEMO_SEARCH_MODE=bge             (set bm25 for a fast dev loop)
#   AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES=1       (the BFF itself still defaults to 0)
#   AGENTRECX_WEB_DEMO_PORT=8787
# Every value can be overridden by exporting it before calling the script.
#
# Side effects:
#   runs/web_demo/bff.pid              PID of the python BFF process (for a PowerShell launcher)
#   runs/web_demo/logs/bff.log         BFF + uvicorn console log (tee'd)
#   runs/web_demo/logs/mcp-<profile>.log  real MCP server stderr (written by the BFF)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export AGENTRECX_WEB_DEMO_SEARCH_MODE="${AGENTRECX_WEB_DEMO_SEARCH_MODE:-bge}"
export AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES="${AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES:-1}"
# The launcher owns runs/web_demo/logs/bff.log via tee, so the BFF skips its own file handler
# (avoids duplicated lines).  A direct `.venv/bin/python -m web_demo` run still writes bff.log.
export AGENTRECX_WEB_DEMO_CONSOLE_ONLY_LOG=1

RUN_DIR="$REPO_ROOT/runs/web_demo"
LOG_DIR="$RUN_DIR/logs"
PID_FILE="$RUN_DIR/bff.pid"
LOG_FILE="$LOG_DIR/bff.log"
mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "BFF already running (pid ${old_pid}). Stop it with: kill ${old_pid}" >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi

{
  echo "=== AgentRec-X Web Demo BFF $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "repo=${REPO_ROOT} search_mode=${AGENTRECX_WEB_DEMO_SEARCH_MODE}" \
       "memory_writes=${AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES}" \
       "port=${AGENTRECX_WEB_DEMO_PORT:-8787}" \
       "qwen_requested=${QWEN_LIVE_ENABLED:-0}"
} | tee -a "$LOG_FILE"

# Process substitution keeps $! pointing at the python process (not at tee), so the PID file and
# the cleanup trap control the real BFF process and never leave an orphaned uvicorn/MCP tree.
.venv/bin/python -m web_demo > >(tee -a "$LOG_FILE") 2>&1 &
BFF_PID=$!
echo "$BFF_PID" > "$PID_FILE"

cleanup() {
  if kill -0 "$BFF_PID" 2>/dev/null; then
    kill -TERM "$BFF_PID" 2>/dev/null || true
    for _ in $(seq 1 100); do
      kill -0 "$BFF_PID" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$BFF_PID" 2>/dev/null; then
      echo "BFF did not stop gracefully; sending SIGKILL" >&2
      kill -KILL "$BFF_PID" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
}
trap cleanup EXIT INT TERM

wait "$BFF_PID"
