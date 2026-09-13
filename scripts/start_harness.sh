#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/root/AgentRec-X"
SESSION="harness"
PORT="38080"
NVM_DIR="/root/.nvm"

echo "========================================"
echo " AgentRec-X Harness Startup"
echo "========================================"

cd "$PROJECT"

# Load Node/NVM for non-interactive SSH sessions
export NVM_DIR
if [ -s "$NVM_DIR/nvm.sh" ]; then
    . "$NVM_DIR/nvm.sh"
fi

echo
echo "[1/5] Repository"
echo "Path: $(pwd)"
git status --short || true

echo
echo "[2/5] GPU"
if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; then
    :
else
    echo "ERROR: GPU check failed."
    exit 1
fi

echo
echo "[3/5] Python / CUDA"
python - <<'PY'
import sys
import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "[4/5] DeepSeek Harness"

# If the tmux session exists, check whether the web service is actually alive.
if tmux has-session -t "$SESSION" 2>/dev/null; then
    HTTP_CODE="$(
        curl -sS --max-time 2 \
            -o /dev/null \
            -w '%{http_code}' \
            "http://127.0.0.1:${PORT}" 2>/dev/null || true
    )"

    if [ -n "$HTTP_CODE" ] && [ "$HTTP_CODE" != "000" ]; then
        echo "Harness is already running."
        echo "tmux session: $SESSION"
        echo "HTTP status: $HTTP_CODE"
    else
        echo "Stale tmux session detected. Restarting..."
        tmux kill-session -t "$SESSION" || true
    fi
fi

# Start only if no valid session remains.
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" \
        "cd '$PROJECT' && source '$NVM_DIR/nvm.sh' && exec npx @deepseek-ai/dsh web --port '$PORT' --no-open"

    echo "Harness process started."

    # Wait up to ~20 seconds for HTTP service.
    READY=0
    for i in $(seq 1 20); do
        HTTP_CODE="$(
            curl -sS --max-time 1 \
                -o /dev/null \
                -w '%{http_code}' \
                "http://127.0.0.1:${PORT}" 2>/dev/null || true
        )"

        if [ -n "$HTTP_CODE" ] && [ "$HTTP_CODE" != "000" ]; then
            READY=1
            break
        fi

        sleep 1
    done

    if [ "$READY" -ne 1 ]; then
        echo "ERROR: Harness did not become reachable."
        echo
        echo "Last tmux output:"
        tmux capture-pane -pt "$SESSION" | tail -n 30 || true
        exit 1
    fi

    echo "Harness HTTP status: $HTTP_CODE"
fi

echo
echo "[5/5] Status"
tmux ls | grep "$SESSION" || true

echo
echo "========================================"
echo " Harness ready"
echo " Remote: http://127.0.0.1:${PORT}"
echo "========================================"
