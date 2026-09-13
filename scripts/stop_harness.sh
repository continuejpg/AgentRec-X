#!/usr/bin/env bash
set -u

PROJECT="/root/AgentRec-X"
SESSION="harness"

echo "========================================"
echo " AgentRec-X Harness Shutdown"
echo "========================================"

cd "$PROJECT"

echo
echo "[1/2] Stopping Harness"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    # Ask the foreground process to stop gracefully.
    tmux send-keys -t "$SESSION" C-c
    sleep 2

    # Remove session if it still exists.
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux kill-session -t "$SESSION"
    fi

    echo "Harness stopped."
else
    echo "Harness is not running."
fi

echo
echo "[2/2] Git status"
git status --short || true

echo
echo "========================================"
echo " Harness stopped."
echo " Review Git status before AutoDL shutdown."
echo "========================================"
