#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/IT/CODE/PROJECTION/agentrec/X
mkdir -p runs/web_demo/logs
export AGENTRECX_WEB_DEMO_SEARCH_MODE=bge
export AGENTRECX_WEB_DEMO_ALLOW_MEMORY_WRITES=1
export AGENTRECX_WEB_DEMO_DEFAULT_PROFILE=profile_0057
export AGENTRECX_WEB_DEMO_CONSOLE_ONLY_LOG=1
nohup setsid .venv/bin/python -m web_demo > runs/web_demo/logs/bff.log 2>&1 < /dev/null &
echo $! > runs/web_demo/bff.pid
echo "started $(cat runs/web_demo/bff.pid)"
