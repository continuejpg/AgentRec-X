"""P14-C0 Qwen service probe / frozen model manifest.

Read-only: it inspects environment/configuration and (if configured) sends one harmless
HTTP request to the configured planner service.  It never starts/stops an AutoDL instance
and never fabricates a model response.  Result goes to runs/p14c0/qwen_manifest.json.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p14c0/qwen_manifest.json"
CONFIG = REPO / "runs/p14c0/config.json"


def _tcp(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _service_status(url: str) -> dict:
    if not url:
        return {"configured": False, "reachable": False, "detail": "no URL configured"}
    try:
        health = url.rstrip("/") + "/plan"
        request = urllib.request.Request(health, data=b"{}", headers={"Content-Type": "application/json"},
                                         method="POST")
        with urllib.request.urlopen(request, timeout=4) as response:
            return {"configured": True, "reachable": True, "url": url,
                    "http_status": response.status,
                    "detail": "planner endpoint answered (not used for a Qwen trajectory in this blocked run)"}
    except Exception as exc:  # noqa: BLE001
        return {"configured": True, "reachable": False, "url": url,
                "detail": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.is_file() else {}
    url = (os.environ.get("AGENTRECX_QWEN_PLANNER_URL") or "").strip()
    enabled = (os.environ.get("AGENTRECX_QWEN_PLANNER") or "").strip().casefold() in {
        "1", "true", "on", "yes", "enabled"}
    payload = {
        "schema": "agentrecx.p14c0.qwen_manifest.v1",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": config.get("model", {}),
        "runtime_config": config.get("budget", {}),
        "service": {
            "enabled": enabled,
            "url_configured": bool(url),
            "probe": _service_status(url) if enabled else {
                "configured": False, "reachable": False,
                "detail": "AGENTRECX_QWEN_PLANNER is not enabled; no Qwen inference was attempted",
            },
            "autodl_ssh": {
                "hosts": ["connect.nmb1.seetacloud.com:22199", "connect.bjb3.seetacloud.com:25658"],
                "probe": "banner exchange: Connection to UNKNOWN port -1: Connection refused",
                "meaning": "instances are not running/reachable; starting/stopping AutoDL is forbidden in this task",
            },
        },
        "trajectories": [],
        "status": "BLOCKED",
        "reason": "real Qwen Base inference could not be executed: no configured local service and the "
                  "AutoDL instances refused connections; no fake Qwen output is produced",
        "next_action": "When an existing Qwen3.5-9B Base serving endpoint is available, set "
                       "AGENTRECX_QWEN_PLANNER=1 and AGENTRECX_QWEN_PLANNER_URL=<url> and run "
                       "`python -m experiments.p14c0.run_dev --planner both`; the same frozen config and "
                       "executor/tool/Gate stack will be reused.",
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "reason")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
