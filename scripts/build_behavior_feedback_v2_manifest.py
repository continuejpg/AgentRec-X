"""Build the V2 environment/GPU evidence files from real processes and artifacts.

Run inside WSL after the remote Qwen service and local BFF are running.  The SSH private
key path is read from AGENTRECX_REMOTE_KEY and is never written to the evidence files.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs/behavior_feedback_v2"
BFF = os.environ.get("BFF_URL", "http://127.0.0.1:8787")
BGE_INDEX = REPO / "runs/p14d2/bge_index.npz"
BGE_MODEL = REPO / "models/bge-base-en-v1.5"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def http_json(url: str, timeout: float = 30.0) -> dict:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def sha256(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.relative_to(REPO)), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def remote_ssh(script: str) -> dict:
    key = os.environ.get("AGENTRECX_REMOTE_KEY")
    host = os.environ.get("AGENTRECX_REMOTE_HOST")
    port = os.environ.get("AGENTRECX_REMOTE_PORT", "22")
    if not key or not host:
        return {"available": False, "reason": "AGENTRECX_REMOTE_KEY / AGENTRECX_REMOTE_HOST not set"}
    command = [
        "ssh", "-p", str(port), "-i", key, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15", f"root@{host}", "bash -s",
    ]
    completed = subprocess.run(command, input=script, text=True, capture_output=True, timeout=60)
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr[-1000:],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    health = http_json(BFF + "/api/health")
    config = http_json(BFF + "/api/demo/config")

    remote_script = r'''
set -e
echo "HOST $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader
echo "MODEL_PID $(pgrep -f 'training.agent_sft.model_server' | head -1 || true)"
echo "PLAN_PORT"
(python3 - <<'PY' 2>/dev/null || true
PY
) || true
tail -5 /root/autodl-tmp/qwen_live_model_server.log 2>/dev/null || true
echo "TORCH $(/root/autodl-tmp/agentrecx-sft/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>/dev/null || true)"
echo "GPU_PEAK_LOG"
if [ -f /root/autodl-tmp/gpu_peak_v2.log ]; then awk -F, 'NR==1{max=0} {gsub(/ MiB/,"",$2); v=$2+0; if(v>max) max=v} END{print "MAX_MIB", max}' /root/autodl-tmp/gpu_peak_v2.log; fi
'''
    remote = remote_ssh(remote_script)
    gpu: dict = {"available": remote.get("available")}
    if remote.get("available"):
        stdout = remote.get("stdout", "")
        for line in stdout.splitlines():
            if line.startswith("HOST "):
                gpu["remote_host_label"] = line[5:].strip()
            elif line.startswith("NVIDIA"):
                parts = [part.strip() for part in line.split(",")]
                gpu.update({
                    "name": parts[0], "memory_total_mib": int(parts[1].split()[0]),
                    "memory_used_mib": int(parts[2].split()[0]),
                    "utilization_gpu_percent": int(parts[3].split()[0]),
                    "driver_version": parts[4],
                })
            elif line.startswith("MODEL_PID "):
                gpu["model_server_pid"] = line[9:].strip() or None
            elif line.startswith("TORCH "):
                gpu["torch"] = line[6:].strip()
            elif line.startswith("MAX_MIB "):
                gpu["peak_observed_mib"] = int(line[8:].strip())
        gpu["model_server_log_tail"] = "\n".join(
            line for line in stdout.splitlines() if "proposal server" in line
        )
    gpu["monitor_note"] = "peak_observed_mib is the max 2-second sample in gpu_peak_v2.log"
    gpu["generated_at"] = now_iso()

    chrome = None
    try:
        chrome = http_json("http://127.0.0.1:9222/json/version", timeout=5).get("Browser")
    except Exception as exc:  # noqa: BLE001
        chrome = f"unavailable: {type(exc).__name__}"

    bge_config = json.loads((BGE_MODEL / "config.json").read_text(encoding="utf-8")) if (BGE_MODEL / "config.json").exists() else {}
    environment = {
        "schema": "agentrecx.behavior_feedback_v2.environment_manifest.v1",
        "generated_at": now_iso(),
        "code": {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "git_status_short": (
                [] if len(subprocess.check_output(
                    ["git", "status", "--short"], cwd=REPO, text=True
                ).splitlines()) > 100 else subprocess.check_output(
                    ["git", "status", "--short"], cwd=REPO, text=True
                ).splitlines()
            ),
        },
        "runtime": {
            "bff_search_mode": health.get("search_mode"),
            "bff_ready": health.get("ready"),
            "servers": health.get("servers"),
            "qwen_live": health.get("qwen_live"),
            "behavior": config.get("behavior"),
            "chrome": chrome,
            "frontend": "Next.js on 127.0.0.1:3000",
        },
        "devices": {
            "qwen": "remote NVIDIA GPU (see gpu_health.json)",
            "bge": "local CPU/WSL",
            "bff": "local CPU/WSL",
            "planner_transport": "WSL-local loopback SSH tunnel; no public endpoint",
        },
        "remote": {k: v for k, v in remote.items() if k not in {"stdout"}},
    }
    (OUT / "gpu_health.json").write_text(json.dumps(gpu, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "environment_manifest.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    bge = {
        "schema": "agentrecx.behavior_feedback_v2.bge_index.v1",
        "generated_at": now_iso(),
        "path": str(BGE_INDEX.relative_to(REPO)),
        "index": sha256(BGE_INDEX),
        "artifacts": {"identities": 76263, "embedding_dim": 768},
        "encoder": {
            "model_id": bge_config.get("_name_or_path", "BAAI/bge-base-en-v1.5"),
            "local_path": str(BGE_MODEL.relative_to(REPO)),
            "config": {key: bge_config.get(key) for key in ("model_type", "hidden_size", "max_position_embeddings")},
        },
        "retrieval_mode": "bge",
        "score_kind": "bge_dense",
        "notes": ["Existing accepted P14-D2 index; not rebuilt, not retrained."],
    }
    (OUT / "bge_environment.json").write_text(json.dumps(bge, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"gpu": gpu, "bge": bge["index"], "chrome": chrome}, indent=1))


if __name__ == "__main__":
    main()
