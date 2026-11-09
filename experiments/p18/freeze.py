"""P18 freeze manifest: record core source/config/index/model hashes before one reserved run."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p18/freeze_manifest.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    core_paths = [
        "recommendation/control/catalog_search.py", "recommendation/mcp/server.py",
        "recommendation/skills/registry.py", "recommendation/skills/shopping.py",
        "recommendation/skills/clarification.py", "experiments/p14e/catalogue.py",
        "experiments/p14c0/electronics_executor.py", "experiments/p14c1/search_backend.py",
        "recommendation/control/rag_evidence.py", "recommendation/preference/decision_gate.py",
        "recommendation/preference/history.py", "recommendation/shopping_companion/memory.py",
        "data/p14e2/benchmark/reserved_test.agent.jsonl",
        "data/p14e2/benchmark/reserved_test.evaluator.jsonl",
        "runs/p14d2/bge_index.npz", "runs/p14d2/query_vectors.npz",
    ]
    files = {}
    for rel in core_paths:
        path = REPO / rel
        if path.is_file():
            files[rel] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest = {
        "schema": "agentrecx.p18.freeze_manifest.v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_hash,
        "python": sys.version,
        "modes": {
            "CPU_MCP": "stdio MCP + local BGE/BM25 + Rule Planner/Gate + P15 + P16; no Qwen",
            "Qwen_Agent": "existing Qwen3.5-9B Base Agent mode; requires GPU/service; this run is replay-only",
        },
        "search": {"catalogue_size": 76263, "index": "runs/p14d2/bge_index.npz",
                    "bge_model": "models/bge-base-en-v1.5", "bge_revision": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"},
        "data": {"task_manifest": "data/p14e2/benchmark/manifest.json",
                  "dev_tasks": 20, "reserved_tasks": 20},
        "policy": {
            "reserved_run": "one-time only after this freeze; no model/rule/benchmark changes based on its result",
            "no_training": True, "no_index_rebuild": True, "no_reserved_retune": True,
        },
        "files": files,
        "core_file_count": len(files),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"frozen_at": manifest["frozen_at"], "git_head": git_hash, "core_file_count": len(files),
                      "total_bytes": sum(value["bytes"] for value in files.values())}, indent=1))


if __name__ == "__main__":
    main()
