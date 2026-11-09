"""Build the fixed BM25 Top-20 + Product Type/Role candidate pools for Eval100."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.p14b2.build_pools import build_pool
from experiments.p14c0.run_dev import _memory_read
from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p14b3_eval100"


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    tasks = load_jsonl(REPO / "data/p14b3_eval100/eval100.agent.jsonl")
    pools = []
    for task in tasks:
        preferences, _history = _memory_read(catalogue, task)
        pools.append(build_pool(catalogue, task, preferences))
    (OUT / "pools").mkdir(parents=True, exist_ok=True)
    path = OUT / "pools/eval100.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in pools),
                    encoding="utf-8")
    stats = {"tasks": len(pools), "candidates": sum(len(row["candidates"]) for row in pools),
             "empty_pools": sum(1 for row in pools if not row["candidates"]),
             "tasks_with_preferences": sum(1 for row in pools if row["preference_summary"]),
             "role_rejected": sum(row["role_rejected"] for row in pools)}
    (OUT / "pool_manifest.json").write_text(json.dumps(
        {"schema": "agentrecx.p14b3eval100.pool_manifest.v1", "pool": "BM25 top-20 + Product Type/Role filter",
         "stats": stats, "labels_in_pools": False}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
