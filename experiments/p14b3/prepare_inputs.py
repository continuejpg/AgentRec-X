"""P14-B3: copy the frozen P14-B2 pools into P14-B3 input artifacts (agent-visible only)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "runs/p14b2/pools"
OUT = REPO / "runs/p14b3/pools"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "agentrecx.p14b3.input_manifest.v1",
                "source": "runs/p14b2/pools (same BM25 top-20 + Product Type/Role filter as P14-B2)",
                "pools": [], "labels_in_pool": False,
                "preference_text": "positive/negative active preferences from M9/M5; withdrawn excluded"}
    for split in ("train_smoke", "dev"):
        source = SRC / f"{split}.jsonl"
        target = OUT / f"{split}.jsonl"
        shutil.copyfile(source, target)
        rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
        manifest["pools"].append({"split": split, "tasks": len(rows),
                                  "candidates": sum(len(row["candidates"]) for row in rows),
                                  "tasks_with_preferences": sum(1 for row in rows if row["preference_summary"])})
    (REPO / "runs/p14b3/input_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                                         encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
