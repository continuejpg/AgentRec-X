"""CPU preflight for P2-SFT dataset: schema, evidence, leakage and manifest checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .format import validate_dataset_row
from .prepare_data import DATA_DIR, GENERATOR_VERSION

_ASIN = re.compile(r"\bB[0-9A-Z]{9}\b")


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def _jaccard(a: str, b: str) -> float:
    left, right = set(_normalise(a).split()), set(_normalise(b).split())
    return len(left & right) / max(1, len(left | right))


def audit(data_dir: Path) -> dict[str, Any]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("manifest generator version does not match this code")
    rows = {split: _read(data_dir / f"{split}.jsonl") for split in ("train", "val", "test")}
    expected = {"train": 2400, "val": 240, "test": 200}
    counts = {split: len(value) for split, value in rows.items()}
    if counts != expected:
        raise ValueError(f"unexpected counts: {counts} != {expected}")
    hashes = {split: hashlib.sha256((data_dir / f"{split}.jsonl").read_bytes()).hexdigest() for split in rows}
    if hashes != manifest["sha256"]:
        raise ValueError("dataset hash mismatch against manifest")
    errors: list[str] = []
    asin_hits = 0
    for split, samples in rows.items():
        for index, row in enumerate(samples):
            for error in validate_dataset_row(row):
                errors.append(f"{split}[{index}]: {error}")
            assistant = str(row.get("assistant", ""))
            if _ASIN.search(assistant):
                asin_hits += 1
            if row.get("sample_id") != f"p2sft_{split}_{index:05d}":
                errors.append(f"{split}[{index}]: non-deterministic sample id")
    frozen_text: set[str] = set()
    frozen_paths = [Path(__file__).resolve().parents[2] / "benchmarks/agent/tasks" / name for name in ("dev.json", "frozen.json")]
    for path in frozen_paths:
        if path.is_file():
            for task in json.loads(path.read_text(encoding="utf-8")):
                for turn in task["turns"]:
                    frozen_text.add(str(turn["user_input"]).strip().casefold())
    overlaps = {
        split: sorted({row["user_message"].casefold() for row in samples if row["user_message"].casefold() in frozen_text})
        for split, samples in rows.items()
    }
    near_duplicates: list[tuple[str, str, float]] = []
    train_messages = [row["user_message"] for row in rows["train"]]
    for test_row in rows["test"]:
        for train_message in train_messages:
            score = _jaccard(test_row["user_message"], train_message)
            if score >= 0.90:
                near_duplicates.append((test_row["sample_id"], train_message, score))
                break
    template_overlap = {
        "test_train": sorted({row["template_id"] for row in rows["test"]} & {row["template_id"] for row in rows["train"]}),
        "test_val": sorted({row["template_id"] for row in rows["test"]} & {row["template_id"] for row in rows["val"]}),
        "train_val": sorted({row["template_id"] for row in rows["train"]} & {row["template_id"] for row in rows["val"]}),
    }
    report = {
        "status": "PASS" if not errors and not asin_hits and not any(overlaps.values()) and not template_overlap["test_train"] and not template_overlap["test_val"] else "FAIL",
        "counts": counts,
        "sha256": hashes,
        "schema_errors": errors[:20],
        "asin_hits_in_assistant": asin_hits,
        "frozen_benchmark_message_overlaps": overlaps,
        "template_overlap": template_overlap,
        "near_duplicate_test_train_count": len(near_duplicates),
        "near_duplicate_examples": near_duplicates[:5],
        "manifest_leakage_checks": manifest.get("leakage_checks", {}),
        "context_source": manifest.get("context_source"),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="P2-SFT CPU data preflight")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.data_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
