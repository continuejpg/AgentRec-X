"""Merge Rule/Base/SFT parser evaluation outputs into one comparison table.

This script runs only after AutoDL has produced each provider's directory; it refuses
missing providers rather than fabricating empty rows or scores.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROVIDERS = ("rule", "base", "sft")
METRIC_KEYS = (
    "schema_valid_rate",
    "operation_exact_match",
    "operation_macro_f1",
    "slot_accuracy",
    "retraction_accuracy",
    "unsupported_proposal_rate",
    "failure_or_abstention_rate",
    "p50_latency_ms",
    "p95_latency_ms",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Rule/Base/SFT parser metrics")
    parser.add_argument("--rule-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--sft-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dirs = {"rule": args.rule_dir, "base": args.base_dir, "sft": args.sft_dir}
    rows = []
    for provider in PROVIDERS:
        path = dirs[provider] / "parser_metrics.json"
        if not path.is_file():
            raise SystemExit(f"missing {provider} parser metrics: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("provider") != provider:
            raise SystemExit(f"provider mismatch in {path}: {metrics.get('provider')!r}")
        rows.append({"provider": provider, **{key: metrics.get(key) for key in METRIC_KEYS}})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["provider", *METRIC_KEYS])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
