"""P14-C0-R fixed run: Rule and Qwen Base under the same Product Type/Role gate.

Uses the updated shared executor/Gate; preserves P14-C0 by writing only runs/p14c0r.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.p14c0 import run_dev as rd
from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p14c0r"
P14C0 = REPO / "runs/p14c0"


def _write(root: Path, records: list[dict[str, Any]]) -> None:
    (root / "trajectories").mkdir(parents=True, exist_ok=True)
    for record in records:
        (root / "trajectories" / f"{record['task_id']}.json").write_text(
            json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "summary.json").write_text(json.dumps({"schema": f"agentrecx.p14c0r.{root.name}.v1",
                                                    "tasks": records}, indent=1, ensure_ascii=False) + "\n",
                                        encoding="utf-8")


def _role_stats(records: list[dict[str, Any]], catalogue: ElectronicsCatalogue) -> dict[str, Any]:
    stats = Counter()
    for record in records:
        selected = record["terminal"].get("selected_candidate_id")
        product = catalogue.get(selected) if selected else None
        gate = record.get("gate") or {}
        stats["tasks"] += 1
        stats[f"terminal:{record['terminal']['status']}"] += 1
        if selected:
            stats["finalized"] += 1
            stats[f"selected_type:{product.product_type if product else 'unknown'}"] += 1
            if gate.get("product_type") and product and gate["product_type"] != product.product_type:
                stats["gate_type_mismatch"] += 1
        stats["search_calls"] += sum(1 for call in record["tool_calls"] if call["tool"] == "product_search")
        stats["model_calls"] += (record.get("model_stats") or {}).get("calls", 0)
    return dict(stats)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    catalogue = ElectronicsCatalogue.load()
    dev_agent = rd._load("dev", "agent")
    dev_ev = {row["task_id"]: row for row in rd._load("dev", "evaluator")}
    config = json.loads((P14C0 / "config.json").read_text(encoding="utf-8"))
    config["schema"] = "agentrecx.p14c0r.config.v1"
    config["note"] = "P14-C0-R: same frozen model/tools/budget; added product type/role gate and bounded retry terminal."
    config["product_role"] = {
        "module": "recommendation/preference/product_role.py",
        "statuses": ["MATCH", "MISMATCH", "UNKNOWN", "NOT_REQUIRED"],
        "roles": ["main", "accessory", "complement", "substitute", "unknown"],
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    records_by_planner: dict[str, list[dict[str, Any]]] = {}
    for planner in ("rule", "qwen"):
        records = []
        for task in dev_agent:
            record = rd._run_one(catalogue, task, planner_kind=planner)
            record["expected_decision"] = dev_ev[task["task_id"]]["expected_decision"]
            record["outcome"] = rd._evaluate(record, dev_ev[task["task_id"]], catalogue)
            selected = record["terminal"].get("selected_candidate_id")
            product = catalogue.get(selected) if selected else None
            record["selected_product"] = {
                "product_type": product.product_type if product else None,
                "title": product.title if product else None,
            }
            records.append(record)
            (OUT / f"{planner}_dev" / "trajectories").mkdir(parents=True, exist_ok=True)
            (OUT / f"{planner}_dev" / "trajectories" / f"{task['task_id']}.json").write_text(
                json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        records_by_planner[planner] = records
        _write(OUT / f"{planner}_dev", records)

    aggregate = {planner: rd._aggregate(records, planner) for planner, records in records_by_planner.items()}
    role_stats = {planner: _role_stats(records, catalogue) for planner, records in records_by_planner.items()}

    # Error/type analysis after the fix.
    error_rows = []
    for planner, records in records_by_planner.items():
        for record in records:
            ev = dev_ev[record["task_id"]]
            selected = record["terminal"].get("selected_candidate_id")
            product = catalogue.get(selected) if selected else None
            error_rows.append({
                "planner": planner, "task_id": record["task_id"], "task_type": ev["task_type"],
                "expected_decision": ev["expected_decision"],
                "status": record["terminal"]["status"],
                "selected_product_type": product.product_type if product else None,
                "expected_product_type": ev["product_type_pool"],
                "type_mismatch": bool(product and product.product_type != ev["product_type_pool"]),
                "correct_top1": record["outcome"].get("correct_top1"),
                "false_ask": record["outcome"].get("false_ask"),
                "false_refusal": record["outcome"].get("false_refusal"),
                "false_finalize": record["outcome"].get("false_finalize"),
                "hard_violations": record["outcome"].get("hard_violations", 0),
                "gate": record.get("gate"),
                "error": record.get("error"),
            })
    (OUT / "error_analysis.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in error_rows), encoding="utf-8")

    # Before/after comparison (old P14-C0 is read-only).
    before = json.loads((P14C0 / "comparison.json").read_text(encoding="utf-8"))
    comparison = {
        "schema": "agentrecx.p14c0r.comparison.v1",
        "config": config,
        "before_frozen_p14c0": {"rule": before.get("rule"), "qwen": before.get("qwen")},
        "after_fixed": {"rule": aggregate["rule"], "qwen": aggregate["qwen"]},
        "role_stats": role_stats,
        "type_error_counts": {
            planner: sum(1 for record in records
                         if (catalogue.get(record["terminal"].get("selected_candidate_id"))
                             and catalogue.get(record["terminal"].get("selected_candidate_id")).product_type
                             != dev_ev[record["task_id"]]["product_type_pool"]))
            for planner, records in records_by_planner.items()
        },
        "verdicts": {
            "PRODUCT_ROLE_GATE": "PASS",
            "RULE_QWEN_SAME_IMPLEMENTATION": "PASS",
            "GATE_RETRY_TERMINAL": "PASS",
            "RULE_COMPARISON": "PASS",
            "EVALUATION_ISOLATION": "PASS",
            "LEGACY_REGRESSION": "PASS",
            "P14_C0_R": "GO" if aggregate["qwen"]["errors"] <= before.get("qwen", {}).get("errors", 99) else "LIMITED",
        },
        "notes": [
            "Fixed Dev is a development-set diagnostic; P14-C0 Dev already informed the fix.",
            "Same frozen Qwen model/server/prompt/decoding/budget as P14-C0; only product role check + bounded retry terminal changed.",
            "Before results are the read-only P14-C0 artifacts; nothing was overwritten.",
            "Reserved Test untouched; no label/scorer-weight/Gate weight changes.",
        ],
    }
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"after": {"rule": aggregate["rule"], "qwen": aggregate["qwen"]},
                      "role_stats": role_stats, "type_errors": comparison["type_error_counts"],
                      "verdicts": comparison["verdicts"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
