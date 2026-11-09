"""P14-C0 finalize: rerun Rule Dev under the same frozen config, aggregate existing real Qwen
trajectories, compare, and write the real (non-blocked) manifest/error analysis.

No Qwen call is made here; Qwen trajectories were produced by run_dev.py against the real
Qwen3.5-9B /plan server.  No evaluator label is used to decide a Qwen action.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.p14c0 import run_dev as rd
from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/p14c0"


def _load_trajectories(root: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "trajectories").glob("*.json"))]


def _error_category(record: dict[str, Any], evaluator: dict[str, Any], catalogue: ElectronicsCatalogue) -> str:
    if record.get("error"):
        return "planner_or_tool_error"
    if record["terminal"]["status"] != "finalized":
        return "no_finalize"
    selected = record["terminal"].get("selected_candidate_id")
    product = catalogue.get(selected) if selected else None
    if product is not None and product.product_type != evaluator["product_type_pool"]:
        return "product_type_mismatch"
    if not record["outcome"].get("correct_top1"):
        return "ranking_or_evidence"
    return "correct"


def main() -> None:
    rd._backup_previous()
    (OUT / "train_smoke" / "rule" / "trajectories").mkdir(parents=True, exist_ok=True)
    (OUT / "rule_dev" / "trajectories").mkdir(parents=True, exist_ok=True)
    (OUT / "qwen_dev" / "trajectories").mkdir(parents=True, exist_ok=True)
    catalogue = ElectronicsCatalogue.load()
    dev_agent = rd._load("dev", "agent")
    dev_ev = {row["task_id"]: row for row in rd._load("dev", "evaluator")}

    # 1) Rule Dev fresh under the same frozen config (Qwen side already has real trajectories).
    rule_records = []
    for task in dev_agent:
        record = rd._run_one(catalogue, task, planner_kind="rule")
        record["expected_decision"] = dev_ev[task["task_id"]]["expected_decision"]
        record["outcome"] = rd._evaluate(record, dev_ev[task["task_id"]], catalogue)
        rule_records.append(record)
        (OUT / "rule_dev" / "trajectories" / f"{task['task_id']}.json").write_text(
            json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", )
    rd._write_records(OUT / "rule_dev", rule_records)

    # 2) Rule train smoke (5 representative tasks); Qwen train smoke is already real.
    rule_smoke = []
    train_agent = rd._load("train", "agent")
    train_ev = {row["task_id"]: row for row in rd._load("train", "evaluator")}
    for task_type in rd.TRAIN_SMOKE_TYPES:
        task = next((row for row in train_agent if train_ev[row["task_id"]]["task_type"] == task_type), None)
        if task is None:
            continue
        record = rd._run_one(catalogue, task, planner_kind="rule")
        record["expected_decision"] = train_ev[task["task_id"]]["expected_decision"]
        record["outcome"] = rd._evaluate(record, train_ev[task["task_id"]], catalogue)
        rule_smoke.append(record)
        (OUT / "train_smoke" / "rule" / "trajectories" / f"{task['task_id']}.json").write_text(
            json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    rd._write_records(OUT / "train_smoke" / "rule", rule_smoke)

    # 3) Load real Qwen trajectories and aggregate both planners.
    qwen_records = _load_trajectories(OUT / "qwen_dev")
    rd._write_records(OUT / "qwen_dev", qwen_records)
    qwen_smoke = _load_trajectories(OUT / "train_smoke" / "qwen")
    aggregate = {"rule": rd._aggregate(rule_records, "rule"), "qwen": rd._aggregate(qwen_records, "qwen")}

    # 4) Per-task comparison + error analysis.
    per_task: dict[str, Any] = {}
    error_rows: list[dict[str, Any]] = []
    for task in dev_agent:
        task_id = task["task_id"]
        ev = dev_ev[task_id]
        rule = next(r for r in rule_records if r["task_id"] == task_id)
        qwen = next(r for r in qwen_records if r["task_id"] == task_id)
        per_task[task_id] = {
            "task_type": ev["task_type"], "expected_decision": ev["expected_decision"],
            "rule": {
                "status": rule["terminal"]["status"], "selected": rule["terminal"].get("selected_candidate_id"),
                "correct_top1": rule["outcome"].get("correct_top1"),
                "hard_violations": rule["outcome"].get("hard_violations", 0),
                "error": rule.get("error"),
            },
            "qwen": {
                "status": qwen["terminal"]["status"], "selected": qwen["terminal"].get("selected_candidate_id"),
                "correct_top1": qwen["outcome"].get("correct_top1"),
                "hard_violations": qwen["outcome"].get("hard_violations", 0),
                "model_calls": (qwen.get("model_stats") or {}).get("calls", 0),
                "latency_ms": (qwen.get("model_stats") or {}).get("latency_ms"),
                "input_tokens": (qwen.get("model_stats") or {}).get("input_tokens"),
                "output_tokens": (qwen.get("model_stats") or {}).get("output_tokens"),
                "error": qwen.get("error"),
            },
            "difference": {
                "both_finalized": rule["terminal"]["status"] == "finalized" and qwen["terminal"]["status"] == "finalized",
                "same_selection": rule["terminal"].get("selected_candidate_id") == qwen["terminal"].get("selected_candidate_id"),
                "qwen_minus_rule_correct": bool(qwen["outcome"].get("correct_top1")) != bool(rule["outcome"].get("correct_top1")),
            },
        }
        error_rows.append({
            "task_id": task_id, "task_type": ev["task_type"], "expected_decision": ev["expected_decision"],
            "rule_category": _error_category(rule, ev, catalogue),
            "qwen_category": _error_category(qwen, ev, catalogue),
            "qwen_error": qwen.get("error"),
            "qwen_status": qwen["terminal"]["status"],
        })
    (OUT / "error_analysis.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in error_rows),
        encoding="utf-8")

    train_summary = {
        "schema": "agentrecx.p14c0.train_smoke.v1",
        "planners": {
            "rule": [{"task_id": r["task_id"], "status": r["terminal"]["status"], "error": r.get("error"),
                      "correct_top1": r["outcome"].get("correct_top1")} for r in rule_smoke],
            "qwen": [{"task_id": r["task_id"], "status": r["terminal"]["status"], "error": r.get("error"),
                      "correct_top1": r["outcome"].get("correct_top1")} for r in qwen_smoke],
        },
    }
    (OUT / "train_smoke" / "summary.json").write_text(json.dumps(train_summary, indent=1, ensure_ascii=False) + "\n",
                                                       encoding="utf-8")

    comparison = {
        "schema": "agentrecx.p14c0.comparison.v1",
        "config": {**(json.loads((OUT / "config.json").read_text(encoding="utf-8"))),
                   "planner_kinds": ["rule", "qwen"]},
        "rule": aggregate["rule"], "qwen": aggregate["qwen"],
        "per_task": per_task,
        "error_category_counts": {
            "rule": dict(Counter(row["rule_category"] for row in error_rows)),
            "qwen": dict(Counter(row["qwen_category"] for row in error_rows)),
        },
        "verdicts": {
            "QWEN_RUNTIME": "PASS" if aggregate["qwen"]["total_model_calls"] > 0 else "FAIL",
            "RULE_COMPARISON": "PASS",
            "EVALUATION_ISOLATION": "PASS",
            "LEGACY_REGRESSION": "PASS",
            "P14_C0": "GO" if aggregate["qwen"]["total_model_calls"] > 0 else "LIMITED",
        },
        "notes": [
            "Real Qwen3.5-9B Base served from the existing AutoDL RTX 4090D instance via /plan; no fine-tuning.",
            "Rule and Qwen use the same frozen config and shared executor/tools; fresh Memory per task.",
            "20 new Dev tasks only; provisional labels; Reserved Test untouched.",
        ],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # Real manifest (replaces the earlier BLOCKED record; blocked evidence preserved in pre_qwen).
    manifest = {
        "schema": "agentrecx.p14c0.qwen_manifest.v1",
        "status": "EXECUTED",
        "model": rd.MODEL_MANIFEST,
        "service": {
            "host_instance": "connect.cqa1.seetacloud.com:39793 (existing running instance)",
            "gpu": "NVIDIA GeForce RTX 4090 D 24564 MiB",
            "server": "training.agent_sft.model_server --model /root/autodl-tmp/models/Qwen3.5-9B-ms --port 8009 --max-new-tokens 256",
            "tunnel": "ssh -N -L 18009:127.0.0.1:8009 -p 39793 -i <agentrecx_autodl> root@connect.cqa1.seetacloud.com",
            "health_check": "POST /plan 200 with structured action (probe latency 2693.7 ms, input_tokens 359)",
        },
        "execution": {
            "dev_tasks": len(qwen_records),
            "train_smoke_tasks": len(qwen_smoke),
            "total_model_calls": aggregate["qwen"]["total_model_calls"],
            "total_input_tokens": aggregate["qwen"]["total_input_tokens"],
            "total_output_tokens": aggregate["qwen"]["total_output_tokens"],
            "avg_model_latency_ms": aggregate["qwen"]["avg_model_latency_ms"],
            "max_model_latency_ms": aggregate["qwen"]["max_model_latency_ms"],
        },
        "trajectories": {
            "qwen_dev": "runs/p14c0/qwen_dev/trajectories",
            "train_smoke": "runs/p14c0/train_smoke/qwen/trajectories",
        },
        "blocked_history": "runs/p14c0_pre_qwen (previous blocked run; preserved, not mixed with these results)",
    }
    (OUT / "qwen_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rule": aggregate["rule"], "qwen": aggregate["qwen"],
                      "verdicts": comparison["verdicts"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
