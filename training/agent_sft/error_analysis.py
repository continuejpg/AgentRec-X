"""Per-operation error analysis for existing Base/SFT/Rule parser predictions.

This is a read-only analysis of the frozen 200-sample Parser Test; it does not run any
model and does not modify predictions or Gold.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import parse_and_validate

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "training/agent_sft/data/test.jsonl"
RUNS = {
    "rule": ROOT / "training/agent_sft/eval/rule_local/predictions.jsonl",
    "base": ROOT / "training/agent_sft/eval/base/predictions.jsonl",
    "sft": ROOT / "training/agent_sft/eval/sft/predictions.jsonl",
}
OUT = ROOT / "training/agent_sft/reports"
OPS = ["set_goal", "update_goal", "add_preference", "update_preference", "retract_preference", "set_constraint", "update_constraint", "retract_constraint", "resolve_reference", "clarify", "noop", "abstain"]


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ops(payload: Any) -> tuple[tuple[str, str, str, str, str], ...]:
    if not isinstance(payload, dict):
        return ()
    result = []
    for op in payload.get("operations", []) if isinstance(payload.get("operations"), list) else []:
        if isinstance(op, dict):
            result.append((str(op.get("op", "")), str(op.get("attribute", "")), str(op.get("value", "")), str(op.get("direction", "")), str(op.get("scope", ""))))
    return tuple(result)


def type_confusion(gold: tuple[tuple[str, ...], ...], pred: tuple[tuple[str, ...], ...]) -> dict[str, int]:
    key = "none" if not pred else pred[0][0]
    gold_type = "none" if not gold else gold[0][0]
    return {"gold_type": gold_type, "pred_type": key, "count": 1}


def main() -> int:
    test = {row["sample_id"]: row for row in load(DATA)}
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for provider, path in RUNS.items():
        preds = {row["sample_id"]: row for row in load(path)}
        counts = {kind: {"gold": 0, "pred": 0, "tp": 0} for kind in OPS}
        errors: Counter[str] = Counter()
        confusion: Counter[tuple[str, str]] = Counter()
        exact = valid = 0
        cases = []
        for sample_id, gold_row in test.items():
            p = preds.get(sample_id, {})
            raw = str(p.get("prediction_raw", ""))
            pred, validation_errors = parse_and_validate(raw, gold_row["user_message"])
            gold = json.loads(gold_row["assistant"])
            gops, pops = ops(gold), ops(pred)
            gold_types = {op[0] for op in gops}
            pred_types = {op[0] for op in pops} if not validation_errors else set()
            for kind in OPS:
                counts[kind]["gold"] += int(kind in gold_types)
                counts[kind]["pred"] += int(kind in pred_types)
                counts[kind]["tp"] += int(kind in gold_types and kind in pred_types)
            if not validation_errors:
                valid += 1
            if not validation_errors and pops == gops:
                exact += 1
            if not validation_errors and pops != gops:
                if not pops:
                    errors["missing_operations"] += 1
                elif not gops:
                    errors["extra_operations"] += 1
                else:
                    g_first, p_first = gops[0], pops[0]
                    if g_first[0] != p_first[0]:
                        errors["wrong_operation_type"] += 1
                    elif g_first[1] != p_first[1]:
                        errors["wrong_attribute"] += 1
                    elif str(g_first[2]).casefold() != str(p_first[2]).casefold():
                        errors["wrong_value"] += 1
                    elif g_first[3] != p_first[3]:
                        errors["wrong_direction"] += 1
                    elif g_first[4] != p_first[4]:
                        errors["wrong_scope"] += 1
                    elif g_first != p_first:
                        errors["evidence_or_extra_fields"] += 1
                    else:
                        errors["multi_operation_or_order"] += 1
                if len(cases) < 12:
                    cases.append({"sample_id": sample_id, "scenario": gold_row["scenario"], "user_message": gold_row["user_message"], "gold": gold_row["assistant"], "prediction_raw": raw, "validation_errors": validation_errors})
            confusion[(("none" if not gops else gops[0][0]), ("none" if not pops else pops[0][0]))] += 1
        per_op = {}
        for kind, c in counts.items():
            precision = c["tp"] / c["pred"] if c["pred"] else 0.0
            recall = c["tp"] / c["gold"] if c["gold"] else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_op[kind] = {**c, "precision": precision, "recall": recall, "f1": f1}
        supported = [kind for kind in OPS if per_op[kind]["gold"] > 0]
        macro_all = sum(per_op[kind]["f1"] for kind in OPS) / len(OPS)
        macro_supported = sum(per_op[kind]["f1"] for kind in supported) / max(1, len(supported))
        summary[provider] = {
            "samples": len(test),
            "valid_json": valid,
            "exact_match": exact / max(1, len(test)),
            "macro_f1_all_12": macro_all,
            "macro_f1_supported": macro_supported,
            "supported_operation_types": supported,
            "error_categories": dict(errors),
            "per_operation": per_op,
        }
        for kind in OPS:
            all_rows.append({"provider": provider, "operation": kind, **per_op[kind]})
        for (gold_type, pred_type), count in sorted(confusion.items()):
            all_rows.append({"provider": provider, "operation": "__confusion__", "gold_operation": gold_type, "predicted_operation": pred_type, "count": count})
        (OUT / f"P2_SFT_ERROR_CASES_{provider}.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "P2_SFT_OPERATION_METRICS.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["provider", "operation", "gold", "pred", "tp", "precision", "recall", "f1", "gold_operation", "predicted_operation", "count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT / "P2_SFT_ERROR_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({provider: {k: v for k, v in data.items() if k not in {"per_operation", "supported_operation_types"}} for provider, data in summary.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
