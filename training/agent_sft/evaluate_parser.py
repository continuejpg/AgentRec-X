"""Offline parser evaluation: Rule / Base Qwen3 / SFT adapter on independent test data.

Local CPU can run `--provider rule`; Base/SFT require GPU and the AutoDL dependencies.
All providers use the same prompt, max_new_tokens=256, non-thinking template and greedy
decoding.  Every sample prediction and error is written alongside the summary metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from .format import prompt_text
from .rule_parser import rule_text
from .schema import parse_and_validate

OPS = [
    "set_goal", "update_goal", "add_preference", "update_preference", "retract_preference",
    "set_constraint", "update_constraint", "retract_constraint", "resolve_reference",
    "clarify", "noop", "abstain",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _canonical_ops(payload: Any) -> tuple[tuple[str, str, str, str, str], ...]:
    if not isinstance(payload, dict) or not isinstance(payload.get("operations"), list):
        return ()
    result = []
    for op in payload["operations"]:
        if not isinstance(op, dict):
            continue
        result.append((
            str(op.get("op", "")),
            str(op.get("attribute", "")),
            str(op.get("value", "")),
            str(op.get("direction", "")),
            str(op.get("scope", "")),
        ))
    return tuple(result)


def _op_types(payload: Any) -> set[str]:
    return {str(op.get("op")) for op in (payload.get("operations", []) if isinstance(payload, dict) else []) if isinstance(op, dict)}


def _slot_counts(gold: Any, pred: Any) -> tuple[int, int, int]:
    gold_ops = _canonical_ops(gold)
    pred_ops = _canonical_ops(pred)
    gold_slots = [(op[1], op[2], op[3]) for op in gold_ops]
    pred_slots = [(op[1], op[2], op[3]) for op in pred_ops]
    matched = 0
    used = [False] * len(pred_slots)
    for slot in gold_slots:
        for index, candidate in enumerate(pred_slots):
            if not used[index] and candidate == slot:
                used[index] = True
                matched += 1
                break
    return matched, len(gold_slots), len(pred_slots)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def evaluate(rows: list[dict[str, Any]], predictor: Callable[[dict[str, Any]], tuple[str, float]]) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    valid_count = exact_count = parse_fail = unsupported = abstained = 0
    gold_type_counts: dict[str, int] = {kind: 0 for kind in OPS}
    pred_type_counts: dict[str, int] = {kind: 0 for kind in OPS}
    type_tp: dict[str, int] = {kind: 0 for kind in OPS}
    gold_slots = pred_slots = matched_slots = 0
    retract_total = retract_correct = 0
    for row in rows:
        raw, elapsed = predictor(row)
        latencies.append(elapsed * 1000.0)
        pred, errors = parse_and_validate(raw, str(row["user_message"]))
        valid = not errors
        if valid:
            valid_count += 1
        else:
            parse_fail += 1
            unsupported += 1
        gold_payload = json.loads(row["assistant"])
        gold_ops = _canonical_ops(gold_payload)
        pred_ops = _canonical_ops(pred)
        exact = bool(valid) and pred_ops == gold_ops
        if exact:
            exact_count += 1
        matched, gold_slot_count, pred_slot_count = _slot_counts(gold_payload, pred)
        gold_slots += gold_slot_count
        pred_slots += pred_slot_count
        matched_slots += matched
        gold_types = _op_types(gold_payload)
        pred_types = _op_types(pred) if valid else set()
        for kind in OPS:
            if kind in gold_types:
                gold_type_counts[kind] += 1
            if kind in pred_types:
                pred_type_counts[kind] += 1
            if kind in gold_types and kind in pred_types:
                type_tp[kind] += 1
        if gold_types == {"retract_preference"} or gold_types == {"retract_constraint"}:
            retract_total += 1
            if valid and pred_types == gold_types:
                retract_correct += 1
        if valid and pred_types <= {"abstain"} and gold_types != {"abstain"}:
            abstained += 1
        predictions.append({
            "sample_id": row["sample_id"],
            "split": row["split"],
            "scenario": row["scenario"],
            "user_message": row["user_message"],
            "gold": row["assistant"],
            "prediction_raw": raw,
            "prediction": json.dumps(pred, ensure_ascii=False, sort_keys=True) if pred is not None else "",
            "schema_valid": valid,
            "errors": errors,
            "operation_exact_match": exact,
            "latency_ms": elapsed * 1000.0,
        })
    micro_precision = sum(type_tp.values()) / max(1, sum(pred_type_counts.values()))
    micro_recall = sum(type_tp.values()) / max(1, sum(gold_type_counts.values()))
    micro_f1 = 2 * micro_precision * micro_recall / max(1e-12, micro_precision + micro_recall)
    per_type_f1: dict[str, float] = {}
    for kind in OPS:
        precision = type_tp[kind] / max(1, pred_type_counts[kind])
        recall = type_tp[kind] / max(1, gold_type_counts[kind])
        per_type_f1[kind] = 2 * precision * recall / max(1e-12, precision + recall)
    macro_f1 = sum(per_type_f1.values()) / len(OPS)
    metrics = {
        "samples": len(rows),
        "schema_valid_rate": valid_count / max(1, len(rows)),
        "operation_exact_match": exact_count / max(1, len(rows)),
        "operation_macro_f1": macro_f1,
        "operation_micro_f1": micro_f1,
        "slot_accuracy": matched_slots / max(1, gold_slots),
        "retraction_accuracy": retract_correct / max(1, retract_total),
        "unsupported_proposal_rate": unsupported / max(1, len(rows)),
        "failure_or_abstention_rate": (parse_fail + abstained) / max(1, len(rows)),
        "p50_latency_ms": _percentile(latencies, 0.5),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "retraction_samples": retract_total,
    }
    return {"metrics": metrics, "predictions": predictions, "per_type_f1": per_type_f1}


def _rule_predictor(row: dict[str, Any]) -> tuple[str, float]:
    started = time.perf_counter()
    raw = rule_text(str(row["user_message"]))
    return raw, time.perf_counter() - started


def _model_predictor(model_name: str, adapter: str | None, max_new_tokens: int) -> Callable[[dict[str, Any]], tuple[str, float]]:
    import torch
    from .modeling import MODEL_ID, load_text_model, load_tokenizer

    if model_name != MODEL_ID and "Qwen3.5-9B" not in model_name:
        raise SystemExit(f"this evaluator is Qwen3.5-only; expected {MODEL_ID} or a local Qwen3.5-9B path, got {model_name}")
    tokenizer = load_tokenizer(model_name, trust_remote_code=True)
    model = load_text_model(model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def predict(row: dict[str, Any]) -> tuple[str, float]:
        prompt = prompt_text(tokenizer, row)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        elapsed = time.perf_counter() - started
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), elapsed

    return predict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate preference-operation parser")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--provider", choices=("rule", "base", "sft"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = load_jsonl(args.data)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    if args.provider == "rule":
        predictor = _rule_predictor
    else:
        if args.provider == "sft" and args.adapter is None:
            raise SystemExit("--adapter is required for --provider sft")
        predictor = _model_predictor(args.model, str(args.adapter) if args.adapter else None, args.max_new_tokens)
    result = evaluate(rows, predictor)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"provider": args.provider, "model": args.model if args.provider != "rule" else "rule", "adapter": str(args.adapter) if args.adapter else None, **result["metrics"]}
    (args.out_dir / "parser_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.out_dir / "parser_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader(); writer.writerow(metrics)
    with (args.out_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["predictions"][0]))
        writer.writeheader(); writer.writerows(result["predictions"])
    (args.out_dir / "predictions.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in result["predictions"]) + "\n", encoding="utf-8")
    (args.out_dir / "per_type_f1.json").write_text(json.dumps(result["per_type_f1"], indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
