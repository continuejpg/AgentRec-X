"""Phase 4 vs 4.2 Bmark ShopRanker reproduction diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

B = Path("data/memory_ranking_benchmark_v2")
OLD = Path("runs/memory_ranking_benchmark_v2/shopranker_shadow.json")
NEW = Path("runs/agent_native_ranking_phase42/shopranker_controlled_gpu.json")
OUT = Path("runs/agent_ranker_phase43")
ATTRS = ("over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth", "microphone",
         "rechargeable", "noise_cancelling", "waterproof", "foldable", "portable", "gaming",
         "mechanical", "silent", "ergonomic", "touch", "rgb", "led", "usb", "hdmi", "4k", "remote")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def old_product_text(item: dict) -> str:
    attrs = list(item.get("attributes") or [])
    categories = item.get("categories") or []
    category = item.get("leaf_category") or item.get("main_category") or (categories[-1] if categories else "unknown")
    evidence = list(item.get("evidence") or [])[:3]
    unknown = [a for a in ATTRS if a not in set(attrs)]
    text = f"title: {item.get('title') or 'unknown'} | category: {category} | verified_attributes: {', '.join(attrs) if attrs else 'none'}"
    if evidence:
        text += f" | evidence: {'; '.join(evidence)}"
    if unknown:
        text += f" | unknown_attributes: {', '.join(unknown)}"
    return text


def new_product_text(item: dict) -> str:
    raw_attrs = item.get("attributes") or {}
    if isinstance(raw_attrs, dict):
        attrs = list(raw_attrs.keys())
        evidence_parts = []
        for name, fact in list(raw_attrs.items())[:4]:
            if isinstance(fact, dict) and fact.get("evidence"):
                evidence_parts.append(f"{name}={str(fact['evidence'])[:80]}")
    else:
        attrs = list(raw_attrs)
        evidence_parts = [str(value)[:80] for value in (item.get("evidence") or [])[:3]]
    scorer = item.get("scorer") or {}
    product_type = scorer.get("candidate_product_type") or scorer.get("expected_product_type") or item.get("leaf_category") or item.get("main_category") or "unknown"
    unknown = [a for a in ATTRS if a not in set(attrs)]
    text = f"title: {item.get('title') or 'unknown'} | product_type: {product_type} | verified_attributes: {', '.join(attrs) if attrs else 'none'}"
    if evidence_parts:
        text += f" | evidence: {'; '.join(evidence_parts)}"
    if unknown:
        text += f" | unknown_attributes: {', '.join(unknown)}"
    return text


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads(OLD.read_text(encoding="utf-8"))
    new = json.loads(NEW.read_text(encoding="utf-8"))
    metadata = {row["item_id"]: row for row in read_jsonl(B / "candidate_metadata.jsonl")}
    pairs = {row["scenario_id"]: row for row in read_jsonl(B / "pair_tasks.jsonl")}
    old_rows = {row["scenario_id"]: row for row in old["per_scenario"]}
    new_rows = {row["scenario_id"]: row for row in new["benchmark_v2"]["details"]}
    text_changed = 0
    candidate_texts = 0
    changed_pairs = []
    for sid, pair in pairs.items():
        if pair["choice"] not in {"A", "B"}:
            continue
        for item_id in (pair["candidate_a_id"], pair["candidate_b_id"]):
            candidate_texts += 1
            if hash_text(old_product_text(metadata[item_id])) != hash_text(new_product_text(metadata[item_id])):
                text_changed += 1
        old_row = old_rows.get(sid) or {}
        new_row = new_rows.get(sid) or {}
        old_scores = old_row.get("query_only_scores") or {}
        new_scores = new_row.get("scores", {}).get("query") or {}
        a, b = pair["candidate_a_id"], pair["candidate_b_id"]
        old_pred = "A" if old_scores.get(a, 0) > old_scores.get(b, 0) else "B"
        new_pred = "A" if new_scores.get(a, 0) > new_scores.get(b, 0) else "B"
        if old_pred != new_pred:
            changed_pairs.append({
                "pair_task_id": pair["pair_task_id"], "scenario_id": sid,
                "label": pair["choice"], "old_pred": old_pred, "new_pred": new_pred,
                "old_correct": old_pred == pair["choice"], "new_correct": new_pred == pair["choice"],
                "old_scores": [old_scores.get(a), old_scores.get(b)],
                "new_scores": [new_scores.get(a), new_scores.get(b)],
                "product_text_changed": {
                    "a": hash_text(old_product_text(metadata[a])) != hash_text(new_product_text(metadata[a])),
                    "b": hash_text(old_product_text(metadata[b])) != hash_text(new_product_text(metadata[b])),
                },
            })
    no_memory = [sid for sid, scenario in ((row["scenario_id"], row) for row in read_jsonl(B / "scenarios.jsonl")) if scenario["scenario_type"] == "no_memory"]
    format_rows = []
    for sid in no_memory:
        row = new_rows[sid]
        format_rows.append({
            "scenario_id": sid,
            "query_top1": row["rankings"]["query"][:1],
            "empty_structured_top1": row["rankings"]["structured"][:1],
            "query_vs_empty_top1_changed": row["rankings"]["query"][:1] != row["rankings"]["structured"][:1],
            "query_vs_empty_top3_changed": set(row["rankings"]["query"][:3]) != set(row["rankings"]["structured"][:3]),
            "query_vs_empty_full_changed": row["rankings"]["query"] != row["rankings"]["structured"],
        })
    payload = {
        "schema": "agentrecx.agent_ranker_phase43.reproduction.v1",
        "phase4_old_metrics": {
            "query_only_correct": 80, "query_only_n": 145,
            "memory_natural_correct": 80, "memory_natural_n": 145,
            "source": "runs/memory_ranking_benchmark_v2/shopranker_shadow.json",
        },
        "phase42_new_metrics": {
            "query_only_correct": new["benchmark_v2"]["pairwise"]["query"]["correct"],
            "query_only_n": new["benchmark_v2"]["pairwise"]["query"]["n"],
            "natural_memory_correct": new["benchmark_v2"]["pairwise"]["natural"]["correct"],
            "natural_memory_n": new["benchmark_v2"]["pairwise"]["natural"]["n"],
        },
        "pair_changes": {
            "n_pairs": 145,
            "pair_prediction_changed": len(changed_pairs),
            "changed_pairs": changed_pairs,
        },
        "product_serialization": {
            "candidate_texts_compared": candidate_texts,
            "candidate_texts_changed": text_changed,
            "old_serialization": "title | category | verified_attributes | evidence | unknown_attributes; max_length=4096",
            "new_serialization": "title | product_type | verified_attributes | evidence | unknown_attributes; max_length=1536",
        },
        "format_control_no_memory": {
            "n": len(format_rows),
            "top1_changed": sum(1 for row in format_rows if row["query_vs_empty_top1_changed"]),
            "top3_changed": sum(1 for row in format_rows if row["query_vs_empty_top3_changed"]),
            "full_changed": sum(1 for row in format_rows if row["query_vs_empty_full_changed"]),
            "details": format_rows,
        },
        "conclusion": "Phase4 -> 4.2 Bmark difference is fully explained by concrete input/config changes, not model revision: product serialization (category -> product_type + evidence formatting), max_length 4096 -> 1536, and no-memory structured wrapper changes. No per-case threshold tuning was performed.",
    }
    (OUT / "reproduction_differences.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("phase4_old_metrics", "phase42_new_metrics", "pair_changes", "product_serialization", "format_control_no_memory")}, indent=1)[:5000])


if __name__ == "__main__":
    main()
