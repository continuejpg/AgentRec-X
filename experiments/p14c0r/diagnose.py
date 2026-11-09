"""P14-C0-R diagnostics: attribute the 9 Rule / 7 Qwen type errors from P14-C0 (read-only).

For each wrong recommend selection we ask whether the BM25-hit union from the original
trajectory contained a verified candidate of the *requested product type*:

* none  -> recall / reachability problem (candidate not in examined hit union);
* yes   -> scorer-ranked or Gate-allowed wrong type (constraint missing in the frozen run).

Original P14-C0 artifacts are not modified.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.p14b1.diagnose import verified_ids
from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[2]
P14C0 = REPO / "runs/p14c0"
OUT = REPO / "runs/p14c0r/diagnostics"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    catalogue = ElectronicsCatalogue.load()
    evaluator = {row["task_id"]: row for row in
                 (json.loads(line) for line in (REPO / "data/p14e2/benchmark/dev.evaluator.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    summary: dict[str, Any] = {"schema": "agentrecx.p14c0r.type_error_diagnosis.v1", "planners": {}}
    for planner in ("rule", "qwen"):
        root = P14C0 / f"{planner}_dev" / "trajectories"
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]
        rows = []
        for record in records:
            ev = evaluator[record["task_id"]]
            if ev["expected_decision"] != "recommend":
                continue
            selected = record["terminal"].get("selected_candidate_id")
            product = catalogue.get(selected) if selected else None
            if record["outcome"].get("correct_top1"):
                continue
            hit_ids: set[str] = set()
            search_calls = 0
            for call in record["tool_calls"]:
                if call["tool"] == "product_search":
                    search_calls += 1
                    hit_ids.update(call["output"].get("hit_ids") or [])
            verified_type = {
                item_id for item_id in verified_ids(ev, catalogue)
                if catalogue.get(item_id) is not None
                and catalogue.get(item_id).product_type == ev["product_type_pool"]
            }
            in_hits = verified_type & hit_ids
            if not in_hits:
                category = "recall_no_correct_type_verified_candidate_in_hits"
            elif product is not None and product.product_type != ev["product_type_pool"]:
                category = "ranking_or_gate_allowed_wrong_type"
            else:
                category = "ranking_or_evidence_other"
            rows.append({
                "task_id": record["task_id"], "task_type": ev["task_type"],
                "expected_product_type": ev["product_type_pool"],
                "selected_product_type": product.product_type if product else None,
                "selected_role": None,
                "correct_type_verified_in_hit_union": len(in_hits),
                "search_calls": search_calls,
                "final_selected": selected,
                "category": category,
                "planner": planner,
            })
        summary["planners"][planner] = {"wrong_recommend": len(rows),
                                        "categories": dict(Counter(row["category"] for row in rows)),
                                        "rows": rows}
    (OUT / "type_error_diagnosis.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for planner in ("rule", "qwen") for row in summary["planners"][planner]["rows"]),
        encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({planner: summary["planners"][planner]["categories"] for planner in summary["planners"]},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
