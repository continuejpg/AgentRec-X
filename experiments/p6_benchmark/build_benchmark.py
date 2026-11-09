"""Build the P6 Benchmark v0 protocol artifacts (CPU only)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from recommendation.api.app import ServiceSettings
from recommendation.catalog import MetadataIndex
from recommendation.evaluation.split import MIN_SEQUENCE_LENGTH

ROOT = Path("data/p6_benchmark_v0")
SEQUENCES = Path("data/processed/Sports_and_Outdoors_sequences.json")
MAX_DEV = 160
MAX_HOLDOUT = 40


def split_for_user(user_id: str) -> str:
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return "holdout" if int(digest, 16) % 5 == 0 else "dev"


def _classical_rows() -> tuple[list[dict], dict]:
    payload = json.loads(SEQUENCES.read_text(encoding="utf-8"))
    dev: list[dict] = []
    holdout: list[dict] = []
    by_split = Counter()
    seen_items: set[str] = set()
    for record in payload["sequences"]:
        items = list(record.get("item_ids") or ())
        asins = list(record.get("parent_asins") or ())
        times = list(record.get("unix_ms") or ())
        if len(items) < MIN_SEQUENCE_LENGTH or len(asins) != len(items) or len(times) != len(items):
            continue
        if 0 in items or any(not isinstance(item, int) for item in items):
            continue
        # Strict chronology: every observed interaction is ordered; the two targets are
        # the final two interactions, so no future event enters history.
        if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
            continue
        user_id = str(record["user_id"])
        split = split_for_user(user_id)
        if split == "dev" and len(dev) >= MAX_DEV:
            continue
        if split == "holdout" and len(holdout) >= MAX_HOLDOUT:
            continue
        histories = asins[:-2]
        target_validation = asins[-2]
        target_test = asins[-1]
        # Exclude cases where the target identity is already in history only after
        # recording an explicit overlap flag; the accepted protocol keeps the target
        # eligible by removing it from the seen mask, so overlap is not silently hidden.
        row = {
            "schema_version": "agentrecx.p6.classical_case.v0",
            "case_id": f"classical-{split}-{len(dev) + len(holdout):05d}",
            "split": split,
            "user_id": user_id,
            "user_int_id": int(record["user_int_id"]),
            "history_parent_asins": histories,
            "history_item_ids": items[:-2],
            "validation_parent_asin": target_validation,
            "validation_item_id": items[-2],
            "test_parent_asin": target_test,
            "test_item_id": items[-1],
            "validation_time_ms": times[-2],
            "test_time_ms": times[-1],
            "history_len": len(histories),
            "target_in_history": target_test in histories or target_validation in histories,
        }
        (dev if split == "dev" else holdout).append(row)
        by_split[split] += 1
        seen_items.update(asins)
        if len(dev) >= MAX_DEV and len(holdout) >= MAX_HOLDOUT:
            break
    dev_targets = {row["test_parent_asin"] for row in dev}
    holdout_targets = {row["test_parent_asin"] for row in holdout}
    report = {
        "artifact": str(SEQUENCES),
        "num_users_total": payload["num_users"],
        "num_items": payload["num_items"],
        "eligible_cases": by_split,
        "dev": len(dev),
        "holdout": len(holdout),
        "user_overlap": len({row["user_id"] for row in dev} & {row["user_id"] for row in holdout}),
        "dev_test_targets": len(dev_targets),
        "holdout_test_targets": len(holdout_targets),
        "test_target_overlap": len(dev_targets & holdout_targets),
        "history_target_overlap_cases": sum(row["target_in_history"] for row in dev + holdout),
    }
    return dev + holdout, report


def _catalogue_records() -> list:
    return list(MetadataIndex.load(ServiceSettings.from_env().catalog_path).records.values())


def _dynamic_tasks(records: list) -> tuple[list[dict], list[dict]]:
    def pick(keyword: str, offset: int = 0) -> list[str]:
        rows = [
            str(record.parent_asin)
            for record in records
            if keyword in str(getattr(record, "title", "") or "").casefold()
            and str(getattr(record, "price_text", "") or "")
        ]
        rows.sort()
        return rows[offset:offset + 3]

    dev_specs = [
        ("budget_update", "dev", "I need a water filter under $40 for camping.", ["update price_max to 60", "remove budget"]),
        ("preference_retract", "dev", "I need a water filter and prefer lighter options.", ["retract lighter preference"]),
        ("goal_switch", "dev", "I need a water filter for hiking.", ["switch goal to water bottle"]),
        ("role_constraint", "dev", "I need a replacement filter cartridge for my water filter.", ["request replacement accessory role"]),
        ("normal_termination", "dev", "Show me camping cookware under $50.", ["finalize if eligible candidates exist"]),
        ("insufficient_candidates", "dev", "I need a water filter under $1.", ["do not loosen hard constraint"]),
    ]
    holdout_specs = [
        ("holdout_budget_update", "holdout", "I need a sleeping pad under $60 for backpacking.", ["update price_max to 80"]),
        ("holdout_preference_retract", "holdout", "I need a lantern and prefer lighter options.", ["retract weight preference"]),
        ("holdout_goal_switch", "holdout", "I need a water bottle for hiking.", ["switch goal to water filter"]),
        ("holdout_insufficient", "holdout", "I need a cookware set under $2.", ["preserve hard constraint and report UNKNOWN"]),
    ]
    tasks = []
    for template_id, split, query, turns in dev_specs + holdout_specs:
        keyword = next((word for word in ("water filter", "cookware", "sleeping pad", "lantern", "water bottle", "stove") if word in query.casefold()), "water filter")
        tasks.append({
            "schema_version": "agentrecx.p6.dynamic_task.v0",
            "task_id": f"dynamic-{split}-{template_id}",
            "split": split,
            "synthetic_user_request": True,
            "template_id": template_id,
            "source": {"catalogue": "amazon_reviews_2023:meta_categories", "not_frozen": True},
            "initial_request": query,
            "turns": [{"message": query}] + [{"message": turn} for turn in turns],
            "catalogue_seed_asins": pick(keyword),
            "checks": {
                "must_preserve_hard_constraints": True,
                "must_not_present_role_mismatch": True,
                "must_report_unknown_when_no_eligible": "under $1" in query or "under $2" in query,
                "forbidden_presentation": ["hard-constraint violator", "role MISMATCH as match"],
            },
            "max_steps": 6,
            "max_tool_calls": 4,
        })
    return [t for t in tasks if t["split"] == "dev"], [t for t in tasks if t["split"] == "holdout"]


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows, report = _classical_rows()
    for split in ("dev", "holdout"):
        part = [row for row in rows if row["split"] == split]
        (ROOT / f"classical_{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in part), encoding="utf-8"
        )
    records = _catalogue_records()
    dev_tasks, holdout_tasks = _dynamic_tasks(records)
    for split, tasks in (("dev", dev_tasks), ("holdout", holdout_tasks)):
        (ROOT / f"dynamic_{split}.jsonl").write_text(
            "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), encoding="utf-8"
        )
    protocol = {
        "schema_version": "agentrecx.p6.benchmark_protocol.v0",
        "status": "DEV_ONLY",
        "classical": {
            "protocol": "temporal_leave_two_out",
            "candidate_set": "full catalogue excluding seen history (target re-admitted)",
            "split_policy": "user_id hash: 80% dev / 20% holdout; user-disjoint",
            "metrics": ["Recall@10", "Recall@50", "NDCG@10", "MRR@10", "CatalogueCoverage"],
            "leakage_rule": "history is items[:-2]; targets are items[-2], items[-1]; chronological timestamps required",
            "target_item_overlap_is_reported": True,
        },
        "dynamic": {
            "task_source": "synthetic shopping requests over real catalogue",
            "split_policy": "template_id disjoint dev/holdout",
            "metrics": ["VerifiedTaskSuccess", "HardConstraintViolationRate", "ProductRoleCorrectness", "PreferenceUpdateConsistency", "EligibleCandidateCoverage", "ToolCallsPerTask", "UnknownRate"],
            "no_frozen_tasks": True,
        },
        "arms": {"traditional_recommendation": "Serving V1", "rule_agent": "default Rule Agent", "qwen_base_planner": "P3D Base /plan", "planner_sft_v0": "P4B Adapter /plan"},
        "fixed_controls": {"catalogue": "Sports_and_Outdoors", "candidate_sources": ["history", "catalog_search"], "max_steps": 6, "max_tool_calls": 4, "eligibility": "unchanged", "k": 5},
        "holdout_policy": "holdout is never used for SFT, prompt tuning, or model selection",
    }
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["dynamic"] = {"dev": len(dev_tasks), "holdout": len(holdout_tasks)}
    (ROOT / "leakage_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
