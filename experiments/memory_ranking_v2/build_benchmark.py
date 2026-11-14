"""Build Memory Ranking Benchmark v2 and Silver pairwise annotation tasks.

Base queries are real ESCI test US small-version queries with at least one exact
catalog overlap product.  Candidate pools are fixed BGE Top-20 over the existing
76,918-item Electronics catalog.  Memory states are synthetic_preference_scenario
unless explicitly marked otherwise.

No human labels are fabricated: HUMAN_GOLD is zero.  Rule-derived labels are SILVER
or MECHANISTIC and are kept separate from EXTERNAL_RELEVANCE and UNJUDGED.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/memory_ranking_benchmark_v2"
POOLS = OUT / "bge_pools.jsonl"
METADATA = OUT / "candidate_metadata.jsonl"
QUERIES = OUT / "base_queries.jsonl"
ATTRS = ("over_ear", "on_ear", "in_ear", "wireless", "wired", "bluetooth", "microphone",
         "rechargeable", "noise_cancelling", "waterproof", "foldable", "portable", "gaming",
         "mechanical", "silent", "ergonomic", "touch", "rgb", "led", "usb", "hdmi", "4k", "remote")
QUERY_ATTR_HINTS = {
    "over_ear": ("over-ear", "over ear", "overear"),
    "on_ear": ("on-ear", "on ear", "onear"),
    "in_ear": ("in-ear", "in ear", "earbud", "inear"),
    "wireless": ("wireless", "bluetooth"),
    "wired": ("wired", "corded"),
    "microphone": ("microphone", "mic"),
    "rechargeable": ("rechargeable", "battery"),
    "noise_cancelling": ("noise cancel", "noise-cancel"),
    "waterproof": ("waterproof", "water resistant"),
    "portable": ("portable", "travel"),
    "gaming": ("gaming", "gamer"),
    "mechanical": ("mechanical",),
    "usb": ("usb",),
    "hdmi": ("hdmi",),
    "4k": ("4k", "uhd"),
    "remote": ("remote",),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _split_for(base_query_id: str) -> str:
    bucket = int(hashlib.md5(base_query_id.encode()).hexdigest(), 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "val"
    return "test"


def _query_mentions(query: str) -> set[str]:
    text = query.lower()
    return {attr for attr, needles in QUERY_ATTR_HINTS.items() if any(needle in text for needle in needles)}


def _choose_attr(items: list[dict[str, Any]], mentions: set[str]) -> str | None:
    counts = Counter(attr for item in items for attr in item.get("attributes") or [])
    valid = [attr for attr in ATTRS if 2 <= counts[attr] <= len(items) - 2]
    for attr in sorted(mentions, key=lambda value: -counts[value]):
        if counts[attr] >= 2:
            return attr
    if valid:
        return max(valid, key=lambda attr: counts[attr])
    # fallback: any attribute represented in the pool
    present = [attr for attr in ATTRS if counts[attr] >= 1]
    return max(present, key=lambda attr: counts[attr]) if present else None


def _pick_pair(
    items: list[dict[str, Any]], attr: str | None, *, prefer_external: bool = False
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Pick two real candidates with different evidence patterns."""
    if prefer_external:
        labeled = [item for item in items if item.get("esci_labels")]
        for left in labeled:
            for right in labeled:
                if left["item_id"] == right["item_id"]:
                    continue
                left_labels = {entry["esci_label"] for entry in left["esci_labels"]}
                right_labels = {entry["esci_label"] for entry in right["esci_labels"]}
                if "E" in left_labels and "I" in right_labels:
                    return left, right, "external_relevance"
    attrs_by_id = {item["item_id"]: set(item.get("attributes") or []) for item in items}
    if attr:
        with_attr = [item for item in items if attr in attrs_by_id[item["item_id"]]]
        without = [item for item in items if attr not in attrs_by_id[item["item_id"]]]
        if with_attr and without:
            # closest BGE ranks make the comparison less trivial
            with_attr = sorted(with_attr, key=lambda item: item.get("bge_rank", 999))
            without = sorted(without, key=lambda item: item.get("bge_rank", 999))
            left = with_attr[0]
            right = min(without, key=lambda item: abs(item.get("bge_rank", 999) - left.get("bge_rank", 999)))
            return left, right, "attribute_vs_unknown"
    ordered = sorted(items, key=lambda item: item.get("bge_rank", 999))
    return ordered[0], ordered[1], "adjacent_bge"


def _safe_alternative(item: dict[str, Any], memory_attr: str) -> bool:
    attrs = set(item.get("attributes") or [])
    if memory_attr == "over_ear":
        return bool(attrs & {"in_ear", "on_ear"})
    if memory_attr == "in_ear":
        return bool(attrs & {"over_ear", "on_ear"})
    if memory_attr == "on_ear":
        return bool(attrs & {"in_ear", "over_ear"})
    if memory_attr == "wireless":
        return "wired" in attrs
    if memory_attr == "wired":
        return "wireless" in attrs
    return False


def _memory_state(positive: list[str], negative: list[str]) -> dict[str, Any]:
    return {
        "positive": [{"value": value, "polarity": "positive", "kind": "synthetic_attribute"} for value in positive],
        "negative": [{"value": value, "polarity": "negative", "kind": "synthetic_attribute"} for value in negative],
        "source": "synthetic_preference_scenario",
    }


def _label_for(
    *,
    scenario_type: str,
    left: dict[str, Any],
    right: dict[str, Any],
    positive: list[str],
    negative: list[str],
    esci_used: bool,
    requires_clarification: bool,
) -> tuple[str | None, str, str]:
    """Return (choice, provenance, reason). Choice is A/B/BOTH/UNJUDGED/ASK."""
    if requires_clarification:
        return "ASK", "MECHANISTIC", "conflicting active preferences; clarification required"
    if esci_used and left.get("esci_label") and right.get("esci_label"):
        return "A" if left["esci_label"] in {"E", "S"} and right["esci_label"] == "I" else "B", "EXTERNAL_RELEVANCE", "ESCI exact relevance labels"
    left_attrs = set(left.get("attributes") or [])
    right_attrs = set(right.get("attributes") or [])
    if positive:
        attr = positive[-1]
        left_match = attr in left_attrs
        right_match = attr in right_attrs
        if left_match != right_match:
            winner = "A" if left_match else "B"
            return winner, "SILVER", "verified positive preference match vs unknown"
        return "UNJUDGED", "UNJUDGED", "both/neither candidates have verified positive evidence"
    if negative:
        attr = negative[-1]
        left_conflict = attr in left_attrs
        right_conflict = attr in right_attrs
        if left_conflict != right_conflict:
            safe = right if left_conflict else left
            if _safe_alternative(safe, attr):
                return ("B" if left_conflict else "A"), "SILVER", "negative preference conflict vs explicit safe alternative"
        return "UNJUDGED", "UNJUDGED", "negative preference has no explicit safe alternative"
    return "UNJUDGED", "UNJUDGED", "no memory / insufficient evidence"


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    queries = _read_jsonl(QUERIES)
    pools = {row["base_query_id"]: row for row in _read_jsonl(POOLS)}
    metadata = {row["item_id"]: row for row in _read_jsonl(METADATA)}
    scenarios: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    annotation: list[dict[str, Any]] = []
    type_counts = Counter()
    provenance_counts = Counter()
    base_query_ids = [row["base_query_id"] for row in queries]

    # 100 base queries x 3 scenarios = 300 pair tasks.
    extra_types = ["negative_memory", "unknown_evidence", "preference_update", "preference_withdrawal",
                   "preference_conflict", "query_memory_competition"]
    for index, query_row in enumerate(queries):
        pool = pools.get(query_row["base_query_id"])
        if not pool:
            continue
        item_ids = pool["candidate_pool_ids"]
        items = []
        for rank, item_id in enumerate(item_ids, start=1):
            item = dict(metadata.get(item_id) or {"item_id": item_id, "title": None, "attributes": [], "metadata_missing": True})
            item["bge_rank"] = rank
            item["bge_score"] = float(pool["bge_scores"][rank - 1])
            item["esci_labels"] = [entry for entry in query_row.get("esci_exact_catalog_labels") or [] if str(entry["product_id"]) == item_id]
            item["esci_label"] = item["esci_labels"][0]["esci_label"] if item["esci_labels"] else None
            items.append(item)
        mentions = _query_mentions(query_row["query"])
        attr = _choose_attr(items, mentions)
        counts = Counter(attr_value for item in items for attr_value in item.get("attributes") or [])
        alternatives = [value for value in ATTRS if value != attr and counts[value] >= 2]
        second = alternatives[0] if alternatives else attr
        split = _split_for(query_row["base_query_id"])
        scenario_type = "no_memory"
        extra = extra_types[index % len(extra_types)]
        scenario_specs = [("no_memory", [], []), ("positive_memory", [attr] if attr else [], [])]
        if extra == "negative_memory" and attr:
            scenario_specs.append((extra, [], [attr]))
        elif extra == "unknown_evidence" and attr:
            scenario_specs.append((extra, [attr], []))
        elif extra == "preference_update" and attr:
            scenario_specs.append((extra, [second], []))
        elif extra == "preference_withdrawal" and attr:
            scenario_specs.append((extra, [], []))
        elif extra == "preference_conflict" and attr:
            scenario_specs.append((extra, [attr, "in_ear" if attr != "in_ear" else "over_ear"], []))
        elif extra == "query_memory_competition" and attr:
            scenario_specs.append((extra, [second], []))
        else:
            scenario_specs.append(("positive_memory", [attr] if attr else [], []))
        for slot, (scenario_type, positive, negative) in enumerate(scenario_specs):
            scenario_id = f"{query_row['base_query_id']}:s{slot}"
            requires_clarification = scenario_type == "preference_conflict"
            if scenario_type == "preference_update" and attr:
                memory_before_positive = [attr]
            elif scenario_type == "preference_withdrawal" and attr:
                memory_before_positive = [attr]
            elif scenario_type == "preference_conflict" and attr:
                memory_before_positive = [attr]
            else:
                memory_before_positive = []
            left, right, pick_reason = _pick_pair(items, positive[-1] if positive else (negative[-1] if negative else None))
            if scenario_type == "no_memory" or scenario_type == "preference_withdrawal":
                left, right, pick_reason = _pick_pair(items, None, prefer_external=True)
            # Alternate candidate order so a model cannot learn "A is always the winner".
            if (index + slot) % 2 == 1:
                left, right = right, left
            esci_used = bool(left.get("esci_label") and right.get("esci_label"))
            choice, provenance, reason = _label_for(
                scenario_type=scenario_type, left=left, right=right,
                positive=positive, negative=negative, esci_used=esci_used,
                requires_clarification=requires_clarification,
            )
            type_counts[scenario_type] += 1
            provenance_counts[provenance] += 1
            query_relation = "none"
            if positive:
                if scenario_type == "query_memory_competition":
                    query_relation = "competition"
                elif any(value in mentions for value in positive):
                    query_relation = "complement"
                else:
                    query_relation = "memory_only"
            scenario = {
                "scenario_id": scenario_id,
                "base_query_id": query_row["base_query_id"],
                "query_id": query_row["query_id"],
                "query": query_row["query"],
                "query_mentions": sorted(mentions),
                "query_memory_relation": query_relation,
                "scenario_type": scenario_type,
                "scenario_source": "synthetic_preference_scenario" if positive or negative else "no_memory",
                "memory_before": _memory_state(memory_before_positive, []),
                "memory_after": _memory_state(positive, negative),
                "requires_clarification": requires_clarification,
                "split": split,
                "candidate_pool_ids": item_ids,
                "eligible_candidate_ids": item_ids,
                "pair_ids": [left["item_id"], right["item_id"]],
                "pair_choice": choice,
                "label_provenance": provenance,
                "label_reason": reason,
                "pick_reason": pick_reason,
                "source": "esci_query_bge_catalog_scenario",
            }
            scenarios.append(scenario)
            pair_task = {
                "pair_task_id": f"{scenario_id}:pair",
                "base_query_id": query_row["base_query_id"],
                "scenario_id": scenario_id,
                "query": query_row["query"],
                "scenario_type": scenario_type,
                "scenario_source": scenario["scenario_source"],
                "memory_after": scenario["memory_after"],
                "memory_before": scenario["memory_before"],
                "requires_clarification": requires_clarification,
                "split": split,
                "candidate_a": left,
                "candidate_b": right,
                "choice": choice,
                "label_provenance": provenance,
                "label_reason": reason,
                "candidate_a_id": left["item_id"],
                "candidate_b_id": right["item_id"],
            }
            pairs.append(pair_task)
            annotation.append({
                "pair_task_id": pair_task["pair_task_id"],
                "base_query_id": query_row["base_query_id"],
                "query": query_row["query"],
                "scenario_type": scenario_type,
                "memory_after": scenario["memory_after"],
                "scenario_source": scenario["scenario_source"],
                "requires_clarification": requires_clarification,
                "candidate_a": {
                    "item_id": left["item_id"], "title": left.get("title"),
                    "categories": left.get("categories"), "attributes": left.get("attributes"),
                    "evidence": left.get("evidence"), "unknown_attributes": [a for a in ATTRS if a not in set(left.get("attributes") or [])],
                },
                "candidate_b": {
                    "item_id": right["item_id"], "title": right.get("title"),
                    "categories": right.get("categories"), "attributes": right.get("attributes"),
                    "evidence": right.get("evidence"), "unknown_attributes": [a for a in ATTRS if a not in set(right.get("attributes") or [])],
                },
            })

    # Write benchmark files.
    def _write_jsonl(name: str, rows: list[dict[str, Any]]) -> None:
        (OUT / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    _write_jsonl("scenarios.jsonl", scenarios)
    _write_jsonl("pair_tasks.jsonl", pairs)
    _write_jsonl("annotation_tasks.jsonl", annotation)
    with (OUT / "annotation_template.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_task_id", "choice", "confidence_1_5", "notes"])
        for row in annotation:
            writer.writerow([row["pair_task_id"], "", "", ""])
    (OUT / "annotation_guidelines.md").write_text(
        """# Memory Ranking Benchmark v2 Annotation Guidelines

For each `pair_task_id`, read the query, the active memory state, and the two real
Electronics products. Choose exactly one:

- `A`: product A better satisfies the query and active memory.
- `B`: product B better satisfies the query and active memory.
- `BOTH`: both are acceptable.
- `INSUFFICIENT`: evidence is insufficient / cannot judge.

Rules:

1. Use only the displayed metadata and evidence. Do not assume an attribute is absent
   when it is marked UNKNOWN.
2. Treat the active memory state as the user's current preference; `scenario_type`
   tells you whether this is a normal, update, withdrawal, or conflict case.
3. Conflict scenarios marked `requires_clarification` must be answered
   `INSUFFICIENT`, not forced to A/B.
4. Do not use BGE rank, ShopRanker score, Scorer score, or any model output; they are
   deliberately hidden from the annotation package.
5. The annotation package contains **no generated labels**; HUMAN_GOLD is zero until a
   real reviewer completes it.
""",
        encoding="utf-8",
    )

    # ESCI exact labels for pair candidates are provenance, not a full memory label.
    counts = {
        "base_queries": len(queries),
        "scenarios": len(scenarios),
        "pair_tasks": len(pairs),
        "candidate_items": len(metadata),
        "candidate_pool_rows": sum(len(pools[row["base_query_id"]]["candidate_pool_ids"]) for row in queries if row["base_query_id"] in pools),
        "unique_candidate_ids": len({item_id for row in queries for item_id in pools[row["base_query_id"]]["candidate_pool_ids"]}),
        "memory_states": len({row["memory_after"].__repr__() for row in scenarios}),
        "scenario_types": dict(type_counts),
        "label_provenance": dict(provenance_counts),
        "human_gold": 0,
        "external_relevance": provenance_counts.get("EXTERNAL_RELEVANCE", 0),
        "silver": provenance_counts.get("SILVER", 0),
        "mechanistic": provenance_counts.get("MECHANISTIC", 0),
        "unjudged": provenance_counts.get("UNJUDGED", 0),
        "splits": dict(Counter(row["split"] for row in scenarios)),
        "base_query_split": dict(Counter(_split_for(query["base_query_id"]) for query in queries)),
    }
    manifest = {
        "schema": "agentrecx.memory_ranking_benchmark_v2.manifest.v1",
        "version": "memory_ranking_benchmark_v2",
        "generated_from": {
            "queries": "ESCI test us small_version, exact catalog-overlap selection",
            "candidate_pool": "fixed Original BGE Top-20 over 76,918 catalog items",
            "metadata": "enrichment/electronics_metadata_sidecar.jsonl",
            "catalog": "data/p13r/electronics/catalog_meta.jsonl",
            "catalog_items": 76918,
            "bge_index": "p13r/retrieval/bge_electronics_vectors.npy (76,918 x 768)",
        },
        "counts": counts,
        "label_policy": {
            "HUMAN_GOLD": 0,
            "EXTERNAL_RELEVANCE": "ESCI exact query-product labels only; not memory Gold",
            "SILVER": "rule-derived from verified attribute evidence and scenario state",
            "MECHANISTIC": "conflict / state-consistency cases; no preference Gold",
            "UNJUDGED": "insufficient evidence or no usable memory signal",
        },
        "split_policy": "split by base_query_id; all memory states of the same base query stay in the same split",
        "annotation_package": {
            "tasks": len(annotation),
            "files": ["annotation_tasks.jsonl", "annotation_guidelines.md", "annotation_template.csv"],
            "hidden": ["BGE rank/score", "ShopRanker score", "PreferenceAwareScorer score", "generated labels"],
        },
        "comparable_pairs": len(pairs),
        "candidate_items_with_metadata": sum(1 for row in metadata.values() if not row.get("metadata_missing")),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=1, ensure_ascii=False))
