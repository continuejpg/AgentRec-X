"""P14-D1-R: enrich real pre-target history with recovered Amazon metadata and build pools."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.p14e.catalogue import ElectronicsCatalogue, extract_attributes

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14d1"
OUT_DATA = REPO / "data/p14d1r"
RUNS = REPO / "runs/p14d1r"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A streamed metadata recovery may end with a truncated final line; skip only that line.
            continue
    return rows


def _txt(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ". ".join(str(item) for item in value if str(item).strip())
    return str(value or "")


def _first_sentence(text: str) -> str:
    for delimiter in (". ", "! ", "? "):
        if delimiter in text:
            return text.split(delimiter, 1)[0].strip() + "."
    return text[:180].strip()


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    tasks = load_jsonl(DATA / "dev_tasks.jsonl")
    targets = {row["task_id"]: row for row in load_jsonl(DATA / "dev_targets.jsonl")}
    natural_d1 = {row["task_id"]: row for row in load_jsonl(REPO / "runs/p14d1/pools/natural.jsonl")}
    target5_d1 = {row["task_id"]: row for row in load_jsonl(REPO / "runs/p14d1/pools/target5.jsonl")}

    metadata: dict[str, dict[str, Any]] = {}
    sources = []
    for path in sorted(OUT_DATA.glob("*history_meta.jsonl")):
        n = 0
        for row in load_jsonl(path):
            item_id = str(row.get("item_id"))
            text = _txt(row.get("text") or row.get("metadata") or "")
            title = _txt(row.get("title")) or _first_sentence(text)
            previous = metadata.get(item_id)
            # Prefer official metadata rows over the sampled C4 cache when both exist.
            if previous is None or row.get("source", "").startswith("Amazon-Reviews"):
                metadata[item_id] = {"title": title, "text": text,
                                     "metadata_category": row.get("metadata_category"),
                                     "source": row.get("source") or path.name}
                n += 1
        sources.append({"file": path.name, "rows": n})
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    (OUT_DATA / "history_metadata.jsonl").write_text(
        "".join(json.dumps({"item_id": item_id, **info}, ensure_ascii=False, sort_keys=True) + "\n"
                for item_id, info in sorted(metadata.items())), encoding="utf-8")

    # Enrich each task history and compute fixed summaries and a conservative weak interest proxy.
    enriched_tasks = []
    coverage_rows = []
    for task in tasks:
        target = targets[task["task_id"]]
        target_id = target["target_parent_asin"]
        history = []
        for item in task["history"]:
            info = metadata.get(item["item_id"])
            product = catalogue.get(item["item_id"])
            title = (info or {}).get("title") or (product.title if product else None) or item.get("title")
            text = (info or {}).get("text") or (product.text if product else "") or ""
            history.append({**item, "title": title, "text": text,
                            "metadata_source": (info or {}).get("source", "none"),
                            "metadata_category": (info or {}).get("metadata_category")})
        summary = "\n".join(
            f"- [{entry.get('category', 'Unknown')}] {(entry.get('title') or entry['item_id'])[:160]} "
            f"({'verified' if entry.get('verified_purchase') else 'unverified'})"
            for entry in history[-12:]
        )
        # Weak proxy: observed attributes only, positive-only, no target features, no negatives.
        counts: Counter[tuple[str, str]] = Counter()
        for entry in history:
            product = catalogue.get(entry["item_id"])
            if product is not None:
                for attribute in product.attributes or {}:
                    counts[("attribute", attribute)] += 1
                if product.form_factor:
                    counts[("form_factor", product.form_factor)] += 1
            elif entry.get("text"):
                attributes, form = extract_attributes(entry["text"], entry.get("title") or "")
                for attribute in attributes:
                    counts[("attribute", attribute)] += 1
                if form:
                    counts[("form_factor", form)] += 1
        proxy = [{"kind": kind, "value": value, "polarity": "positive",
                  "evidence_count": count, "source": "history_proxy_observed_attributes"}
                 for (kind, value), count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:5]]
        enriched_tasks.append({"task": task, "history": history, "summary": summary, "proxy": proxy,
                               "target": target, "target_id": target_id})
        coverage_rows.append({"task_id": task["task_id"],
                              "history_items": len(history),
                              "history_with_metadata": sum(1 for entry in history if entry.get("title") or entry.get("text")),
                              "history_with_catalogue_product": sum(1 for entry in history if catalogue.get(entry["item_id"]) is not None),
                              "history_proxy_preferences": len(proxy),
                              "history_verified": sum(1 for entry in history if entry.get("verified_purchase"))})

    # Deterministic shuffled-history donor: other user, length/category-distribution matched, recipient target removed.
    category_counts = [Counter(entry.get("category") for entry in row["history"]) for row in enriched_tasks]
    sorted_universe = sorted(category_counts)
    shuffled = []
    for index, row in enumerate(enriched_tasks):
        best = None
        best_score = None
        for donor_index, donor in enumerate(enriched_tasks):
            if donor["task"]["user_id"] == row["task"]["user_id"]:
                continue
            donor_hist = [entry for entry in donor["history"] if entry["item_id"] != row["target_id"]]
            if not donor_hist:
                continue
            length_diff = abs(len(donor_hist) - len(row["history"]))
            donor_cat = Counter(entry.get("category") for entry in donor_hist)
            cat_diff = sum(abs(donor_cat.get(category, 0) - category_counts[index].get(category, 0))
                           for category in set(donor_cat) | set(category_counts[index]))
            score = (length_diff, cat_diff, donor["task"]["task_id"])
            if best is None or score < best_score:
                best = (donor_hist, donor)
                best_score = score
        if best is None:
            donor_hist = row["history"]
        else:
            donor_hist = best[0]
        donor_summary = "\n".join(
            f"- [{entry.get('category', 'Unknown')}] {(entry.get('title') or entry['item_id'])[:160]} "
            f"({'verified' if entry.get('verified_purchase') else 'unverified'})"
            for entry in donor_hist[-12:]
        )
        shuffled.append({"task_id": row["task"]["task_id"], "history_length": len(donor_hist),
                         "summary": donor_summary, "donor_user": best[1]["task"]["user_id"] if best else None})
    shuffled_by_id = {row["task_id"]: row for row in shuffled}

    # Candidate pools stay identical to P14-D1; only history fields are added.
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "pools").mkdir(parents=True, exist_ok=True)
    for name, source in (("natural", natural_d1), ("target5", target5_d1)):
        output = []
        for row in enriched_tasks:
            tid = row["task"]["task_id"]
            if tid not in source:
                continue
            pool = source[tid]
            output.append({**pool,
                           "history_summary_real": row["summary"],
                           "history_summary_shuffled": shuffled_by_id[tid]["summary"],
                           "history_proxy_preferences": row["proxy"],
                           "history_coverage": {key: value for key, value in
                                                next(item for item in coverage_rows if item["task_id"] == tid).items()
                                                if key != "task_id"}})
        (RUNS / f"pools/{name}.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in output), encoding="utf-8")

    recovered_unique = len(metadata)
    total_unique = len({entry["item_id"] for row in enriched_tasks for entry in row["history"]})
    manifest = {
        "schema": "agentrecx.p14d1r.history_metadata.v1",
        "sources": sources,
        "history_items": sum(len(row["history"]) for row in enriched_tasks),
        "unique_history_items": total_unique,
        "unique_items_with_metadata": recovered_unique,
        "coverage_ratio": round(recovered_unique / total_unique, 4) if total_unique else 0.0,
        "tasks": len(enriched_tasks),
        "tasks_with_any_metadata": sum(1 for row in coverage_rows if row["history_with_metadata"] > 0),
        "items_with_catalogue_product": sum(row["history_with_catalogue_product"] for row in coverage_rows),
        "before_metadata_coverage": {"history_items_in_catalogue": 126, "tasks_with_any_history_metadata": 50,
                                     "total_history_items": 14945, "unique_history_items": 14655},
        "after_metadata_coverage": {
            "history_items_with_metadata": sum(row["history_with_metadata"] for row in coverage_rows),
            "tasks_with_any_metadata": sum(1 for row in coverage_rows if row["history_with_metadata"] > 0),
            "unique_items_with_metadata": recovered_unique,
        },
        "tasks_with_proxy_preferences": sum(1 for row in coverage_rows if row["history_proxy_preferences"] > 0),
        "shuffled_control": {"definition": "other-user history; length/category matched; recipient target removed",
                             "donors_available": sum(1 for row in shuffled if row["donor_user"]),
                             "tasks": len(shuffled)},
        "per_task": coverage_rows,
        "notes": ["recovered metadata is a local read-only artifact; target identity never used",
                  "no synthetic users or preferences", "UNKNOWN remains UNKNOWN"],
    }
    (OUT_DATA / "history_metadata_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                                             encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("sources", "history_items", "unique_history_items",
                                                      "unique_items_with_metadata", "coverage_ratio",
                                                      "tasks_with_any_metadata", "items_with_catalogue_product",
                                                      "after_metadata_coverage", "tasks_with_proxy_preferences")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
