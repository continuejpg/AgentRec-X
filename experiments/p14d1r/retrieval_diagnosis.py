"""P14-D1-R retrieval diagnosis: BM25 query variants + BGE artifact alignment check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.p14e.catalogue import ElectronicsCatalogue

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/p14d1"
RUNS = REPO / "runs/p14d1r"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    catalogue = ElectronicsCatalogue.load()
    tasks = load_jsonl(DATA / "dev_tasks.jsonl")
    targets = {row["task_id"]: row for row in load_jsonl(DATA / "dev_targets.jsonl")}
    metadata = {row["item_id"]: row for row in load_jsonl(REPO / "data/p14d1r/history_metadata.jsonl")}
    rows = []
    for task in tasks:
        tid = task["task_id"]
        target_id = targets[tid]["target_parent_asin"]
        titles = []
        for entry in task["history"][-8:]:
            info = metadata.get(entry["item_id"])
            if info and info.get("title"):
                titles.append(info["title"])
        variants = {
            "original": task["query_original"],
            "rewritten": task["query_rewritten"] or task["query_original"],
            "original_plus_rewritten": task["query_original"] + " " + (task["query_rewritten"] or ""),
            "history_expanded": task["query_original"] + " " + " ".join(titles)[:500],
        }
        row = {"task_id": tid, "target_in_catalogue": bool(targets[tid]["target_in_catalogue"]), "variants": {}}
        for name, query in variants.items():
            result = catalogue.search_source.search(query, limit=100)
            ids = [hit.parent_asin.split(":", 1)[-1] for hit in result.hits]
            target_rank = ids.index(target_id) + 1 if target_id in ids else None
            row["variants"][name] = {"target_rank": target_rank,
                                     "target_in_top20": bool(target_rank and target_rank <= 20),
                                     "target_in_top100": bool(target_rank)}
        rows.append(row)
    summary = {}
    for name in ("original", "rewritten", "original_plus_rewritten", "history_expanded"):
        summary[name] = {"target_in_top20": sum(1 for row in rows if row["variants"][name]["target_in_top20"]),
                         "target_in_top100": sum(1 for row in rows if row["variants"][name]["target_in_top100"])}

    # BGE check without loading dense vectors.
    bge: dict[str, Any] = {"available": False, "aligned": False, "reason": "index missing"}
    index_path = REPO / "runs/p8_bge/bge_base_dense.npz"
    if index_path.is_file():
        z = np.load(index_path, allow_pickle=False)
        identities = z["identities"]
        meta = json.loads(str(z["metadata"][0]))
        catalogue_ids = {product.item_id for product in catalogue._products.values()}
        intersection = len(set(identities) & catalogue_ids)
        sample_titles = [str(t) for t in z["titles"][:3]]
        bge = {"available": True, "aligned": False, "index_path": str(index_path),
               "item_count": int(len(identities)), "sample_titles": sample_titles,
               "intersection_with_p13r_item_ids": intersection,
               "catalogue_fingerprint": meta.get("catalogue_fingerprint"),
               "reason": "index item_count/identity set does not match P13-R Electronics catalogue; "
                         "using it would require rebuilding a full catalogue BGE index (forbidden this round)"}
    result = {"schema": "agentrecx.p14d1r.retrieval_diagnosis.v1", "tasks": len(rows),
              "target_in_catalogue": sum(1 for row in rows if row["target_in_catalogue"]),
              "bm25_query_variants": summary, "bge": bge,
              "protocol": "natural retrieval only; target never injected",
              "notes": ["history_expanded uses recovered pre-history titles only as a retrieval diagnostic",
                        "the candidate pools for reranker ablation remain the original BM25 Top-20 pools"],
              "per_task": rows}
    (RUNS / "retrieval_diagnosis.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("tasks", "target_in_catalogue", "bm25_query_variants", "bge")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
